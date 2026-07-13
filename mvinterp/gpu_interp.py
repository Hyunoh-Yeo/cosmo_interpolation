"""GPU per-particle interpolation across cosmologies (CuPy / CUDA).

Given M training cosmologies theta_m=(Om0,w0,wa) and, per cosmology, a per-particle
field Y_m (displacement or velocity, shape (N,3)), predict the field at new theta*.

Two methods, both ending in one GEMM over the particle catalog:

  method='linear'  (BASELINE, node-exact)
      Proper linear interpolation on the scattered (Om0,w0,wa) nodes: Delaunay
      triangulation -> barycentric weights of the simplex containing theta*.
      Prediction is a convex blend of the <=4 surrounding cosmologies:
            Y(theta*) = Wbary(theta*) @ Y          # Wbary: (Q, M), <=4 nonzero/row
      Reproduces the simulated cosmologies exactly and works on the real irregular
      Multiverse grid. This is what "simple linear interpolation" means and is the
      deliverable to beat. (Outside the convex hull -> nearest cosmology.)

  method='rbf'   (IN DEVELOPMENT, the "better model")
      Multiquadric RBF + linear polynomial tail via a reusable operator L:
            coeffs(tile) = L @ Y(tile);  Y(theta*) = qbasis(theta*) @ coeffs
      Smooth, captures curvature linear interpolation misses.

Triangulation / weights are tiny host (SciPy) work on the M cosmology points; the
heavy per-particle blend is a cuBLAS GEMM on the GPU, streamed in particle tiles
(W is never formed globally -> scales to N=2048^3).
"""
import numpy as np
from scipy.spatial import Delaunay

from .backend import xp, to_device, asnumpy, free_pool


def _pairwise_dist(a, b):
    """Euclidean distances between rows of a (Q,d) and b (M,d) -> (Q,M)."""
    a2 = (a * a).sum(1)[:, None]
    b2 = (b * b).sum(1)[None, :]
    d2 = a2 + b2 - 2.0 * a @ b.T
    return xp.sqrt(xp.clip(d2, 0.0, None))


_KERNELS = {
    "multiquadric": lambda r, e: xp.sqrt(r * r + e * e),
    "inverse_multiquadric": lambda r, e: 1.0 / xp.sqrt(r * r + e * e),
    "gaussian": lambda r, e: xp.exp(-(r * r) / (e * e)),
}


def _rbf_operator(ts, eps, kernel, reg, poly_fn):
    """Reusable RBF operator L = A^{-1}[:, :M] for the augmented kernel+polynomial system."""
    M = ts.shape[0]
    Phi = _KERNELS[kernel](_pairwise_dist(ts, ts), eps) + reg * xp.eye(M, dtype=ts.dtype)
    Pm = poly_fn(ts)
    p = Pm.shape[1]
    A = xp.zeros((M + p, M + p), dtype=ts.dtype)
    A[:M, :M] = Phi
    A[:M, M:] = Pm
    A[M:, :M] = Pm.T
    return xp.linalg.inv(A)[:, :M]                            # (M+p, M); RHS tail is zero


def _rbf_query(ts_q, ts_tr, eps, kernel, poly_fn):
    K = _KERNELS[kernel](_pairwise_dist(ts_q, ts_tr), eps)
    return xp.concatenate([K, poly_fn(ts_q)], axis=1)


class CosmologyInterpolator:
    def __init__(self, method="rbf", reg=1e-6, epsilon=None,
                 kernel="multiquadric", dtype=xp.float32):
        assert method in ("linear", "quadratic", "rbf")
        assert kernel in _KERNELS
        self.method = method
        self.reg = reg
        self.epsilon = epsilon       # RBF shape param; None -> mean pairwise dist heuristic
        self.kernel = kernel         # multiquadric | inverse_multiquadric | gaussian
        self.dtype = dtype
        self._ready = False

    @staticmethod
    def _poly(ts):
        ones = xp.ones((ts.shape[0], 1), dtype=ts.dtype)
        return xp.concatenate([ones, ts], axis=1)        # [1, Om, w0, wa]

    @staticmethod
    def _poly2(ts):
        # 2nd-order Taylor basis in theta: [1, x, x^2, cross terms] -> 10 columns
        o = xp.ones((ts.shape[0], 1), dtype=ts.dtype)
        a, b, c = ts[:, 0:1], ts[:, 1:2], ts[:, 2:3]
        return xp.concatenate([o, a, b, c, a*a, b*b, c*c, a*b, a*c, b*c], axis=1)

    def fit_basis(self, theta_train):
        """Build the cosmology-only structure (cheap; depends on M, not N)."""
        theta_h = np.asarray(theta_train, dtype=np.float64)
        self._M = theta_h.shape[0]
        self._mu_h = theta_h.mean(0, keepdims=True)
        self._sd_h = theta_h.std(0, keepdims=True) + 1e-12
        self._ts_h = (theta_h - self._mu_h) / self._sd_h     # standardised, host

        if self.method == "linear":
            self._tri = Delaunay(self._ts_h)                 # triangulate cosmology space
            self._L = None                                   # prediction = Wbary @ Y
        elif self.method == "quadratic":
            ts = to_device(self._ts_h, self.dtype)
            P2 = self._poly2(ts)                             # (M,10)
            G = P2.T @ P2 + self.reg * xp.eye(P2.shape[1], dtype=self.dtype)
            self._L = xp.linalg.solve(G, P2.T)              # (10, M) least-squares operator
        else:
            ts = to_device(self._ts_h, self.dtype)
            if self.epsilon is None:
                D = _pairwise_dist(ts, ts)
                self._eps = self.dtype(D[~xp.eye(self._M, dtype=bool)].mean())
            else:
                self._eps = self.dtype(self.epsilon)
            self._L = _rbf_operator(ts, self._eps, self.kernel, self.reg, self._poly)
            self._theta_s = ts
        self._ready = True
        return self

    # ---- barycentric (linear) weights on the host ----
    def _bary_weights(self, tq):
        """tq: standardised query thetas (Q,3) host -> weight matrix (Q,M) host."""
        tri, M, ndim = self._tri, self._M, tq.shape[1]
        s = tri.find_simplex(tq)
        W = np.zeros((tq.shape[0], M))
        ins = s >= 0
        if ins.any():
            si = s[ins]
            T = tri.transform[si]                            # (q, ndim+1, ndim)
            b = np.einsum("qij,qj->qi", T[:, :ndim, :], tq[ins] - T[:, ndim, :])
            bary = np.concatenate([b, 1.0 - b.sum(1, keepdims=True)], axis=1)
            verts = tri.simplices[si]                        # (q, ndim+1)
            rows = np.repeat(np.where(ins)[0], ndim + 1)
            W[rows, verts.ravel()] = bary.ravel()
        if (~ins).any():                                     # outside hull -> nearest
            oi = np.where(~ins)[0]
            dd = np.linalg.norm(tq[oi][:, None, :] - self._ts_h[None], axis=2)
            W[oi, dd.argmin(1)] = 1.0
        return W

    def _query_basis(self, theta_query):
        tq_h = np.atleast_2d(np.asarray(theta_query, dtype=np.float64))
        ts_h = (tq_h - self._mu_h) / self._sd_h
        if self.method == "linear":
            return tq_h.shape[0], to_device(self._bary_weights(ts_h), self.dtype)
        if self.method == "quadratic":
            return tq_h.shape[0], self._poly2(to_device(ts_h, self.dtype))
        ts = to_device(ts_h, self.dtype)
        qb = _rbf_query(ts, self._theta_s, self._eps, self.kernel, self._poly)
        return tq_h.shape[0], qb                                # (Q, M+4)

    # ---- prediction ----
    def predict_tile(self, Y_tile, theta_query):
        """One particle tile. Y_tile: (M, t, 3) -> (Q, t, 3) on device."""
        assert self._ready, "call fit_basis() first"
        Y = to_device(Y_tile, self.dtype)
        t = Y.shape[1]
        Yflat = Y.reshape(self._M, t * 3)
        Q, qb = self._query_basis(theta_query)
        coeffs = Yflat if self._L is None else self._L @ Yflat
        return (qb @ coeffs).reshape(Q, t, 3)

    def predict(self, Y_train, theta_query, tile=2_000_000):
        """Stream the whole catalog in tiles. Y_train: (M,N,3) -> (Q,N,3) host."""
        assert self._ready, "call fit_basis() first"
        N = Y_train.shape[1]
        Q = np.atleast_2d(np.asarray(theta_query)).shape[0]
        out = xp.empty((Q, N, 3), dtype=self.dtype)
        for s in range(0, N, tile):
            e = min(s + tile, N)
            out[:, s:e, :] = self.predict_tile(Y_train[:, s:e, :], theta_query)
        free_pool()
        return asnumpy(out)

    def fit(self, theta_train, Y_train=None):
        """Convenience alias for fit_basis (Y is supplied later to predict)."""
        return self.fit_basis(theta_train)


def select_epsilon(theta_train, Y_sample, kernel="multiquadric",
                   scales=(0.25, 0.5, 1.0, 2.0, 4.0), reg=1e-6, dtype=xp.float32):
    """Pick the RBF shape parameter by leave-one-cosmology-out cross-validation.

    Drop each cosmology in turn, reconstruct it from the others, and measure the
    error on a small particle subsample -- exactly the "predict an unsimulated
    cosmology" task. Returns the best epsilon (float) over base_dist * scales,
    where base_dist is the mean pairwise distance in standardised theta space.

    theta_train: (M,3);  Y_sample: (M, n_sample, 3) a cheap subsample of the field.
    """
    theta_h = np.asarray(theta_train, dtype=np.float64)
    mu = theta_h.mean(0, keepdims=True)
    sd = theta_h.std(0, keepdims=True) + 1e-12
    ts = to_device((theta_h - mu) / sd, dtype)
    M = ts.shape[0]
    Y = to_device(Y_sample, dtype).reshape(M, -1)
    base = float(asnumpy(_pairwise_dist(ts, ts)[~xp.eye(M, dtype=bool)].mean()))
    poly = CosmologyInterpolator._poly

    best_eps, best_err = None, np.inf
    for sc in scales:
        eps = dtype(base * sc)
        err = 0.0
        for k in range(M):
            idx = [i for i in range(M) if i != k]
            tsk, Yk = ts[idx], Y[idx]
            L = _rbf_operator(tsk, eps, kernel, reg, poly)
            qb = _rbf_query(ts[k:k + 1], tsk, eps, kernel, poly)
            pred = qb @ (L @ Yk)
            err += float(asnumpy(xp.sqrt(((pred - Y[k:k + 1]) ** 2).mean())))
        err /= M
        if err < best_err:
            best_eps, best_err = float(base * sc), err
    return best_eps
