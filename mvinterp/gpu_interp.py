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


class CosmologyInterpolator:
    def __init__(self, method="rbf", reg=1e-6, epsilon=None, dtype=xp.float32):
        assert method in ("linear", "rbf")
        self.method = method
        self.reg = reg
        self.epsilon = epsilon       # RBF shape param; default = mean pairwise dist
        self.dtype = dtype
        self._ready = False

    @staticmethod
    def _poly(ts):
        ones = xp.ones((ts.shape[0], 1), dtype=ts.dtype)
        return xp.concatenate([ones, ts], axis=1)        # [1, Om, w0, wa]

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
        else:
            self._mu = to_device(self._mu_h, self.dtype)
            self._sd = to_device(self._sd_h, self.dtype)
            ts = to_device(self._ts_h, self.dtype)
            Pm = self._poly(ts)                              # (M,4)
            D = _pairwise_dist(ts, ts)
            if self.epsilon is None:
                off = D[~xp.eye(self._M, dtype=bool)]
                self._eps = self.dtype(off.mean())
            else:
                self._eps = self.dtype(self.epsilon)
            Phi = xp.sqrt(D * D + self._eps**2) + self.reg * xp.eye(self._M, dtype=self.dtype)
            m = self._M
            A = xp.zeros((m + 4, m + 4), dtype=self.dtype)
            A[:m, :m] = Phi
            A[:m, m:] = Pm
            A[m:, :m] = Pm.T
            self._L = xp.linalg.inv(A)[:, :m]                # (M+4, M); RHS tail is zero
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
        ts = to_device(ts_h, self.dtype)
        Pq = self._poly(ts)                                  # (Q,4)
        D = _pairwise_dist(ts, self._theta_s)
        K = xp.sqrt(D * D + self._eps**2)
        return tq_h.shape[0], xp.concatenate([K, Pq], axis=1)   # (Q, M+4)

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
