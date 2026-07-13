"""CPU comparison twin of demo.py -- runs the 4-way accuracy comparison locally
(no CuPy needed), so you can demo it on a laptop.

The linear / quadratic / rbf methods are reimplemented here in NumPy, mirroring
`gpu_interp.py` exactly (multiquadric RBF + linear-poly tail; node-exact
barycentric linear; 2nd-order Taylor quadratic; LOO-CV shape-parameter tuning).
They were verified against SciPy's LinearNDInterpolator / RBFInterpolator. The GP
model is the real `GPCosmologyInterpolator` (torch). The GPU `demo.py` stays
authoritative for scale + timing; this script is for accuracy comparison anywhere.

Run:
    python -m mvinterp.compare            # n=48, all 4 models + GP uncertainty
    python -m mvinterp.compare --plot     # save compare_error.png
"""
import argparse
import numpy as np
from scipy.spatial import Delaunay

from .make_mock import build_dataset, truth_at

try:
    from .gp_interp import GPCosmologyInterpolator
except Exception:
    GPCosmologyInterpolator = None


# ---- shared helpers (mirror gpu_interp.py) ----
def _pdist(a, b):
    d2 = (a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2.0 * a @ b.T
    return np.sqrt(np.clip(d2, 0.0, None))


def _poly1(x):
    return np.concatenate([np.ones((x.shape[0], 1)), x], 1)


def _poly2(x):
    a, b, c = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return np.concatenate([np.ones((x.shape[0], 1)), a, b, c,
                           a * a, b * b, c * c, a * b, a * c, b * c], 1)


def _rbf_L(ts, eps, reg=1e-6):
    M = len(ts)
    Phi = np.sqrt(_pdist(ts, ts) ** 2 + eps * eps) + reg * np.eye(M)
    Pm = _poly1(ts)
    A = np.zeros((M + 4, M + 4)); A[:M, :M] = Phi; A[:M, M:] = Pm; A[M:, :M] = Pm.T
    return np.linalg.inv(A)[:, :M]


def _rbf_q(tq, ts, eps):
    return np.concatenate([np.sqrt(_pdist(tq, ts) ** 2 + eps * eps), _poly1(tq)], 1)


def _select_eps(ts, Ys, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Leave-one-cosmology-out CV (mirrors gpu_interp.select_epsilon)."""
    M = len(ts); Y = Ys.reshape(M, -1)
    base = _pdist(ts, ts)[~np.eye(M, dtype=bool)].mean()
    best = (base, np.inf)
    for sc in scales:
        eps = base * sc; err = 0.0
        for k in range(M):
            idx = [i for i in range(M) if i != k]
            L = _rbf_L(ts[idx], eps); qb = _rbf_q(ts[k:k + 1], ts[idx], eps)
            err += np.sqrt(((qb @ (L @ Y[idx])) - Y[k:k + 1]) ** 2).mean()
        if err / M < best[1]:
            best = (eps, err / M)
    return best[0]


class _Interp:
    """NumPy linear/quadratic/rbf -- same math as gpu_interp.CosmologyInterpolator."""
    def __init__(self, method, eps=None):
        self.method, self.eps = method, eps

    def fit(self, theta):
        th = np.asarray(theta, float)
        self.mu = th.mean(0, keepdims=True); self.sd = th.std(0, keepdims=True) + 1e-12
        self.ts = (th - self.mu) / self.sd
        if self.method == "linear":
            self.tri = Delaunay(self.ts)
        elif self.method == "rbf":
            self.L = _rbf_L(self.ts, self.eps)
        return self

    def predict(self, Y, tq):
        tqs = (np.atleast_2d(np.asarray(tq, float)) - self.mu) / self.sd
        M = len(self.ts); Yf = Y.reshape(M, -1)
        if self.method == "linear":
            s = self.tri.find_simplex(tqs); W = np.zeros((tqs.shape[0], M)); nd = 3
            ins = s >= 0
            T = self.tri.transform[s[ins]]
            b = np.einsum("qij,qj->qi", T[:, :nd, :], tqs[ins] - T[:, nd, :])
            bary = np.concatenate([b, 1 - b.sum(1, keepdims=True)], 1)
            v = self.tri.simplices[s[ins]]
            W[np.repeat(np.where(ins)[0], nd + 1), v.ravel()] = bary.ravel()
            if (~ins).any():
                oi = np.where(~ins)[0]
                W[oi, np.linalg.norm(tqs[oi][:, None] - self.ts[None], axis=2).argmin(1)] = 1.0
            out = W @ Yf
        elif self.method == "quadratic":
            P = _poly2(self.ts)
            C = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ Yf)
            out = _poly2(tqs) @ C
        else:  # rbf
            out = _rbf_q(tqs, self.ts, self.eps) @ (self.L @ Yf)
        return out.reshape(-1, *Y.shape[1:])


# ---- comparison driver ----
def _mi(d, box):
    return (d + 0.5 * box) % box - 0.5 * box


def _poserr(q, dp, tp, box):
    return np.sqrt((_mi(np.mod(q + dp, box) - tp, box) ** 2).sum(-1))


def run_case(name, ds, train_idx, test_theta, true_pos, box, gp_iters=100):
    q = ds["q"]
    disp = _mi(ds["pos"] - q[None], box)
    theta_tr = ds["theta"][train_idx]
    disp_tr = disp[train_idx]
    ts_tr = (theta_tr - theta_tr.mean(0, keepdims=True)) / (theta_tr.std(0, keepdims=True) + 1e-12)

    rng = np.random.default_rng(0)
    samp = rng.choice(disp_tr.shape[1], size=min(2000, disp_tr.shape[1]), replace=False)
    eps = _select_eps(ts_tr, disp_tr[:, samp, :])

    res = {}
    for m in ("linear", "quadratic", "rbf"):
        dp = _Interp(m, eps).fit(theta_tr).predict(disp_tr, test_theta)[0]
        res[m] = _poserr(q, dp, true_pos, box)
    gp_std = None
    if GPCosmologyInterpolator is not None:
        g = GPCosmologyInterpolator(iters=gp_iters).fit(theta_tr, disp_tr[:, samp, :])
        g.prepare(theta_tr)
        res["GP"] = _poserr(q, g.predict(disp_tr, np.atleast_2d(test_theta))[0], true_pos, box)
        gp_std = float(g.predict_uncertainty(np.atleast_2d(test_theta))[0])

    methods = ["linear", "quadratic", "rbf"] + (["GP"] if "GP" in res else [])
    rms = np.sqrt((disp_tr ** 2).sum(-1).mean())
    base = np.median(res["linear"])
    print(f"\n=== {name}  (theta = {np.round(test_theta, 3).tolist()}) ===")
    print(f"  train cosmologies: {len(train_idx)}   rms displacement: {rms:.3f} cMpc/h"
          f"   rbf eps: {eps:.2f}")
    print(f"  {'model':12s} {'pos med':>9s} {'pos 95%':>9s} {'vs linear':>10s}")
    for m in methods:
        pe = res[m]
        print(f"  {m:12s} {np.median(pe):9.4f} {np.percentile(pe,95):9.4f} "
              f"{base/np.median(pe):9.1f}x")
    if gp_std is not None:
        print(f"  GP uncertainty (posterior std): {gp_std:.4f} cMpc/h")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48, help="particles per side (n^3)")
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boxsize", type=float, default=1024.0)
    ap.add_argument("--gp-iters", type=int, default=100)
    ap.add_argument("--plot", action="store_true", help="save compare_error.png")
    args = ap.parse_args()

    Om, w0, wa = [0.21, 0.26, 0.31, 0.36], [-1.3, -1.0, -0.7], [-0.6, 0.0, 0.6]
    grid = np.array([(o, a, b) for o in Om for a in w0 for b in wa], float)
    print(f"CPU comparison (no CuPy).  grid: {args.n}^3 = {args.n**3} particles x "
          f"{len(grid)} cosmologies"
          f"{'' if GPCosmologyInterpolator else '   [GP skipped: torch/gpytorch missing]'}")
    ds = build_dataset(grid, z=args.z, n=args.n, boxsize=args.boxsize, seed=args.seed)
    box = args.boxsize

    tt = np.array([0.285, -0.85, 0.30])
    tp, _ = truth_at(ds, tt)
    resA = run_case("Test A: off-grid intermediate cosmology", ds, np.arange(len(grid)),
                    tt, tp, box, args.gp_iters)

    held = int(np.argmin(np.linalg.norm(grid - np.array([0.26, -1.0, 0.0]), axis=1)))
    tr = np.array([i for i in range(len(grid)) if i != held])
    resB = run_case("Test B: leave-one-out (interior grid point)", ds, tr,
                    grid[held], ds["pos"][held], box, args.gp_iters)

    if args.plot:
        _plot(resA, resB)


def _plot(resA, resB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = {"linear": "tab:gray", "quadratic": "tab:green",
              "rbf": "tab:blue", "GP": "tab:red"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, res) in zip(axes, [("off-grid", resA), ("leave-one-out", resB)]):
        for m, pe in res.items():
            ax.hist(pe, bins=60, histtype="step", color=colors.get(m), label=m, density=True)
        ax.set_xlabel("per-particle position error [cMpc/h]")
        ax.set_ylabel("pdf"); ax.set_title(title); ax.legend()
    fig.suptitle("4-way interpolation error (mock)")
    fig.tight_layout(); fig.savefig("compare_error.png", dpi=130)
    print("\nsaved compare_error.png")


if __name__ == "__main__":
    main()
