"""CPU comparison of the interpolation methods — runs anywhere (no CuPy).

Methods are the standard library implementations; only the GPU/tiled version in
`gpu_interp.py` is hand-written (SciPy is CPU-only and cannot tile over 2048^3):

  linear    -> scipy.interpolate.LinearNDInterpolator   (Delaunay barycentric, node-exact)
  quadratic -> numpy least squares on a 2nd-order polynomial basis
  rbf       -> scipy.interpolate.RBFInterpolator (multiquadric + degree-1 tail),
               shape parameter chosen by leave-one-cosmology-out CV
  GP        -> gp_interp.GPCosmologyInterpolator (torch/gpytorch) + uncertainty

Run:
    python -m mvinterp.compare            # n=48, all 4 models
    python -m mvinterp.compare --plot     # save compare_error.png
"""
import argparse
import numpy as np
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, RBFInterpolator

from .make_mock import build_dataset, truth_at

try:
    from .gp_interp import GPCosmologyInterpolator
except Exception:
    GPCosmologyInterpolator = None


def _standardize(theta, mu=None, sd=None):
    if mu is None:
        mu, sd = theta.mean(0, keepdims=True), theta.std(0, keepdims=True) + 1e-12
    return (theta - mu) / sd, mu, sd


def _poly2(x):
    a, b, c = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return np.concatenate([np.ones((x.shape[0], 1)), a, b, c,
                           a * a, b * b, c * c, a * b, a * c, b * c], 1)


def select_epsilon(ts, Y, scales=(0.25, 0.5, 1.0, 2.0, 4.0)):
    """Leave-one-cosmology-out CV for SciPy's RBF shape parameter."""
    M = len(ts)
    # scipy's multiquadric uses eps as an inverse length -> scan around 1/mean-distance
    d = np.linalg.norm(ts[:, None, :] - ts[None, :, :], axis=2)
    base = 1.0 / d[~np.eye(M, dtype=bool)].mean()
    best, best_err = base, np.inf
    for sc in scales:
        eps, err = base * sc, 0.0
        for k in range(M):
            idx = [i for i in range(M) if i != k]
            try:
                f = RBFInterpolator(ts[idx], Y[idx], kernel="multiquadric",
                                    epsilon=eps, degree=1)
                err += np.sqrt(((f(ts[k:k + 1]) - Y[k:k + 1]) ** 2).mean())
            except Exception:
                err = np.inf
                break
        if err / M < best_err:
            best, best_err = eps, err / M
    return best


def predict(method, theta_tr, Y, tq, eps=None):
    """Y: (M, N, 3) -> (1, N, 3) at the query cosmology."""
    ts, mu, sd = _standardize(np.asarray(theta_tr, float))
    tqs = (np.atleast_2d(np.asarray(tq, float)) - mu) / sd
    M = len(ts)
    Yf = Y.reshape(M, -1)

    if method == "linear":
        out = LinearNDInterpolator(ts, Yf)(tqs)
        bad = np.isnan(out).any(-1)                       # outside the convex hull
        if bad.any():
            out[bad] = NearestNDInterpolator(ts, Yf)(tqs[bad])
    elif method == "quadratic":
        P = _poly2(ts)
        C = np.linalg.solve(P.T @ P + 1e-6 * np.eye(P.shape[1]), P.T @ Yf)
        out = _poly2(tqs) @ C
    elif method == "rbf":
        out = RBFInterpolator(ts, Yf, kernel="multiquadric", epsilon=eps, degree=1)(tqs)
    else:
        raise ValueError(method)
    return out.reshape(-1, *Y.shape[1:])


def _mi(d, box):
    return (d + 0.5 * box) % box - 0.5 * box


def run_case(name, ds, train_idx, test_theta, true_pos, box, gp_iters=100):
    q = ds["q"]
    disp = _mi(ds["pos"] - q[None], box)
    theta_tr, disp_tr = ds["theta"][train_idx], disp[train_idx]
    ts_tr, _, _ = _standardize(theta_tr)

    rng = np.random.default_rng(0)
    samp = rng.choice(disp_tr.shape[1], size=min(2000, disp_tr.shape[1]), replace=False)
    eps = select_epsilon(ts_tr, disp_tr[:, samp, :].reshape(len(theta_tr), -1))

    res = {}
    for m in ("linear", "quadratic", "rbf"):
        dp = predict(m, theta_tr, disp_tr, test_theta, eps)[0]
        res[m] = np.sqrt((_mi(np.mod(q + dp, box) - true_pos, box) ** 2).sum(-1))

    gp_std = None
    if GPCosmologyInterpolator is not None:
        g = GPCosmologyInterpolator(iters=gp_iters).fit(theta_tr, disp_tr[:, samp, :])
        g.prepare(theta_tr)
        dp = g.predict(disp_tr, np.atleast_2d(test_theta))[0]
        res["GP"] = np.sqrt((_mi(np.mod(q + dp, box) - true_pos, box) ** 2).sum(-1))
        gp_std = float(g.predict_uncertainty(np.atleast_2d(test_theta))[0])

    methods = ["linear", "quadratic", "rbf"] + (["GP"] if "GP" in res else [])
    base = np.median(res["linear"])
    rms = np.sqrt((disp_tr ** 2).sum(-1).mean())
    print("\n=== %s  (theta = %s) ===" % (name, np.round(test_theta, 3).tolist()))
    print("  train cosmologies: %d   rms displacement: %.3f cMpc/h   rbf eps: %.3f"
          % (len(train_idx), rms, eps))
    print("  %-12s %9s %9s %10s" % ("model", "pos med", "pos 95%", "vs linear"))
    for m in methods:
        print("  %-12s %9.4f %9.4f %9.1fx"
              % (m, np.median(res[m]), np.percentile(res[m], 95), base / np.median(res[m])))
    if gp_std is not None:
        print("  GP uncertainty (posterior std): %.4f cMpc/h" % gp_std)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--z", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--boxsize", type=float, default=1024.0)
    ap.add_argument("--gp-iters", type=int, default=100)
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    Om, w0, wa = [0.21, 0.26, 0.31, 0.36], [-1.3, -1.0, -0.7], [-0.6, 0.0, 0.6]
    grid = np.array([(o, a, b) for o in Om for a in w0 for b in wa], float)
    print("CPU comparison (SciPy + GPyTorch).  %d^3 particles x %d cosmologies%s"
          % (args.n, len(grid), "" if GPCosmologyInterpolator else "   [GP skipped]"))
    ds = build_dataset(grid, z=args.z, n=args.n, boxsize=args.boxsize, seed=args.seed)
    box = args.boxsize

    tt = np.array([0.285, -0.85, 0.30])
    tp, _ = truth_at(ds, tt)
    resA = run_case("Test A: off-grid intermediate cosmology", ds,
                    np.arange(len(grid)), tt, tp, box, args.gp_iters)

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
