"""End-to-end demo: build mock Multiverse snapshots, interpolate to unsimulated
cosmologies on the GPU, and compare RBF vs linear accuracy + timing.

Run on the GPU node (`grammar`):
    python -m mvinterp.demo                 # quick (n=32 -> 32768 particles)
    python -m mvinterp.demo --n 128 --plot  # bigger, save diagnostic figure
Requires CuPy (CUDA). Mock generation/growth run on the host (NumPy/SciPy); all
interpolation runs on the GPU.
"""
import argparse
import numpy as np

from .backend import backend_name, timed, device_mem_gb
from .make_mock import build_dataset, truth_at
from .gpu_interp import CosmologyInterpolator


def make_grid():
    """A small Multiverse-like (Om0, w0, wa) grid: 4 x 3 x 3 = 36 cosmologies."""
    Om = [0.21, 0.26, 0.31, 0.36]
    w0 = [-1.3, -1.0, -0.7]
    wa = [-0.6, 0.0, 0.6]
    return np.array([(o, a, b) for o in Om for a in w0 for b in wa], dtype=float)


def minimal_image(d, box):
    return (d + 0.5 * box) % box - 0.5 * box


def pos_error(pred_pos, true_pos, box):
    d = minimal_image(pred_pos - true_pos, box)
    return np.sqrt((d * d).sum(-1))      # per-particle, cMpc/h


def vec_error(pred, true):
    d = pred - true
    return np.sqrt((d * d).sum(-1))


def run_case(name, ds, train_idx, test_theta, true_pos, true_vel, box, tile):
    q = ds["q"]
    # interpolate DISPLACEMENT (small, no periodic wrap), not absolute position
    disp = minimal_image(ds["pos"] - q[None, :, :], box)      # (M,N,3)
    theta_all = ds["theta"]

    theta_tr = theta_all[train_idx]
    disp_tr = disp[train_idx]
    vel_tr = ds["vel"][train_idx]

    results = {}
    for method in ("linear", "rbf"):
        ip = CosmologyInterpolator(method=method).fit_basis(theta_tr)
        dp = ip.predict(disp_tr, test_theta, tile=tile)[0]
        pp = np.mod(q + dp, box)
        pe = pos_error(pp, true_pos, box)

        iv = CosmologyInterpolator(method=method).fit_basis(theta_tr)
        vp = iv.predict(vel_tr, test_theta, tile=tile)[0]
        ve = vec_error(vp, true_vel)
        results[method] = (pe, ve)

    rms_disp = np.sqrt((disp_tr ** 2).sum(-1).mean())
    rms_vel = np.sqrt((vel_tr ** 2).sum(-1).mean())

    label = {"linear": "linear (baseline)", "rbf": "rbf (in dev)"}
    print(f"\n=== {name}  (theta = Om0={test_theta[0]:.3f}, "
          f"w0={test_theta[1]:.3f}, wa={test_theta[2]:.3f}) ===")
    print(f"  train cosmologies: {len(train_idx)}   "
          f"rms displacement: {rms_disp:.3f} cMpc/h   rms vpec: {rms_vel:.1f} km/s")
    print(f"  {'method':18s} {'pos med':>9s} {'pos 95%':>9s} "
          f"{'(pos med / rms)':>16s} {'vel med':>9s}")
    for m in ("linear", "rbf"):
        pe, ve = results[m]
        print(f"  {label[m]:18s} {np.median(pe):9.4f} {np.percentile(pe,95):9.4f} "
              f"{np.median(pe)/rms_disp:16.4f} {np.median(ve):8.2f} km/s")
    impr = np.median(results["linear"][0]) / np.median(results["rbf"][0])
    print(f"  -> linear interpolation is the deliverable baseline; "
          f"the in-dev RBF is {impr:.1f}x more accurate here")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32, help="particles per side (n^3 total)")
    ap.add_argument("--z", type=float, default=0.0, help="snapshot redshift")
    ap.add_argument("--boxsize", type=float, default=1024.0, help="cMpc/h")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tile", type=int, default=2_000_000, help="particles per GPU tile")
    ap.add_argument("--plot", action="store_true", help="save error_diagnostic.png")
    args = ap.parse_args()

    print(f"backend: {backend_name()}")
    print(f"grid: {args.n}^3 = {args.n**3} particles/cosmology   tile: {args.tile}")

    grid = make_grid()
    with timed("build mock dataset (host)"):
        ds = build_dataset(grid, z=args.z, n=args.n, boxsize=args.boxsize, seed=args.seed)
    box = args.boxsize

    # --- Test A: off-grid intermediate cosmology (the real goal) ---
    test_theta = np.array([0.285, -0.85, 0.30])
    tp, tv = truth_at(ds, test_theta)
    all_idx = np.arange(len(grid))
    resA = run_case("Test A: off-grid intermediate cosmology", ds, all_idx,
                    test_theta, tp, tv, box, args.tile)

    # --- Test B: leave-one-out on an interior grid cosmology ---
    held = int(np.argmin(np.linalg.norm(grid - np.array([0.26, -1.0, 0.0]), axis=1)))
    train_idx = np.array([i for i in range(len(grid)) if i != held])
    resB = run_case("Test B: leave-one-out (interior grid point)", ds, train_idx,
                    grid[held], ds["pos"][held], ds["vel"][held], box, args.tile)

    # --- Timing: predict the full catalog for a batch of cosmologies (GPU) ---
    disp = minimal_image(ds["pos"] - ds["q"][None], box)
    ip = CosmologyInterpolator(method="rbf").fit_basis(ds["theta"])
    batch = np.array([[0.285, -0.85, 0.30], [0.24, -1.1, -0.2], [0.33, -0.9, 0.4],
                      [0.29, -1.2, 0.1], [0.22, -0.8, -0.5], [0.35, -1.05, 0.25],
                      [0.27, -0.95, 0.0], [0.31, -0.7, 0.5]])
    ip.predict(disp, batch[:1], tile=args.tile)        # warm-up (compile kernels)
    store = {}
    with timed(f"predict {len(batch)} cosmologies x {args.n**3} particles", store):
        _ = ip.predict(disp, batch, tile=args.tile)
    key = next(k for k in store if k.startswith("predict"))
    pps = len(batch) * args.n**3 / store[key]
    used, total = device_mem_gb()
    print(f"  throughput: {pps:.3e} particle-cosmologies / s")
    print(f"  GPU memory: {used:.2f} / {total:.2f} GB in use")

    if args.plot:
        _plot(resA, resB)


def _plot(resA, resB):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, (title, res) in zip(axes, [("off-grid", resA), ("leave-one-out", resB)]):
        for m, c in [("linear", "tab:orange"), ("rbf", "tab:blue")]:
            ax.hist(res[m][0], bins=60, histtype="step", color=c, label=m, density=True)
        ax.set_xlabel("per-particle position error  [cMpc/h]")
        ax.set_ylabel("pdf")
        ax.set_title(title)
        ax.legend()
    fig.suptitle("Per-particle interpolation error: RBF vs linear")
    fig.tight_layout()
    fig.savefig("error_diagnostic.png", dpi=130)
    print("\nsaved error_diagnostic.png")


if __name__ == "__main__":
    main()
