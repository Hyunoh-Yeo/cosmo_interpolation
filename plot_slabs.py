"""Plot density-map .npy files (from snapshot_view.py) side by side as a slide.

Run LOCALLY (needs matplotlib). Bring the .npy files back from grammar first.
    python plot_slabs.py slab_A.npy slab_B.npy [slab_C.npy] [--box 1024] [--out slabs.png]
Each panel is a thin z-slab's projected density (log scale) — the cosmic web.
Similar cosmologies (shared ICs) should show the SAME structures in the SAME places.
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="one or more .npy density maps")
    ap.add_argument("--box", type=float, default=1024.0, help="cMpc/h")
    ap.add_argument("--out", default="slabs.png")
    ap.add_argument("--titles", nargs="*", default=None, help="optional per-panel titles")
    args = ap.parse_args()

    n = len(args.files)
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    axes = axes[0]
    for i, (ax, f) in enumerate(zip(axes, args.files)):
        H = np.load(f)
        img = np.log1p(H).T                          # log density, transpose so x→right, y→up
        ax.imshow(img, origin="lower", cmap="magma",
                  extent=[0, args.box, 0, args.box], interpolation="nearest")
        title = args.titles[i] if args.titles and i < len(args.titles) else \
            f.replace(".npy", "")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x [cMpc/h]")
        if i == 0:
            ax.set_ylabel("y [cMpc/h]")
    fig.suptitle("z-slab projected density (log) — cosmic web", fontsize=13)
    fig.tight_layout()
    fig.savefig(args.out, dpi=140)
    print("saved", args.out)


if __name__ == "__main__":
    main()
