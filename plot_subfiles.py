"""Plot the per-subfile z distribution saved by check_subfiles.py.

Two panels:
  (top)    every subfile's z histogram, overlaid -- shows the slabs tiling the box
  (bottom) a 2D image [subfile number vs z]     -- a diagonal band = clean ordered
           slabs; smearing/gaps/overlaps jump out immediately

    python3 plot_subfiles.py subfiles_z.npz --out subfiles_z.png

Run with a matplotlib python (e.g. /opt/anaconda3/bin/python3).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="subfiles_z.png")
    args = ap.parse_args()

    d = np.load(args.npz)
    H, centers, sub = d["H"], d["centers"], d["sub_index"]
    los, his, box = d["los"], d["his"], float(d["box"])
    nsub = H.shape[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8),
                                   gridspec_kw={"height_ratios": [1, 1.3], "hspace": 0.28})

    # (top) overlaid per-subfile z histograms
    cmap = plt.cm.viridis
    for r in range(nsub):
        ax1.plot(centers, H[r], lw=0.7, color=cmap(r / max(nsub - 1, 1)), alpha=0.8)
    ax1.set_xlabel("z position [cMpc/h]")
    ax1.set_ylabel("particles / bin (sampled)")
    ax1.set_title("Per-subfile z distribution (%d subfiles, colored by index)" % nsub)
    ax1.set_xlim(0, box)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(sub.min(), sub.max()))
    fig.colorbar(sm, ax=ax1, label="subfile #", pad=0.01)

    # (bottom) image: rows = subfile, cols = z-bin (log so faint tails show)
    im = ax2.imshow(np.log1p(H), aspect="auto", origin="lower", cmap="magma",
                    extent=[0, box, sub.min(), sub.max()], interpolation="nearest")
    ax2.plot(los, sub, color="cyan", lw=0.8, label="z_min")
    ax2.plot(his, sub, color="lime", lw=0.8, label="z_max")
    ax2.set_xlabel("z position [cMpc/h]")
    ax2.set_ylabel("subfile #")
    ax2.set_title("z distribution vs subfile index  (tight diagonal = clean ordered slabs)")
    ax2.legend(loc="lower right", fontsize=8)
    fig.colorbar(im, ax=ax2, label="log(1+count)", pad=0.01)

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved %s" % args.out)

    # quick text summary of slab regularity
    thick = his - los
    print("slab thickness: mean %.3f  min %.3f  max %.3f cMpc/h" % (thick.mean(), thick.min(), thick.max()))
    if nsub > 1:
        gaps = los[1:] - his[:-1]
        print("consecutive gap: max %.3f  overlap(max) %.3f cMpc/h" % (gaps.max(), (-gaps).max()))


if __name__ == "__main__":
    main()
