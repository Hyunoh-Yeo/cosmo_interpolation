"""Map the 'sensitive locations' saved by verify_shared_ic.py --save-movers.

Input is the (N, 5) array (indx, |dr|, x, y, z). Shows WHERE in the 3D box the
shared-IC premise is least accurate, as the three 2D projections (x-y, x-z, y-z),
each point colored by |dr| (temperature). Looking at all three keeps z from being
hidden -- a cluster that looks tight in x-y might be spread along z, or vice versa.

    python3 plot_movers.py movers.npy --box 1024 --out movers.png

Run with a matplotlib python (e.g. /opt/anaconda3/bin/python3).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 140, "axes.unicode_minus": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npy")
    ap.add_argument("--box", type=float, default=1024.0)
    ap.add_argument("--out", default="movers.png")
    args = ap.parse_args()

    a = np.load(args.npy)
    if a.ndim != 2 or a.shape[1] != 5:
        raise SystemExit("expected (N,5): indx, |dr|, x, y, z")
    dr, x, y, z = a[:, 1], a[:, 2], a[:, 3], a[:, 4]
    order = np.argsort(dr)                      # draw biggest movers on top
    dr, x, y, z = dr[order], x[order], y[order], z[order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.4))
    s = 10 if len(dr) < 5000 else 3
    proj = [("x", "y", x, y), ("x", "z", x, z), ("y", "z", y, z)]
    sc = None
    for ax, (la, lb, pa, pb) in zip(axes, proj):
        sc = ax.scatter(pa, pb, c=dr, s=s, cmap="inferno", vmin=dr.min(), vmax=dr.max())
        ax.set_xlabel("%s [cMpc/h]" % la); ax.set_ylabel("%s [cMpc/h]" % lb)
        ax.set_xlim(0, args.box); ax.set_ylim(0, args.box); ax.set_aspect("equal")
        ax.set_title("%s-%s projection" % (la, lb))
    fig.colorbar(sc, ax=axes, label="|dr| [cMpc/h]", pad=0.01, fraction=0.025)
    fig.suptitle("Sensitive locations: %d particles with |dr| > %.1f cMpc/h "
                 "(z shown via x-z and y-z)" % (len(dr), dr.min()), y=1.02)

    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved %s" % args.out)
    print("movers: %d   |dr| range %.2f .. %.2f cMpc/h" % (len(dr), dr.min(), dr.max()))
    print("z range of movers: %.1f .. %.1f cMpc/h" % (z.min(), z.max()))

    # crude clustering readout: how concentrated are they?
    from collections import Counter
    cell = 20.0
    cells = Counter(zip((x // cell).astype(int), (y // cell).astype(int), (z // cell).astype(int)))
    top = cells.most_common(5)
    print("most populated 20 cMpc/h cells (x,y,z index -> count):")
    for (cx, cy, cz), c in top:
        print("   (%4d,%4d,%4d) -> %d" % (cx * cell, cy * cell, cz * cell, c))


if __name__ == "__main__":
    main()
