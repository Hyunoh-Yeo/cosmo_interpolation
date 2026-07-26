"""Show a GOTPM snapshot z-slab as an ASCII density map in the terminal.

A subfile is a thin z-slab (~4.5 cMpc/h), so this is a cross-section of the
cosmic web: bright = clusters/halos, faint threads = filaments, blank = voids.
Needs numpy + gotpm.py (scp both); no matplotlib needed on grammar. Also saves
the full-resolution density map as .npy so you can scp it back and plot it nicely.
"""
import argparse
import numpy as np
from gotpm import read_records, decode_positions


def read_pos(path):
    rec, p = read_records(path)
    return decode_positions(rec, p), float(p["Boxsize(Mpc/h)"])


def paint_density(pos, box, bins, mas="tsc"):
    """Project a thin z-slab onto a 2D (x,y) grid of particle counts.

    Uses pypower's CatalogMesh for proper mass assignment (ngp/cic/tsc); falls back
    to a plain 2D histogram (= NGP) if pypower isn't installed.

    We paint a 3D mesh over the slab and sum along z. That is exactly 2D assignment
    in (x,y): each particle's z-weights form a partition of unity, so they sum to 1.
    """
    if mas != "ngp-hist":
        try:
            from pypower import CatalogMesh
        except ImportError:
            print("(pypower 없음 -> NGP histogram2d로 대체)")
            mas = "ngp-hist"

    if mas == "ngp-hist":
        H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=bins,
                                 range=[[0, box], [0, box]])
        return H, "ngp (histogram2d)"

    zlo, zhi = float(pos[:, 2].min()), float(pos[:, 2].max())
    thick = max(zhi - zlo, 1e-3)
    nz = max(4, int(round(thick / (box / bins))))       # keep cells roughly cubic
    mesh = CatalogMesh(data_positions=pos,
                       boxsize=[box, box, thick],
                       boxcenter=[box / 2.0, box / 2.0, 0.5 * (zlo + zhi)],
                       nmesh=[bins, bins, nz],
                       resampler=mas, interlacing=0,
                       position_type="pos", wrap=True, dtype="f4")
    arr = np.asarray(mesh.to_mesh())                     # counts per cell
    return arr.sum(axis=2), "%s (pypower CatalogMesh, nz=%d)" % (mas, nz)


def ascii_map(H, cols=90, rows=45):
    ny, nx = H.shape
    cols, rows = min(cols, nx), min(rows, ny)
    ys = np.linspace(0, ny, rows + 1).astype(int)
    xs = np.linspace(0, nx, cols + 1).astype(int)
    M = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            blk = H[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            M[i, j] = blk.mean() if blk.size else 0.0
    v = np.log1p(M)
    v = (v - v.min()) / (v.max() - v.min() + 1e-12)
    chars = " .:-=+*#%@"
    return "\n".join("".join(chars[int(k * (len(chars) - 1))] for k in row) for row in v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="one subfile (a z-slab)")
    ap.add_argument("--bins", type=int, default=512, help="density-map resolution")
    ap.add_argument("--out", default="slab.npy", help="save full-res map here")
    ap.add_argument("--png", default=None,
                    help="also save an image here (needs matplotlib; skipped if absent)")
    ap.add_argument("--mas", default="tsc",
                    choices=["ngp", "cic", "tsc", "ngp-hist"],
                    help="mass assignment (pypower); 'ngp-hist' = plain histogram2d")
    args = ap.parse_args()

    pos, box = read_pos(args.path)
    zlo, zhi = float(pos[:, 2].min()), float(pos[:, 2].max())
    H, mas_used = paint_density(pos, box, args.bins, args.mas)
    cell = box / args.bins
    print("particles: %d   z-slab: %.2f-%.2f cMpc/h   box: %.0f cMpc/h"
          % (len(pos), zlo, zhi, box))
    print("mass assignment: %s   (painted total = %.0f)" % (mas_used, H.sum()))
    # --- basic statistics of the projected density field ---
    delta = H / H.mean() - 1.0                       # overdensity  δ = ρ/ρ̄ − 1
    pcts = np.percentile(H.ravel(), [50, 90, 99, 99.9])
    print("cell size: %.1f cMpc/h   density/cell: mean %.2f  max %.0f" % (cell, H.mean(), H.max()))
    print("overdensity δ: std %.3f   void fraction(δ<-0.5): %.1f%%   empty cells: %.1f%%"
          % (delta.std(), 100 * (delta < -0.5).mean(), 100 * (H == 0).mean()))
    print("density percentiles 50/90/99/99.9: %.0f / %.0f / %.0f / %.0f\n" % tuple(pcts))
    print(ascii_map(H))
    np.save(args.out, H)
    print("\nsaved %dx%d density map -> %s" % (args.bins, args.bins, args.out))

    if args.png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("(matplotlib 없음 -> PNG 건너뜀; .npy를 로컬로 가져가 그리세요)")
            return
        fig, ax = plt.subplots(figsize=(6.4, 6.2))
        ax.imshow(np.log1p(H).T, origin="lower", cmap="magma",
                  extent=[0, box, 0, box], interpolation="nearest")
        ax.set_xlabel("x [cMpc/h]"); ax.set_ylabel("y [cMpc/h]")
        ax.set_title("z-slab projected density (log)\n%s" % args.path.split("/")[-2],
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(args.png, dpi=140)
        print("saved image -> %s" % args.png)


if __name__ == "__main__":
    main()
