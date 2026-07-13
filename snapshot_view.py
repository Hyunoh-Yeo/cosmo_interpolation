"""Show a GOTPM snapshot z-slab as an ASCII density map in the terminal.

A subfile is a thin z-slab (~4.5 cMpc/h), so this is a cross-section of the
cosmic web: bright = clusters/halos, faint threads = filaments, blank = voids.
Self-contained (numpy only) so it runs on grammar without matplotlib. Also saves
the full-resolution density map as .npy so you can scp it back and plot it nicely.
"""
import argparse
import numpy as np

REC = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4"), ("indx", "<i8")])


def read_header(path):
    params = {}
    with open(path, "rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError("reached EOF before header end")
            t = line.decode("latin-1", "replace").rstrip("\n")
            if t.strip() == "#End of Ascii Header":
                return params, f.tell()
            if t.startswith("define") and "=" in t:
                k, v = t[len("define"):].split("=", 1)
                params[k.strip()] = v.strip()


def read_pos(path):
    p, off = read_header(path)
    nx, ny, nz = int(p["Nx"]), int(p["Ny"]), int(p["Nz"])
    box = float(p["Boxsize(Mpc/h)"])
    mx, mxmy = int(p["Mx"]), int(p["MxMy"])
    with open(path, "rb") as f:
        f.seek(off)
        rec = np.fromfile(f, dtype=REC)
    ix = rec["indx"] % mx
    iy = (rec["indx"] % mxmy) // mx
    iz = rec["indx"] // mxmy
    pos = np.empty((rec.size, 3), np.float32)
    pos[:, 0] = np.mod(rec["x"] + ix, nx) * (box / nx)
    pos[:, 1] = np.mod(rec["y"] + iy, ny) * (box / ny)
    pos[:, 2] = np.mod(rec["z"] + iz, nz) * (box / nz)
    return pos, box


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
    args = ap.parse_args()

    pos, box = read_pos(args.path)
    zlo, zhi = float(pos[:, 2].min()), float(pos[:, 2].max())
    H, _, _ = np.histogram2d(pos[:, 0], pos[:, 1], bins=args.bins,
                             range=[[0, box], [0, box]])
    cell = box / args.bins
    print("particles: %d   z-slab: %.2f-%.2f cMpc/h   box: %.0f cMpc/h"
          % (len(pos), zlo, zhi, box))
    # --- basic statistics of the projected density field ---
    delta = H / H.mean() - 1.0                       # overdensity  δ = ρ/ρ̄ − 1
    pcts = np.percentile(H.ravel(), [50, 90, 99, 99.9])
    print("cell size: %.1f cMpc/h   density/cell: mean %.2f  max %.0f" % (cell, H.mean(), H.max()))
    print("overdensity δ: std %.3f   void fraction(δ<-0.5): %.1f%%   empty cells: %.1f%%"
          % (delta.std(), 100 * (delta < -0.5).mean(), 100 * (H == 0).mean()))
    print("density percentiles 50/90/99/99.9: %.0f / %.0f / %.0f / %.0f\n" % tuple(pcts))
    print(ascii_map(H))
    np.save(args.out, H)
    print("\nsaved %dx%d density map -> %s   (scp to your Mac to plot a real image)"
          % (args.bins, args.bins, args.out))


if __name__ == "__main__":
    main()
