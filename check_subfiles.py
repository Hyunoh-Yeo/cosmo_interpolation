"""Check how a snapshot's subfiles tile the box, and dump the per-subfile z
distribution for plotting.

run_window assumes subfile j is a contiguous z-slab, in order, non-overlapping
(so A's slab j maps to B's slabs j+-w). This tool certifies that: it prints each
subfile's z-range (gaps/overlaps flagged) AND histograms z for every subfile into
a single .npz you can scp back and plot with plot_subfiles.py.

    # on grammar (numpy + gotpm.py):
    python3 check_subfiles.py COSMO_DIR --prefix SyncINITIAL --step 1881
    python3 check_subfiles.py COSMO_DIR --prefix SyncINITIAL --step 1881 --every 25   # faster

    # back on the Mac:
    python3 plot_subfiles.py subfiles_z.npz --out subfiles_z.png

Needs numpy + gotpm.py.
"""
import argparse
import numpy as np
from gotpm import list_subfiles, read_records, decode_positions, geom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cosmo_dir")
    ap.add_argument("--prefix", default="SyncINITIAL")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--every", type=int, default=1, help="sample every Nth subfile")
    ap.add_argument("--stride", type=int, default=256,
                    help="mmap stride within each subfile (z-distribution needs a sample, not all)")
    ap.add_argument("--zbins", type=int, default=1024, help="histogram bins across the box")
    ap.add_argument("--out", default="subfiles_z.npz", help="save per-subfile z histograms here")
    args = ap.parse_args()

    files = list_subfiles(args.cosmo_dir, args.step, args.prefix)
    if not files:
        raise SystemExit("no subfiles for %s.%05d in %s" % (args.prefix, args.step, args.cosmo_dir))
    idxs = list(range(0, len(files), args.every))

    # box size from the first header, to fix a common histogram grid
    _, p0 = read_records(files[0], stride=args.stride)
    box = geom(p0)[3]
    edges = np.linspace(0.0, box, args.zbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    print("%d subfiles total; sampling every %d; box %.0f cMpc/h\n" % (len(files), args.every, box))
    print("%4s %10s %10s %10s %10s   %s" % ("sub", "z_min", "z_max", "thick", "Np(samp)", "note"))

    H = np.zeros((len(idxs), args.zbins), np.int64)     # [subfile, z-bin] counts
    los = np.zeros(len(idxs)); his = np.zeros(len(idxs))
    prev_hi = None
    for row, i in enumerate(idxs):
        rec, p = read_records(files[i], stride=args.stride)
        z = decode_positions(rec, p)[:, 2]
        H[row] = np.histogram(z, bins=edges)[0]
        lo, hi = float(z.min()), float(z.max())
        los[row], his[row] = lo, hi
        note = ""
        if prev_hi is not None:
            gap = lo - prev_hi
            if gap > 1.0:
                note = "GAP %.1f" % gap
            elif gap < -1.0:
                note = "OVERLAP %.1f" % (-gap)
        prev_hi = hi
        print("%4d %10.3f %10.3f %10.3f %10d   %s" % (i, lo, hi, hi - lo, rec.size, note))

    ordered = bool(np.all(np.diff(los) > 0))
    print("\nz_min increases with subfile number: %s" % ordered)
    print("overall z coverage: %.2f .. %.2f cMpc/h (box ~0..%.0f)" % (los.min(), his.max(), box))
    if ordered and args.every == 1:
        gaps = los[1:] - his[:-1]
        ok = gaps.max() < 2.0 and (-gaps).max() < 2.0
        print("max gap %.3f   max overlap %.3f   =>  run_window assumption %s"
              % (gaps.max(), (-gaps).max(), "HOLDS" if ok else "SHAKY -- inspect notes"))
    elif ordered:
        print("=> ordered on sampled subfiles; rerun with --every 1 to certify every gap")

    np.savez(args.out, H=H, centers=centers, sub_index=np.array(idxs),
             los=los, his=his, box=box)
    print("\nsaved per-subfile z histograms -> %s  (scp to Mac; plot with plot_subfiles.py)" % args.out)


if __name__ == "__main__":
    main()
