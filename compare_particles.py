"""Test the shared-IC assumption: do the SAME particle IDs sit at nearly the same
position across two cosmologies at the same redshift?

Reads subfile 0 (a z-slab) of each cosmology, matches particles by GLOBAL index,
and measures |Δr| between the two. If the assumption holds, most particles overlap
(same z-slab) and |Δr| is small (a few cMpc/h, << 10). Outliers = "sensitive" spots.

Needs only numpy + gotpm.py (scp both to grammar). Saves |Δr| to .npy for a histogram.
NOTE: superseded by verify_shared_ic.py, which scans the whole box without the
slab-boundary bias this one-subfile version has. Kept as the simple reference.
    python3 compare_particles.py fileA fileB [--out dr.npy]
"""
import argparse
import numpy as np
from gotpm import read_records, decode_positions


def read(path):
    rec, p = read_records(path)
    return rec["indx"], decode_positions(rec, p), float(p["Boxsize(Mpc/h)"]), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fileA"); ap.add_argument("fileB")
    ap.add_argument("--out", default="dr.npy")
    args = ap.parse_args()

    iA, pA, box, hA = read(args.fileA)
    iB, pB, _, hB = read(args.fileB)
    zA = 48 / float(hA["Anow"]) - 1
    print("A: Om=%s w0=%s wa=%s  z=%.3f  (%d particles in slab)" %
          (hA["OmegaMatter0"], hA.get("w0 of DE (CPL)"), hA.get("wa of DE (CPL)"), zA, iA.size))
    print("B: Om=%s w0=%s wa=%s        (%d particles in slab)" %
          (hB["OmegaMatter0"], hB.get("w0 of DE (CPL)"), hB.get("wa of DE (CPL)"), iB.size))

    common, ia, ib = np.intersect1d(iA, iB, assume_unique=True, return_indices=True)
    d = pA[ia] - pB[ib]
    d = (d + box / 2) % box - box / 2                 # minimal image (periodic)
    dist = np.sqrt((d * d).sum(1))

    print("\ncommon particle IDs in both slabs: %d  (%.1f%% of slab A)" %
          (common.size, 100 * common.size / iA.size))
    print("|Δr| between the two cosmologies (cMpc/h):")
    print("  median %.3f   90%% %.3f   99%% %.3f   99.9%% %.3f   max %.2f" %
          (np.median(dist), np.percentile(dist, 90), np.percentile(dist, 99),
           np.percentile(dist, 99.9), dist.max()))
    for thr in (1, 5, 10):
        print("  fraction moved > %2d cMpc/h: %.3f%%" % (thr, 100 * (dist > thr).mean()))
    np.save(args.out, dist)
    print("\nsaved |Δr| array -> %s  (scp to Mac for a histogram)" % args.out)


if __name__ == "__main__":
    main()
