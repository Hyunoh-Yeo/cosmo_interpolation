"""Full-box test of the shared-initial-conditions premise.

Does the same Lagrangian particle sit at (nearly) the same place in two different
cosmologies?  `compare_particles.py` answered this for ONE subfile, but a subfile
is a slab and matching *within* a slab silently drops the particles that crossed
its boundary -- i.e. exactly the ones that moved the most.  Its maximum |dr| is
therefore a lower bound, not a measurement.  This tool scans the whole box by
particle identity (see run_window), so nothing is selected by where it ended up.

    # full scan of both snapshots -- submit it, it reads ~275 GB x2
    python3 verify_shared_ic.py DIR_A DIR_B --prefix SyncINITIAL --step 1881

    # smoke test on a few slabs first
    python3 verify_shared_ic.py DIR_A DIR_B --prefix SyncINITIAL --step 1881 --max-sub 8

Needs numpy + gotpm.py (scp both to grammar).
"""
import argparse
import glob
import os
import numpy as np
from gotpm import read_records, decode_positions

THRESHOLDS = (0.5, 1.0, 2.0, 5.0, 10.0)     # cMpc/h; 10 is Dr. Hong's criterion
BINS = np.concatenate([[0.0], np.logspace(-4, 3, 351)])
NTOP = 20


def load(path):
    rec, p = read_records(path)
    return rec["indx"], decode_positions(rec, p), float(p["Boxsize(Mpc/h)"]), p


def subfiles(d, prefix, step):
    f = sorted(glob.glob(os.path.join(d, "%s.%05d?????" % (prefix, step))))
    if not f:
        raise SystemExit("no %s.%05d* subfiles in %s" % (prefix, step, d))
    return f


def minimal_image(d, box):
    return (d + 0.5 * box) % box - 0.5 * box


class Accum:
    """Streaming |dr| statistics: histogram + threshold counts + worst offenders."""

    def __init__(self):
        self.hist = np.zeros(len(BINS) - 1, np.int64)
        self.over = {t: 0 for t in THRESHOLDS}
        self.n = 0
        self.top_dr = np.zeros(0)
        self.top_id = np.zeros(0, np.int64)
        self.top_pos = np.zeros((0, 3), np.float32)

    def add(self, dr, ids, pos):
        self.n += dr.size
        self.hist += np.histogram(dr, bins=BINS)[0]
        for t in THRESHOLDS:
            self.over[t] += int((dr > t).sum())
        k = min(NTOP, dr.size)
        sel = np.argpartition(dr, -k)[-k:]
        self.top_dr = np.concatenate([self.top_dr, dr[sel]])
        self.top_id = np.concatenate([self.top_id, ids[sel]])
        self.top_pos = np.concatenate([self.top_pos, pos[sel]])
        keep = np.argsort(self.top_dr)[-NTOP:]
        self.top_dr, self.top_id, self.top_pos = \
            self.top_dr[keep], self.top_id[keep], self.top_pos[keep]

    def percentile(self, q):
        """From the histogram (bin-resolution, adequate on a log grid)."""
        c = np.cumsum(self.hist)
        i = int(np.searchsorted(c, q / 100.0 * self.n))
        return BINS[min(i + 1, len(BINS) - 1)]

    def report(self, box):
        print("\n|dr| over %d particles [cMpc/h]   (interparticle spacing %.3f)"
              % (self.n, box / 2048.0))
        for q in (50, 90, 99, 99.9, 99.99):
            print("   %7s%%  <= %9.4f" % (q, self.percentile(q)))
        print("   %7s   %9.4f" % ("max", self.top_dr.max() if self.top_dr.size else np.nan))
        print("\nfraction exceeding:")
        for t in THRESHOLDS:
            n = self.over[t]
            print("   > %5.1f cMpc/h : %12d  (%.8f%%)" % (t, n, 100.0 * n / max(self.n, 1)))
        print("\n%d largest movers -- the 'sensitive locations':" % self.top_dr.size)
        print("   %14s %10s %28s" % ("indx", "|dr|", "position in A [cMpc/h]"))
        for j in np.argsort(self.top_dr)[::-1]:
            print("   %14d %10.4f   (%7.2f, %7.2f, %7.2f)"
                  % (self.top_id[j], self.top_dr[j], *self.top_pos[j]))


def run_window(fa, fb, w, box):
    """Unbiased full-box scan by particle identity.

    Multiverse subfiles are Eulerian z-slabs (grouped by present-day position), so a
    given particle sits in different subfiles in cosmology A vs B, and matching within
    one subfile would silently drop the boundary-crossers -- the biggest movers. But a
    particle moves far less than a slab thickness, so A's slab j maps into B's slabs
    j-w .. j+w (periodic in z). Searching that window matches every particle by ID --
    nothing dropped -- with only 2w+1 slabs resident. Anything still unmatched moved
    more than w slab-thicknesses, which is itself the headline result.
    """
    acc, cache, unmatched = Accum(), {}, 0
    every = max(1, len(fa) // 25)          # ~25 progress lines over a full snapshot
    nb = len(fb)
    for j in range(len(fa)):
        # the box is periodic in z, so slab 0 and slab nb-1 are neighbours
        win = [(j + d) % nb for d in range(-w, w + 1)]
        for m in win:
            if m not in cache:
                im, xm, box, _ = load(fb[m])
                o = np.argsort(im)
                cache[m] = (im[o], xm[o])
        for m in list(cache):
            if m not in win:
                del cache[m]

        ia, xa, box, _ = load(fa[j])
        xb = np.full_like(xa, np.nan)
        todo = np.ones(ia.size, bool)
        for m in win:
            im, xm = cache[m]
            k = np.searchsorted(im, ia[todo])
            k[k >= im.size] = 0
            hit = im[k] == ia[todo]
            rows = np.flatnonzero(todo)[hit]
            xb[rows] = xm[k[hit]]
            todo[rows] = False
            if not todo.any():
                break
        found = ~todo
        unmatched += int(todo.sum())
        acc.add(np.sqrt((minimal_image(xa[found] - xb[found], box) ** 2).sum(1)),
                ia[found], xa[found])
        if (j + 1) % every == 0 or j == len(fa) - 1:
            print("  %d/%d subfiles, %d compared, %d unmatched"
                  % (j + 1, len(fa), acc.n, unmatched), flush=True)

    if unmatched:
        print("\n!! %d particles (%.6f%%) were not found within +-%d slabs, i.e. they moved\n"
              "   more than ~%.1f cMpc/h. Re-run with a larger --window to measure them."
              % (unmatched, 100.0 * unmatched / (acc.n + unmatched), w, w * 4.3))
    else:
        print("\nevery particle was matched within +-%d slabs -- no unmeasured tail." % w)
    return acc, box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--prefix", default="SyncINITIAL")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--window", type=int, default=3,
                    help="search +-W of B's slabs (W*4.3 cMpc/h of reach); raise if any unmatched")
    ap.add_argument("--max-sub", type=int, default=None, help="limit #subfiles (testing only)")
    args = ap.parse_args()

    fa = subfiles(args.dir_a, args.prefix, args.step)
    fb = subfiles(args.dir_b, args.prefix, args.step)
    if len(fa) != len(fb):
        print("!! different subfile counts (%d vs %d)" % (len(fa), len(fb)))
    if args.max_sub:
        fa, fb = fa[:args.max_sub], fb[:args.max_sub]

    ia, xa, box, ha = load(fa[0])
    _, _, _, hb = load(fb[0])
    za, zb = 48.0 / float(ha["Anow"]) - 1.0, 48.0 / float(hb["Anow"]) - 1.0
    print("A: Om=%s w0=%s wa=%s  z=%.4f" % (ha["OmegaMatter0"],
          ha.get("w0 of DE (CPL)"), ha.get("wa of DE (CPL)"), za))
    print("B: Om=%s w0=%s wa=%s  z=%.4f" % (hb["OmegaMatter0"],
          hb.get("w0 of DE (CPL)"), hb.get("wa of DE (CPL)"), zb))
    if abs(za - zb) > 1e-3:
        print("!! redshifts differ -- part of any |dr| is growth, not cosmology")
    print("scanning %d subfile pairs of %s.%05d (window +-%d) ...\n"
          % (len(fa), args.prefix, args.step, args.window), flush=True)

    acc, box = run_window(fa, fb, args.window, box)
    acc.report(box)


if __name__ == "__main__":
    main()
