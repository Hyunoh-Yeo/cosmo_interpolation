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
from gotpm import read_records, decode_positions, geom

THRESHOLDS = (0.5, 1.0, 2.0, 5.0, 10.0)     # cMpc/h; 10 is the reference criterion
BINS = np.concatenate([[0.0], np.logspace(-4, 3, 351)])
COS_BINS = np.linspace(-1.0, 1.0, 101)      # alignment of dr with the particle's own psi
PSI_BINS = np.concatenate([[0.0], np.logspace(-2, 2, 41)])   # |psi| bins, cMpc/h
NTOP = 20


def load(path):
    rec, p = read_records(path)
    return rec["indx"], decode_positions(rec, p), float(p["Boxsize(Mpc/h)"]), p


# The suite is mid-migration: a cosmology lives in one of these, not both, and a
# directory can exist in the other path as an empty shell (logs only). So resolve a
# bare name by looking for actual particle files rather than by directory existence.
ROOTS = ("/gpfs/mhee7173_snu/Multiverse_CPL",
         "/multiverse/mhee7173_snu/MultiverseCPL")


def subfiles(d, prefix, step):
    """Subfile paths for a snapshot. `d` may be a full path or a bare cosmology name
    (e.g. MV_Om0.26_-1_0), in which case both storage roots are searched."""
    cands = [d] if os.path.sep in d else [os.path.join(r, d) for r in ROOTS]
    for c in cands:
        f = sorted(glob.glob(os.path.join(c, "%s.%05d?????" % (prefix, step))))
        if f:
            if c != d:
                print("resolved %s -> %s" % (d, c))
            return f
    raise SystemExit("no %s.%05d* subfiles for '%s' (looked in: %s)"
                     % (prefix, step, d, ", ".join(cands)))


def minimal_image(d, box):
    return (d + 0.5 * box) % box - 0.5 * box


def lagrangian(indx, params):
    """The particle's initial grid position [N,3] in cMpc/h, decoded from indx."""
    nx, ny, nz, box, mx, mxmy = geom(params)
    q = np.empty((indx.size, 3), np.float32)
    q[:, 0] = (indx % mx) * (box / nx)
    q[:, 1] = ((indx % mxmy) // mx) * (box / ny)
    q[:, 2] = (indx // mxmy) * (box / nz)
    return q


class Accum:
    """Streaming |dr| statistics: histogram + threshold counts + worst offenders.

    If `save_above` is set, EVERY particle with |dr| > save_above is kept (indx,
    |dr|, position in A) so the "sensitive locations" can be mapped, not just the
    top-20. Pick the threshold so the kept set stays manageable (e.g. 5 cMpc/h ->
    ~10k particles here, not the full billions)."""

    def __init__(self, save_above=None):
        self.hist = np.zeros(len(BINS) - 1, np.int64)
        self.over = {t: 0 for t in THRESHOLDS}
        self.n = 0
        self.top_dr = np.zeros(0)
        self.top_id = np.zeros(0, np.int64)
        self.top_pos = np.zeros((0, 3), np.float32)
        self.save_above = save_above
        self.mv_dr, self.mv_id, self.mv_pos = [], [], []      # movers above threshold
        # per-axis displacement sums -> is the motion isotropic or directional?
        self.d_sum = np.zeros(3)          # sum of signed dx, dy, dz
        self.d_sq = np.zeros(3)           # sum of dx^2, dy^2, dz^2
        self.slab = []                    # (j, n, median, max) per A-slab, for PBC checks
        # Is dr aligned with the particle's OWN displacement psi = x_A - q (q from indx)?
        # If cosmology only rescales the displacement (psi_B = (1+e) psi_A) then
        # dr = -e psi_A, i.e. exactly (anti)parallel. cos_hist records the alignment;
        # psi_bin accumulates |dr| binned by |psi|, which says where dr grows and why.
        self.cos_hist = np.zeros(len(COS_BINS) - 1, np.int64)
        self.cos_sum = 0.0
        self.psi_n = np.zeros(len(PSI_BINS) - 1, np.int64)     # count per |psi| bin
        self.psi_dr = np.zeros(len(PSI_BINS) - 1)              # sum |dr| per bin
        self.psi_cos = np.zeros(len(PSI_BINS) - 1)             # sum cos per bin

    def add(self, dr, ids, pos, dvec=None, slab_j=None, psi=None):
        self.n += dr.size
        self.hist += np.histogram(dr, bins=BINS)[0]
        for t in THRESHOLDS:
            self.over[t] += int((dr > t).sum())
        if dvec is not None and dvec.size:
            self.d_sum += dvec.sum(0)
            self.d_sq += (dvec.astype(np.float64) ** 2).sum(0)
        if psi is not None and psi.size:
            pn = np.sqrt((psi ** 2).sum(1))
            ok = (pn > 1e-6) & (dr > 1e-6)
            if ok.any():
                c = (dvec[ok] * psi[ok]).sum(1) / (dr[ok] * pn[ok])
                np.clip(c, -1.0, 1.0, out=c)
                self.cos_hist += np.histogram(c, bins=COS_BINS)[0]
                self.cos_sum += float(c.sum())
                k = np.clip(np.searchsorted(PSI_BINS, pn[ok], "right") - 1,
                            0, len(PSI_BINS) - 2)
                self.psi_n += np.bincount(k, minlength=len(PSI_BINS) - 1)
                self.psi_dr += np.bincount(k, weights=dr[ok], minlength=len(PSI_BINS) - 1)
                self.psi_cos += np.bincount(k, weights=c, minlength=len(PSI_BINS) - 1)
        if slab_j is not None:
            self.slab.append((slab_j, dr.size,
                              float(np.median(dr)) if dr.size else np.nan,
                              float(dr.max()) if dr.size else np.nan))
        if self.save_above is not None:
            m = dr > self.save_above
            if m.any():
                self.mv_dr.append(dr[m]); self.mv_id.append(ids[m]); self.mv_pos.append(pos[m])
        k = min(NTOP, dr.size)
        sel = np.argpartition(dr, -k)[-k:]
        self.top_dr = np.concatenate([self.top_dr, dr[sel]])
        self.top_id = np.concatenate([self.top_id, ids[sel]])
        self.top_pos = np.concatenate([self.top_pos, pos[sel]])
        keep = np.argsort(self.top_dr)[-NTOP:]
        self.top_dr, self.top_id, self.top_pos = \
            self.top_dr[keep], self.top_id[keep], self.top_pos[keep]

    def save_movers(self, path, box):
        """Write (indx, dr, x, y, z) for every saved mover -> .npy of shape (N, 5)."""
        if not self.mv_dr:
            print("no movers above %.2f cMpc/h to save" % self.save_above)
            return
        ids = np.concatenate(self.mv_id).astype(np.float64)
        dr = np.concatenate(self.mv_dr).astype(np.float64)
        pos = np.concatenate(self.mv_pos).astype(np.float64)
        out = np.column_stack([ids, dr, pos])
        np.save(path, out)
        print("\nsaved %d movers > %.2f cMpc/h -> %s  (cols: indx, |dr|, x, y, z)"
              % (out.shape[0], self.save_above, path))

    def percentile(self, q):
        """From the histogram (bin-resolution, adequate on a log grid)."""
        c = np.cumsum(self.hist)
        i = int(np.searchsorted(c, q / 100.0 * self.n))
        return BINS[min(i + 1, len(BINS) - 1)]

    def brief(self, reach):
        """One-line running summary, so a long scan can be read before it finishes.

        `reach` is the window's |dr| ceiling: anything beyond it is unmatched rather
        than measured, so the last bucket is open-ended and labelled that way."""
        edges = [t for t in THRESHOLDS if t < reach]
        parts = ["med %.3f" % self.percentile(50), "99.9%% %.3f" % self.percentile(99.9),
                 "max %.3f" % (self.top_dr.max() if self.top_dr.size else np.nan)]
        for lo, hi in zip(edges, edges[1:] + [reach]):
            parts.append("%g-%g: %d" % (lo, hi, self.over[lo] - self.over.get(hi, 0)))
        return "    " + " | ".join(parts)

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
        # --- is the motion directional, or isotropic? ---
        if self.n and self.d_sq.any():
            mean = self.d_sum / self.n
            rms = np.sqrt(self.d_sq / self.n)
            print("\nper-axis displacement [cMpc/h]   (isotropic => the three rms agree,")
            print("                                   and each mean ~ 0)")
            for k, ax in enumerate("xyz"):
                print("   d%s : mean %+9.5f   rms %8.4f" % (ax, mean[k], rms[k]))
            print("   rms spread across axes: %.2f%%   (large => motion has a preferred axis)"
                  % (100.0 * (rms.max() - rms.min()) / max(rms.mean(), 1e-12)))

        # --- which direction does dr point, and where does it grow? ---
        nc = int(self.cos_hist.sum())
        if nc:
            mean_cos = self.cos_sum / nc
            frac_al = float(self.cos_hist[COS_BINS[:-1] >= 0.9].sum()) / nc
            frac_anti = float(self.cos_hist[COS_BINS[1:] <= -0.9].sum()) / nc
            print("\ndirection of dr vs the particle's own displacement psi = x_A - q:")
            print("   mean cos(dr, psi) = %+0.4f   (0 = random, +-1 = pure rescaling of psi)"
                  % mean_cos)
            print("   |cos| > 0.9 : %.1f%% aligned + %.1f%% anti-aligned = %.1f%% collinear"
                  % (100 * frac_al, 100 * frac_anti, 100 * (frac_al + frac_anti)))
            m = self.psi_n > 0
            if m.any():
                print("\n   |psi| [cMpc/h]      <|dr|>    <cos>     particles")
                idx = np.flatnonzero(m)
                for i in idx[::max(1, len(idx) // 8)]:
                    print("   %6.3f - %-6.3f  %9.4f %+8.3f %13d"
                          % (PSI_BINS[i], PSI_BINS[i + 1], self.psi_dr[i] / self.psi_n[i],
                             self.psi_cos[i] / self.psi_n[i], self.psi_n[i]))
                lo, hi = idx[0], idx[-1]
                r_lo = self.psi_dr[lo] / self.psi_n[lo]
                r_hi = self.psi_dr[hi] / self.psi_n[hi]
                print("   => <|dr|> grows %.0fx from the smallest to the largest |psi| bin"
                      % (r_hi / max(r_lo, 1e-9)))

        # --- do the first/last slabs behave like the interior? (periodic-boundary check) ---
        if len(self.slab) > 6:
            s = np.array(self.slab, float)
            edge = np.concatenate([s[:3], s[-3:]])
            mid = s[len(s) // 2 - 2: len(s) // 2 + 3]
            print("\nslab check (median |dr| per A-slab)   edge vs interior:")
            print("   first/last 3 slabs : %s" % np.round(edge[:, 2], 4).tolist())
            print("   middle 5 slabs     : %s" % np.round(mid[:, 2], 4).tolist())
            r = edge[:, 2].mean() / max(mid[:, 2].mean(), 1e-12)
            print("   edge/interior ratio: %.3f   (~1 => periodic wrap is handled correctly)" % r)
            j_worst = int(s[np.argmax(s[:, 3]), 0])
            print("   slab with the largest single |dr|: %d of %d" % (j_worst, len(s)))

        print("\n%d largest movers -- the 'sensitive locations':" % self.top_dr.size)
        print("   %14s %10s %28s" % ("indx", "|dr|", "position in A [cMpc/h]"))
        for j in np.argsort(self.top_dr)[::-1]:
            print("   %14d %10.4f   (%7.2f, %7.2f, %7.2f)"
                  % (self.top_id[j], self.top_dr[j], *self.top_pos[j]))

    def save_stats(self, path, box, unmatched=0):
        """Dump every accumulator to .npz.

        This is also the unit of parallelism: each worker writes one of these for its
        slab range and merge_stats.py adds them up. Every field is a plain sum or a
        top-N merge, so the merged result is bit-identical to a single-process run."""
        np.savez(path, bins=BINS, hist=self.hist, n=self.n, box=box,
                 unmatched=unmatched,
                 slab=np.array(self.slab, float) if self.slab else np.zeros((0, 4)),
                 d_sum=self.d_sum, d_sq=self.d_sq,
                 thresholds=np.array(THRESHOLDS),
                 over=np.array([self.over[t] for t in THRESHOLDS], float),
                 top_dr=self.top_dr, top_id=self.top_id.astype(float), top_pos=self.top_pos,
                 cos_bins=COS_BINS, cos_hist=self.cos_hist, cos_sum=self.cos_sum,
                 psi_bins=PSI_BINS, psi_n=self.psi_n, psi_dr=self.psi_dr,
                 psi_cos=self.psi_cos)
        print("\nsaved full histogram + slab/axis stats -> %s" % path)


def run_window(fa, fb, w, box, save_above=None, ja=0, jb=None):
    """Unbiased full-box scan by particle identity.

    Multiverse subfiles are Eulerian z-slabs (grouped by present-day position), so a
    given particle sits in different subfiles in cosmology A vs B, and matching within
    one subfile would silently drop the boundary-crossers -- the biggest movers. But a
    particle moves far less than a slab thickness, so A's slab j maps into B's slabs
    j-w .. j+w (periodic in z). Searching that window matches every particle by ID --
    nothing dropped -- with only 2w+1 slabs resident.

    The window bounds |dz| ONLY, not |dr|: slabs are cut along z, so a particle that
    moves far in x/y but little in z still lands in the window and its full 3-D |dr|
    is measured exactly (that is how a max above w*4.3 can be reported). What the
    window cannot see is |dz| > w*4.3 -- those are counted as `unmatched`, never as a
    distance. Motion is close to isotropic, so a large |dr| usually implies a large
    |dz| (~|dr|/sqrt(3)) and does get caught; still, the measured tail is incomplete
    whenever unmatched > 0, and the fix is a larger --window.

    [ja, jb) restricts which of A's slabs this call handles -- the unit of parallelism.
    Only A is partitioned; B is still indexed over the full periodic range, so each
    worker sees exactly the window it would have seen in a single-process run and the
    partition introduces no boundary artefacts.
    """
    acc, cache, unmatched = Accum(save_above), {}, 0
    jb = len(fa) if jb is None else min(jb, len(fa))
    ntot = jb - ja
    every = max(1, ntot // 25)             # ~25 progress lines over the assigned range
    nb = len(fb)
    for j in range(ja, jb):
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

        ia, xa, box, hdr = load(fa[j])
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
        dvec = minimal_image(xa[found] - xb[found], box)
        # psi = how far this particle has moved from its own Lagrangian node since z_init.
        # dr is compared against it to see whether the cosmology difference simply
        # rescales that motion (dr parallel to psi) or redirects it.
        psi = minimal_image(xa[found] - lagrangian(ia[found], hdr), box)
        acc.add(np.sqrt((dvec ** 2).sum(1)), ia[found], xa[found],
                dvec=dvec, slab_j=j, psi=psi)
        if (j - ja + 1) % every == 0 or j == jb - 1:
            print("  %d/%d subfiles, %d compared, %d unmatched (|dz|>%.1f)"
                  % (j - ja + 1, ntot, acc.n, unmatched, w * 4.3))
            print(acc.brief(w * 4.3), flush=True)

    if unmatched:
        print("\n!! %d particles (%.6f%%) were not found within +-%d slabs, i.e. their\n"
              "   Z-DISPLACEMENT |dz| exceeds ~%.1f cMpc/h (the window bounds |dz|, not |dr|).\n"
              "   Their |dr| is unmeasured, so the tail below is incomplete -- re-run with a\n"
              "   larger --window to measure them."
              % (unmatched, 100.0 * unmatched / (acc.n + unmatched), w, w * 4.3))
    else:
        print("\nevery particle was matched within +-%d slabs (|dz| <= %.1f) -- no unmeasured tail."
              % (w, w * 4.3))
    return acc, box, unmatched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--prefix", default="SyncINITIAL")
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--window", type=int, default=3,
                    help="search +-W of B's slabs (W*4.3 cMpc/h of reach); raise if any unmatched")
    ap.add_argument("--max-sub", type=int, default=None, help="limit #subfiles (testing only)")
    ap.add_argument("--sub-range", default=None, metavar="START:END",
                    help="handle only A's subfiles [START,END) -- one parallel worker's "
                         "share; combine the --stats-out files with merge_stats.py")
    ap.add_argument("--save-movers", type=float, default=None, metavar="DR",
                    help="save every particle with |dr| > DR cMpc/h (indx,dr,x,y,z) to --movers-out")
    ap.add_argument("--movers-out", default="movers.npy", help="path for --save-movers output")
    ap.add_argument("--stats-out", default=None, metavar="NPZ",
                    help="dump full |dr| histogram + per-slab + per-axis stats here")
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

    ja, jb = 0, len(fa)
    if args.sub_range:
        ja, jb = (int(v) for v in args.sub_range.split(":"))
        jb = min(jb, len(fa))
        if not 0 <= ja < jb:
            raise SystemExit("bad --sub-range %s (have %d subfiles)" % (args.sub_range, len(fa)))
    print("scanning A subfiles [%d,%d) of %d, %s.%05d (window +-%d) ...\n"
          % (ja, jb, len(fa), args.prefix, args.step, args.window), flush=True)

    acc, box, unmatched = run_window(fa, fb, args.window, box, args.save_movers, ja, jb)
    acc.report(box)
    if args.save_movers is not None:
        acc.save_movers(args.movers_out, box)
    if args.stats_out:
        acc.save_stats(args.stats_out, box, unmatched)


if __name__ == "__main__":
    main()
