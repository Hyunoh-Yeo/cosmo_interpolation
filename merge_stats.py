"""Combine the per-worker .npz files from a parallel verify_shared_ic.py run.

Every accumulator is a plain sum (histogram, threshold counts, particle count,
per-axis sums) or a top-N merge, so the merged result is IDENTICAL to what a
single process scanning all slabs would have produced -- not an approximation.

    python3 merge_stats.py part_*.npz --out stats.npz

Prints the same report a single-process run ends with.
"""
import argparse
import glob
import numpy as np

NTOP = 20


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", help="worker .npz files (globs ok)")
    ap.add_argument("--out", default="stats.npz")
    args = ap.parse_args()

    paths = sorted({p for pat in args.parts for p in glob.glob(pat)} or set(args.parts))
    if not paths:
        raise SystemExit("no part files matched")

    hist = n = unmatched = None
    over = d_sum = d_sq = None
    bins = thresholds = None
    box = 0.0
    slabs, top_dr, top_id, top_pos = [], [], [], []

    for p in paths:
        d = np.load(p)
        if hist is None:
            hist = d["hist"].astype(np.int64).copy()
            over = d["over"].astype(float).copy()
            d_sum = d["d_sum"].astype(float).copy()
            d_sq = d["d_sq"].astype(float).copy()
            bins, thresholds = d["bins"], d["thresholds"]
            n = int(d["n"]); unmatched = int(d["unmatched"]); box = float(d["box"])
            cos_bins, psi_bins = d["cos_bins"], d["psi_bins"]
            cos_hist = d["cos_hist"].astype(np.int64).copy(); cos_sum = float(d["cos_sum"])
            psi_n = d["psi_n"].astype(np.int64).copy()
            psi_dr = d["psi_dr"].astype(float).copy(); psi_cos = d["psi_cos"].astype(float).copy()
        else:
            hist += d["hist"].astype(np.int64)
            over += d["over"].astype(float)
            d_sum += d["d_sum"]; d_sq += d["d_sq"]
            n += int(d["n"]); unmatched += int(d["unmatched"])
            cos_hist += d["cos_hist"].astype(np.int64); cos_sum += float(d["cos_sum"])
            psi_n += d["psi_n"].astype(np.int64)
            psi_dr += d["psi_dr"]; psi_cos += d["psi_cos"]
        slabs.append(d["slab"])
        top_dr.append(d["top_dr"]); top_id.append(d["top_id"]); top_pos.append(d["top_pos"])

    slab = np.concatenate([s for s in slabs if s.size]) if any(s.size for s in slabs) \
        else np.zeros((0, 4))
    slab = slab[np.argsort(slab[:, 0])] if slab.size else slab
    tdr = np.concatenate(top_dr); tid = np.concatenate(top_id); tpos = np.concatenate(top_pos)
    keep = np.argsort(tdr)[-NTOP:]
    tdr, tid, tpos = tdr[keep], tid[keep], tpos[keep]

    np.savez(args.out, bins=bins, hist=hist, n=n, box=box, unmatched=unmatched,
             slab=slab, d_sum=d_sum, d_sq=d_sq, thresholds=thresholds, over=over,
             top_dr=tdr, top_id=tid, top_pos=tpos,
             cos_bins=cos_bins, cos_hist=cos_hist, cos_sum=cos_sum,
             psi_bins=psi_bins, psi_n=psi_n, psi_dr=psi_dr, psi_cos=psi_cos)

    # ---- report, same shape as a single-process run ----
    c = np.cumsum(hist)
    def pct(q):
        i = int(np.searchsorted(c, q / 100.0 * n))
        return bins[min(i + 1, len(bins) - 1)]

    print("merged %d workers, A subfiles covered: %d" % (len(paths), len(slab)))
    print("\n|dr| over %d particles [cMpc/h]   (interparticle spacing %.3f)"
          % (n, box / 2048.0))
    for q in (50, 90, 99, 99.9, 99.99):
        print("   %7s%%  <= %9.4f" % (q, pct(q)))
    print("   %7s   %9.4f" % ("max", tdr.max() if tdr.size else np.nan))
    print("\nfraction exceeding:")
    for t, k in zip(thresholds, over):
        print("   > %5.1f cMpc/h : %12d  (%.8f%%)" % (t, int(k), 100.0 * k / max(n, 1)))
    print("\nunmatched (|dz| beyond the window): %d  (%.6f%%)"
          % (unmatched, 100.0 * unmatched / max(n + unmatched, 1)))

    rms = np.sqrt(d_sq / max(n, 1)); mean = d_sum / max(n, 1)
    print("\nper-axis displacement [cMpc/h]")
    for k, ax in enumerate("xyz"):
        print("   d%s : mean %+9.5f   rms %8.4f" % (ax, mean[k], rms[k]))
    print("   rms spread across axes: %.2f%%" % (100 * (rms.max() - rms.min()) / rms.mean()))

    if cos_hist.sum():
        mc = cos_sum / cos_hist.sum()
        col = (cos_hist[cos_bins[:-1] >= 0.9].sum() + cos_hist[cos_bins[1:] <= -0.9].sum())
        print("\nmean cos(dr, psi) = %+0.4f   collinear (|cos|>0.9): %.1f%%"
              % (mc, 100.0 * col / cos_hist.sum()))
        m = psi_n > 0
        if m.any():
            # the table, not just the endpoints: whether <cos> drifts with |psi| is
            # what separates "linear rescaling, spoiled at large psi" from "never
            # aligned at all", and the endpoints alone cannot show that
            print("\n   %14s %10s %9s %15s" % ("|psi| [cMpc/h]", "<|dr|>", "<cos>", "particles"))
            for i in np.flatnonzero(m):
                print("   %6.3f - %-6.3f %9.4f %+9.3f %15d"
                      % (psi_bins[i], psi_bins[i + 1], psi_dr[i] / psi_n[i],
                         psi_cos[i] / psi_n[i], psi_n[i]))
            i0, i1 = np.flatnonzero(m)[[0, -1]]
            print("   => <|dr|> grows %.1fx from |psi| %.3f to %.1f cMpc/h"
                  % ((psi_dr[i1]/psi_n[i1]) / max(psi_dr[i0]/psi_n[i0], 1e-9),
                     psi_bins[i0], psi_bins[i1+1]))

    if len(slab) > 6:
        med = slab[:, 2]
        edge = np.concatenate([med[:3], med[-3:]]).mean()
        mid = med[len(med) // 2 - 2: len(med) // 2 + 3].mean()
        print("\nslab median |dr|: min %.4f  max %.4f   edge/interior %.3f"
              % (med.min(), med.max(), edge / max(mid, 1e-12)))

    print("\n%d largest movers:" % tdr.size)
    print("   %14s %10s %28s" % ("indx", "|dr|", "position in A [cMpc/h]"))
    for j in np.argsort(tdr)[::-1]:
        print("   %14d %10.4f   (%7.2f, %7.2f, %7.2f)" % (int(tid[j]), tdr[j], *tpos[j]))
    print("\nsaved merged stats -> %s" % args.out)


if __name__ == "__main__":
    main()
