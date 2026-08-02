"""Plot the diagnostics dumped by verify_shared_ic.py --stats-out.

Three panels answering the questions raised in the meeting:
  (1) the FULL |dr| histogram, not just percentiles
  (2) median |dr| per A-slab -- if the first/last slabs differ from the interior,
      the periodic wrap is being mishandled
  (3) per-axis rms displacement -- equal bars mean isotropic motion; one tall bar
      means the motion has a preferred direction

    python3 plot_stats.py stats.npz --out stats.png

Run with a matplotlib python (e.g. /opt/anaconda3/bin/python3).
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "axes.unicode_minus": False})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--out", default="stats.png")
    ap.add_argument("--label", default=None, help="title suffix, e.g. 'same Om 0.26'")
    args = ap.parse_args()

    d = np.load(args.npz)
    bins, hist, n, box = d["bins"], d["hist"], int(d["n"]), float(d["box"])
    slab, d_sum, d_sq = d["slab"], d["d_sum"], d["d_sq"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 4.8),
                                        gridspec_kw={"width_ratios": [1.3, 1.3, 0.8]})

    # (1) full |dr| distribution
    c = 0.5 * (bins[:-1] + bins[1:])
    m = hist > 0
    ax1.step(c[m], hist[m], where="mid", color="#2c7fb8", lw=1.6)
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"$|\Delta r|$ [cMpc/h]"); ax1.set_ylabel("particles per bin")
    ax1.axvline(box / 2048.0, color="0.5", ls=":", lw=1.3)
    ax1.text(box / 2048.0 * 1.1, hist[m].max() * 0.3, "spacing\n0.5", fontsize=8, color="0.4")
    ax1.set_title("full $|\\Delta r|$ distribution  (N = %.3g)" % n)
    ax1.grid(alpha=0.25, which="both")

    # (2) per-slab median -- periodic-boundary sanity
    if slab.size:
        j, med = slab[:, 0], slab[:, 2]
        ax2.plot(j, med, ".-", color="#d95f0e", lw=1.0, ms=4)
        lo, hi = med.min(), med.max()
        ax2.axhline(np.median(med), color="0.4", ls="--", lw=1.0,
                    label="overall median %.4f" % np.median(med))
        for k in (0, len(j) - 1):
            ax2.plot(j[k], med[k], "o", color="crimson", ms=8, zorder=3)
        ax2.set_xlabel("A subfile index $j$ (z-slab)")
        ax2.set_ylabel(r"median $|\Delta r|$ [cMpc/h]")
        ax2.set_title("per-slab median  (red = first/last: PBC check)")
        ax2.set_ylim(lo - 0.05 * (hi - lo + 1e-9), hi + 0.05 * (hi - lo + 1e-9))
        ax2.legend(fontsize=8.5); ax2.grid(alpha=0.25)
        edge = np.concatenate([med[:3], med[-3:]]).mean()
        mid = med[len(med) // 2 - 2: len(med) // 2 + 3].mean()
        ax2.text(0.02, 0.04, "edge/interior = %.3f" % (edge / max(mid, 1e-12)),
                 transform=ax2.transAxes, fontsize=9,
                 color="crimson" if abs(edge / max(mid, 1e-12) - 1) > 0.1 else "green")

    # (3) per-axis rms -- isotropic?
    rms = np.sqrt(d_sq / max(n, 1))
    mean = d_sum / max(n, 1)
    ax3.bar(["x", "y", "z"], rms, color=["#4c72b0", "#55a868", "#c44e52"],
            edgecolor="black", lw=0.6)
    for i, r in enumerate(rms):
        ax3.text(i, r, "%.4f" % r, ha="center", va="bottom", fontsize=9)
    ax3.set_ylabel("rms displacement [cMpc/h]")
    spread = 100.0 * (rms.max() - rms.min()) / max(rms.mean(), 1e-12)
    ax3.set_title("per-axis rms\nspread %.1f%% %s"
                  % (spread, "(isotropic)" if spread < 5 else "(DIRECTIONAL)"))
    ax3.grid(alpha=0.25, axis="y")

    if args.label:
        fig.suptitle(args.label, fontsize=13, y=1.02)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved %s" % args.out)
    print("per-axis rms  x %.5f  y %.5f  z %.5f   spread %.2f%%" % (*rms, spread))
    print("per-axis mean x %+.5f  y %+.5f  z %+.5f  (should be ~0)" % tuple(mean))
    if slab.size:
        med = slab[:, 2]
        print("per-slab median: min %.4f  max %.4f  edge/interior %.3f"
              % (med.min(), med.max(),
                 np.concatenate([med[:3], med[-3:]]).mean()
                 / max(med[len(med) // 2 - 2: len(med) // 2 + 3].mean(), 1e-12)))


if __name__ == "__main__":
    main()
