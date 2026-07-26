"""Plot P(k) for one or more cosmologies, plus the ratio to the first one.

The ratio is the quantity that matters for this project: it is the clustering
signal that distinguishes cosmologies, and therefore the thing an interpolated
snapshot has to reproduce.

    python3 plot_pk.py pk_A.npy pk_B.npy --out pk_compare.png

Each .npy is the (2, nk) array written by power_spectrum.py (row0=k, row1=P).
Labels are taken from the filenames unless --labels is given.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def label_from(path):
    b = os.path.basename(path)
    for pre, suf in (("pk_", ""), ("", ".npy")):
        b = b[len(pre):] if b.startswith(pre) else b
        b = b[:-len(suf)] if suf and b.endswith(suf) else b
    return b.replace("MV_", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--kmax", type=float, default=0.3,
                    help="shade k above this as shot-noise dominated")
    ap.add_argument("--out", default="pk_compare.png")
    args = ap.parse_args()

    data = [np.load(f) for f in args.files]
    labels = args.labels or [label_from(f) for f in args.files]
    k0, p0 = data[0]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 7.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08})

    for (k, p), lab in zip(data, labels):
        ax1.loglog(k, p, lw=1.6, label=lab)
    ax1.set_ylabel(r"$P(k)$  $[\mathrm{Mpc}/h]^3$")
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.25, which="both")

    for (k, p), lab in zip(data[1:], labels[1:]):
        ax2.semilogx(k, p / np.interp(k, k0, p0), lw=1.6,
                     label="%s / %s" % (lab, labels[0]))
    ax2.axhline(1.0, color="k", lw=0.8, ls="--")
    ax2.set_xlabel(r"$k$  $[h/\mathrm{Mpc}]$")
    ax2.set_ylabel(r"$P/P_{\rm ref}$")
    ax2.grid(alpha=0.25, which="both")
    if len(data) > 1:
        ax2.legend(fontsize=9)

    for ax in (ax1, ax2):
        ax.axvspan(args.kmax, k0.max(), color="0.85", zorder=0)
    ax1.text(args.kmax * 1.1, ax1.get_ylim()[1] * 0.4,
             "shot-noise dominated\n(lower --stride to extend)",
             fontsize=8.5, color="0.35", va="top")

    fig.suptitle("Multiverse power spectra — real snapshots (pypower CatalogFFTPower)")
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print("saved %s" % args.out)

    if len(data) > 1:
        print("\n%8s  %10s  %10s  %7s" % ("k", labels[0], labels[1], "ratio"))
        for i in range(0, len(k0), max(1, len(k0) // 12)):
            r = np.interp(k0[i], data[1][0], data[1][1]) / p0[i]
            print("%8.4f  %10.4g  %10.4g  %7.4f"
                  % (k0[i], p0[i], np.interp(k0[i], data[1][0], data[1][1]), r))


if __name__ == "__main__":
    main()
