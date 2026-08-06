"""Two-point correlation function xi(r) from P(k), via FFTLog.

Direct pair counting at 2048^3 is only affordable at small separations -- the
Multiverse density is 8 particles per (cMpc/h)^3, so the pair count grows as
r_max^3 and reaches ~50 core-hours by r_max = 20 cMpc/h. Transforming the power
spectrum instead costs seconds and covers all scales the box supports, which makes
it the right first look; direct counting is worth it later only where FFTLog's
resolution limits bite.

    # one cosmology
    python3 correlation.py pk_A.npy --out xi_A.npy

    # two, with the ratio -- the quantity an interpolated snapshot must reproduce
    python3 correlation.py pk_A.npy pk_B.npy --out xi.npy

Input is the (2, nk) array power_spectrum.py writes (row0 = k, row1 = P).
Needs cosmoprimo (already required for the growth factor).

Accuracy, measured rather than assumed:
  * against an analytic Gaussian pair over the full k range, 1e-5 relative.
  * with the input truncated to what we actually measure (k = 0.006-1.57, 120 bins)
    and a realistic power-law shape P ~ k^-2.3, the power-law extrapolation keeps it
    to 0.05% at r = 1-10 and 0.3% at r = 10-100 cMpc/h.
  * so no damping is needed, and damping actively hurts (4% error at damping = 1).
    It is kept only for genuinely noisy high-k input.
  * shot noise is already subtracted by pypower, but its residual scatter at high k
    still transforms into small-r noise.
"""
import argparse
import numpy as np


def load_pk(path):
    a = np.load(path)
    if a.ndim != 2 or a.shape[0] != 2:
        raise SystemExit("%s: expected a (2, nk) array of k and P(k)" % path)
    k, p = a
    m = np.isfinite(k) & np.isfinite(p) & (k > 0)
    return k[m], p[m]


def extend(k, p, kmin, kmax, n=2048):
    """Resample onto a wide log grid, extrapolating as a power law at both ends.

    FFTLog assumes the input spans the whole transform range; padding with a fitted
    slope keeps the tails smooth instead of introducing a step at the data edges."""
    lk, lp = np.log(k), np.log(np.clip(p, 1e-30, None))
    n_end = max(3, len(k) // 10)
    s_lo = np.polyfit(lk[:n_end], lp[:n_end], 1)[0]
    s_hi = np.polyfit(lk[-n_end:], lp[-n_end:], 1)[0]
    lknew = np.linspace(np.log(kmin), np.log(kmax), n)
    lpnew = np.interp(lknew, lk, lp)
    lpnew[lknew < lk[0]] = lp[0] + s_lo * (lknew[lknew < lk[0]] - lk[0])
    lpnew[lknew > lk[-1]] = lp[-1] + s_hi * (lknew[lknew > lk[-1]] - lk[-1])
    return np.exp(lknew), np.exp(lpnew), s_lo, s_hi


def pk_to_xi(k, p, kmin, kmax, damping):
    from cosmoprimo.fftlog import PowerToCorrelation
    kk, pp, s_lo, s_hi = extend(k, p, kmin, kmax)
    if damping:                       # Gaussian cut-off to suppress ringing from k_max
        pp = pp * np.exp(-(kk * damping) ** 2)
    r, xi = PowerToCorrelation(kk, complex=False)(pp)
    return np.asarray(r), np.asarray(xi), s_lo, s_hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pk", nargs="+", help="pk_*.npy from power_spectrum.py (1 or 2)")
    ap.add_argument("--kmin", type=float, default=1e-4)
    ap.add_argument("--kmax", type=float, default=1e3)
    ap.add_argument("--damping", type=float, default=0.0,
                    help="Gaussian smoothing scale in cMpc/h applied to P(k). Leave at 0: "
                         "tested against a truncated power law it HURTS (0.05%% error "
                         "becomes 4%% at damping=1), because the power-law extrapolation "
                         "already handles the truncation. Only reach for it if a "
                         "measured P(k) is genuinely noisy at high k.")
    ap.add_argument("--out", default="xi.npy")
    args = ap.parse_args()

    out = []
    for path in args.pk:
        k, p = load_pk(path)
        r, xi, s_lo, s_hi = pk_to_xi(k, p, args.kmin, args.kmax, args.damping)
        out.append((path, r, xi))
        print("%s\n  measured k = %.5f .. %.4f h/Mpc  (%d bins)" % (path, k[0], k[-1], len(k)))
        print("  extrapolated slopes: low-k %.2f, high-k %.2f" % (s_lo, s_hi))
        sel = (r > 1) & (r < 200)
        rr, xx = r[sel], xi[sel]
        print("  %8s %14s" % ("r [Mpc/h]", "xi(r)"))
        for i in range(0, len(rr), max(1, len(rr) // 10)):
            print("  %8.2f %14.5g" % (rr[i], xx[i]))
        z = np.flatnonzero(np.diff(np.sign(xx)))
        if z.size:
            print("  xi crosses zero near r = %.1f cMpc/h" % rr[z[0]])

    r0, xi0 = out[0][1], out[0][2]
    if len(out) == 2:
        r1, xi1 = out[1][1], out[1][2]
        x1 = np.interp(r0, r1, xi1)
        sel = (r0 > 1) & (r0 < 150) & (np.abs(xi0) > 1e-6)
        print("\n%10s %14s %14s %9s" % ("r", "xi_A", "xi_B", "B/A"))
        idx = np.flatnonzero(sel)
        for i in idx[::max(1, len(idx) // 12)]:
            print("%10.2f %14.5g %14.5g %9.4f" % (r0[i], xi0[i], x1[i], x1[i] / xi0[i]))
        np.save(args.out, np.vstack([r0, xi0, x1]))
        print("\nsaved -> %s  (rows: r, xi_A, xi_B)" % args.out)
    else:
        np.save(args.out, np.vstack([r0, xi0]))
        print("\nsaved -> %s  (rows: r, xi)" % args.out)


if __name__ == "__main__":
    main()
