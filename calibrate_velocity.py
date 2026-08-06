"""Calibrate the GOTPM velocity unit from linear theory.

The sample sources declare Fact1/Fact2/Pfact but never show the conversion, so the
raw code-unit velocities cannot be turned into km/s from the code alone. Physics
supplies the missing constant instead: in linear theory the peculiar velocity of a
particle is tied to its own displacement,

    v_pec = a H(a) f(a) * Psi          Psi = x - q, the displacement from the
                                       Lagrangian grid node encoded in indx

so regressing the raw velocity on Psi gives (conversion factor) x a H f, and a H f
comes from cosmoprimo. With Psi in cMpc/h and v in km/s, a=1 and H0=100h km/s/Mpc,

    v_pec [km/s] = 100 * f(a) * |Psi| [cMpc/h]        (at z = 0)

The relation only holds while the displacement is still linear, so the ratio is
reported per |Psi| bin: read the constant off the small-|Psi| plateau and expect it
to fall for large |Psi| (collapsed regions, where the mapping is no longer linear).

    python3 calibrate_velocity.py SUBFILE [--max-sub N] [--stride S]

Needs numpy + gotpm.py; cosmoprimo only for the f(a) number (skipped if absent).
"""
import argparse
import numpy as np
from gotpm import read_records, decode_positions, geom

PSI_BINS = np.logspace(-1.5, 1.5, 25)      # cMpc/h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subfile", help="one snapshot subfile")
    ap.add_argument("--stride", type=int, default=8,
                    help="keep every Nth record; the fit needs statistics, not every particle")
    args = ap.parse_args()

    rec, p = read_records(args.subfile, stride=args.stride)
    nx, ny, nz, box, mx, mxmy = geom(p)
    x = decode_positions(rec, p)
    v = np.stack([rec["vx"], rec["vy"], rec["vz"]], 1).astype(np.float64)

    q = np.empty_like(x)
    q[:, 0] = (rec["indx"] % mx) * (box / nx)
    q[:, 1] = ((rec["indx"] % mxmy) // mx) * (box / ny)
    q[:, 2] = (rec["indx"] // mxmy) * (box / nz)
    psi = (x - q + 0.5 * box) % box - 0.5 * box          # minimal image
    pn = np.sqrt((psi ** 2).sum(1))
    vn = np.sqrt((v ** 2).sum(1))

    z = 48.0 / float(p["Anow"]) - 1.0
    om = float(p["OmegaMatter0"])
    print("%s\n  Om=%s w0=%s wa=%s  z=%.4f  box=%.0f  N(sampled)=%d"
          % (args.subfile, p["OmegaMatter0"], p.get("w0 of DE (CPL)"),
             p.get("wa of DE (CPL)"), z, box, rec.size))
    for k in ("Pfact", "Fact1", "Fact2", "Astep", "Amax", "Anow"):
        if k in p:
            print("  header %-6s = %s" % (k, p[k]))

    print("\n  |Psi| rms %.3f cMpc/h   |v_raw| rms %.4g (code units)"
          % (np.sqrt((pn ** 2).mean()), np.sqrt((vn ** 2).mean())))

    # --- is v parallel to Psi, as linear theory says? ---
    ok = (pn > 1e-6) & (vn > 1e-12)
    cos = (v[ok] * psi[ok]).sum(1) / (vn[ok] * pn[ok])
    print("  mean cos(v, Psi) = %+0.4f   (linear theory: +1)" % cos.mean())

    # --- ratio per |Psi| bin: v_raw / |Psi| should plateau where growth is linear ---
    kbin = np.clip(np.searchsorted(PSI_BINS, pn[ok], "right") - 1, 0, len(PSI_BINS) - 2)
    cnt = np.bincount(kbin, minlength=len(PSI_BINS) - 1)
    sv = np.bincount(kbin, weights=vn[ok], minlength=len(PSI_BINS) - 1)
    sp = np.bincount(kbin, weights=pn[ok], minlength=len(PSI_BINS) - 1)
    sc = np.bincount(kbin, weights=cos, minlength=len(PSI_BINS) - 1)
    m = cnt > 100
    print("\n  |Psi| [cMpc/h]     <|v_raw|>/<|Psi|>   <cos>    particles")
    for i in np.flatnonzero(m):
        print("   %6.3f - %-6.3f    %14.6g %+8.3f %11d"
              % (PSI_BINS[i], PSI_BINS[i + 1], (sv[i] / cnt[i]) / (sp[i] / cnt[i]),
                 sc[i] / cnt[i], cnt[i]))

    idx = np.flatnonzero(m)
    plateau = np.median([(sv[i] / cnt[i]) / (sp[i] / cnt[i]) for i in idx[:max(3, len(idx) // 3)]])
    print("\n  small-|Psi| plateau: v_raw/|Psi| = %.6g  [code units per cMpc/h]" % plateau)

    try:
        from mvinterp.cpl_growth import CPLGrowth
        g = CPLGrowth(om, float(p.get("w0 of DE (CPL)", -1.0)),
                      float(p.get("wa of DE (CPL)", 0.0)))
        f = float(g.f(z)); E = float(g.H_over_H0(z)); a = 1.0 / (1.0 + z)
        expect = 100.0 * a * E * f          # km/s per cMpc/h of Psi
        print("  linear theory  : v_pec/|Psi| = 100*a*E*f = %.4g km/s per cMpc/h"
              " (f=%.4f, E=%.4f, a=%.4f)" % (expect, f, E, a))
        print("\n  => 1 code unit = %.6g km/s" % (expect / plateau))
        print("     v_pec [km/s] = v_raw * %.6g" % (expect / plateau))
    except Exception as e:
        print("  (cosmoprimo unavailable: %s)" % e)
        print("  multiply the plateau into 100*a*E*f to finish the calibration")


if __name__ == "__main__":
    main()
