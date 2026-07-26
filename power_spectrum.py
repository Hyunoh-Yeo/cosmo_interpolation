"""Power spectrum P(k) of a Multiverse snapshot via pypower's CatalogFFTPower.

P(k) needs the FULL periodic box, so this reads every subfile of a snapshot
(250 of them, ~275 GB) — use `--subsample` to keep only a fraction of particles
(unbiased, just noisier) and/or `--max-sub` for a quick smoke test.

    # smoke test (2 subfiles, fast — the P(k) will NOT be correct, just checks the API)
    python3 power_spectrum.py <cosmo_dir> --step 1150 --max-sub 2 --nmesh 256

    # real run: all subfiles, 2% of particles
    python3 power_spectrum.py <cosmo_dir> --step 1150 --subsample 0.02 --nmesh 512

Compare two cosmologies by running it twice and plotting the two .npy files.
Requires the venv (pypower); run `source ~/setup_cosmo.sh` first.
"""
import argparse
import glob
import os
import numpy as np
from gotpm import read_records, decode_positions


def read_positions(path, frac=1.0, rng=None, stride=1):
    """Absolute positions (N,3) in cMpc/h, optionally subsampled.

    `stride > 1` memory-maps the file and takes every Nth record; `frac < 1` is a
    uniform random draw (needs rng). See gotpm.read_records for the I/O tradeoffs.
    """
    rec, p = read_records(path, stride=stride, frac=frac, rng=rng)
    return decode_positions(rec, p), float(p["Boxsize(Mpc/h)"]), p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cosmo_dir", help="e.g. /gpfs/.../MV_Om0.26_-1.2_-0.8")
    ap.add_argument("--step", type=int, required=True, help="snapshot step, e.g. 1150")
    ap.add_argument("--prefix", default="INITIAL")
    ap.add_argument("--max-sub", type=int, default=None, help="limit #subfiles (testing only)")
    ap.add_argument("--subsample", type=float, default=1.0,
                    help="random fraction of particles to keep (still reads every byte)")
    ap.add_argument("--stride", type=int, default=1,
                    help="keep every Nth record via mmap; >=128 also cuts I/O (e.g. 1280 -> ~10x faster)")
    ap.add_argument("--nmesh", type=int, default=512)
    ap.add_argument("--resampler", default="tsc", choices=["ngp", "cic", "tsc", "pcs"])
    ap.add_argument("--interlacing", type=int, default=2)
    ap.add_argument("--out", default=None, help="output .npy (default: pk_<cosmo>_<step>.npy)")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.cosmo_dir,
                                          "%s.%05d?????" % (args.prefix, args.step))))
    if not files:
        raise SystemExit("no subfiles matched in %s for step %d" % (args.cosmo_dir, args.step))
    if args.max_sub:
        files = files[:args.max_sub]
        print("WARNING: only %d/%d subfiles -> P(k) is NOT physically correct "
              "(smoke test only)" % (len(files), 250))

    rng = np.random.default_rng(0)
    every = max(1, len(files) // 10)          # ~10 progress lines regardless of count
    chunks, box, hdr, ntot = [], None, None, 0
    print("reading %d subfiles from %s ..." % (len(files), args.cosmo_dir), flush=True)
    for i, f in enumerate(files):
        pos, box, hdr = read_positions(f, args.subsample, rng, args.stride)
        chunks.append(pos)
        ntot += pos.shape[0]
        if (i + 1) % every == 0 or i == len(files) - 1:
            print("  read %d/%d subfiles (%d particles so far)"
                  % (i + 1, len(files), ntot), flush=True)
    pos = np.concatenate(chunks); del chunks
    z = 48.0 / float(hdr["Anow"]) - 1.0
    print("total particles: %d   box %.0f cMpc/h   z=%.3f   Om=%s w0=%s wa=%s"
          % (len(pos), box, z, hdr["OmegaMatter0"],
             hdr.get("w0 of DE (CPL)"), hdr.get("wa of DE (CPL)")))

    from pypower import CatalogFFTPower
    kf = 2 * np.pi / box                      # fundamental mode
    knyq = np.pi * args.nmesh / box           # Nyquist
    edges = np.arange(kf, knyq, kf)
    print("computing P(k): nmesh=%d resampler=%s interlacing=%d, k=%.4f..%.3f h/Mpc"
          % (args.nmesh, args.resampler, args.interlacing, edges[0], edges[-1]), flush=True)

    res = CatalogFFTPower(data_positions1=pos, edges=edges, ells=(0,), los="z",
                          boxsize=box, boxcenter=box / 2.0, nmesh=args.nmesh,
                          resampler=args.resampler, interlacing=args.interlacing,
                          position_type="pos", wrap=True, dtype="f4")
    poles = res.poles
    k = np.asarray(poles.k)
    pk = np.asarray(poles.power[0]).real
    shot = float(getattr(poles, "shotnoise", np.nan))

    out = args.out or ("pk_%s_%05d.npy" % (os.path.basename(args.cosmo_dir.rstrip("/")), args.step))
    np.save(out, np.vstack([k, pk]))
    print("\nshot noise: %.4g   (already subtracted by pypower)" % shot)
    print("%10s %14s" % ("k [h/Mpc]", "P(k) [Mpc/h]^3"))
    for i in range(0, len(k), max(1, len(k) // 12)):
        print("%10.4f %14.4g" % (k[i], pk[i]))
    print("\nsaved -> %s   (row0=k, row1=P(k))" % out)


if __name__ == "__main__":
    main()
