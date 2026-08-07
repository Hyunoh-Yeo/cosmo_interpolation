"""Enumerate what the interpolation grid can actually be.

Two things have to be settled before running more comparisons:

  * which cosmologies exist, and in which storage root -- a directory can sit in
    the other path as an empty shell (logs only), so this counts particle files
  * which SyncINITIAL steps each one has -- they are NOT the same everywhere
    (Om0.26_-1.2_-0.8 has 11, Om0.21_-1_0 has 9), so the usable ladder is the
    intersection

It then groups by (w0, wa) and reports which Omega_m values share a step, which is
exactly the set of dOm ladders that can be measured without changing anything else.

    python3 scan_grid.py                      # all steps, all cosmologies
    python3 scan_grid.py --step 1881          # just z=0
    python3 scan_grid.py --step 1881 --pairs  # also print ready-to-run submit lines
"""
import argparse
import glob
import os
import re
from collections import defaultdict

from gotpm import ROOTS

NAME = re.compile(r"^MV_Om([\d.]+)_(-?[\d.]+)_(-?[\d.]+)$")


def parse(name):
    m = NAME.match(name)
    if not m:
        return None
    try:
        return tuple(float(v) for v in m.groups())
    except ValueError:
        return None


def scan(prefix):
    """{(Om, w0, wa): {step: (path, nfiles)}} over both storage roots."""
    found = defaultdict(dict)
    for root in ROOTS:
        for d in sorted(glob.glob(os.path.join(root, "MV_Om*"))):
            if not os.path.isdir(d):
                continue
            key = parse(os.path.basename(d))
            if key is None:                       # s09 variant, checksum files, ...
                continue
            steps = defaultdict(int)
            for f in glob.glob(os.path.join(d, f"{prefix}.??????????")):
                tag = os.path.basename(f).split(".")[-1]
                if len(tag) == 10 and tag.isdigit():
                    steps[int(tag[:5])] += 1
            for s, n in steps.items():
                if n >= 250:                      # a complete snapshot, not a fragment
                    if s in found[key]:
                        print("!! %s step %d present in BOTH roots" % (key, s))
                    found[key][s] = (d, n)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="SyncINITIAL")
    ap.add_argument("--step", type=int, default=None, help="restrict to one step")
    ap.add_argument("--pairs", action="store_true", help="print submit_ic.sh lines")
    args = ap.parse_args()

    found = scan(args.prefix)
    if not found:
        raise SystemExit("no cosmologies found")

    all_steps = sorted({s for v in found.values() for s in v})
    print("%d cosmologies with complete %s snapshots\n" % (len(found), args.prefix))

    if args.step is None:
        print("step coverage (how many cosmologies have each step):")
        for s in all_steps:
            n = sum(1 for v in found.values() if s in v)
            bar = "#" * int(40 * n / len(found))
            print("  %05d  %3d/%d  %s" % (s, n, len(found), bar))
        common = [s for s in all_steps if all(s in v for v in found.values())]
        print("\nsteps present in EVERY cosmology: %s"
              % (", ".join("%05d" % s for s in common) if common else "(none)"))
        # the practical ladder: steps that most cosmologies share
        good = [s for s in all_steps
                if sum(1 for v in found.values() if s in v) >= 0.9 * len(found)]
        print("steps in >=90%% of cosmologies : %s"
              % ", ".join("%05d" % s for s in good))

    step = args.step
    if step is None:
        print("\n(pass --step to see the Omega_m columns at one epoch)")
        return

    have = {k: v for k, v in found.items() if step in v}
    print("=== step %05d: %d cosmologies ===\n" % (step, len(have)))

    # group by (w0, wa) -- these are the columns along which only Omega_m varies
    cols = defaultdict(list)
    for (om, w0, wa) in have:
        cols[(w0, wa)].append(om)
    usable = {k: sorted(v) for k, v in cols.items() if len(v) >= 2}

    print("(w0, wa) columns with 2+ Omega_m values -- these give clean dOm ladders:")
    for (w0, wa), oms in sorted(usable.items(), key=lambda x: -len(x[1])):
        print("  w0=%-6g wa=%-6g : Om = %s" % (w0, wa, ", ".join("%g" % o for o in oms)))
    if not usable:
        print("  (none)")

    singles = {k: v for k, v in cols.items() if len(v) < 2}
    if singles:
        print("\nsingle-Omega_m columns (no ladder possible):")
        for (w0, wa), oms in sorted(singles.items()):
            print("  w0=%-6g wa=%-6g : Om = %g" % (w0, wa, oms[0]))

    if args.pairs:
        print("\n--- ready-to-run pairs (adjacent Omega_m within each column) ---")
        for (w0, wa), oms in sorted(usable.items()):
            for a, b in zip(oms, oms[1:]):
                da, db = have[(a, w0, wa)][step][0], have[(b, w0, wa)][step][0]
                tag = "om%gv%g_w%g" % (a, b, w0)
                tag = tag.replace(".", "").replace("-", "m")
                print("./submit_ic.sh %s %s %s 3 16"
                      % (os.path.basename(da), os.path.basename(db), tag))


if __name__ == "__main__":
    main()
