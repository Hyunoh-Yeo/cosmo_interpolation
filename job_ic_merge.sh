#!/bin/bash
#SBATCH --job-name=ic_merge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --output=%x_%j.log
# Runs after the array job (submitted with --dependency=afterok by submit_ic.sh),
# so it only ever sees a complete set of parts. It still checks the count, because
# afterok would also fire if the array were cancelled between tasks.

set -euo pipefail

TAG=${1:?usage: sbatch job_ic_merge.sh TAG [NWORK]}
NWORK=${2:-16}
cd "$SLURM_SUBMIT_DIR"
source ~/setup_cosmo.sh

WORKDIR="parts_${TAG}"
n=$(ls "$WORKDIR"/part_*.npz 2>/dev/null | wc -l)
if [ "$n" -ne "$NWORK" ]; then
    echo "ERROR: found $n/$NWORK part files in $WORKDIR -- refusing to merge a partial scan."
    echo "Rerun the missing indices with:  sbatch --array=<i> job_ic_array.sh ..."
    exit 1
fi

python3 merge_stats.py "$WORKDIR/part_*.npz" --out "stats_${TAG}.npz"

if ls "$WORKDIR"/mv_*.npy >/dev/null 2>&1; then
    python3 - "$WORKDIR" "movers_${TAG}.npy" <<'PY'
import glob, sys, numpy as np
d, out = sys.argv[1], sys.argv[2]
a = [np.load(p) for p in sorted(glob.glob(d + "/mv_*.npy"))]
a = [x for x in a if x.size]
if a:
    m = np.concatenate(a); np.save(out, m)
    print("merged %d movers -> %s" % (len(m), out))
else:
    print("no movers above the threshold")
PY
fi
echo "[$(date)] $TAG merged -> stats_${TAG}.npz"
