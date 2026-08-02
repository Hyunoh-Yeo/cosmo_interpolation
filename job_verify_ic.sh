#!/bin/bash
#SBATCH --job-name=verify_ic
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --output=%x_%j.log
# A single node has 64 cores and ~500 GB, and each worker needs ~12 GB for its
# window cache, so NWORK is bounded by memory (~40) rather than cores.
#
#   sbatch job_verify_ic.sh DIR_A DIR_B TAG [WINDOW] [NWORK]
#
# e.g. sbatch job_verify_ic.sh \
#        /multiverse/mhee7173_snu/MultiverseCPL/MV_Om0.21_-1_0 \
#        /multiverse/mhee7173_snu/MultiverseCPL/MV_Om0.26_-1_0 om21v26 6 16
#
# Splits A's 250 subfiles across NWORK processes, then merges. The merge is exact:
# every accumulator is a sum or a top-N, so the result equals a single-process scan.

set -euo pipefail

DIR_A=${1:?usage: sbatch job_verify_ic.sh DIR_A DIR_B TAG [WINDOW] [NWORK]}
DIR_B=${2:?}
TAG=${3:?}
WINDOW=${4:-3}
NWORK=${5:-16}
STEP=${STEP:-1881}
PREFIX=${PREFIX:-SyncINITIAL}
NSUB=${NSUB:-250}

cd "$SLURM_SUBMIT_DIR"
source ~/setup_cosmo.sh

WORKDIR="parts_${TAG}"
mkdir -p "$WORKDIR"
echo "[$(date)] $TAG: A=$DIR_A"
echo "[$(date)] $TAG: B=$DIR_B"
echo "[$(date)] $TAG: $NSUB subfiles over $NWORK workers, window +-$WINDOW, step $STEP"

# --- fan out: each worker takes a contiguous slice of A's subfiles ---
per=$(( (NSUB + NWORK - 1) / NWORK ))
pids=()
for ((i = 0; i < NWORK; i++)); do
    a=$(( i * per ))
    b=$(( a + per )); (( b > NSUB )) && b=$NSUB
    (( a >= NSUB )) && break
    python3 verify_shared_ic.py "$DIR_A" "$DIR_B" \
        --prefix "$PREFIX" --step "$STEP" --window "$WINDOW" \
        --sub-range "${a}:${b}" \
        --stats-out "$WORKDIR/part_$(printf %03d "$i").npz" \
        > "$WORKDIR/w$(printf %03d "$i").log" 2>&1 &
    pids+=($!)
done
echo "[$(date)] launched ${#pids[@]} workers"

# --- wait, and fail loudly if any worker died (a silent partial merge would be worse) ---
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail + 1)); done
if (( fail )); then
    echo "[$(date)] ERROR: $fail worker(s) failed -- see $WORKDIR/w*.log; NOT merging"
    exit 1
fi
echo "[$(date)] all workers done, merging"

python3 merge_stats.py "$WORKDIR/part_*.npz" --out "stats_${TAG}.npz"
echo "[$(date)] $TAG complete -> stats_${TAG}.npz"
