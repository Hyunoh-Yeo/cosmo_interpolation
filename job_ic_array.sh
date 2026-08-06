#!/bin/bash
#SBATCH --job-name=ic_scan
#SBATCH --partition=normal
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --output=parts_%x/w%a.log
# One INDEPENDENT job per worker, instead of 16 background processes inside one job.
#
# Why: SLURM can then place the workers on whatever cores and nodes are free, they
# start as slots open up rather than waiting for a whole 64-core node, and a worker
# that dies takes only itself down -- rerun that array index alone with
#     sbatch --array=<i> job_ic_array.sh ...
#
# Submit through submit_ic.sh, which also queues the dependent merge job:
#     ./submit_ic.sh DIR_A DIR_B TAG [WINDOW] [NWORK]
#
# Memory: a worker holds 2w+1 B-slabs (~0.7 GB each) plus the A slab it is on, so
# ~13 GB at window 3 and ~18 GB at window 6. 20 G covers both.

set -euo pipefail

DIR_A=${1:?usage: sbatch job_ic_array.sh DIR_A DIR_B TAG [WINDOW]}
DIR_B=${2:?}
TAG=${3:?}
WINDOW=${4:-3}
STEP=${STEP:-1881}
PREFIX=${PREFIX:-SyncINITIAL}
NSUB=${NSUB:-250}
SAVE_MOVERS=${SAVE_MOVERS:-}

cd "$SLURM_SUBMIT_DIR"
source ~/setup_cosmo.sh

NWORK=${SLURM_ARRAY_TASK_COUNT:-16}
i=${SLURM_ARRAY_TASK_ID}
per=$(( (NSUB + NWORK - 1) / NWORK ))
a=$(( i * per ))
b=$(( a + per )); (( b > NSUB )) && b=$NSUB

if (( a >= NSUB )); then
    echo "task $i: nothing to do (only $NSUB subfiles)"; exit 0
fi

WORKDIR="parts_${TAG}"
mkdir -p "$WORKDIR"
echo "[$(date)] task $i/$NWORK on $(hostname): A subfiles [$a,$b) of $NSUB, window +-$WINDOW"

mv_args=()
[ -n "$SAVE_MOVERS" ] && mv_args=(--save-movers "$SAVE_MOVERS"
                                  --movers-out "$WORKDIR/mv_$(printf %03d "$i").npy")

python3 verify_shared_ic.py "$DIR_A" "$DIR_B" \
    --prefix "$PREFIX" --step "$STEP" --window "$WINDOW" \
    --sub-range "${a}:${b}" \
    --stats-out "$WORKDIR/part_$(printf %03d "$i").npz" \
    "${mv_args[@]}"

echo "[$(date)] task $i done"
