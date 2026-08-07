#!/bin/bash
# Submit one shared-IC scan as an array of independent workers + a dependent merge.
#
#   ./submit_ic.sh DIR_A DIR_B TAG [WINDOW] [NWORK]
#   SAVE_MOVERS=10 ./submit_ic.sh ... 	# also export every particle with |dr| > 10
#
# DIR_A/DIR_B may be bare cosmology names (MV_Om0.21_-1_0); the reader searches both
# storage roots. The merge job only runs if every array task succeeded (afterok) and
# re-checks the part count before merging.

set -euo pipefail

DIR_A=${1:?usage: ./submit_ic.sh DIR_A DIR_B TAG [WINDOW] [NWORK]}
DIR_B=${2:?}
TAG=${3:?}
WINDOW=${4:-3}
NWORK=${5:-16}
last=$(( NWORK - 1 ))

mkdir -p "parts_${TAG}"

aid=$(sbatch --parsable --array=0-${last} --job-name="${TAG}" \
             --output="parts_${TAG}/w%a.log" \
             job_ic_array.sh "$DIR_A" "$DIR_B" "$TAG" "$WINDOW")
echo "array job  : $aid  (${NWORK} independent tasks)"

mid=$(sbatch --parsable --dependency=afterok:${aid} --job-name="${TAG}_merge" \
             job_ic_merge.sh "$TAG" "$NWORK")
echo "merge job  : $mid  (starts only if all tasks succeed)"
echo
echo "watch:   squeue -u \$USER"
echo "result:  stats_${TAG}.npz"
echo "logs:    parts_${TAG}/w0.log .. w${last}.log   (unpadded, from SLURM %a)"
