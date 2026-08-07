#!/bin/bash
#SBATCH --job-name=pk_scan
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=pk_%x_%a.log
# One independent job PER COSMOLOGY, not per subfile: a single P(k) needs every
# particle on one mesh before the FFT, so splitting one spectrum across workers
# would mean sharing the mesh. Splitting across cosmologies instead is free, and
# scanning many cosmologies is what we actually want.
#
# Submit with submit_pk.sh, which builds the array from a list:
#     ./submit_pk.sh cosmologies.txt [STRIDE] [NMESH]
#
# Cost is dominated by reading, so STRIDE is the dial: 1280 keeps 1/1280 of the
# particles (6.7 M) and reads ~1/10 of the bytes; 128 keeps 67 M and reads all of
# them. Shot noise is V/N, so 6.7 M gives 160 and 67 M gives 16 (cMpc/h)^3 --
# pypower subtracts it either way, but it sets where the spectrum stops being
# usable (P > shot noise).

set -euo pipefail

LIST=${1:?usage: sbatch job_pk_array.sh COSMO_LIST [STRIDE] [NMESH]}
STRIDE=${2:-1280}
NMESH=${3:-512}
STEP=${STEP:-1881}
PREFIX=${PREFIX:-SyncINITIAL}
RESAMPLER=${RESAMPLER:-tsc}

cd "$SLURM_SUBMIT_DIR"
source ~/setup_cosmo.sh

# one non-empty, non-comment line per cosmology
COSMO=$(grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p")
if [ -z "$COSMO" ]; then
    echo "task ${SLURM_ARRAY_TASK_ID}: no entry in $LIST"; exit 0
fi

TAG=$(basename "$COSMO" | sed 's/^MV_//')
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): $COSMO"
echo "[$(date)] step $STEP, prefix $PREFIX, stride $STRIDE, nmesh $NMESH, $RESAMPLER"

python3 power_spectrum.py "$COSMO" \
    --prefix "$PREFIX" --step "$STEP" \
    --stride "$STRIDE" --nmesh "$NMESH" --resampler "$RESAMPLER" \
    --out "pk_${TAG}_${STEP}.npy"

echo "[$(date)] done -> pk_${TAG}_${STEP}.npy"
