#!/bin/bash
# Submit a P(k) scan: one independent job per cosmology listed in a text file.
#
#   ./submit_pk.sh cosmologies.txt [STRIDE] [NMESH]
#   STEP=1881 PREFIX=SyncINITIAL ./submit_pk.sh cosmologies.txt 1280 512
#
# The list holds one cosmology per line -- a bare name (MV_Om0.21_-1_0) is resolved
# against both storage roots, a full path is used as given. Blank lines and lines
# starting with # are skipped.
#
# Each task writes pk_<cosmology>_<step>.npy, which correlation.py and plot_pk.py
# both read.

set -euo pipefail

LIST=${1:?usage: ./submit_pk.sh COSMO_LIST [STRIDE] [NMESH]}
STRIDE=${2:-1280}
NMESH=${3:-512}
[ -f "$LIST" ] || { echo "no such list: $LIST"; exit 1; }

n=$(grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | wc -l)
(( n > 0 )) || { echo "$LIST has no cosmologies"; exit 1; }
last=$(( n - 1 ))

echo "cosmologies : $n"
grep -v '^\s*#' "$LIST" | grep -v '^\s*$' | nl -w3 -s'  '
echo

jid=$(sbatch --parsable --array=0-${last} job_pk_array.sh "$LIST" "$STRIDE" "$NMESH")
echo "array job   : $jid"
echo "watch       : squeue -u \$USER"
echo "results     : pk_*_${STEP:-1881}.npy   logs: pk_pk_scan_*.log"
