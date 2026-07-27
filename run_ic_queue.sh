#!/usr/bin/env bash
# Run several shared-IC scans back-to-back (never in parallel -- they each stream
# ~550 GB from GPFS and would only fight over I/O). Waits for any verify_shared_ic
# already running to finish first, so it's safe to launch while one is in progress.
#
#   nohup bash run_ic_queue.sh > queue.log 2>&1 &
#   tail -f queue.log        # watch;  ctrl-C only stops watching, not the queue
#
# Each pair writes its own ic_<label>.log; the report (percentiles, max, >10 Mpc
# count, top-20 movers) is at the bottom of that file when it finishes.

set -u
MV=/multiverse/mhee7173_snu/MultiverseCPL
STEP=1881
PREFIX=SyncINITIAL

# label            DIR_A                     DIR_B
PAIRS=(
  "om21v36_lcdm    $MV/MV_Om0.21_-1_0        $MV/MV_Om0.36_-1_0"      # dOm max, LCDM (the test)
  "om21v36_w14     $MV/MV_Om0.21_-1.4_0      $MV/MV_Om0.36_-1.4_0"    # dOm max, other w0 (robustness)
  "om21_w0scan     $MV/MV_Om0.21_-1_0        $MV/MV_Om0.21_-1.4_0"    # same Om, w0 only (control)
)

# don't compete with a scan already running
while pgrep -f 'verify_shared_ic.py' >/dev/null; do
  echo "[queue] waiting for the running verify_shared_ic to finish ... $(date)"
  sleep 120
done

for entry in "${PAIRS[@]}"; do
  read -r label A B <<<"$entry"
  log="ic_${label}.log"
  echo "[queue] START $label  ($(date))"
  echo "[queue]   A=$A"
  echo "[queue]   B=$B  ->  $log"
  python3 verify_shared_ic.py "$A" "$B" --prefix "$PREFIX" --step "$STEP" > "$log" 2>&1
  echo "[queue] DONE  $label  ($(date))  exit=$?"
  echo "[queue] --- tail of $log ---"
  tail -18 "$log"
  echo "[queue] ======================================================"
done
echo "[queue] all pairs finished ($(date))"
