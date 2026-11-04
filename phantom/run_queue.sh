#!/bin/bash
# Run Phantom runs one after another: run_queue.sh THREADS NAME... (inputs in ~/prod/NAME.in,
# starting dumps in ~/prod/). Each run gets ~/runs_ph/NAME with a manifest (rule 8); an
# existing run folder is never overwritten. Logs to ~/runs_ph/queue.log.
threads=$1; shift
mkdir -p ~/runs_ph
for name in "$@"; do
  d=~/runs_ph/$name
  if [ -e "$d" ]; then echo "[$(date -u +%FT%TZ)] $name: folder exists, skipped" >> ~/runs_ph/queue.log; continue; fi
  mkdir -p "$d" && cd "$d" || continue
  cp ~/phantom/bin/phantom . && cp ~/prod/$name.in cloud.in
  ic=$(grep -m1 "dumpfile =" cloud.in | awk '{print $3}'); cp ~/prod/$ic .
  cat > manifest.json <<MAN
{"name": "$name", "engine": "Phantom 0855f82ff5f0891b3446f6c58abee61eb5210d36 + phantom/nebula_phantom.patch",
 "nebula_git_commit": "$(cat ~/prod/NEBULA_COMMIT)", "patch_md5": "$(md5sum ~/prod/nebula_phantom.patch | cut -d' ' -f1)",
 "phantom_binary_md5": "$(md5sum phantom | cut -d' ' -f1)", "input": "cloud.in", "start_dump": "$ic",
 "ic_note": "$(cat ~/prod/IC_NOTES | grep "^$ic" | cut -d' ' -f2-)",
 "machine": "$(hostname), $(nproc) cpus, $(uname -r)", "omp_threads": $threads,
 "start_utc": "$(date -u +%FT%TZ)"}
MAN
  echo "[$(date -u +%FT%TZ)] $name: start ($threads threads)" >> ~/runs_ph/queue.log
  OMP_NUM_THREADS=$threads ./phantom cloud.in > run.out 2>&1 < /dev/null
  st=$?
  sed -i "s/}$/, \"end_utc\": \"$(date -u +%FT%TZ)\", \"exit_status\": $st}/" manifest.json
  echo "[$(date -u +%FT%TZ)] $name: end, status $st, $(grep -c FATAL run.out) fatal" >> ~/runs_ph/queue.log
done
