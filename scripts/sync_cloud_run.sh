#!/bin/bash
# Copy a run from the Google Cloud machine to this Mac, incrementally (only new or changed
# files; the large checkpoint is skipped: it stays on the VM for resuming).
#     scripts/sync_cloud_run.sh RUN_NAME [VM]   e.g. R6.1_50k, or R6.4_50k_newton nebula-sim-2
# Needs: ~/google-cloud-sdk (gcloud compute config-ssh run once) and the VM running.
set -euo pipefail
RUN="$1"
VM="${2:-nebula-sim}"  # the machine the run is on (default: the first VM)
HOST="$VM.us-central1-b.project-f6231423-0846-4084-823"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# the VM gets a new external IP each time it starts: refresh the SSH config first
"$HOME/google-cloud-sdk/bin/gcloud" compute config-ssh --quiet >/dev/null 2>&1
mkdir -p "$ROOT/runs/$RUN"
rsync -az --exclude checkpoint.pkl --exclude checkpoint.tmp "$HOST:nebula/runs/$RUN/" "$ROOT/runs/$RUN/"
ssh "$HOST" "tail -c 2000 nebula/outputs/recon/$RUN.out" > "$ROOT/runs/$RUN/run_output_tail.txt" 2>/dev/null || true
echo "synced $RUN: $(ls "$ROOT/runs/$RUN/snapshots" | wc -l | tr -d ' ') snapshots; last log line:"
tail -1 "$ROOT/runs/$RUN/log.txt" | cut -c1-200
