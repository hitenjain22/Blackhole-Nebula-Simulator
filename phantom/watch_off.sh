#!/bin/bash
# Power off ~15 min after neither a phantom run, a run queue nor a render (python) is running.
busy() { ps -eo args | grep -q -E "^[.]/phantom |[r]un_queue.sh|[c]inematic|[p]h_to_snapshots"; }
sleep 300
while busy; do sleep 120; done
sleep 900
busy && exec "$0"
echo "[$(date -u +%H:%M:%S)] idle: shutting down" >> "$HOME/watch.log"
sudo shutdown -h now
