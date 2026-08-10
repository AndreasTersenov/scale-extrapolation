#!/bin/bash
# ./check.sh — the one verification command. Runs BOTH test stacks with the
# correct interpreters and login-node-safe CPU pinning. Use before any commit
# that touches code, and whenever you want to know the repo is sound.
set -o pipefail
cd "$(dirname "$0")" || exit 1
rc=0

echo "== gate 1: tests/  (env.sh stack: numpy/scipy/pywt) =="
( [ -f env.sh ] && source env.sh; taskset -c 0-3 python -m pytest tests/ -q ) || rc=1

echo
echo "== gate 2: tests_wfm/ tests_p2/  (wl-challenge-env: JAX, cpu-pinned) =="
JAX_PLATFORMS=cpu taskset -c 0-3 "$HOME/wl-challenge-env/bin/python" -m pytest tests_wfm/ tests_p2/ -q || rc=1

echo
if [ $rc -eq 0 ]; then echo "ALL GREEN"; else echo "FAILURES ABOVE — do not commit code"; fi
exit $rc
