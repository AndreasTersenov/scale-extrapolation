#!/bin/bash
# Stop-hook test gate — CONDITIONAL since 2026-08-08 (low-token/wind-down mode).
#
# Why conditional: the unconditional version ran BOTH pytest stacks (~2-4 min of
# multi-core CPU on a LOGIN NODE) after every single turn, including pure
# documentation/writing turns where nothing executable changed. That is the
# login-node abuse pattern the cluster warned about, and it bought nothing on a
# writing turn. The safety property we actually want is: never end a turn with
# broken code. So: run the suites only when executable files are dirty.
#
# Full manual run (always available, and what CI-of-record uses): ./check.sh
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 0

# Which paths make the gate necessary? Code and tests only — not .md, not results.
CODE_RE='^(tests|tests_wfm|tests_p2|scripts_p2|arms_p2|wfm|sandbox|src|depmeasure)/|\.py$|^pyproject\.toml$|^env\.sh$'
dirty=$(git status --porcelain 2>/dev/null | awk '{print $2}' | grep -E "$CODE_RE" | head -20)

if [ -z "$dirty" ]; then
  # Nothing executable changed — writing/analysis turn. Skip silently.
  exit 0
fi

echo "test-gate: executable files changed, running both suites..." >&2
fail=0

# --- Gate 1: scaledrift suite (env.sh stack) ---
if compgen -G "tests/test_*.py" > /dev/null; then
  [ -f env.sh ] && source env.sh
  out1=$(taskset -c 0-3 python -m pytest tests/ -q --tb=short 2>&1); code1=$?
  if [ $code1 -ne 0 ]; then
    echo "--- gate 1 (tests/, env.sh stack) FAILED ---" >&2
    echo "$out1" | tail -25 >&2
    fail=1
  fi
fi

# --- Gate 2: phase-1/phase-2 suites (JAX env) ---
WLPY="$HOME/wl-challenge-env/bin/python"
p2trees=()
compgen -G "tests_wfm/test_*.py" > /dev/null && p2trees+=(tests_wfm)
compgen -G "tests_p2/test_*.py"  > /dev/null && p2trees+=(tests_p2)
if [ ${#p2trees[@]} -gt 0 ] && [ -x "$WLPY" ]; then
  out2=$(JAX_PLATFORMS=cpu taskset -c 0-3 "$WLPY" -m pytest "${p2trees[@]}" -q --tb=short 2>&1); code2=$?
  if [ $code2 -ne 0 ]; then
    echo "--- gate 2 (${p2trees[*]}, wl-challenge-env) FAILED ---" >&2
    echo "$out2" | tail -25 >&2
    fail=1
  fi
fi

[ $fail -ne 0 ] && exit 2
exit 0
