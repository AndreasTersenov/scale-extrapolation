#!/bin/bash
# Stop-hook test gate — covers BOTH interpreter stacks (fix ordered by the 2026-07-16
# memo ruling: pyproject testpaths had silently restricted every repo-root pytest to
# tests/ only; tests_wfm was never gated).
#   Gate 1: tests/            -> env.sh stack (scaledrift: numpy/scipy/pywt venv)
#   Gate 2: tests_wfm tests_p2 -> ~/wl-challenge-env (JAX; conftest pins affinity+cpu)
cd "${CLAUDE_PROJECT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}" || exit 0
fail=0

# --- Gate 1: scaledrift suite ---
if compgen -G "tests/test_*.py" > /dev/null; then
  [ -f env.sh ] && source env.sh
  out1=$(python -m pytest tests/ -q --tb=short 2>&1); code1=$?
  if [ $code1 -ne 0 ]; then
    echo "--- gate 1 (tests/, env.sh stack) FAILED ---" >&2
    echo "$out1" | tail -25 >&2
    fail=1
  fi
fi

# --- Gate 2: phase-1/phase-2 suites under the JAX env ---
WLPY="$HOME/wl-challenge-env/bin/python"
p2trees=()
compgen -G "tests_wfm/test_*.py" > /dev/null && p2trees+=(tests_wfm)
compgen -G "tests_p2/test_*.py"  > /dev/null && p2trees+=(tests_p2)
if [ ${#p2trees[@]} -gt 0 ] && [ -x "$WLPY" ]; then
  out2=$("$WLPY" -m pytest "${p2trees[@]}" -q --tb=short 2>&1); code2=$?
  if [ $code2 -ne 0 ]; then
    echo "--- gate 2 (${p2trees[*]}, wl-challenge-env) FAILED ---" >&2
    echo "$out2" | tail -25 >&2
    fail=1
  fi
fi

if [ $fail -ne 0 ]; then
  echo "STOP-GATE: tests failing. Fix before finishing, or mark xfail with a written justification in log/." >&2
  exit 2
fi
exit 0
