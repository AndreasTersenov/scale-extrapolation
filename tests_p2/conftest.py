"""Phase-2 test setup (sandbox/, depmeasure/, arms_p2/).

Numpy-first tree: sandbox and B1-estimator code is pure numpy/scipy so this tree runs
under BOTH stacks; the Stop hook runs it under ~/wl-challenge-env alongside tests_wfm.
Thread caps precede numpy for the login-node cgroup limit (see tests/conftest.py,
2026-07-11 incident).
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
