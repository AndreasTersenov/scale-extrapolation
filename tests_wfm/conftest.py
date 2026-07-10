"""Phase-1 (JAX) test setup. Runs in ~/wl-challenge-env, NOT the scaledrift Stop-hook env.

Pins CPU affinity to a few cores BEFORE jax is imported so XLA's CPU threadpool fits the
login node's shared process limit (otherwise jax aborts with pthread_create EAGAIN).
"""
import os
import sys

try:
    os.sched_setaffinity(0, set(range(4)))
except Exception:
    pass
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
