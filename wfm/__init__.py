"""wfm -- wavelet-factorized conditional flow-matching generator (phase-1 toy).

Reuses the flow-matching CORE from ``jax_flows`` (ot interpolation, CFM target, euler/heun
samplers); the conditional architecture and the wavelet factorization are this package's.
``scaledrift`` is NOT imported here -- it stays the frozen measurement instrument, applied
to generated fields in its own env.
"""
import os
import sys

# Make jax_flows importable without an editable install (its pyproject pins a diffrax
# version broken on this cluster; we only need the pure-JAX core, no dep resolution).
_JF = os.path.expanduser("~/software/jax_flows")
if _JF not in sys.path:
    sys.path.append(_JF)

from . import haar  # noqa: E402
