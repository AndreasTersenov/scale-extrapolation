"""Arm F2 group machinery (JAX side; prereg 2026-07-25-f2-groupavg).

D4 elements parameterized as (k, f): rotate by k*90deg (axes H,W), then flip
along W if f=1. The sampler never needs a channel action table: details are
sampled in the transformed frame, ASSEMBLED there, and the assembled field is
transformed back (synthesis-transform-analysis) — exactness rests on the
Haar/D4 commutation, gate-tested in tests_wfm/test_d4_action.py.
"""
from __future__ import annotations

import jax.numpy as jnp

D4_ELEMENTS = tuple((k, f) for f in (0, 1) for k in (0, 1, 2, 3))


def apply_g(x, g):
    """Apply (k, f) to channel-last (B, H, W, C)."""
    k, f = g
    y = jnp.rot90(x, k, axes=(1, 2))
    if f:
        y = y[:, :, ::-1, :]
    return y


def apply_g_inv(x, g):
    """Inverse of apply_g: undo the flip, then rotate back."""
    k, f = g
    y = x[:, :, ::-1, :] if f else x
    return jnp.rot90(y, -k, axes=(1, 2))


def assemble_in_frame(coarse, model_fn, g):
    """Assemble one octave in the g-transformed frame and map back.

    model_fn(c_transformed) -> detail triple (B, h, w, 3) in FIELD units
    (any std scaling applied by the caller inside model_fn)."""
    from wfm import haar
    c_g = apply_g(coarse, g)
    det = model_fn(c_g)
    f_g = haar.idwt2(c_g, (det[..., 0:1], det[..., 1:2], det[..., 2:3]))
    return apply_g_inv(f_g, g)


def assemble_group_assigned(coarse, model_fn, assignments):
    """Per-field group assignment: fields with the same element are assembled
    together in that frame. assignments: int array (B,) indexing D4_ELEMENTS.
    model_fn must be batch-shape-agnostic. Returns the assembled fine field."""
    out = None
    for gi, g in enumerate(D4_ELEMENTS):
        idx = jnp.nonzero(assignments == gi, size=None)[0] \
            if hasattr(jnp, "nonzero") else None
        # JAX nonzero with dynamic size is awkward under jit; this runs
        # un-jitted at inference, so numpy-style indexing is fine.
        import numpy as np
        sel = np.nonzero(np.asarray(assignments) == gi)[0]
        if len(sel) == 0:
            continue
        sub = assemble_in_frame(coarse[sel], model_fn, g)
        if out is None:
            out = np.zeros((coarse.shape[0],) + tuple(sub.shape[1:]),
                           dtype=np.asarray(sub).dtype)
        out[sel] = np.asarray(sub)
    return jnp.asarray(out)
