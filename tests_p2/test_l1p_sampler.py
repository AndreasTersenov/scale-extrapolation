"""L1' plumbing equivalence test (pre-job, per the pre-statement): with the
white base, sample_base_fn must equal sample_tbase EXACTLY — same key usage,
same ODE — on a real (tiny) ConditionalUNet. Also: the colored base changes
the output (the override actually reaches the sampler)."""
import os
import sys

import numpy as np

import jax

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts_p2"))

from arms_p2.c1t.flow import sample_tbase  # noqa: E402
from colored_base import filter_from_ring_amps, make_z_table  # noqa: E402
from l1p_lib import make_colored_base, sample_base_fn, white_base  # noqa: E402
from wfm.model import ConditionalUNet  # noqa: E402


def _toy():
    model = ConditionalUNet(out_channels=3, channels=(4, 8), bottleneck=16,
                            cond_dim=0, cond_mode="film", variance_head=False)
    coarse = jax.random.normal(jax.random.PRNGKey(0), (2, 8, 8, 1))
    params = model.init(jax.random.PRNGKey(1), coarse[..., :3] * 0 +
                        jax.random.normal(jax.random.PRNGKey(2), (2, 8, 8, 3)),
                        np.zeros(2, np.float32), coarse, None)["params"]
    return model, params, coarse


def test_white_base_equals_sample_tbase_exactly():
    model, params, coarse = _toy()
    key = jax.random.PRNGKey(42)
    a = sample_tbase(model.apply, params, key, coarse, 3, n_steps=8)
    b = sample_base_fn(model.apply, params, key, coarse, white_base,
                       n_steps=8)
    assert np.array_equal(np.asarray(a), np.asarray(b)), \
        "white-base sample_base_fn != sample_tbase (key usage drift)"


def test_colored_base_changes_output():
    model, params, coarse = _toy()
    key = jax.random.PRNGKey(42)
    z, x = make_z_table()
    filt = filter_from_ring_amps(np.array([1.0, 3.0, 2.0, 1.0, 0.5]), 8)
    cb = make_colored_base(filt, z, x)
    a = sample_base_fn(model.apply, params, key, coarse, white_base, n_steps=8)
    b = sample_base_fn(model.apply, params, key, coarse, cb, n_steps=8)
    assert not np.array_equal(np.asarray(a), np.asarray(b))
