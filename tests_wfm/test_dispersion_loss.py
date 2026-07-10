"""Guards for the step-(c) conditional-dispersion regularizer.

Checks the per-bin std helper is correct, the regularizer is non-negative (so the total
loss >= the plain CFM loss), everything is finite, and the penalty is LARGER when the
one-step data estimate is dispersion-collapsed than when it matches the data spread.
"""
import jax
import jax.numpy as jnp
import numpy as np

from wfm.cfm import _per_bin_std, cfm_loss, cfm_loss_dispersion
from wfm.model import ConditionalUNet


def test_per_bin_std_matches_numpy():
    rng = np.random.default_rng(0)
    w = rng.normal(size=400)
    idx = rng.integers(0, 4, 400)
    onehot = jax.nn.one_hot(idx, 4)
    cnt = onehot.sum(0) + 1e-6
    got = np.asarray(_per_bin_std(jnp.asarray(w), onehot, cnt))
    exp = np.array([w[idx == k].std() for k in range(4)])
    assert np.allclose(got, exp, atol=1e-4)


def test_regularizer_nonnegative_and_finite():
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32, cond_dim=0)
    key = jax.random.PRNGKey(0)
    detail = jax.random.normal(key, (4, 16, 16, 3))
    coarse = jax.random.normal(key, (4, 16, 16, 1))
    params = model.init(key, detail, jnp.ones(4), coarse)["params"]
    base = cfm_loss(params, model.apply, detail, coarse, key)
    reg = cfm_loss_dispersion(params, model.apply, detail, coarse, key, lam=0.3, n_bins=6)
    assert jnp.isfinite(base) and jnp.isfinite(reg)
    assert float(reg) >= float(base) - 1e-6            # penalty term is >= 0


def test_penalty_larger_for_collapsed_prediction():
    """Synthetic: coarse-dependent data spread; a collapsed x1_hat proxy incurs more penalty.
    Uses _per_bin_std directly (the penalty's core) on collapsed vs faithful predictions."""
    rng = np.random.default_rng(1)
    n = 2000
    c = rng.normal(size=n)
    idx = jnp.clip(jnp.digitize(c, np.quantile(c, np.linspace(0, 1, 7)[1:-1])), 0, 5)
    onehot = jax.nn.one_hot(idx, 6)
    cnt = onehot.sum(0) + 1e-6
    spread = 0.5 + 0.5 * (np.asarray(idx))             # data std grows with coarse bin
    w_data = rng.normal(size=n) * spread
    sd_data = _per_bin_std(jnp.asarray(w_data), onehot, cnt)
    w_collapsed = rng.normal(size=n) * 0.3             # ~flat, under-dispersed
    w_faithful = rng.normal(size=n) * spread
    pen_c = float(jnp.mean((_per_bin_std(jnp.asarray(w_collapsed), onehot, cnt) - sd_data) ** 2))
    pen_f = float(jnp.mean((_per_bin_std(jnp.asarray(w_faithful), onehot, cnt) - sd_data) ** 2))
    assert pen_c > pen_f
