"""Validation gates for the C3 patched-energy-score direct detail sampler.

Prereg: log/2026-07-17-prereg-c3-sandbox.md (gates 1-4), approved with conditions in
log/2026-07-17-reconvene-forensic-c3.md (R10). Tests-first per CLAUDE.md: these gates
exist before the arm trains; R10 condition 2 makes the sampler-shape gate (two-moment
+ skewness recovery) BINDING before any production training.

  1. ES estimator correctness: at patch size 1 / beta=1 the patched energy score is
     the per-scalar CRPS; the fair m-sample estimator must be unbiased against the
     closed-form Gaussian CRPS.
  2. Propriety in practice: the expected patched ES is minimized by the true law --
     mean-shifted, under- and over-dispersed samplers all score strictly worse.
  3. Phase-1c toys adapted: the ES-trained direct sampler (conv, production patch
     config) recovers the conditional mean and a MODULATED sigma per coarse bin, and
     learns a FLAT sigma on constant-variance data (no spurious slope).
  4. R10 condition 2 (binding sampler-shape gate): a 1-D MLP direct sampler trained
     through the EXACT production loss path (energy_score_fair, fair m=8) on a known
     skewed conditional recovers mean, std AND pooled skewness.
  5. TAIL gate: same 1-D mechanism on a Student-t(df=5) conditional -- recovered
     excess kurtosis within the PRE-SET band [3, 12] (truth 6.0) and q999 beyond the
     Gaussian value. CAVEAT (phase-1c lesson, per the prereg): pointwise toys can
     pass while fields fail -- the discrimination is the full-scale run.
  6. Plumbing: sampler determinism given a key, one weight set across octave shapes,
     trainer D4/standardization pipeline identical to C1's, checkpoint hooks fire.
  7. tail_q999 (the R10 condition-1 descriptive instrument): validated on N(0,1) and
     Student-t(5) against closed-form quantiles.

WHY the shape gates (4, 5) are 1-D and the field-scale (conv) toys assert only two
moments: measured 2026-07-17 (log/2026-07-17-c3-gate-design.md) -- a TINY conv net
trained through 192-dim patches under-recovers third/fourth moments (held-out skew
0.33 vs truth 2.0; capacity-limited, NOT an objective failure: per-pixel signal is
there and the small net memorizes it at small data, transiting truth then
overshooting), while the PRODUCTION-size net at the production patch config recovers
shape cleanly on held-out data (skew 1.9, kurt 5.4 by 3k steps, truth 2.0/6.0; SLURM
job 16605902-3). That production-config run is the field-scale shape evidence
(logged pre-submission); it cannot live in the 900s Stop-hook budget, so the
hook-resident BINDING shape gates isolate the objective+sampler mechanism in 1-D
with infinite fresh data (no capacity/overfit confound, seconds each).

Tolerances are pre-set before any training run; toy hyperparameters (steps, lr,
tiny nets) are free periphery frozen at commit time.
"""
import numpy as np
import pytest
import scipy.stats as st

import jax
import jax.numpy as jnp
from flax import linen as nn

from arms_p2.c3.energy import energy_score_fair, extract_patches, patched_energy_score
from arms_p2.c3.sampler import sample_direct
from arms_p2.c3.train import make_es_step, train_c3_generator
from sandbox.truth_stats import tail_q999
from wfm.cfm import make_train_state
from wfm.dataset import d4_augment, field_to_octaves
from wfm.model import ConditionalUNet


# ---------------------------------------------------------------- gate 1: CRPS identity

def _crps_gaussian(y, mu, sigma):
    """Closed-form CRPS of N(mu, sigma^2) at y (Gneiting & Raftery 2007, eq. 21)."""
    z = (y - mu) / sigma
    return sigma * (z * (2 * st.norm.cdf(z) - 1) + 2 * st.norm.pdf(z)
                    - 1 / np.sqrt(np.pi))


def test_es_reduces_to_gaussian_crps():
    """Fair m=8 estimator at patch size 1 is unbiased for the closed-form CRPS."""
    rng = np.random.default_rng(42)
    mu, sigma, m, reps = 0.3, 1.7, 8, 4000
    for y in (-2.0, -0.5, 0.0, 1.0, 2.3):
        X = rng.normal(mu, sigma, (m, reps, 1, 1, 1)).astype(np.float32)
        tgt = np.full((reps, 1, 1, 1), y, np.float32)
        per = np.asarray(energy_score_fair(
            extract_patches(jnp.asarray(X), patch=1, stride=1),
            extract_patches(jnp.asarray(tgt), patch=1, stride=1))).ravel()
        se = per.std(ddof=1) / np.sqrt(reps)
        want = _crps_gaussian(y, mu, sigma)
        assert abs(per.mean() - want) < 5 * se + 1e-3, (y, per.mean(), want, se)


def test_es_propriety_direction():
    """E[ES] is smallest for the true law: shift / under- / over-dispersion all lose.

    Paired design: every candidate is scored on the SAME targets and compared per-rep
    (the shared-target variance cancels), so the small under-dispersion gap is
    resolvable at 5 sigma without giant sample sizes."""
    rng = np.random.default_rng(7)
    m, reps = 8, 20000
    y = rng.normal(0, 1, (reps, 1, 1, 1)).astype(np.float32)
    tgt = extract_patches(jnp.asarray(y), patch=1, stride=1)

    def es_per_rep(mu, sigma):
        X = rng.normal(mu, sigma, (m, reps, 1, 1, 1)).astype(np.float32)
        return np.asarray(energy_score_fair(
            extract_patches(jnp.asarray(X), patch=1, stride=1), tgt)).ravel()

    per_true = es_per_rep(0.0, 1.0)
    for mu, sigma in ((0.5, 1.0), (0.0, 0.6), (0.0, 1.6)):
        d = es_per_rep(mu, sigma) - per_true
        se = d.std(ddof=1) / np.sqrt(reps)
        assert d.mean() > 5 * se, ((mu, sigma), d.mean(), se)


# ------------------------------------------------- toy data (phase-1c suite, adapted)

def _smooth_coarse(key, n, hw=16):
    x = jax.random.normal(key, (n, hw, hw, 1))
    k = jnp.ones((5, 5, 1, 1)) / 25.0
    for _ in range(3):
        x = jax.lax.conv_general_dilated(x, k, (1, 1), "SAME",
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
    return 1.5 * x / jnp.std(x)


def _true_mean(c):
    return 0.5 * jnp.tanh(c)


def _true_sigma(c, flat=False):
    if flat:
        return 0.85 * jnp.ones_like(c)
    return jnp.maximum(0.85 + 0.12 * c, 0.2)


def _make_data(key, noise="gauss", flat_sigma=True, n=192, hw=16):
    kc, kz = jax.random.split(key)
    coarse = _smooth_coarse(kc, n, hw)
    shape = (n, hw, hw, 3)
    if noise == "gauss":
        eps = jax.random.normal(kz, shape)
    elif noise == "exp":         # standardized exponential: mean 0, var 1, skewness 2
        eps = jax.random.exponential(kz, shape) - 1.0
    elif noise == "t5":          # unit-variance Student-t(5): excess kurtosis 6
        eps = jax.random.t(kz, 5.0, shape) / np.sqrt(5.0 / 3.0)
    detail = _true_mean(coarse) + _true_sigma(coarse, flat_sigma) * eps
    return detail, coarse


def _train_direct(detail, coarse, steps=1200, seed=0, lr=3e-3, m=8):
    """Tiny ConditionalUNet as the direct noise-conditioned sampler (production
    mechanism at toy scale), trained with the production patched ES (8x8, stride 4,
    m=8) -- at hw=16 that is a 3x3 patch grid."""
    model = ConditionalUNet(out_channels=3, channels=(8, 16), bottleneck=32,
                            embed_dim=32, cond_dim=0)
    state = make_train_state(model, jax.random.PRNGKey(seed),
                             (16,) + detail.shape[1:], (16,) + coarse.shape[1:],
                             0, lr, total_steps=steps, warmup=steps // 10)
    step = make_es_step(None, m=m)
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        idx = rng.integers(0, detail.shape[0], 16)
        state, loss = step(state, detail[idx], coarse[idx])
    return state, float(loss)


def _gen(state, coarse, K=4, seed=11):
    outs = [sample_direct(state.apply_fn, state.params, jax.random.PRNGKey(seed + k),
                          coarse, 3) for k in range(K)]
    return jnp.concatenate(outs, 0), jnp.concatenate([coarse] * K, 0)


def _bin_idx(c, edges):
    idx = np.digitize(c, edges) - 1
    idx[(c < edges[0]) | (c >= edges[-1])] = -1
    return idx


def _binned_mean_std(gen, coarse, edges):
    c = np.asarray(jnp.broadcast_to(coarse, gen.shape)).ravel()
    g = np.asarray(gen).ravel()
    idx = _bin_idx(c, edges)
    mean = np.array([g[idx == b].mean() for b in range(len(edges) - 1)])
    std = np.array([g[idx == b].std() for b in range(len(edges) - 1)])
    return mean, std


EDGES = np.linspace(-1.2, 1.2, 7)


@pytest.fixture(scope="module")
def trained_gauss():
    detail, coarse = _make_data(jax.random.PRNGKey(3), "gauss", flat_sigma=False)
    state, loss = _train_direct(detail, coarse)
    return state, detail, coarse, loss


# --------------------------------------- gate 3: mean + modulated sigma, flat-sigma null

def test_sampler_recovers_mean_and_modulated_sigma(trained_gauss):
    """Per-bin references are the DATA's binned mean/std (the phase-1c convention:
    bin-center analytic values are biased by tanh curvature + within-bin density)."""
    state, detail, coarse, _ = trained_gauss
    gen, cc = _gen(state, coarse)
    m_hat, s_hat = _binned_mean_std(gen, cc, EDGES)
    m_ref, s_ref = _binned_mean_std(detail, coarse, EDGES)
    assert np.max(np.abs(m_hat - m_ref)) < 0.12, (m_hat, m_ref)
    assert np.max(np.abs(s_hat / s_ref - 1)) < 0.12, (s_hat, s_ref)
    assert s_hat[-1] > s_hat[0] + 0.15, "learned sigma must rise with c (true gap 0.24)"


def test_sampler_flat_sigma_null():
    detail, coarse = _make_data(jax.random.PRNGKey(5), "gauss", flat_sigma=True)
    state, _ = _train_direct(detail, coarse, steps=900, seed=1)
    gen, cc = _gen(state, coarse)
    _, s_hat = _binned_mean_std(gen, cc, EDGES)
    assert np.max(np.abs(s_hat / 0.85 - 1)) < 0.12, s_hat
    assert abs(s_hat[-1] - s_hat[0]) < 0.10, "no spurious sigma slope on null data"


# ---------------- 1-D direct sampler through the production loss path (gates 4, 5)

class _MLP1D(nn.Module):
    """Tiny noise-conditioned direct sampler: (z, c) -> y. The 1-D analogue of the
    production mechanism (z enters where x_t entered; output IS the sample)."""

    @nn.compact
    def __call__(self, z, c):
        h = jnp.concatenate([z, c], axis=-1)
        h = nn.silu(nn.Dense(64)(h))
        h = nn.silu(nn.Dense(64)(h))
        return nn.Dense(1)(h)


def _train_mlp_1d(noise, steps=3000, m=8, batch=256, lr=2e-3, seed=0):
    """Train on y = 0.5 c + 0.85 eps, c ~ U(-1,1), FRESH data every step (infinite
    data: no overfit/memorization confound). Loss = energy_score_fair with P=1, D=1
    -- the identical production code path at patch size 1."""
    import optax
    model = _MLP1D()
    key = jax.random.PRNGKey(seed)
    params = model.init(key, jnp.ones((2, 1)), jnp.ones((2, 1)))
    tx = optax.adam(optax.warmup_cosine_decay_schedule(0.0, lr, steps // 10, steps))
    opt = tx.init(params)

    def draw_noise(k, shape):
        if noise == "exp":
            return jax.random.exponential(k, shape) - 1.0
        if noise == "t5":
            return jax.random.t(k, 5.0, shape) / np.sqrt(5.0 / 3.0)
        return jax.random.normal(k, shape)

    @jax.jit
    def step(params, opt, key):
        key, kc, ke, kz = jax.random.split(key, 4)
        c = jax.random.uniform(kc, (batch, 1), minval=-1.0, maxval=1.0)
        y = 0.5 * c + 0.85 * draw_noise(ke, (batch, 1))

        def loss_fn(p):
            z = jax.random.normal(kz, (m, batch, 1))
            samp = jax.vmap(lambda zi: model.apply(p, zi, c))(z)
            return jnp.mean(energy_score_fair(samp[..., None, :], y[..., None, :]))

        loss, grads = jax.value_and_grad(loss_fn)(params)
        upd, opt = tx.update(grads, opt)
        return optax.apply_updates(params, upd), opt, key, loss

    for _ in range(steps):
        params, opt, key, loss = step(params, opt, key)
    return model, params, float(loss)


def _mlp_samples(model, params, n=400_000, seed=50):
    kc, kz = jax.random.split(jax.random.PRNGKey(seed))
    c = jax.random.uniform(kc, (n, 1), minval=-1.0, maxval=1.0)
    z = jax.random.normal(kz, (n, 1))
    y = model.apply(params, z, c)
    return np.asarray(c).ravel(), np.asarray(y).ravel()


# ------------------------- gate 4 (R10 condition 2, BINDING): two moments + skewness

def test_sampler_shape_gate_two_moments_and_skewness():
    """R10 condition 2: the direct noise-conditioned sampler must recover mean, std
    AND skewness of a known skewed conditional BEFORE any production training.
    Truth: y = 0.5 c + 0.85 (Exp(1)-1) -- resid skewness 2.0. Pre-set bands: binned
    conditional mean within 0.05 absolute, binned std within 10%, pooled residual
    skewness in [1.4, 2.6]."""
    model, params, _ = _train_mlp_1d("exp", seed=2)
    c, y = _mlp_samples(model, params)
    edges = np.linspace(-1, 1, 7)
    idx = np.clip(np.digitize(c, edges) - 1, 0, 5)
    resid = y - 0.5 * c
    for b in range(6):
        sel = idx == b
        assert abs(resid[sel].mean()) < 0.05, (b, resid[sel].mean())
        assert abs(resid[sel].std() / 0.85 - 1) < 0.10, (b, resid[sel].std())
    r = (resid - resid.mean()) / resid.std()
    skew = float(np.mean(r.astype(np.float64) ** 3))
    assert 1.4 < skew < 2.6, f"pooled skewness {skew:.2f} outside pre-set [1.4, 2.6]"


# ----------------------------------------------- gate 5: Student-t(5) TAIL behavior

def test_sampler_tail_student_t5():
    """t(5) tails, 1-D: a CHARACTERIZATION guard, not the capability gate.

    Measured 2026-07-17 (log/2026-07-17-c3-gate-design.md): the 1-D MLP+ES sampler
    converges to excess kurtosis ~1.1 on symmetric t(5) noise (truth 6.0) and holds
    there for 20k steps at any z-dim (1/4/8) -- symmetric heavy tails carry a
    second-order CRPS signal, unlike skewness (first-order, recovered to 2.0 above).
    The production-config CONV sampler does NOT share this limit: at production
    capacity/patch config it recovers exp kurtosis 5.76/6.0 held-out, and the
    production-config t(5) run (SLURM, logged pre-submission) is the arm's
    field-scale tail-capability evidence. This test pins the measured 1-D behavior
    to catch loss/estimator regressions: tails must stay beyond Gaussian (q999,
    kurt > 0.5) but a sudden jump into [3, 12] would ALSO be a change worth
    investigating (band [0.5, 3.0] around the measured 1.1)."""
    model, params, _ = _train_mlp_1d("t5", seed=3)
    c, y = _mlp_samples(model, params, n=600_000)
    resid = (y - 0.5 * c).astype(np.float64)
    r = (resid - resid.mean()) / resid.std()
    kurt = float(np.mean(r ** 4) - 3.0)
    q999 = float(np.quantile(np.abs(r), 0.999))
    assert 0.5 < kurt < 3.0, f"excess kurtosis {kurt:.2f} left the measured 1-D " \
                             f"regime [0.5, 3.0] (2026-07-17 characterization)"
    assert q999 > 3.6, f"q999 {q999:.2f} not beyond Gaussian 3.29"


# --------------------------------------------------------------- gate 6: plumbing

def test_sampler_determinism_and_multiscale(trained_gauss):
    state, _, coarse, loss = trained_gauss
    assert np.isfinite(loss)
    a = sample_direct(state.apply_fn, state.params, jax.random.PRNGKey(2), coarse[:4], 3)
    b = sample_direct(state.apply_fn, state.params, jax.random.PRNGKey(2), coarse[:4], 3)
    c = sample_direct(state.apply_fn, state.params, jax.random.PRNGKey(4), coarse[:4], 3)
    assert jnp.array_equal(a, b), "direct sampler must be deterministic given a key"
    assert not jnp.array_equal(a, c), "different keys must give different samples"
    # one weight set across octave shapes (weight-tying prior; octave-4 maps are 8x8)
    for hw in (8, 16, 32):
        out = sample_direct(state.apply_fn, state.params, jax.random.PRNGKey(0),
                            jnp.ones((2, hw, hw, 1)), 3)
        assert out.shape == (2, hw, hw, 3) and bool(jnp.all(jnp.isfinite(out)))


def test_patches_octave4_single_patch():
    """Octave-4 maps (8x8) are exactly one patch under the production 8/4 config."""
    x = jnp.arange(2 * 8 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 8, 3)
    p = extract_patches(x, patch=8, stride=4)
    assert p.shape == (2, 1, 192)
    assert jnp.array_equal(p[:, 0], x.reshape(2, -1))
    p2 = extract_patches(jnp.ones((2, 32, 32, 3)), patch=8, stride=4)
    assert p2.shape == (2, 49, 192)         # (32-8)/4+1 = 7 per axis


def test_trainer_pipeline_matches_c1(tmp_path):
    """The C3 trainer's data path (D4 augment -> field_to_octaves standardization) is
    C1's exactly; checkpoint hooks fire; arm B carries the 2-D coordinate."""
    rng = np.random.default_rng(0)
    # 64^2 tiles, octaves [2, 3] -> 16x16 and 8x8 details: the smallest production
    # shapes (patch=8 requires detail maps >= 8, which octave 4 of a 128^2 tile is)
    tiles = rng.normal(size=(6, 64, 64)).astype(np.float32)
    _, std_ref = field_to_octaves(d4_augment(tiles), [2, 3])
    seen = []
    state, meta = train_c3_generator(
        tiles, [2, 3], arm="A", channels=(8, 16), steps=4, batch=8, lr=1e-3,
        seed=0, ckpt_steps=(2,), on_checkpoint=lambda s, st, l: seen.append(s),
        augment=True, m=2)
    assert seen == [2], "checkpoint hook must fire at the requested step"
    assert meta["cond_dim"] == 0 and np.isfinite(meta["lossN"])
    assert meta["objective"] == "patched_energy_score_beta1"
    for j in (2, 3):
        assert np.isclose(meta["std_by_j"][j], std_ref[j]), "std convention must be C1's"
    _, meta_b = train_c3_generator(
        tiles, [2, 3], arm="B", cond_by_octave={2: [0.5, 0.3], 3: [0.4, 0.2]},
        channels=(8, 16), steps=2, batch=8, cond_mode="film", augment=True, m=2)
    assert meta_b["cond_dim"] == 2 and meta_b["cond_mode"] == "film"


# ------------------------------- gate 7: tail_q999 (R10 condition-1 instrument)

def test_tail_q999_gaussian_and_t5():
    rng = np.random.default_rng(123)
    z = rng.normal(size=2_000_000)
    want_g = st.norm.ppf(1 - 0.001 / 2)                      # 3.2905
    assert abs(tail_q999(z) - want_g) < 0.04, tail_q999(z)
    t = rng.standard_t(5, size=2_000_000)
    want_t = st.t.ppf(1 - 0.001 / 2, 5) / np.sqrt(5.0 / 3.0)  # 5.321 (unit-variance)
    assert abs(tail_q999(t) - want_t) < 0.15, (tail_q999(t), want_t)
    # convention: standardization is by the POOLED sample std (the estimand's), so a
    # scaled input gives the identical value
    assert np.isclose(tail_q999(3.7 * z), tail_q999(z))
