"""Verify the score-from-velocity identity on the GRF (Gaussian) control, before it is
used for SDE sampling. For the OT path x_t=(1-t)x0 + t*x1 with x0~N(0,I) and Gaussian data
x1~N(m, s^2) (the GRF case), both the exact velocity and the exact score are analytic; we
check s(x,t) == (t*v(x,t) - x)/(1-t) to machine precision, and against a finite-difference
of log p_t.
"""
import numpy as np

from wfm.cfm import score_from_velocity


def _analytic(m, s2, t, x):
    """Exact velocity and score for x1~N(m, s2), x0~N(0,1), linear path."""
    Vt = (1 - t) ** 2 + t ** 2 * s2                 # Var(x_t)
    dx = x - t * m
    # E[x1|x]=m+(t*s2/Vt)dx, E[x0|x]=((1-t)/Vt)dx ; v = E[x1-x0|x]
    v = m + (t * s2 - (1 - t)) / Vt * dx
    score = -dx / Vt                                # d/dx log N(x; t m, Vt)
    return v, score


def test_identity_matches_analytic_gaussian():
    x = np.linspace(-4, 4, 41)
    for m in (0.0, 0.7, -1.3):
        for s2 in (0.25, 1.0, 3.0):
            for t in (0.05, 0.3, 0.6, 0.9):
                v, score = _analytic(m, s2, t, x)
                s_hat = score_from_velocity(v, x, t)
                assert np.allclose(np.asarray(s_hat), score, atol=1e-10), (m, s2, t)


def test_identity_matches_finite_difference_score():
    """The identity's score equals a finite-difference of log p_t (Gaussian data)."""
    rng = np.random.default_rng(0)
    m, s2 = 0.4, 1.7
    for t in (0.1, 0.5, 0.85):
        Vt = (1 - t) ** 2 + t ** 2 * s2
        x = rng.normal(size=20)
        v, _ = _analytic(m, s2, t, x)
        s_hat = np.asarray(score_from_velocity(v, x, t))
        h = 1e-4
        logp = lambda z: -0.5 * (z - t * m) ** 2 / Vt
        fd = (logp(x + h) - logp(x - h)) / (2 * h)
        assert np.allclose(s_hat, fd, atol=1e-4)
