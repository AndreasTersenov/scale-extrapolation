"""Tests-first for the Stage-0 branch adjudication (prereg 62db5f0 + A1).

The pcure lesson (R36.2): verdict logic gets verified against the
pre-statement BEFORE it ever sees data. Every branch of the prereg table is
exercised here, including A1 (sign-flip), the #11 at-bar band, the mixed rule,
and the panel-blocked-CLOSED cascade.
"""
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts_p2"))

from stage0_p3_score import adjudicate_stage0  # noqa: E402

REFS = {"2.5": (0.1313, 0.031), "3.0": (0.1462, 0.0308)}
PANEL_OK = {"2.5": [0.02, -0.01, 0.03], "3.0": [0.01, -0.02, 0.02]}
PANEL_SIGNCONS = {"2.5": [0.05, 0.04, 0.06], "3.0": [0.01, -0.02, 0.02]}


def ex(e25, s25, ci25, e30, s30, ci30):
    return {"2.5": {"excess": e25, "se": s25, "ci95": list(ci25)},
            "3.0": {"excess": e30, "se": s30, "ci95": list(ci30)}}


def test_closed():
    r = adjudicate_stage0(ex(0.01, 0.02, (-0.03, 0.05),
                             0.01, 0.02, (-0.03, 0.05)), REFS, PANEL_OK)
    assert r["branch"] == "S0-CLOSED"


def test_flipped_takes_precedence():
    # ci95 entirely below zero at one nu: A1 fires even if the other nu
    # would read UNCHANGED
    r = adjudicate_stage0(ex(-0.08, 0.02, (-0.12, -0.03),
                             0.13, 0.03, (0.07, 0.19)), REFS, PANEL_OK)
    assert r["branch"] == "S0-FLIPPED"
    assert r["flipped_at"] == ["2.5"]


def test_shrunk():
    r = adjudicate_stage0(ex(0.05, 0.012, (0.027, 0.074),
                             0.05, 0.012, (0.027, 0.074)), REFS, PANEL_OK)
    assert r["branch"] == "S0-SHRUNK"


def test_unchanged():
    r = adjudicate_stage0(ex(0.13, 0.03, (0.07, 0.19),
                             0.145, 0.03, (0.085, 0.205)), REFS, PANEL_OK)
    assert r["branch"] == "S0-UNCHANGED"


def test_mixed_worse_governs():
    r = adjudicate_stage0(ex(0.04, 0.01, (0.02, 0.06),
                             0.13, 0.03, (0.07, 0.19)), REFS, PANEL_OK)
    assert r["branch"] == "MIXED/AT-BAR"
    assert r["governing"] == "UNCHANGED"


def test_at_bar_band():
    # e within 0.5*se of r/2 at both nus, categories agree: still MIXED/AT-BAR
    r = adjudicate_stage0(ex(0.066, 0.02, (0.027, 0.105),
                             0.070, 0.02, (0.031, 0.109)), REFS, PANEL_OK)
    assert r["branch"] == "MIXED/AT-BAR"
    assert r["per_nu"]["2.5"]["at_bar"] and r["per_nu"]["3.0"]["at_bar"]


def test_panel_blocks_closed_cascades_to_shrunk():
    r = adjudicate_stage0(ex(0.03, 0.025, (-0.02, 0.08),
                             0.03, 0.025, (-0.02, 0.08)), REFS, PANEL_SIGNCONS)
    assert r["branch"] == "S0-SHRUNK"
    assert "panel blocked CLOSED" in r["note"]
