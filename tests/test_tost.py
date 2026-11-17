"""The seed-level TOST equivalence core (``analysis/reports/ch2/report_ch2_tost.py``).

Guards the statistics behind the thesis' equivalence claim. Two failures matter
here. The first is silent NaN: the between-seed SD of a difference is sometimes
exactly zero, which makes ``se == 0`` and the t-statistic undefined -- the guarded
branch must still yield the right four-way verdict. The second is a wrong verdict
boundary, so every case below is hand-checkable, and the non-degenerate p-values
are cross-validated against ``statsmodels`` rather than against themselves.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.reports.ch2.report_ch2_tost import (  # noqa: E402
    DELTA, VERDICTS, _paired_stats, _welch_stats, classify, mde_paired,
)

TOL = 1e-9


def _verdict(st):
    return classify(st["p_nhst"], st["p_tost"])


# ---------------------------------------------------------------------------
# The four verdicts, one hand-checkable case each
# ---------------------------------------------------------------------------

def test_equivalent_nondegenerate_matches_statsmodels():
    """A small, tightly estimated difference is `equivalent`, and p_tost is right.

    The cross-check is the point: `statsmodels` is an independent implementation
    of the same two one-sided tests, so agreement to 1e-9 rules out a sign or
    tail error in the manual formulas.
    """
    a = [.93, .94, .93, .94, .93]
    b = [.92, .92, .93, .92, .93]
    st = _paired_stats(a, b, DELTA, 0.05)
    assert st["diff"] == pytest.approx(0.01, abs=TOL)
    assert _verdict(st) == "equivalent"
    assert not st["degenerate"]

    sm = pytest.importorskip("statsmodels.stats.weightstats")
    expected, _, _ = sm.ttost_paired(np.asarray(a), np.asarray(b), -DELTA, DELTA)
    assert st["p_tost"] == pytest.approx(expected, abs=TOL)


def test_different_effect_is_flagged_and_ci_excludes_zero():
    a = [.98] * 5
    b = [.80, .79, .78, .81, .80]
    st = _paired_stats(a, b, DELTA, 0.05)
    assert _verdict(st) == "different"
    assert st["p_tost"] >= 0.05                      # equivalence NOT declared
    assert st["ci90_low"] > 0.0                      # CI excludes 0


def test_trivial_is_degenerate_nonzero_below_delta():
    """Zero between-seed variance, difference below the bound -> `trivial`."""
    st = _paired_stats([.95] * 5, [.93] * 5, DELTA, 0.05)
    assert st["se"] == 0.0
    assert st["diff"] == pytest.approx(0.02, abs=TOL)
    assert st["degenerate"] is True
    assert _verdict(st) == "trivial"
    assert (st["ci90_low"], st["ci90_high"]) == (st["diff"], st["diff"])


def test_equivalent_is_degenerate_exact_equality():
    st = _paired_stats([.94] * 5, [.94] * 5, DELTA, 0.05)
    assert st["diff"] == 0.0
    assert st["degenerate"] is True
    assert _verdict(st) == "equivalent"


def test_degenerate_difference_at_the_bound_is_not_equivalent():
    """|diff| == DELTA is the boundary and must fall OUTSIDE the region.

    The operands are chosen so the subtraction is exact in binary (.05 - .0);
    the natural-looking .95 - .90 lands on 0.0499999... and would test the
    wrong side of the bound by accident.
    """
    st = _paired_stats([.05] * 5, [.0] * 5, DELTA, 0.05)
    assert st["diff"] == DELTA                       # exact, not approx
    assert st["p_tost"] == 1.0
    assert _verdict(st) == "different"


def test_degenerate_difference_above_the_bound_is_different():
    st = _paired_stats([.95] * 5, [.87] * 5, DELTA, 0.05)
    assert st["se"] == 0.0
    assert st["diff"] > DELTA
    assert _verdict(st) == "different"


# ---------------------------------------------------------------------------
# No NaN/inf escapes the guarded branches
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b", [
    ([.94] * 5, [.94] * 5),                       # se == 0, diff == 0
    ([.95] * 5, [.93] * 5),                       # se == 0, diff != 0
    ([.93, .94, .93, .94, .93], [.92, .92, .93, .92, .93]),
])
def test_paired_stats_are_always_finite(a, b):
    st = _paired_stats(a, b, DELTA, 0.05)
    for k, v in st.items():
        if isinstance(v, float):
            assert math.isfinite(v), f"{k} is not finite"


def test_equivalence_agrees_with_the_ninety_percent_ci():
    """`p_tost < alpha` must hold exactly when the 90% CI sits inside the bound.

    The two are the same statement; if the CI is built from a different critical
    value than the one-sided tests use, they silently disagree near the boundary.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        a = rng.uniform(0.70, 1.00, 5)
        b = a + rng.normal(0.0, 0.03, 5)
        st = _paired_stats(a, b, DELTA, 0.05)
        equiv = st["p_tost"] < 0.05
        inside = -DELTA < st["ci90_low"] and st["ci90_high"] < DELTA
        assert equiv == inside, st


def test_welch_fallback_handles_zero_variance():
    """The unpaired branch needs the same guard; it is dead on current data."""
    st = _welch_stats([.95] * 5, [.93] * 4, DELTA, 0.05)
    assert st["se"] == 0.0
    assert st["degenerate"] is True
    assert _verdict(st) == "trivial"


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

def test_mde_is_zero_without_noise_and_monotonic_in_it():
    assert mde_paired(0.0, 5) == 0.0
    vals = [mde_paired(sd, 5) for sd in (0.005, 0.01, 0.02, 0.05)]
    assert all(x < y for x, y in zip(vals, vals[1:]))


def test_mde_shrinks_with_more_seeds():
    assert mde_paired(0.02, 20) < mde_paired(0.02, 5)


# ---------------------------------------------------------------------------
# End-to-end on the real probe scores (skipped when they are absent)
# ---------------------------------------------------------------------------

def _probe_scores_present() -> bool:
    from analysis.report_common import EVAL_ROOT
    return any((EVAL_ROOT / "probe_scores").glob("*.parquet"))


@pytest.mark.skipif(not _probe_scores_present(),
                    reason="ep12 probe score parquets not present")
def test_real_contrast_table_is_well_formed():
    from analysis.reports.ch2.report_ch2_tost import MIN_SEEDS, build_data_long, run_contrasts

    long = build_data_long()
    assert set(long.columns) == {"grammar", "subcase", "split", "seed", "accuracy"}
    assert long["accuracy"].between(0.0, 1.0).all()

    res = run_contrasts(long)
    assert len(res) > 0
    numeric = res.select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all(), "NaN/inf in the contrast table"
    assert (res["n_seeds"] >= MIN_SEEDS).all()
    assert set(res["verdict"]) <= set(VERDICTS)
    # The cross-grammar n is the seed count, never the 300-pair count.
    assert res["n_seeds"].max() <= 20


@pytest.mark.skipif(not _probe_scores_present(),
                    reason="ep12 probe score parquets not present")
def test_per_seed_accuracies_match_the_seed_variance_section():
    """`data_long` must be the same five dots Section 4.3.1 plots, not a re-derivation."""
    from analysis.report_common import ANALYSIS_ROOT
    from analysis.reports.ch2.report_ch2_tost import build_data_long

    ref_csv = (ANALYSIS_ROOT / "evaluation" / "structural_probes" / "ch2"
               / "sub2_seed_variance" / "tables" / "seed_points.csv")
    if not ref_csv.exists():
        pytest.skip("sub2_seed_variance has not been run")

    ref = (pd.read_csv(ref_csv)
           .rename(columns={"phenomenon": "subcase", "acc": "accuracy"})
           .set_index(["grammar", "subcase", "split", "seed"])["accuracy"]
           .sort_index())
    got = (build_data_long()
           .set_index(["grammar", "subcase", "split", "seed"])["accuracy"]
           .sort_index())
    assert got.index.equals(ref.index)
    assert np.allclose(got.to_numpy(), ref.to_numpy(), atol=TOL)
