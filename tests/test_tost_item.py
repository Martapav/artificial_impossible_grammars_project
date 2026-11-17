"""The item-level two-proportion TOST (``analysis/reports/ch2/report_ch2_tost_item.py``).

Guards the companion to the seed-level test. Three things can go wrong here: the
``se == 0`` branch at ceiling (several gate cells score 1.0 in every seed), the
N_req certification arithmetic, and the effective-N premise itself -- the whole
analysis is only valid because the probe item set is FIXED across seeds, so that
premise is asserted against the scored data rather than assumed.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("pandas")
pytest.importorskip("scipy")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from analysis.reports.ch2.report_ch2_tost_item import (  # noqa: E402
    DELTA, N_ITEMS, classify_item, required_n, two_proportion_tost,
)

TOL = 1e-9


def test_se_matches_the_unpooled_two_proportion_formula():
    st = two_proportion_tost(0.90, 0.88)
    expected = math.sqrt(0.90 * 0.10 / 300 + 0.88 * 0.12 / 300)
    assert st["se_item"] == pytest.approx(expected, abs=TOL)
    assert st["diff"] == pytest.approx(0.02, abs=TOL)


def test_tost_p_values_match_the_normal_tails():
    p_a, p_b = 0.91, 0.895
    st = two_proportion_tost(p_a, p_b)
    d, se = st["diff"], st["se_item"]
    assert st["p_tost"] == pytest.approx(
        max(stats.norm.sf((d + DELTA) / se), stats.norm.cdf((d - DELTA) / se)), abs=TOL)
    assert st["p_nhst"] == pytest.approx(2 * stats.norm.sf(abs(d) / se), abs=TOL)


def test_ci90_uses_the_one_sided_critical_value():
    """The 90% interval is the 1 - 2*alpha interval TOST at alpha=0.05 corresponds to."""
    st = two_proportion_tost(0.80, 0.78)
    half = stats.norm.ppf(0.95) * st["se_item"]
    assert st["ci90_low"] == pytest.approx(st["diff"] - half, abs=TOL)
    assert st["ci90_high"] == pytest.approx(st["diff"] + half, abs=TOL)


def test_zero_variance_at_ceiling_is_guarded():
    """Both proportions pinned at 1.0 gives se == 0; must not divide by zero."""
    st = two_proportion_tost(1.0, 1.0)
    assert st["se_item"] == 0.0
    assert st["diff"] == 0.0
    assert classify_item(st["p_tost"], st["p_nhst"]) == "equivalent"
    for v in st.values():
        assert math.isfinite(v)


@pytest.mark.parametrize("p_tost,p_nhst,expected", [
    (0.01, 0.40, "equivalent"),
    (0.01, 0.01, "trivial"),
    (0.40, 0.01, "different"),
    (0.40, 0.40, "inconclusive"),
])
def test_verdict_quadrants(p_tost, p_nhst, expected):
    assert classify_item(p_tost, p_nhst) == expected


def test_required_n_grows_as_the_difference_approaches_the_bound():
    near = required_n(0.90, 0.86, -0.04)[0]
    far = required_n(0.90, 0.89, -0.01)[0]
    assert near > far


def test_difference_at_or_beyond_the_bound_is_uncertifiable():
    """No probe size certifies equivalence around a point already outside the bound."""
    n_req, certifiable = required_n(0.83, 0.78, 0.05)
    assert certifiable is False
    assert np.isnan(n_req)


def test_required_n_actually_certifies_at_that_size():
    """The returned N must make the 90% CI fit inside the bound -- the point of it."""
    p_a, p_b, diff = 0.90, 0.88, 0.02
    n_req, certifiable = required_n(p_a, p_b, diff)
    assert certifiable
    se = math.sqrt(p_a * (1 - p_a) / n_req + p_b * (1 - p_b) / n_req)
    assert abs(diff) + stats.norm.ppf(0.95) * se <= DELTA + 1e-6


# ---------------------------------------------------------------------------
# The premise the whole analysis rests on
# ---------------------------------------------------------------------------

def _probe_scores_present() -> bool:
    from analysis.report_common import EVAL_ROOT
    return any((EVAL_ROOT / "probe_scores").glob("*.parquet"))


@pytest.mark.skipif(not _probe_scores_present(), reason="probe score parquets not present")
def test_probe_item_set_is_fixed_across_seeds():
    """Every seed must score an identical item set, or N = 300 is the wrong unit.

    If the probe set were regenerated per seed the independent unit would be up to
    1500 clustered observations and this module's two-proportion test would be
    misspecified, so this premise is asserted, not assumed.
    """
    from analysis.report_common import EVAL_ROOT

    for grammar in ("H", "P", "Hprime", "Lprime"):
        shards = sorted((EVAL_ROOT / "probe_scores").glob(f"{grammar}__s*.parquet"))
        assert len(shards) == 5, f"{grammar}: expected 5 seed shards, found {len(shards)}"
        seen = [frozenset(pd.read_parquet(s, columns=["item_uid"])["item_uid"]) for s in shards]
        assert all(x == seen[0] for x in seen), f"{grammar}: item set varies across seeds"


@pytest.mark.skipif(not _probe_scores_present(), reason="probe score parquets not present")
def test_per_seed_accuracies_are_integral_counts_over_300():
    from analysis.report_common import ANALYSIS_ROOT

    csv = (ANALYSIS_ROOT / "evaluation" / "structural_probes" / "ch2" / "sub4_tost"
           / "tables" / "data_long.csv")
    if not csv.exists():
        pytest.skip("sub4_tost has not been run")
    k = pd.read_csv(csv)["accuracy"].to_numpy() * N_ITEMS
    assert np.abs(k - np.round(k)).max() < 1e-6
