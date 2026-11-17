"""The binding forced-draw credit: the entropy of draws binding marginalizes.

Guards the two-component correction (``analysis/binding_credit.py``) that makes
the binding differential surprisal a non-negative KL again: the target-block
marginalization and the geometry-resample log Z. The failure it prevents is the
one this whole correction exists to remove — a silently negative ``D_KL(p‖q)``
on ``anaphoric_binding_{pron,refl}``.
"""
from __future__ import annotations

import math

import pytest

pytest.importorskip("pandas")
import pandas as pd  # noqa: E402

from analysis.binding_credit import (  # noqa: E402
    SUBTYPES, _block_cost, _zipf_neglogp, apply_credit, credit_map,
)
from grammars.tracing import TracedRandom  # noqa: E402


# ---------------------------------------------------------------------------
# Draw pricing matches the generator's `pick`
# ---------------------------------------------------------------------------

def test_zipf_neglogp_matches_traced_pick_open_class():
    """The analytic price equals what the tracer logs for a Zipf `pick`."""
    from grammars.grammar_H.lexicon import cat1_items, load, pick
    pool = cat1_items(load())
    rng = TracedRandom(0)
    # Per drawn item the traced −log p and the analytic price are identical, so a
    # small sample suffices to assert the exact match (not a statistical one).
    n = 500
    emp = 0.0
    ana = 0.0
    for _ in range(n):
        rng.reset_trace()
        item = pick(rng, pool)
        emp += rng.trace.neg_logp
        ana += _zipf_neglogp(item, pool)
    assert math.isclose(emp / n, ana / n, rel_tol=1e-12)


def test_zipf_neglogp_uniform_for_closed_class():
    pool = [{"form": f"x{i}"} for i in range(5)]      # no zipf_rank -> uniform
    assert math.isclose(_zipf_neglogp(pool[2], pool), math.log(5))


# ---------------------------------------------------------------------------
# The block-cost formula
# ---------------------------------------------------------------------------

_C1 = [{"zipf_rank": r + 1, "countability": "countable"} for r in range(50)]
_C2 = [{"zipf_rank": r + 1} for r in range(30)]


def test_noncountable_head_is_lexeme_plus_number_only():
    """No CAT2 loop for a non-countable head — just head lexeme + INFL number."""
    head = {"zipf_rank": 1, "countability": "mass"}
    cost = _block_cost(head, [], _C1, _C2, p_cat2=0.4)
    assert math.isclose(cost, _zipf_neglogp(head, _C1) + math.log(2))


def test_countable_head_adds_loop_and_modifiers():
    """A countable head pays the terminating Bernoulli even with zero modifiers,
    and strictly more with a modifier attached."""
    head = _C1[0]
    zero = _block_cost(head, [], _C1, _C2, p_cat2=0.4)
    one = _block_cost(head, [_C2[3]], _C1, _C2, p_cat2=0.4)
    base = _zipf_neglogp(head, _C1) + math.log(2)
    assert zero > base                          # terminating False Bernoulli
    assert math.isclose(zero, base - math.log(1 - 0.4))
    # one modifier adds: one continuation Bernoulli + one CAT2 pick
    assert math.isclose(one - zero,
                        -math.log(0.4) + _zipf_neglogp(_C2[3], _C2))


# ---------------------------------------------------------------------------
# apply_credit
# ---------------------------------------------------------------------------

@pytest.fixture
def credit_parquet(tmp_path):
    path = tmp_path / "binding_credit.parquet"
    pd.DataFrame([
        dict(grammar="H", subtype="anaphoric_binding_refl", credit=12.5),
        dict(grammar="H", subtype="anaphoric_binding_pron", credit=9.8),
    ]).to_parquet(path, index=False)
    return path


def test_apply_credit_subtracts_only_on_binding(credit_parquet):
    ell = pd.DataFrame({
        "item_uid": list("abcd"),
        "construction": ["neutral", "wh_movement",
                         "anaphoric_binding_refl", "anaphoric_binding_pron"],
        "ell_corpus": [10.0, 20.0, 30.0, 40.0],
    })
    out = apply_credit(ell, "H", path=credit_parquet)
    # non-binding unchanged
    assert out.loc[0, "ell_corrected"] == 10.0
    assert out.loc[1, "ell_corrected"] == 20.0
    # binding reduced by exactly the credit
    assert math.isclose(out.loc[2, "ell_corrected"], 30.0 - 12.5)
    assert math.isclose(out.loc[3, "ell_corrected"], 40.0 - 9.8)


def test_apply_credit_raises_when_binding_credit_missing(credit_parquet):
    """A silent zero on a binding construction would reinstate the artefact."""
    ell = pd.DataFrame({
        "item_uid": ["z"],
        "construction": ["anaphoric_binding_refl"],
        "ell_corpus": [30.0],
    })
    with pytest.raises(KeyError, match="no binding credit"):
        apply_credit(ell, "Lprime", path=credit_parquet)   # Lprime absent here


def test_credit_map_keys(credit_parquet):
    cm = credit_map(credit_parquet)
    assert cm[("H", "anaphoric_binding_refl")] == 12.5
    assert set(SUBTYPES) == {"anaphoric_binding_refl", "anaphoric_binding_pron"}


# ---------------------------------------------------------------------------
# End to end on the real generator (small, but exercises both components)
# ---------------------------------------------------------------------------

def test_measure_credit_structure_tree():
    """Tree family: both credits positive; refl carries a geometry log Z, pron
    does not (its loop has no geometry check). Runs the real generator, so it is
    the slowest test here (a few thousand items)."""
    from analysis.binding_credit import measure_credit
    df = measure_credit("H", n=6000, seed=3).set_index("subtype")
    for s in SUBTYPES:
        assert df.loc[s, "credit"] > 0
        assert df.loc[s, "block"] > 0
    assert df.loc["anaphoric_binding_refl", "logz"] > 0.5
    assert df.loc["anaphoric_binding_pron", "logz"] == 0.0
    # refl's total credit exceeds pron's by roughly the log Z it alone pays
    assert df.loc["anaphoric_binding_refl", "credit"] > \
        df.loc["anaphoric_binding_pron", "credit"]
