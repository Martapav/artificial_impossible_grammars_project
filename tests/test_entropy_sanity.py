"""A3 sanity checks on the grammar floor.

Three questions, each of which would invalidate `ℓ(x)` if answered wrong:

1. **Is the arithmetic right?** For a draw whose distribution we set ourselves,
   the mean of the logged `−log p` must converge to that distribution's analytic
   entropy. This is an *independent* check: the analytic value comes from the
   rule constants, the measured value from the tracing layer, and nothing is
   shared between the two paths.
2. **Is `Ĥ(p)` stable?** It is a sample mean, so it must agree across independent
   seeds to within Monte Carlo error. Systematic drift would mean the trace
   depends on something other than the derivation.
3. **Is `ℓ(x)` always finite?** A zero-probability drawn outcome makes `ℓ(x)`
   diverge and always indicates a bug.

Note on filtering: checks of type 1 run on RAW `_one_item` draws, never on corpus
items. The corpus applies a construction quota, so the phenomenon distribution in
an accepted batch is the *quota*, not the generator's own categorical — measuring
entropy there would recover the quota and prove nothing.
"""
from __future__ import annotations

import math
import statistics

import pytest

from grammars.grammar_H.generate import _one_item as h_item
from grammars.grammar_H.lexicon import load as load_h
from grammars.grammar_H.rules import PHENOMENA, PHENOMENON_PROBS, P_AUX
from grammars.grammar_P.generate import _one_item as p_item
from grammars.grammar_P.lexicon import load as load_p
from grammars.tracing import TracedRandom

N_DRAWS = 4000


def _entropy(probs) -> float:
    return -math.fsum(p * math.log(p) for p in probs if p > 0)


def _collect(item_fn, lex, n=N_DRAWS, seed=20260731):
    """Per-label −log p across `n` UNFILTERED draws, plus per-item ℓ(x)."""
    rng = TracedRandom(seed)
    by_label: dict[str, list[float]] = {}
    ells: list[float] = []
    for _ in range(n):
        rng.reset_trace()
        item_fn(rng, lex)
        for d in rng.trace.draws:
            by_label.setdefault(d.label, []).append(-d.logp)
        ells.append(rng.trace.neg_logp)
    return by_label, ells


@pytest.fixture(scope="module")
def h_draws():
    return _collect(h_item, load_h())


@pytest.fixture(scope="module")
def p_draws():
    return _collect(p_item, load_p())


# ---------------------------------------------------------------------------
# 1. Analytic cross-check
# ---------------------------------------------------------------------------

def test_phenomenon_entropy_matches_analytic(h_draws):
    """Mean −log p of the phenomenon draw == H(PHENOMENON_PROBS).

    Independent: the expected value is computed from the rule constants, the
    measured value from the tracing layer.
    """
    by_label, _ = h_draws
    measured = statistics.fmean(by_label["phenomenon"])
    analytic = _entropy(PHENOMENON_PROBS[k] for k in PHENOMENA)
    se = statistics.stdev(by_label["phenomenon"]) / math.sqrt(len(by_label["phenomenon"]))
    assert abs(measured - analytic) < 4 * se, (measured, analytic, se)


def test_insitu_aux_entropy_matches_binary_entropy(h_draws):
    """Conditional on being drawn, the in-situ aux flag is Bernoulli(P_AUX)."""
    by_label, _ = h_draws
    vals = by_label["insitu_aux"]
    measured = statistics.fmean(vals)
    analytic = _entropy((P_AUX, 1 - P_AUX))
    se = statistics.stdev(vals) / math.sqrt(len(vals))
    assert abs(measured - analytic) < 4 * se, (measured, analytic, se)


def test_binding_subtype_is_a_fair_coin(h_draws):
    """`bind_subtype` is an explicit 0.5 draw, so every logged value is log 2."""
    by_label, _ = h_draws
    assert all(math.isclose(v, math.log(2), rel_tol=1e-12)
               for v in by_label["bind_subtype"])


# ---------------------------------------------------------------------------
# 2. Stability of Ĥ(p) across seeds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("item_fn,loader", [(h_item, load_h), (p_item, load_p)])
def test_hhat_stable_across_seeds(item_fn, loader):
    """Independent seeds must agree to within Monte Carlo error."""
    lex = loader()
    means, ses = [], []
    for seed in (1, 2, 3):
        _, ells = _collect(item_fn, lex, n=2500, seed=seed)
        means.append(statistics.fmean(ells))
        ses.append(statistics.stdev(ells) / math.sqrt(len(ells)))
    spread = max(means) - min(means)
    # Difference of two independent means has SE = sqrt(se_i^2 + se_j^2).
    tol = 4 * math.sqrt(2) * max(ses)
    assert spread < tol, (means, ses, spread, tol)


# ---------------------------------------------------------------------------
# 3. Finiteness / no impossible draws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fixture", ["h_draws", "p_draws"])
def test_ell_is_finite_and_positive(fixture, request):
    _, ells = request.getfixturevalue(fixture)
    assert all(0.0 < e < math.inf for e in ells)


@pytest.mark.parametrize("fixture", ["h_draws", "p_draws"])
def test_every_logged_draw_has_positive_probability(fixture, request):
    """−log p must be finite and non-negative for every draw, i.e. 0 < p ≤ 1."""
    by_label, _ = request.getfixturevalue(fixture)
    for label, vals in by_label.items():
        assert all(0.0 <= v < math.inf for v in vals), label


@pytest.mark.parametrize("fixture", ["h_draws", "p_draws"])
def test_ell_equals_sum_of_its_labels(fixture, request):
    """Bookkeeping: total ℓ(x) is exactly the per-label attribution summed."""
    by_label, ells = request.getfixturevalue(fixture)
    assert math.isclose(math.fsum(math.fsum(v) for v in by_label.values()),
                        math.fsum(ells), rel_tol=1e-9)
