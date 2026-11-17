"""TracedRandom must observe draws without perturbing them (entropy plan §2.2).

The whole point of the tracing layer is that it can be switched on without
changing a single generated character: if the RNG stream shifted, the corpora
would change and every downstream artifact (probes, sidecars, the separability
audit) would have to be rebuilt. These tests pin that guarantee, and check that
the recorded probabilities are the right numbers.
"""
from __future__ import annotations

import math
import random

import pytest

from grammars.tracing import AcceptanceCounter, TracedRandom

SEED = 20260731


# ---------------------------------------------------------------------------
# Stream preservation
# ---------------------------------------------------------------------------

def test_choice_stream_identical():
    seq = list(range(17))
    plain, traced = random.Random(SEED), TracedRandom(SEED)
    assert [plain.choice(seq) for _ in range(200)] == \
           [traced.choice(seq) for _ in range(200)]


def test_choices_stream_identical():
    pop = list("abcdefg")
    w = [5, 1, 9, 3, 2, 8, 4]
    plain, traced = random.Random(SEED), TracedRandom(SEED)
    assert [plain.choices(pop, weights=w, k=1) for _ in range(200)] == \
           [traced.choices(pop, weights=w, k=1) for _ in range(200)]


def test_bernoulli_matches_raw_comparison():
    """`rng.bernoulli(p)` consumes exactly what `rng.random() < p` consumed."""
    plain, traced = random.Random(SEED), TracedRandom(SEED)
    expected = [plain.random() < 0.3 for _ in range(200)]
    got = [traced.bernoulli(0.3, "b") for _ in range(200)]
    assert expected == got


def test_categorical_matches_cumulative_scan():
    """`categorical` reproduces the `r = rng.random(); acc += ...` idiom."""
    probs = {"neutral": 0.70, "aux": 0.10, "wh": 0.10, "bind": 0.10}

    def scan(r):
        acc = 0.0
        for name, p in probs.items():
            acc += p
            if r < acc:
                return name
        return list(probs)[-1]

    plain, traced = random.Random(SEED), TracedRandom(SEED)
    expected = [scan(plain.random()) for _ in range(300)]
    got = [traced.categorical(probs, "phenomenon") for _ in range(300)]
    assert expected == got


def test_interleaved_stream_identical():
    """Mixed draw kinds in sequence stay aligned with the untraced stream."""
    seq, pop, w = [1, 2, 3, 4, 5], list("xyz"), [2, 3, 5]
    plain, traced = random.Random(SEED), TracedRandom(SEED)
    a, b = [], []
    for _ in range(120):
        a += [plain.choice(seq), plain.random() < 0.4,
              plain.choices(pop, weights=w, k=1)[0]]
        b += [traced.choice(seq), traced.bernoulli(0.4, "b"),
              traced.choices(pop, weights=w, k=1)[0]]
    assert a == b


# ---------------------------------------------------------------------------
# Recorded probabilities are correct
# ---------------------------------------------------------------------------

def test_uniform_choice_logp():
    seq = list(range(8))
    rng = TracedRandom(SEED)
    for _ in range(50):
        rng.choice(seq)
    assert all(math.isclose(d.logp, -math.log(8)) for d in rng.trace.draws)


def test_choice_marginalizes_duplicate_values():
    """p is the probability of the VALUE drawn, not of a slot."""
    rng = TracedRandom(SEED)
    seq = ["a", "a", "a", "b"]          # p(a)=3/4, p(b)=1/4
    for _ in range(200):
        rng.choice(seq)
    seen = {round(math.exp(d.logp), 6) for d in rng.trace.draws}
    assert seen <= {0.75, 0.25} and seen


def test_weighted_choices_logp():
    rng = TracedRandom(SEED)
    pop, w = ["a", "b", "c"], [1.0, 3.0, 6.0]
    expect = {"a": 0.1, "b": 0.3, "c": 0.6}
    for _ in range(300):
        out = rng.choices(pop, weights=w, k=1)[0]
        d = rng.trace.draws[-1]
        assert math.isclose(math.exp(d.logp), expect[out], rel_tol=1e-9)


def test_bernoulli_logp_both_branches():
    rng = TracedRandom(SEED)
    for _ in range(400):
        rng.bernoulli(0.25, "b")
    ps = sorted({round(math.exp(d.logp), 6) for d in rng.trace.draws})
    assert ps == [0.25, 0.75]


def test_neg_logp_is_sum_and_nonnegative():
    rng = TracedRandom(SEED)
    for _ in range(100):
        rng.choice(list(range(5)))
        rng.bernoulli(0.5, "b")
    t = rng.trace
    assert t.n_draws == 200
    assert math.isclose(t.neg_logp, -math.fsum(d.logp for d in t.draws))
    assert t.neg_logp > 0


def test_impossible_outcome_raises():
    """A zero-probability drawn outcome means a bug, not an infinite ℓ(x).

    Unreachable for ``bernoulli`` (at p=0 the complement, probability 1, is
    always what gets drawn). It IS reachable for ``categorical`` when the
    probabilities under-sum and the zero-mass final entry absorbs the shortfall
    — exactly the malformed-table case worth failing on.
    """
    rng = TracedRandom(SEED)
    with pytest.raises(ValueError, match="positive probability"):
        for _ in range(200):                      # some r must exceed 0.3
            rng.categorical({"a": 0.3, "b": 0.0}, "malformed")


def test_bernoulli_extremes_are_legal():
    """p=0 and p=1 are degenerate but well defined: the drawn outcome has p=1."""
    rng = TracedRandom(SEED)
    assert rng.bernoulli(0.0, "never") is False
    assert rng.bernoulli(1.0, "always") is True
    assert all(math.isclose(d.logp, 0.0) for d in rng.trace.draws)


# ---------------------------------------------------------------------------
# Trace scoping
# ---------------------------------------------------------------------------

def test_reset_trace_drops_rejected_attempt():
    """Draws burnt on a rejected attempt must not enter the accepted ℓ(x)."""
    rng = TracedRandom(SEED)
    for _ in range(10):
        rng.choice([1, 2, 3, 4])
    rng.reset_trace()                       # attempt rejected; start over
    for _ in range(3):
        rng.choice([1, 2, 3, 4])
    assert rng.trace.n_draws == 3
    assert math.isclose(rng.trace.neg_logp, 3 * math.log(4))


def test_take_trace_returns_and_clears():
    rng = TracedRandom(SEED)
    rng.choice([1, 2])
    t = rng.take_trace()
    assert t.n_draws == 1 and rng.trace.n_draws == 0


def test_by_label_attribution():
    rng = TracedRandom(SEED)
    for _ in range(4):
        rng.bernoulli(0.5, "aux")
    for _ in range(2):
        rng.bernoulli(0.5, "pp")
    by = rng.trace.by_label()
    assert set(by) == {"aux", "pp"}
    assert math.isclose(by["aux"], 4 * math.log(2))
    assert math.isclose(by["pp"], 2 * math.log(2))


# ---------------------------------------------------------------------------
# Acceptance bookkeeping (the rejection-sampling correction)
# ---------------------------------------------------------------------------

def test_acceptance_counter_tracks_both_normalizations():
    """Eligibility measures the grammar; accepted counts measure the quota.

    The correction in `analysis/entropy_baseline.py` needs the former. Here
    `wh_movement` is eligible on 30 of 100 draws but the quota keeps only 10, so
    the two numbers differ by 3x — which is exactly the reweighting the
    correction has to undo.
    """
    c = AcceptanceCounter()
    for _ in range(70):
        c.note_attempt("neutral", eligible=True); c.note_accept("neutral")
    for _ in range(10):
        c.note_attempt("wh_movement", eligible=True); c.note_accept("wh_movement")
    for _ in range(20):
        c.note_attempt("wh_movement", eligible=True)   # eligible, quota already full
    s = c.summary()
    assert s["attempts"] == 100
    assert s["accepted_total"] == 80
    assert s["drawn_by_construction"]["wh_movement"] == 30
    assert s["eligible_by_construction"]["wh_movement"] == 30
    assert math.isclose(s["p_eligible_by_construction"]["wh_movement"], 0.30)
    assert math.isclose(s["corpus_proportion"]["wh_movement"], 10 / 80)


def test_acceptance_counter_separates_ineligible_from_quota_rejected():
    """A draw rejected for length is NOT eligible; one rejected by quota is."""
    c = AcceptanceCounter()
    c.note_attempt("wh", eligible=True)        # would have been kept
    c.note_attempt("wh", eligible=False)       # out of length window
    c.note_attempt("wh_skipped", eligible=False)
    s = c.summary()
    assert s["drawn_by_construction"]["wh"] == 2
    assert s["eligible_by_construction"]["wh"] == 1
    assert "wh_skipped" not in s["eligible_by_construction"]
