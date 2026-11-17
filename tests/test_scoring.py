"""Tests for grammars/scoring.py — scoring a derivation under π_corpus (A2.1).

The headline property is the round-trip: a derivation the corpus generator
produced, re-scored through the corpus code path, must reproduce the traced
generator's own ℓ(x). Not approximately — bit-exactly. The same code computes
the same distribution at the same site and ``math.fsum`` is deterministic, so
any drift means the replay took a different path and the number is not π_corpus.

The rest of the file pins the two failure modes apart. A2.3 turns them into
different null reasons, so a test suite that let one masquerade as the other
would let a broken join look like a legitimately undefined value.
"""
from __future__ import annotations

import math

import pytest

from grammars.scoring import (
    Choice, Derivation, DerivationRecorder, OutOfSupport, ScoringRandom,
    StructuralDivergence, score_derivation,
)

GRAMMARS = ("H", "P", "Hprime", "Lprime")


def _parts(grammar):
    """(one_item, lex) for a grammar, mirroring how the emission path calls it."""
    if grammar in ("H", "Hprime"):
        from grammars.grammar_H.generate import _one_item
        from grammars.grammar_H.lexicon import load
        lex = load()
        if grammar == "H":
            return _one_item, lex
        from grammars.grammar_H_prime.transforms import apply as apply_fn
    else:
        from grammars.grammar_P.generate import _one_item
        from grammars.grammar_P.lexicon import load
        lex = load()
        if grammar == "P":
            return _one_item, lex
        from grammars.grammar_L_prime.transforms import apply as apply_fn
    return (lambda rng, lexicon: _one_item(rng, lexicon, apply_fn=apply_fn)), lex


# ─────────────────────────────────────────────────────────────────────────────
# Round trip: the corpus generator scoring its own output
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_roundtrip_reproduces_traced_ell_exactly(grammar):
    one_item, lex = _parts(grammar)
    rec = DerivationRecorder(42)
    for _ in range(40):
        rec.reset_trace()
        one_item(rec, lex)
        recorded_ell, recorded_n = rec.trace.neg_logp, rec.trace.n_draws
        result = score_derivation(rec.derivation(), lambda r: one_item(r, lex))
        assert result.neg_logp == recorded_ell
        assert result.n_choices == recorded_n


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_roundtrip_reproduces_per_label_attribution(grammar):
    """Not just the total: the mass must sit at the same sites."""
    one_item, lex = _parts(grammar)
    rec = DerivationRecorder(7)
    for _ in range(15):
        rec.reset_trace()
        one_item(rec, lex)
        recorded = {k: v for k, v in sorted(rec.trace.by_label().items())}
        result = score_derivation(rec.derivation(), lambda r: one_item(r, lex))
        assert result.by_label == recorded


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_grammatical_corpus_items_have_positive_ell_and_choices(grammar):
    """A5: ℓ(x) > 0 and n_choices > 0 on every grammatical corpus item."""
    one_item, lex = _parts(grammar)
    rec = DerivationRecorder(3)
    for _ in range(30):
        rec.reset_trace()
        one_item(rec, lex)
        result = score_derivation(rec.derivation(), lambda r: one_item(r, lex))
        assert result.neg_logp > 0.0
        assert result.n_choices > 0
        assert math.isfinite(result.neg_logp)


def test_replay_consumes_the_whole_script():
    one_item, lex = _parts("H")
    rec = DerivationRecorder(11)
    rec.reset_trace()
    one_item(rec, lex)
    d = rec.derivation()
    rng = ScoringRandom(d)
    one_item(rng, lex)
    assert rng.exhausted
    assert rng.remaining == 0


# ─────────────────────────────────────────────────────────────────────────────
# Recorder invariants
# ─────────────────────────────────────────────────────────────────────────────

def test_recorder_is_stream_identical_to_plain_random():
    """The recorder must not perturb what is generated (gate A2.0(a))."""
    import random

    from grammars.grammar_H.generate import _one_item
    from grammars.grammar_H.lexicon import load

    lex = load()
    # ONE stream for both sides: reset_trace clears the record, not the RNG, so
    # a fresh random.Random per item would compare against a restarted stream.
    plain_rng = random.Random(5)
    plain = [_one_item(plain_rng, lex)[0] for _ in range(20)]
    rec = DerivationRecorder(5)
    traced = []
    for _ in range(20):
        rec.reset_trace()
        traced.append(_one_item(rec, lex)[0])
    assert traced == plain


def test_rewind_truncates_choices_with_the_trace():
    """Regression: the desync that fabricated A2.0(b) collisions.

    ``TracedRandom.rewind`` drops draws; if the choice script does not drop them
    too, every later index refers to the wrong site.
    """
    rec = DerivationRecorder(1)
    rec.bernoulli(0.5, "a")
    mark = rec.mark()
    rec.bernoulli(0.5, "b")
    rec.choice([1, 2, 3], "c")
    assert len(rec.derivation()) == 3
    rec.rewind(mark)
    d = rec.derivation()
    assert len(d) == 1 and d.labels == ("a",)
    assert len(rec.trace.draws) == 1


def test_derivation_desync_fails_loudly():
    rec = DerivationRecorder(1)
    rec.bernoulli(0.5, "a")
    rec._choices.append(Choice("ghost", "bernoulli", True))   # simulate a desync
    with pytest.raises(AssertionError, match="desync"):
        rec.derivation()


def test_choice_is_recorded_as_an_index_not_a_value():
    """Objects are rebuilt on replay, so only an index can recover the slot."""
    rec = DerivationRecorder(2)
    population = [object() for _ in range(5)]
    drawn = rec.choice(population, "obj")
    (entry,) = rec.derivation().choices
    assert entry.kind == "choice"
    assert isinstance(entry.value, int)
    assert population[entry.value] is drawn


def test_reset_trace_clears_the_script():
    rec = DerivationRecorder(1)
    rec.bernoulli(0.5, "a")
    rec.reset_trace()
    assert len(rec.derivation()) == 0


# ─────────────────────────────────────────────────────────────────────────────
# Structural divergence — the derivation is not of this generator
# ─────────────────────────────────────────────────────────────────────────────

def test_script_exhausted_is_structural_divergence():
    d = Derivation((Choice("a", "bernoulli", True),))

    def replay(rng):
        rng.bernoulli(0.5, "a")
        rng.bernoulli(0.5, "b")        # one draw more than the script has

    with pytest.raises(StructuralDivergence, match="exhausted"):
        score_derivation(d, replay)


def test_leftover_script_is_structural_divergence():
    d = Derivation((Choice("a", "bernoulli", True),
                    Choice("b", "bernoulli", True)))

    with pytest.raises(StructuralDivergence, match="left over"):
        score_derivation(d, lambda rng: rng.bernoulli(0.5, "a"))


def test_leftover_is_tolerated_when_exhaustion_not_required():
    d = Derivation((Choice("a", "bernoulli", True),
                    Choice("b", "bernoulli", True)))
    result = score_derivation(d, lambda rng: rng.bernoulli(0.5, "a"),
                              require_exhausted=False)
    assert result.n_choices == 1


def test_label_mismatch_is_structural_divergence():
    d = Derivation((Choice("elsewhere", "bernoulli", True),))
    with pytest.raises(StructuralDivergence, match="divergence at draw 0"):
        score_derivation(d, lambda rng: rng.bernoulli(0.5, "here"))


def test_kind_mismatch_is_structural_divergence():
    d = Derivation((Choice("a", "categorical", "x"),))
    with pytest.raises(StructuralDivergence, match="divergence at draw 0"):
        score_derivation(d, lambda rng: rng.bernoulli(0.5, "a"))


def test_index_outside_reconstructed_population_is_structural_divergence():
    """A shorter candidate list means the code paths diverged, not p=0."""
    d = Derivation((Choice("site", "choice", 7),))
    with pytest.raises(StructuralDivergence, match="outside the reconstructed"):
        score_derivation(d, lambda rng: rng.choice([1, 2, 3], "site"))


def test_untraced_primitive_is_structural_divergence():
    d = Derivation((Choice("a", "bernoulli", True),))
    with pytest.raises(StructuralDivergence, match="not a traced primitive"):
        score_derivation(d, lambda rng: rng.random())


def test_bool_is_not_accepted_as_a_choice_index():
    """``True == 1`` in Python; a bernoulli value must not silently index."""
    d = Derivation((Choice("site", "choice", True),))
    with pytest.raises(StructuralDivergence, match="expected an index"):
        score_derivation(d, lambda rng: rng.choice([1, 2, 3], "site"))


# ─────────────────────────────────────────────────────────────────────────────
# Out of support — well-formed derivation, zero probability under π_corpus
# ─────────────────────────────────────────────────────────────────────────────

def test_categorical_outcome_not_offered_is_out_of_support():
    d = Derivation((Choice("phenomenon", "categorical", "wh_movement"),))

    def replay(rng):
        rng.categorical({"neutral": 1.0}, "phenomenon")

    with pytest.raises(OutOfSupport, match="not an outcome"):
        score_derivation(d, replay)


def test_zero_probability_categorical_is_out_of_support():
    d = Derivation((Choice("phenomenon", "categorical", "wh_movement"),))

    def replay(rng):
        rng.categorical({"neutral": 1.0, "wh_movement": 0.0}, "phenomenon")

    with pytest.raises(OutOfSupport, match="undefined here, not infinite"):
        score_derivation(d, replay)


def test_impossible_bernoulli_outcome_is_out_of_support():
    """A forced site whose corpus probability is exactly 0."""
    d = Derivation((Choice("aux_site", "bernoulli", True),))
    with pytest.raises(OutOfSupport):
        score_derivation(d, lambda rng: rng.bernoulli(0.0, "aux_site"))

    d_false = Derivation((Choice("aux_site", "bernoulli", False),))
    with pytest.raises(OutOfSupport):
        score_derivation(d_false, lambda rng: rng.bernoulli(1.0, "aux_site"))


def test_out_of_support_never_returns_infinity():
    """The whole point of raising: an ∞ would dominate any downstream mean."""
    d = Derivation((Choice("a", "bernoulli", True),))
    try:
        score_derivation(d, lambda rng: rng.bernoulli(0.0, "a"))
    except OutOfSupport:
        pass
    else:
        pytest.fail("expected OutOfSupport")


# ─────────────────────────────────────────────────────────────────────────────
# Probability semantics
# ─────────────────────────────────────────────────────────────────────────────

def test_choice_marginalizes_duplicate_slots():
    """Two slots holding the same form are one surface outcome."""
    d = Derivation((Choice("c", "choice", 0),))
    result = score_derivation(d, lambda rng: rng.choice(["a", "a", "b", "c"], "c"))
    assert result.neg_logp == pytest.approx(-math.log(0.5))


def test_choices_uses_weights():
    d = Derivation((Choice("lex", "choices", [1]),))
    result = score_derivation(
        d, lambda rng: rng.choices(["x", "y"], weights=[3.0, 1.0], k=1,
                                   label="lex"))
    assert result.neg_logp == pytest.approx(-math.log(0.25))


def test_choices_accepts_cum_weights():
    d = Derivation((Choice("lex", "choices", [1]),))
    result = score_derivation(
        d, lambda rng: rng.choices(["x", "y"], cum_weights=[3.0, 4.0], k=1,
                                   label="lex"))
    assert result.neg_logp == pytest.approx(-math.log(0.25))


def test_choices_k_mismatch_is_structural_divergence():
    d = Derivation((Choice("lex", "choices", [0]),))
    with pytest.raises(StructuralDivergence, match="k=2"):
        score_derivation(d, lambda rng: rng.choices(["x", "y"], k=2, label="lex"))


def test_bernoulli_scores_the_complement_for_false():
    d = Derivation((Choice("a", "bernoulli", False),))
    result = score_derivation(d, lambda rng: rng.bernoulli(0.25, "a"))
    assert result.neg_logp == pytest.approx(-math.log(0.75))


def test_ell_is_the_sum_over_sites():
    d = Derivation((Choice("a", "bernoulli", True),
                    Choice("b", "bernoulli", True)))

    def replay(rng):
        rng.bernoulli(0.5, "a")
        rng.bernoulli(0.25, "b")

    result = score_derivation(d, replay)
    assert result.neg_logp == pytest.approx(-math.log(0.5) - math.log(0.25))
    assert result.n_choices == 2


# ─────────────────────────────────────────────────────────────────────────────
# Retry loops
# ─────────────────────────────────────────────────────────────────────────────

def test_scoring_rewind_resets_the_script_cursor():
    """A rewound retry must re-read the script from the same point."""
    d = Derivation((Choice("keep", "bernoulli", True),
                    Choice("retry", "bernoulli", False)))

    def replay(rng):
        rng.bernoulli(0.5, "keep")
        mark = rng.mark()
        rng.bernoulli(0.5, "retry")
        rng.rewind(mark)              # discard the attempt
        rng.bernoulli(0.5, "retry")   # and take it again

    result = score_derivation(d, replay)
    assert result.n_choices == 2


def test_duplicate_screen_retry_does_not_inflate_ell():
    """The screen's discarded attempts must not appear in the recorded script."""
    from grammars.grammar_H.generate import _one_item
    from grammars.grammar_H.lexicon import load

    lex = load()
    rec = DerivationRecorder(19)
    for _ in range(25):
        rec.reset_trace()
        _one_item(rec, lex)
        d = rec.derivation()
        # Every scripted choice is one logged draw of the SURVIVING trajectory.
        assert len(d) == rec.trace.n_draws


# ─────────────────────────────────────────────────────────────────────────────
# Serialization
# ─────────────────────────────────────────────────────────────────────────────

def test_derivation_json_round_trip():
    one_item, lex = _parts("P")
    rec = DerivationRecorder(23)
    rec.reset_trace()
    one_item(rec, lex)
    d = rec.derivation()
    revived = Derivation.from_json(d.to_json())
    assert revived.labels == d.labels
    assert [c.value for c in revived.choices] == [c.value for c in d.choices]
    assert (score_derivation(revived, lambda r: one_item(r, lex)).neg_logp
            == rec.trace.neg_logp)
