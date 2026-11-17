"""Tests for scoring structural-probe items under π_corpus (A2.2).

Two things are load-bearing here and neither is obvious from the code alone:

1. Making the probe builder traceable must not move the RNG stream. If it did,
   every probe corpus ever built from a seed would change, which is a
   regeneration decision and not a side effect of instrumentation.
2. A returned ℓ(x) must belong to the item it is attached to. The replay is
   therefore required to reproduce the probe item's own surface; consuming the
   script is not enough, because a script can replay cleanly into a different
   sentence.
"""
from __future__ import annotations

import random

import pytest

from corpusgen.structural_probes import build_structural_probe
from grammars.probe_scoring import (
    MAPPED_SUBCASES, REASON_FORCED_STRUCTURE, REASON_SURFACE_MISMATCH,
    REASON_TRANSFORM_CHOICE, REASON_UNGRAMMATICAL, SUBCASE_CONSTRUCTION,
    ProbeScore, corpus_script, coverage, score_probe_member,
    score_probe_records,
)
from grammars.scoring import DerivationRecorder

GRAMMARS = ("H", "P", "Hprime", "Lprime")

# Minimum share of a mapped subcase's records that must carry a score. Set from
# the measured worst cell after C7.1 — P aux 26/30 — with one record of slack.
# See test_mapped_subcases_score_and_reproduce_the_surface.
COVERAGE_FLOOR_NUM, COVERAGE_FLOOR_DEN = 25, 30


def _one_item_for(grammar):
    if grammar in ("H", "Hprime"):
        from grammars.grammar_H.generate import _one_item
        if grammar == "H":
            return _one_item
        from grammars.grammar_H_prime.transforms import apply as apply_fn
    else:
        from grammars.grammar_P.generate import _one_item
        if grammar == "P":
            return _one_item
        from grammars.grammar_L_prime.transforms import apply as apply_fn
    return lambda rng, lex: _one_item(rng, lex, apply_fn=apply_fn)


def _lex_for(grammar):
    if grammar in ("H", "Hprime"):
        from grammars.grammar_H.lexicon import load
    else:
        from grammars.grammar_P.lexicon import load
    return load()


def _probe(grammar, n=8, seed=42):
    """Probe items plus captured derivations, built with a recording rng."""
    derivations = []
    items = build_structural_probe(
        grammar, seed=seed, min_length=2, max_length=25, n_per_subcase=n,
        rng=DerivationRecorder(seed), derivations_out=derivations)
    return items, derivations


# ─────────────────────────────────────────────────────────────────────────────
# Stream identity — instrumentation must not change the probe
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_recording_rng_builds_the_same_probe(grammar):
    kw = dict(seed=42, min_length=2, max_length=25, n_per_subcase=6)
    plain = build_structural_probe(grammar, **kw)
    traced = build_structural_probe(grammar, rng=DerivationRecorder(42), **kw)
    assert [i["sentence"] for i in traced] == [i["sentence"] for i in plain]
    assert [i["item_uid"] for i in traced] == [i["item_uid"] for i in plain]


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_default_rng_matches_an_explicit_seeded_one(grammar):
    kw = dict(seed=7, min_length=2, max_length=25, n_per_subcase=4)
    assert ([i["sentence"] for i in build_structural_probe(grammar, **kw)]
            == [i["sentence"] for i in
                build_structural_probe(grammar, rng=random.Random(7), **kw)])


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_capture_does_not_change_the_items(grammar):
    kw = dict(seed=42, min_length=2, max_length=25, n_per_subcase=4)
    captured = []
    a = build_structural_probe(grammar, rng=DerivationRecorder(42),
                               derivations_out=captured, **kw)
    b = build_structural_probe(grammar, rng=DerivationRecorder(42), **kw)
    assert [i["sentence"] for i in a] == [i["sentence"] for i in b]
    assert len(captured) * 2 == len(a)


# ─────────────────────────────────────────────────────────────────────────────
# Derivation capture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_every_accepted_pair_has_a_base_derivation(grammar):
    _, derivations = _probe(grammar, n=4)
    assert derivations
    for record in derivations:
        assert len(record["base_derivation"]) > 0
        assert record["subcase"] in SUBCASE_CONSTRUCTION


def test_rejected_attempts_do_not_leak_into_the_record():
    """`reset_trace` per attempt + the screen rewind keep ℓ(x) to one attempt.

    A base derivation carrying several build attempts would show up as a script
    the corpus path cannot consume; that it replays exactly (below) is the
    positive evidence. Here we check the record is not absurdly long.
    """
    _, derivations = _probe("H", n=6)
    lengths = [len(r["base_derivation"]) for r in derivations]
    assert max(lengths) < 400


# ─────────────────────────────────────────────────────────────────────────────
# Scoring — the mapped subcases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_mapped_subcases_score_and_reproduce_the_surface(grammar):
    """Every mapped subcase scores, with ONE documented exception.

    ``corpus_script`` asserts ``aux_site = matrix``. In the token family P's
    rule fronts the FIRST CAT3AUX in left-to-right scan, so whenever a probe
    base's first aux is the EMBEDDED one, the corpus derivation runs through
    ``_force_embedded_aux`` — a block of draws this splice does not supply —
    and the replay yields a different sentence. That is caught as
    ``surface_mismatch``, which is the §"Verification contract" guard working.

    This test used to assert 100% coverage everywhere, which held only because
    it samples 8 records per subcase. At n=30 the case is visible at IOD both
    before and after C7.1 — P aux 29/30 → 26/30, L′ aux 29/30 → 27/30 when the
    embedded-aux draw went from 0.30 to ``P_AUX_EMBEDDED = 0.60``; H and H′ stay
    30/30 (H's rule reads the matrix Type2, so the matrix-site script always
    holds). The density is NOT adjusted to restore coverage — probe coverage is
    not the objective, and the mapping's limit is documented rather than hidden.

    The assertion is therefore narrowed, not dropped: a fallout is tolerated
    only for ``aux`` in the token family and only with that exact reason, and
    every mapped subcase must clear a **25/30 coverage floor** — the measured
    worst cell (P aux 26/30) with one record of slack. At this test's n=8 that
    admits exactly one fallout per subcase, which is what P and L′ actually
    produce (7/8), so the floor has no headroom left: any further loss fails. A
    wrong mapping — a different reason, a tree-family fallout, or coverage
    sliding — still fails.
    """
    _items, derivations = _probe(grammar, n=8)
    one_item, lex = _one_item_for(grammar), _lex_for(grammar)
    token_family = grammar in ("P", "Lprime")
    seen, scored = set(), {}
    for record in derivations:
        subcase = record["subcase"]
        if subcase not in MAPPED_SUBCASES:
            continue
        seen.add(subcase)
        tally = scored.setdefault(subcase, [0, 0])
        tally[1] += 1
        score = score_probe_member(record, "grammatical", one_item, lex)
        if score.defined:
            tally[0] += 1
            assert score.neg_logp > 0.0
            assert score.n_choices > 0
            continue
        assert subcase == "aux" and token_family, (
            subcase, score.reason, score.detail)
        assert score.reason == REASON_SURFACE_MISMATCH, (
            subcase, score.reason, score.detail)
    assert seen == set(MAPPED_SUBCASES)
    for subcase, (ok, total) in scored.items():
        # Floor: 25/30 of records must score. Expressed as a ratio rather than a
        # count because this test samples 8 per subcase while the figure was
        # measured at 30 — integer arithmetic, no float rounding at the edge.
        assert ok * COVERAGE_FLOOR_DEN >= COVERAGE_FLOOR_NUM * total, (
            subcase, ok, total)


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_ungrammatical_members_are_null_with_that_reason(grammar):
    _, derivations = _probe(grammar, n=4)
    one_item, lex = _one_item_for(grammar), _lex_for(grammar)
    for record in derivations:
        score = score_probe_member(record, "ungrammatical", one_item, lex)
        assert not score.defined
        assert score.reason == REASON_UNGRAMMATICAL
        assert score.neg_logp is None
        assert score.n_choices is None


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_unmapped_subcases_report_their_reason_not_a_number(grammar):
    _, derivations = _probe(grammar, n=6)
    one_item, lex = _one_item_for(grammar), _lex_for(grammar)
    for record in derivations:
        if record["subcase"] in MAPPED_SUBCASES:
            continue
        score = score_probe_member(record, "grammatical", one_item, lex)
        assert score.neg_logp is None
        expected = (REASON_TRANSFORM_CHOICE if record["subcase"] == "pron"
                    else REASON_FORCED_STRUCTURE)
        assert score.reason == expected


def test_a_score_is_never_infinite():
    for grammar in GRAMMARS:
        _, derivations = _probe(grammar, n=4)
        one_item, lex = _one_item_for(grammar), _lex_for(grammar)
        for score in score_probe_records(derivations, one_item, lex):
            assert score.neg_logp is None or score.neg_logp == score.neg_logp
            assert score.neg_logp is None or score.neg_logp != float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Script construction
# ─────────────────────────────────────────────────────────────────────────────

def test_corpus_script_prepends_the_phenomenon():
    _, derivations = _probe("H", n=4)
    record = next(r for r in derivations if r["subcase"] == "neutral_agreement")
    script = corpus_script(record["base_derivation"], "neutral_agreement")
    assert script.choices[0].label == "phenomenon"
    assert script.choices[0].value == "neutral"


def test_aux_script_drops_the_insitu_aux_factor():
    """Under a matrix aux site the corpus never draws insitu_aux."""
    _, derivations = _probe("H", n=6)
    record = next(r for r in derivations if r["subcase"] == "aux")
    script = corpus_script(record["base_derivation"], "aux")
    labels = [c.label for c in script.choices]
    assert labels[0] == "phenomenon"
    assert labels[1] == "aux_site"
    assert "insitu_aux" not in labels


def test_neutral_script_keeps_and_moves_the_insitu_aux_factor():
    _, derivations = _probe("H", n=6)
    record = next(r for r in derivations
                  if r["subcase"] == "neutral_positional")
    base = record["base_derivation"]
    script = corpus_script(base, "neutral_positional")
    base_labels = [c.label for c in base.choices]
    if "insitu_aux" in base_labels:
        labels = [c.label for c in script.choices]
        assert labels[1] == "insitu_aux"
        assert labels.count("insitu_aux") == 1


def test_binding_script_carries_the_subtype():
    _, derivations = _probe("H", n=6)
    record = next(r for r in derivations if r["subcase"] == "refl")
    script = corpus_script(record["base_derivation"], "refl")
    subtype = next(c for c in script.choices if c.label == "bind_subtype")
    assert subtype.value is True
    pron = corpus_script(record["base_derivation"], "pron")
    assert next(c for c in pron.choices
                if c.label == "bind_subtype").value is False


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def test_coverage_tallies_every_member():
    _, derivations = _probe("P", n=6)
    one_item, lex = _one_item_for("P"), _lex_for("P")
    scores = score_probe_records(derivations, one_item, lex)
    tally = coverage(scores)
    assert sum(tally.values()) == len(scores) == 2 * len(derivations)
    assert tally["defined"] > 0
    assert tally[REASON_UNGRAMMATICAL] == len(derivations)


def test_item_uid_matches_the_probe_records():
    items, derivations = _probe("H", n=4)
    one_item, lex = _one_item_for("H"), _lex_for("H")
    scores = score_probe_records(derivations, one_item, lex)
    assert ([s.item_uid for s in scores]
            == [i["item_uid"] for i in items])


def test_probe_score_defined_flag_tracks_reason():
    assert ProbeScore("p", "grammatical", "aux", 1.0, 2).defined
    assert not ProbeScore("p", "grammatical", "aux", reason="x").defined
