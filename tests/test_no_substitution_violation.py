"""The no-substitution binding violation: a licensed configuration left bare.

The claim under test is that the ungrammatical member is a string the generator
can never emit. That is checkable against the generator's own screens, which are
the operational definition of the invariant:

* H/H′ — `has_nucleus_duplicate`: same CAT1 stem as subject and object of an
  EMBEDDED nucleus. Surface-level, so it applies to both members; the
  substituted member has a CAT1PRON object and does not trip it.
* P/L′ — `has_licensed_duplicate`: applied by `grammar_P.generate` to the
  **base**, before the transform, to reject chance duplicates. On a *substituted*
  string it returns True by construction, because `binding.cats_and_forms`
  reverts CAT1PRON to CAT1 carrying `coref_form` to reconstruct the duplicate the
  transform consumed. So it is only a grammaticality oracle for UNsubstituted
  strings — the same restriction `test_reformalized_rules` applies when it checks
  the screen over corpus items.

Refl only: H/H′ Pron substitution is optional at the grammar level
(TRANSFORM_RULES §3.1), so a bare cross-nucleus duplicate is grammatical there.
`BINDING_MODES` withholds the mode from pron in both families to keep the H-vs-P
comparison on identical item sets.
"""
from __future__ import annotations

import random

import pytest

from corpusgen.structural_probes import (
    BINDING_MODES,
    _draw_base,
    _make_binding_pair,
    _registry,
    _surface,
    build_structural_probe,
)
from grammars import metadata as M
from grammars.grammar_H.transforms import has_nucleus_duplicate
from grammars.grammar_P.transforms import has_licensed_duplicate

GRAMMARS = ("H", "P", "Hprime", "Lprime")
N_PAIRS = 40


def _pairs(grammar: str, n: int = N_PAIRS, seed: int = 4242):
    family, load_lex = _registry()[grammar]
    lex = load_lex()
    rng = random.Random(seed)
    out = []
    for _ in range(6000):
        if len(out) >= n:
            break
        pair = _make_binding_pair(_draw_base(family, rng, lex), family,
                                  "Refl", "no_substitution", rng, lex)
        if pair is not None:
            out.append(pair)
    assert len(out) == n, f"{grammar}: only built {len(out)}/{n} pairs"
    return family, out


# --- the invariant ---------------------------------------------------------

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_ungrammatical_member_trips_the_generation_screen(grammar):
    """The bare licensed configuration is exactly what generation resamples away.

    If this fails the violation is not a violation: the string would be one the
    generator could have emitted, and the model would be right to accept it.
    """
    family, pairs = _pairs(grammar)
    screen = has_nucleus_duplicate if family == M.FAMILY_TREE else has_licensed_duplicate
    tripped = sum(bool(screen(u)) for _, u, _ in pairs)
    assert tripped == len(pairs), (
        f"{grammar}: only {tripped}/{len(pairs)} ungrammatical members carry a "
        f"licensed unsubstituted duplicate"
    )


@pytest.mark.parametrize("grammar", ("H", "Hprime"))
def test_grammatical_member_does_not_trip_the_screen(grammar):
    """Only assertable for the tree family — see the module docstring for why
    `has_licensed_duplicate` is not a surface oracle on substituted P strings."""
    _, pairs = _pairs(grammar)
    tripped = sum(bool(has_nucleus_duplicate(g)) for g, _, _ in pairs)
    assert tripped == 0, f"{grammar}: {tripped} grammatical members trip the screen"


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_members_differ_only_in_whether_the_pron_was_written(grammar):
    """Both members are coindexed at the SAME licensed site; the grammatical one
    is pronominalized and the ungrammatical one is not."""
    family, pairs = _pairs(grammar)
    for g, u, violation in pairs:
        assert violation == "no_substitution"
        g_words = _surface(g, family).split()
        u_words = _surface(u, family).split()
        assert len(g_words) == len(u_words), "members must stay index aligned"
        differing = [i for i, (a, b) in enumerate(zip(g_words, u_words)) if a != b]
        assert len(differing) == 1, (
            f"{grammar}: expected exactly one differing word, got {differing}"
        )
        i = differing[0]
        assert g_words[i].split("#")[0] == "mm", (
            f"grammatical member should carry the reflexive at the site, "
            f"got {g_words[i]!r}"
        )
        assert u_words[i].split("#")[0] != "mm"


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_ungrammatical_member_repeats_its_antecedent(grammar):
    """The bare site must be a genuine coreferential duplicate — the same stem as
    some earlier CAT1 — not just any full nominal."""
    family, pairs = _pairs(grammar)
    for g, u, _ in pairs:
        g_words = _surface(g, family).split()
        u_words = _surface(u, family).split()
        i = next(k for k, (a, b) in enumerate(zip(g_words, u_words)) if a != b)
        stem = u_words[i].split("#")[0]
        earlier = {w.split("#")[0] for w in u_words[:i]}
        assert stem in earlier, (
            f"{grammar}: bare site {stem!r} does not repeat an earlier stem"
        )


# --- allocation ------------------------------------------------------------

def test_no_substitution_is_refl_only():
    """Pron must refuse the mode: H/H′ pron substitution is optional, so the
    'violation' would be a grammatical string there."""
    family, load_lex = _registry()["P"]
    lex = load_lex()
    rng = random.Random(0)
    base = _draw_base(family, rng, lex)
    with pytest.raises(ValueError, match="Refl only"):
        _make_binding_pair(base, family, "Pron", "no_substitution", rng, lex)
    assert "no_substitution" not in BINDING_MODES["pron"]
    assert "no_substitution" in BINDING_MODES["refl"]


@pytest.mark.parametrize("n_per_subcase", (2, 3, 30, 31, 100, 101))
@pytest.mark.parametrize("subcase", tuple(BINDING_MODES))
def test_mode_allocation_is_balanced(subcase, n_per_subcase):
    """The block allocation itself — pure arithmetic, so tested directly rather
    than by building thousands of probe items."""
    import collections

    modes = BINDING_MODES[subcase]
    counts = collections.Counter({m: 0 for m in modes})
    counts.update(
        modes[(made * len(modes)) // n_per_subcase] for made in range(n_per_subcase)
    )
    assert sum(counts.values()) == n_per_subcase
    # Balanced to within one, counting modes that got nothing.
    assert max(counts.values()) - min(counts.values()) <= 1, dict(counts)
    if n_per_subcase >= len(modes):
        assert all(counts[m] > 0 for m in modes), f"{subcase}: {dict(counts)}"


def test_both_members_carry_a_critical_span():
    """Both members must be stamped with the SAME critical span.

    The sidecar's target-finder locates the construction's CAT1PRON, and the
    unsubstituted member has none — so without the `_divergence_targets`
    override it comes back with `targets: []`, `neg_logq_crit` is NaN, and the
    whole `refl:no_substitution` term disappears from `S[r]` without an error.
    That is what the first build of this violation actually did.
    """
    from models.tokenize import tokenizer

    items = build_structural_probe("H", seed=999, min_length=2, max_length=25,
                                   n_per_subcase=30)
    pairs = {}
    for item in items:
        pairs.setdefault(item["pair_id"], {})[item["variant"]] = item

    checked = 0
    for members in pairs.values():
        u = members["ungrammatical"]
        if u["violation"] != "no_substitution":
            continue
        checked += 1
        g = members["grammatical"]
        g_targets = g["_meta"]["targets"]
        u_targets = u["_meta"]["targets"]
        assert g_targets and u_targets, "both members need a target"
        g_span = g_targets[0]["model_token_span"]
        u_span = u_targets[0]["model_token_span"]
        assert g_span == u_span, f"spans must align: {g_span} vs {u_span}"

        # and the span must sit on the token that actually differs
        g_ids = tokenizer(g["sentence"])["input_ids"]
        u_ids = tokenizer(u["sentence"])["input_ids"]
        assert len(g_ids) == len(u_ids)
        divergence = [i for i, (a, b) in enumerate(zip(g_ids, u_ids)) if a != b]
        assert divergence == list(range(g_span[0], g_span[1])), (
            f"span {g_span} does not cover the divergence {divergence}"
        )
    assert checked == 10, f"expected 10 no_substitution pairs, saw {checked}"


def test_built_probe_carries_every_mode():
    """One end-to-end build confirming the allocation reaches the written items."""
    import collections

    items = build_structural_probe("H", seed=999, min_length=2, max_length=25,
                                   n_per_subcase=30)
    counts = collections.Counter(
        (i["subcase"], i["violation"]) for i in items
        if i["variant"] == "ungrammatical" and i["subcase"] in BINDING_MODES
    )
    assert counts[("refl", "no_substitution")] == 10
    assert counts[("refl", "wrong_item")] == 10
    assert counts[("refl", "wrong_site")] == 10
    assert counts[("pron", "wrong_item")] == 15
    assert counts[("pron", "wrong_site")] == 15
    assert ("pron", "no_substitution") not in counts
