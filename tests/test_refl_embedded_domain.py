"""Pins for the reformalized H-family reflexive rule (A′, 2026-07-29).

Rule under test (TRANSFORM_RULES.md §3.1):

    A CAT1PRON-Refl must be bound within its nucleus, AND that nucleus must be
    EMBEDDED — its Type0 dominated by at least one Type4, at any depth.

Written against the RATIONALE, not against the implementation: each test names
the structural configuration it pins, so a passing suite means the code decides
"is this nucleus embedded?", not merely "is this a direct object" (the defect
A′ replaces — the old rule licensed exactly the set of direct objects, leaving
its rulewise separability cell with zero negatives).

Consequences also pinned here (decisions confirmed 2026-07-29):
  * Pron is UNCHANGED, so a MATRIX subject-object coreferential pair is now
    licensed by NEITHER sub-rule — a blocked configuration (§3.4 rewritten).
  * ``has_nucleus_duplicate`` narrows to embedded nuclei, mirroring Grammar P's
    ``has_licensed_duplicate`` (which screens only LICENSED offsets and lets
    unlicensed duplicates surface as blocking evidence).
  * Refl conditioning is REJECTION SAMPLING on A′'s predicate (§1c refactor,
    2026-08-05): draw naturally, keep if the base has an embedded nucleus with
    both a subject and an object, both full Type1. Nothing is built. The former
    plan forced matrix transitivity, the Type4's host, its attachment level and
    its depth — four properties A′ names nowhere — so three tests here now pin
    the ABSENCE of those forcings rather than their presence.
  * Consequently the object-host share falls to ~0.24 marginally, and §3.1
    accepts it: the count-channel cancellation is PARTIAL and the residual is
    handled in probe design and analysis, never by distorting generation. The
    surviving host invariant is conditional on a matrix object existing.
  * The AUX forcing path is independent, but its host is now drawn too (§4).
"""

import collections
import itertools
import random

import pytest

from grammars.grammar_H.nodes import Node
from grammars.grammar_H.transforms import (
    _type0s, _nucleus, has_nucleus_duplicate,
)
from grammars.licensing import binding_hfamily_licensed_ids
from grammars.grammar_H.generate import generate_with_metadata as H_meta
from grammars.grammar_P.generate import generate_with_metadata as P_meta

N = 3000
SEED = 7


# ── tree builders (distinct node_ids: licensing returns node_ids) ────────────

_ids = itertools.count(1)


def _term(label, form="x", role="head", subclass=None, feats=None):
    lex = {"form": form}
    if subclass:
        lex["subclass"] = subclass
    return Node(label=label, head_cat=label, lex=lex, feats=feats or {},
                children=[], role=role, licensor_id=None, node_id=next(_ids))


def _t1(role, form="x", extra=None):
    return Node(label="Type1", head_cat="CAT1", lex=None, feats={},
                children=[_term("CAT1", form)] + (extra or []),
                role=role, licensor_id=None, node_id=next(_ids))


def _t0(subj, vp_children, root=True):
    t2 = Node(label="Type2", head_cat="CAT3", lex=None, feats={},
              children=vp_children, role="vp", licensor_id=None,
              node_id=next(_ids))
    return Node(label="Type0", head_cat="CAT3", lex=None, feats={},
                children=[subj, t2], role="root" if root else "emb",
                licensor_id=None, node_id=next(_ids))


def _t3(attachment, comp):
    return Node(label="Type3", head_cat="CAT6", lex=None,
                feats={"attachment": attachment},
                children=[_term("CAT6"), comp], role=f"{attachment}_adjunct",
                licensor_id=None, node_id=next(_ids))


def _t4(inner):
    return Node(label="Type4", head_cat="CAT9", lex=None, feats={},
                children=[_term("CAT9"), inner], role="rel_clause",
                licensor_id=None, node_id=next(_ids))


def _refl_sites(tree):
    return binding_hfamily_licensed_ids(tree, "Refl")


# ── the domain condition ─────────────────────────────────────────────────────

def test_matrix_nucleus_object_is_not_licensed():
    """A bare matrix transitive clause has NO reflexive site.

    This is the whole point of A′: under the old rule this object licensed.
    """
    obj = _t1("object", "b")
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), obj])
    assert _refl_sites(tree) == set()


def test_embedded_nucleus_object_is_licensed():
    """The object of a Type4-embedded nucleus licenses."""
    eobj = _t1("object", "e")
    inner = _t0(_t1("subject", "d"), [_term("CAT3"), eobj], root=False)
    subj = _t1("subject", "a", extra=[_t4(inner)])
    tree = _t0(subj, [_term("CAT3")])
    assert _refl_sites(tree) == {eobj.node_id}


def test_matrix_object_still_unlicensed_when_the_sentence_has_embedding():
    """THE discriminating case.

    A sentence that CONTAINS an embedded clause does not thereby license its
    MATRIX object. The condition is on the nucleus's own clause, not on the
    sentence. A rule that merely asked "does a CAT9 appear?" would fail here.
    """
    eobj = _t1("object", "e")
    inner = _t0(_t1("subject", "d"), [_term("CAT3"), eobj], root=False)
    subj = _t1("subject", "a", extra=[_t4(inner)])
    mobj = _t1("object", "b")
    tree = _t0(subj, [_term("CAT3"), mobj])
    assert _refl_sites(tree) == {eobj.node_id}
    assert mobj.node_id not in _refl_sites(tree)


def test_depth_two_nucleus_is_licensed():
    """A′ is depth >= 1, at ANY depth — not depth exactly 1."""
    d2obj = _t1("object", "g")
    d2 = _t0(_t1("subject", "f"), [_term("CAT3"), d2obj], root=False)
    d1subj = _t1("subject", "d", extra=[_t4(d2)])
    d1 = _t0(d1subj, [_term("CAT3")], root=False)
    subj = _t1("subject", "a", extra=[_t4(d1)])
    tree = _t0(subj, [_term("CAT3")])
    assert _refl_sites(tree) == {d2obj.node_id}


def test_every_embedded_nucleus_object_licenses_not_just_the_deepest():
    """Two sibling embedded clauses: both their objects license."""
    o1 = _t1("object", "e")
    i1 = _t0(_t1("subject", "d"), [_term("CAT3"), o1], root=False)
    o2 = _t1("object", "g")
    i2 = _t0(_t1("subject", "f"), [_term("CAT3"), o2], root=False)
    subj = _t1("subject", "a", extra=[_t4(i1)])
    mobj = _t1("object", "b", extra=[_t4(i2)])
    tree = _t0(subj, [_term("CAT3"), mobj])
    assert _refl_sites(tree) == {o1.node_id, o2.node_id}


def test_host_of_the_type4_does_not_matter():
    """Subject-hosted and object-hosted embedding license identically.

    Pins that A′ is a DOMINANCE condition, not an attachment-site condition —
    the reason forced embedding may be host-mixed without changing licensing.
    """
    o_sub = _t1("object", "e")
    i_sub = _t0(_t1("subject", "d"), [_term("CAT3"), o_sub], root=False)
    t_sub = _t0(_t1("subject", "a", extra=[_t4(i_sub)]),
                [_term("CAT3"), _t1("object", "b")])

    o_obj = _t1("object", "e")
    i_obj = _t0(_t1("subject", "d"), [_term("CAT3"), o_obj], root=False)
    t_obj = _t0(_t1("subject", "a"),
                [_term("CAT3"), _t1("object", "b", extra=[_t4(i_obj)])])

    assert _refl_sites(t_sub) == {o_sub.node_id}
    assert _refl_sites(t_obj) == {o_obj.node_id}


# ── nucleus locality survives the change ─────────────────────────────────────

def test_embedded_pp_complement_is_not_licensed():
    """The nucleus is subject+verb+object only (D2): a PP complement inside the
    embedded clause is outside it, so embedding alone does not license it."""
    ecomp = _t1("pp", "g")
    inner = _t0(_t1("subject", "d"),
                [_term("CAT3"), _t3("vp", ecomp)], root=False)
    tree = _t0(_t1("subject", "a", extra=[_t4(inner)]), [_term("CAT3")])
    assert _refl_sites(tree) == set()


def test_embedded_subject_is_not_licensed():
    """Refl substitutes the SECOND occurrence (the object), never the subject."""
    esubj = _t1("subject", "d")
    inner = _t0(esubj, [_term("CAT3")], root=False)
    tree = _t0(_t1("subject", "a", extra=[_t4(inner)]), [_term("CAT3")])
    assert _refl_sites(tree) == set()


def test_intransitive_embedded_clause_has_no_site():
    inner = _t0(_t1("subject", "d"), [_term("CAT3")], root=False)
    tree = _t0(_t1("subject", "a", extra=[_t4(inner)]),
               [_term("CAT3"), _t1("object", "b")])
    assert _refl_sites(tree) == set()


def test_licensed_set_is_a_proper_subset_of_the_objects():
    """The defect A′ fixes, stated as a test.

    The old rule returned EXACTLY the set of direct objects, so its rulewise
    separability cell had zero negatives. A′ must leave at least one object
    unlicensed in a configuration that has both.
    """
    eobj = _t1("object", "e")
    inner = _t0(_t1("subject", "d"), [_term("CAT3"), eobj], root=False)
    mobj = _t1("object", "b")
    tree = _t0(_t1("subject", "a", extra=[_t4(inner)]), [_term("CAT3"), mobj])

    objects = {n.node_id for t0 in _type0s(tree)
               for n in [_nucleus(t0)[1]] if n is not None}
    licensed = _refl_sites(tree)
    assert licensed < objects, "licensed must be a PROPER subset of the objects"


# ── decision 1(a): the matrix same-nucleus pair is licensed by neither ───────

def test_matrix_same_nucleus_pair_is_licensed_by_neither_subrule():
    """Pron is unchanged, so a matrix subject-object coreferential pair is now
    a BLOCKED configuration: not refl (not embedded), not pron (same nucleus).
    """
    obj = _t1("object", "a")          # same stem as the subject: a duplicate
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), obj])
    assert obj.node_id not in binding_hfamily_licensed_ids(tree, "Refl")
    assert obj.node_id not in binding_hfamily_licensed_ids(tree, "Pron")


def test_pron_cross_nucleus_licensing_is_unchanged():
    """A′ touches Refl only; the cross-nucleus Pron relation still licenses.

    The pair must be non-dominating (the rule excludes a nominal and one it
    contains), so the embedded clause hangs off the SUBJECT while the partner
    is the matrix OBJECT.
    """
    esubj = _t1("subject", "e")
    inner = _t0(esubj, [_term("CAT3")], root=False)
    msubj = _t1("subject", "a", extra=[_t4(inner)])
    mobj = _t1("object", "e")         # same stem as the embedded subject
    tree = _t0(msubj, [_term("CAT3"), mobj])
    lic = binding_hfamily_licensed_ids(tree, "Pron")
    assert esubj.node_id in lic and mobj.node_id in lic


# ── decision 2: the duplicate screen narrows to embedded nuclei ──────────────

def test_screen_ignores_a_matrix_same_nucleus_duplicate():
    """A matrix same-stem subject/object pair is no longer a refl intermediate,
    so it must be allowed to surface — Grammar P's blocking-evidence pattern."""
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), _t1("object", "a")])
    assert not has_nucleus_duplicate(tree)


def test_screen_still_flags_an_embedded_same_nucleus_duplicate():
    inner = _t0(_t1("subject", "d"), [_term("CAT3"), _t1("object", "d")],
                root=False)
    tree = _t0(_t1("subject", "a", extra=[_t4(inner)]), [_term("CAT3")])
    assert has_nucleus_duplicate(tree)


# ── corpus level ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def h_items():
    return H_meta(N, seed=SEED)


@pytest.fixture(scope="module")
def p_items():
    return P_meta(N, seed=SEED)


def _all_nodes(node):
    """Every node in the tree, pre-order."""
    yield node
    for child in node.children:
        yield from _all_nodes(child)


def _refl_nucleus_clause(tree):
    """The Type0 whose nucleus object is the realized Refl, or None."""
    for t0 in _type0s(tree):
        _subj, obj, _ = _nucleus(t0)
        if (obj is not None and obj.label == "CAT1PRON"
                and obj.lex.get("subclass") == "Refl"):
            return t0
    return None


def test_corpus_every_reflexive_sits_in_an_embedded_nucleus(h_items):
    seen = 0
    for _s, lab, tree in h_items:
        if lab != "anaphoric_binding_refl":
            continue
        t0 = _refl_nucleus_clause(tree)
        assert t0 is not None, "refl item without a realized reflexive object"
        assert t0 is not tree, "reflexive in the MATRIX nucleus — A′ violated"
        seen += 1
    assert seen > 0, "no refl items generated"


def test_corpus_refl_forced_embedding_is_host_mixed(h_items):
    """Both attachment sites must occur among refl items' embedded clauses.

    Subject-anchoring alone would make a left-side verb count separate licensed
    from unlicensed objects almost perfectly (measured U=0.93, bal-acc 0.998).
    """
    hosts = collections.Counter()
    for _s, lab, tree in h_items:
        if lab != "anaphoric_binding_refl":
            continue
        subj = next(c for c in tree.children if c.role == "subject")
        t2 = next(c for c in tree.children if c.label == "Type2")
        obj = next((c for c in t2.children if c.role == "object"), None)
        if any(c.label == "Type4" for c in subj.children):
            hosts["subject"] += 1
        if obj is not None and any(c.label == "Type4" for c in obj.children):
            hosts["object"] += 1
    assert hosts["subject"] > 0 and hosts["object"] > 0, \
        f"forced embedding is not host-mixed: {dict(hosts)}"


def test_corpus_refl_matrix_transitivity_is_natural(h_items):
    """Matrix transitivity in the refl slice must sit at its NATURAL rate.

    This test is the inverse of the one it replaces, which asserted that EVERY
    refl base carries a matrix object. That was true only because generation
    forced ``build_type0(force_transitive=True)`` for refl — a property rule A′
    names nowhere (§3.1: "The matrix clause is drawn naturally. Matrix
    transitivity is named nowhere in A′"). Forcing it to probability 1 inside the
    slice made it a cue available in place of the rule, undetectable by any
    measurement drawn from the same corpus (§1c).

    So the invariant is that it VARIES, near the ~48.5% neutral base rate.
    Mutation check: reinstating the forcing drives this to 1.0 and fails here.
    """
    rates = []
    for _s, lab, tree in h_items:
        if lab != "anaphoric_binding_refl":
            continue
        t2 = next(c for c in tree.children if c.label == "Type2")
        rates.append(any(c.role == "object" for c in t2.children))
    assert len(rates) > 100, f"too few refl items to judge ({len(rates)})"
    share = sum(rates) / len(rates)
    assert 0.35 <= share <= 0.62, (
        f"matrix-transitive share {share:.2f} in the refl slice; expected the "
        f"natural ~0.485. At ~1.0 the build-time forcing has been reinstated")


def test_corpus_refl_bases_carry_an_unlicensed_nominal(h_items):
    """Every refl base must still contain the rule's negative.

    The forced-transitive plan guaranteed this via the matrix object. Under
    rejection sampling the matrix object is present only at its natural rate, so
    the contrast has to be re-established from the rule itself: A′ licenses the
    OBJECT OF AN EMBEDDED NUCLEUS and nothing else, and every base carries at
    least a matrix subject, which A′ never licenses. §3.1 accepts the loss of the
    object-hosted variant explicitly — "a downstream consequence of the base, not
    a licensing failure" — but the per-sentence contrast itself must survive, or
    the probe's licensed-vs-unlicensed design has nothing to contrast within an
    item.
    """
    for _s, lab, tree in h_items:
        if lab != "anaphoric_binding_refl":
            continue
        licensed = set(binding_hfamily_licensed_ids(tree, "Refl"))
        nominals = [n for n in _all_nodes(tree)
                    if n.label in ("Type1", "CAT1PRON")]
        assert any(n.node_id not in licensed for n in nominals), \
            "refl base has no unlicensed nominal — no within-item contrast"


def test_corpus_refl_host_is_drawn_not_pinned(h_items):
    """Pin the MECHANISM (a Q_EMB_OBJECT draw), not a marginal split.

    The test this replaces required an object-host share of 0.35–0.65 among refl
    items. That marginal was only reachable because the matrix was forced
    transitive, so an object always existed for the draw to choose. With the
    matrix natural the same mechanism yields ~0.24 (0.5 x ~0.485), and §3.1
    accepts it: "the mix is subject-weighted, so the cancellation is PARTIAL and
    a residual count cue is expected ... it is not cancelled by forcing the
    matrix".

    The invariant that survives is conditional: GIVEN a matrix object, the host
    is chosen by a fair Q_EMB_OBJECT draw. That is stable under any matrix
    transitivity rate, and it is what "drawn, never pinned" actually asserts.
    Pinning the host to the subject drives this to 0 and fails here.
    """
    hosts = collections.Counter()
    for _s, lab, tree in h_items:
        if lab != "anaphoric_binding_refl":
            continue
        subj = next(c for c in tree.children if c.role == "subject")
        t2 = next(c for c in tree.children if c.label == "Type2")
        obj = next((c for c in t2.children if c.role == "object"), None)
        if obj is None:
            continue                      # no draw was available to make
        if any(c.label == "Type4" for c in obj.children):
            hosts["object"] += 1
        elif any(c.label == "Type4" for c in subj.children):
            hosts["subject"] += 1
    total = hosts["subject"] + hosts["object"]
    assert total > 50, f"too few host draws to judge ({total})"
    share = hosts["object"] / total
    assert 0.35 <= share <= 0.65, (
        f"object-host share {share:.2f} GIVEN a matrix object; the host is "
        f"supposed to be a fair Q_EMB_OBJECT=0.5 draw, not pinned")


def test_corpus_no_embedded_nucleus_duplicates_survive(h_items):
    for _s, _lab, tree in h_items:
        assert not has_nucleus_duplicate(tree)


def test_corpus_refl_length_parity_between_families(h_items, p_items):
    """Decision 3: P refl items force an embedded clause for LENGTH PARITY.

    The two families' refl slices must not drift apart in length the way they
    would if only H forced embedding.
    """
    def lens(items, fam):
        return [len(s.split()) for s, lab, _ in items
                if lab == "anaphoric_binding_refl"]
    h = lens(h_items, "H")
    p = lens(p_items, "P")
    assert h and p
    mh, mp = sum(h) / len(h), sum(p) / len(p)
    assert abs(mh - mp) / max(mh, mp) < 0.20, \
        f"refl length parity broken: H mean {mh:.1f} vs P mean {mp:.1f}"


def test_corpus_aux_items_are_untouched_by_the_refl_change(h_items):
    """The aux rule is independent: fronted auxes are still the matrix one and
    H still exercises its structural aux blocking."""
    labels = collections.Counter(lab for _s, lab, _ in h_items)
    assert labels["auxiliary_movement_skipped"] > 0
    for _s, lab, tree in h_items:
        if lab == "auxiliary_movement":
            assert tree.children[0].label == "CAT3AUX"
