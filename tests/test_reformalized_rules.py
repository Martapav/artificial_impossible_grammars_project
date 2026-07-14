"""Unit + corpus-level regression tests for the reformalized transform rules.

Reference: grammars/LINEAR_HIERARCHICAL_TRIVIALITY_AUDIT.md (assessment) and
grammars/TRANSFORM_RULES.md. Pins:
  * P wh licensing (third CAT1-position / 1st-after-CAT5 — audit §F),
    ordinal counting of CAT1PRON, Refl never a target, blockability;
  * H wh licensing (matrix object / inside a Type3 dominated by a Type4 —
    audit §F), matrix subject and embedded subjects never licensed,
    matrix-attached Type3s never licensed, blockability;
  * P binding: distance-licensed substitution, feature copy from the
    antecedent, subject-substitution agreement fixup, blockability;
  * H binding: nucleus-local Refl, cross-nucleus Pron, feature copy;
  * corpus-level divergence directions and the duplicate screens.
"""

import collections
import random

import pytest

from grammars.grammar_H.nodes import Node
from grammars.grammar_H.transforms import (
    _collect_wh_candidates as h_wh_candidates,
    _type0s,
    _nucleus,
    has_nucleus_duplicate,
)
from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import load as load_lex_P
from grammars.grammar_P.transforms import (
    _wh_candidates as p_wh_candidates,
    _binding as p_binding,
    has_licensed_duplicate,
)
from grammars.grammar_H.generate import generate_with_metadata as H_meta
from grammars.grammar_P.generate import generate_with_metadata as P_meta

N = 3000
SEED = 7


# ── P wh licensing (unit) ────────────────────────────────────────────────────

def T(cat, form="x", role="", cid=0, subclass=None, feats=None):
    lex = {"form": form}
    if subclass:
        lex["subclass"] = subclass
    return Token(cat, lex, feats=feats or {}, role=role, clause_id=cid)


def test_p_wh_bare_svo_is_blocked():
    toks = [T("CAT1"), T("CAT4"), T("CAT3"), T("CAT1")]
    assert p_wh_candidates(toks) == []


def test_p_wh_cat5_licenses_next_nominal():
    toks = [T("CAT1"), T("CAT4"), T("CAT3"), T("CAT5"), T("CAT1")]
    assert p_wh_candidates(toks) == [4]


def test_p_wh_third_position():
    toks = [T("CAT1"), T("CAT3"), T("CAT6"), T("CAT1"), T("CAT1")]
    assert p_wh_candidates(toks) == [4]


def test_p_wh_cat9_licenses_nothing():
    # Audit §F: the second-after-CAT9 condition is removed. Positions here:
    # 0, 2 (embedded subject), 5 (matrix object). Only the third overall (5)
    # is licensed; CAT9 contributes nothing.
    toks = [T("CAT1"), T("CAT9", cid=1), T("CAT1", cid=1),
            T("CAT3", cid=1), T("CAT3"), T("CAT1")]
    assert p_wh_candidates(toks) == [5]
    # With only two CAT1-positions and no CAT5, embedding alone no longer
    # makes wh applicable.
    toks = [T("CAT1"), T("CAT9", cid=1), T("CAT1", cid=1),
            T("CAT3", cid=1), T("CAT3")]
    assert p_wh_candidates(toks) == []


def test_p_wh_pron_counts_and_refl_never_fronts():
    # CAT1PRON counts as a CAT1-position for ordinals...
    toks = [T("CAT1"), T("CAT1PRON", subclass="Refl"), T("CAT3"), T("CAT1")]
    assert p_wh_candidates(toks) == [3]  # third position, full CAT1
    # ...a Pron at a licensed position is a target, a Refl is not.
    base = [T("CAT1"), T("CAT4"), T("CAT3"), T("CAT5")]
    assert p_wh_candidates(base + [T("CAT1PRON", subclass="Pron")]) == [4]
    assert p_wh_candidates(base + [T("CAT1PRON", subclass="Refl")]) == []


# ── H wh licensing (unit) ────────────────────────────────────────────────────

def _term(label, form="x", role="head", subclass=None, feats=None, nid=0):
    lex = {"form": form}
    if subclass:
        lex["subclass"] = subclass
    return Node(label=label, head_cat=label, lex=lex, feats=feats or {},
                children=[], role=role, licensor_id=None, node_id=nid)


def _t1(role, form="x", extra=None, nid=0):
    return Node(label="Type1", head_cat="CAT1", lex=None, feats={},
                children=[_term("CAT1", form)] + (extra or []),
                role=role, licensor_id=None, node_id=nid)


def _t0(subj, vp_children, root=True):
    t2 = Node(label="Type2", head_cat="CAT3", lex=None, feats={},
              children=vp_children, role="vp", licensor_id=None, node_id=0)
    return Node(label="Type0", head_cat="CAT3", lex=None, feats={},
                children=[subj, t2], role="root" if root else "emb",
                licensor_id=None, node_id=0)


def _t3(attachment, comp):
    return Node(label="Type3", head_cat="CAT6", lex=None,
                feats={"attachment": attachment},
                children=[_term("CAT6"), comp], role=f"{attachment}_adjunct",
                licensor_id=None, node_id=0)


def _t4(inner):
    return Node(label="Type4", head_cat="CAT9", lex=None, feats={},
                children=[_term("CAT9"), inner], role="rel_clause",
                licensor_id=None, node_id=0)


def test_h_wh_matrix_object_licensed_subject_not():
    obj = _t1("object", "b")
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), obj])
    assert [c[0] for c in h_wh_candidates(tree)] == [obj]


def test_h_wh_intransitive_simple_is_blocked():
    tree = _t0(_t1("subject", "a"), [_term("CAT3")])
    assert h_wh_candidates(tree) == []


def test_h_wh_type3_licensed_only_under_type4():
    # A subject-attached (matrix) Type3 comp is NOT licensed; a Type3 comp
    # inside a Type4 IS. The embedded subject is no longer licensed (§F).
    comp_np = _t1("pp", "c")
    subj = _t1("subject", "a", extra=[_t3("np", comp_np)])
    # embedded clause with an intransitive VP carrying a vp-PP
    ecomp = _t1("pp", "e")
    esubj = _t1("subject", "d")
    inner = _t0(esubj, [_term("CAT3"), _t3("vp", ecomp)], root=False)
    subj.children.append(_t4(inner))
    tree = _t0(subj, [_term("CAT3")])
    got = {id(c[0]) for c in h_wh_candidates(tree)}
    assert got == {id(ecomp)}


def test_h_wh_embedded_arguments_not_licensed():
    # Neither the embedded subject nor the embedded object licenses (§F):
    # embedded material fronts only from inside a Type4-dominated Type3.
    eobj = _t1("object", "e")
    inner = _t0(_t1("subject", "d"), [_term("CAT3"), eobj], root=False)
    subj = _t1("subject", "a", extra=[_t4(inner)])
    tree = _t0(subj, [_term("CAT3")])
    assert h_wh_candidates(tree) == []


def test_h_wh_pron_frontable_refl_not():
    pron = _term("CAT1PRON", "m", role="object", subclass="Pron")
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), pron])
    assert [c[0] for c in h_wh_candidates(tree)] == [pron]
    refl = _term("CAT1PRON", "mm", role="object", subclass="Refl")
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), refl])
    assert h_wh_candidates(tree) == []


# ── P binding (unit) ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lex_p():
    return load_lex_P()


def _feats(n, g):
    return {"INFL1_number": n, "INFL1_gender": g}


def test_p_refl_distance_3_copy_and_obligatory_target(lex_p):
    rng = random.Random(0)
    toks = [T("CAT1", "a", role="subject", feats=_feats("3", "1")),
            T("CAT4"), T("CAT3", feats=_feats("3", "1")),
            T("CAT1", "b", role="object", feats=_feats("4", "2"))]
    label = p_binding(toks, rng, lex_p, subtype="Refl")
    assert label == "anaphoric_binding_refl"
    assert toks[3].cat == "CAT1PRON" and toks[3].lex["subclass"] == "Refl"
    assert toks[3].feats == _feats("3", "1")   # copied from the antecedent


def test_p_refl_blocked_without_distance_3_pair(lex_p):
    rng = random.Random(0)
    toks = [T("CAT1", "a", feats=_feats("3", "1")), T("CAT3"),
            T("CAT1", "b", feats=_feats("4", "2"))]  # distance 2 only
    assert p_binding(toks, rng, lex_p, subtype="Refl") == "anaphoric_binding_skipped"
    # ...but it is a licensed Pron geometry.
    assert p_binding(toks, rng, lex_p, subtype="Pron") == "anaphoric_binding_pron"


def test_p_binding_subject_substitution_fixes_agreement(lex_p):
    rng = random.Random(0)
    # anchor(0) ... embedded subject at +3 with different features
    toks = [T("CAT1", "a", role="subject", feats=_feats("3", "1")),
            T("CAT4"), T("CAT9", cid=1),
            T("CAT1", "b", role="subject", cid=1, feats=_feats("4", "2")),
            T("CAT3", cid=1, feats={**_feats("4", "2"), "INFL3_tense": "7"}),
            T("CAT3", feats={**_feats("3", "1"), "INFL3_tense": "8"})]
    label = p_binding(toks, rng, lex_p, subtype="Refl")
    assert label == "anaphoric_binding_refl"
    assert toks[3].cat == "CAT1PRON" and toks[3].feats == _feats("3", "1")
    # the embedded verb realigned with the substituted subject's features
    assert toks[4].feats["INFL1_number"] == "3"
    assert toks[4].feats["INFL1_gender"] == "1"
    # the matrix verb untouched
    assert toks[5].feats["INFL1_number"] == "3"


def test_p_duplicate_screen_flags_licensed_distances_only():
    a, b = {"form": "z"}, {"form": "z"}
    base = [Token("CAT1", a), Token("CAT4", {"form": "u"}),
            Token("CAT3", {"form": "k"}), Token("CAT1", b)]
    assert has_licensed_duplicate(base)          # distance 3
    far = [Token("CAT1", a), Token("CAT4", {"form": "u"}),
           Token("CAT3", {"form": "k"}), Token("CAT5", {"form": "o"}),
           Token("CAT5", {"form": "p"}), Token("CAT1", b)]
    assert not has_licensed_duplicate(far)       # distance 5: legal


# ── Corpus-level invariants and divergence directions ────────────────────────

@pytest.fixture(scope="module")
def h_items():
    return H_meta(N, seed=SEED)


@pytest.fixture(scope="module")
def p_items():
    return P_meta(N, seed=SEED)


def _p_pron_anchor_ok(toks, j, offsets):
    for off in offsets:
        i = j - off
        if i >= 0 and toks[i].cat == "CAT1" \
                and toks[i].feats["INFL1_number"] == toks[j].feats["INFL1_number"] \
                and toks[i].feats["INFL1_gender"] == toks[j].feats["INFL1_gender"]:
            return i
    return None


def test_p_corpus_binding_invariants_and_divergence(p_items):
    refl = pron = refl_cross = pron_svo = 0
    for _, lab, toks in p_items:
        if lab == "anaphoric_binding_refl":
            j = max(i for i, t in enumerate(toks) if t.cat == "CAT1PRON")
            i = _p_pron_anchor_ok(toks, j, (3,))
            assert i is not None, "refl without a matching antecedent at -3"
            refl += 1
            if toks[i].clause_id != toks[j].clause_id:
                refl_cross += 1   # P-refl where H demands pron
        if lab == "anaphoric_binding_pron":
            j = max(i for i, t in enumerate(toks) if t.cat == "CAT1PRON"
                    and t.lex.get("subclass") == "Pron")
            i = _p_pron_anchor_ok(toks, j, (2, 4))
            assert i is not None, "pron without a matching antecedent at -2/-4"
            pron += 1
            if toks[i].clause_id == toks[j].clause_id \
                    and toks[i].role == "subject" and toks[j].role == "object":
                pron_svo += 1     # P-pron where H demands refl
    assert refl > 0 and pron > 0
    assert refl_cross > 0, "cross-clause refl (divergence direction) never occurred"
    assert pron_svo > 0, "same-nucleus pron (divergence direction) never occurred"


def test_p_corpus_aux_embedded_fronting_occurs(p_items):
    fronted = [toks for _, lab, toks in p_items if lab == "auxiliary_movement"]
    assert all(t[0].cat == "CAT3AUX" for t in fronted)
    emb = sum(1 for t in fronted if t[0].clause_id != 0)
    assert emb > 0, "embedded-aux fronting (H-skip divergence) never occurred"


def test_p_corpus_wh_blockable_and_no_untransformed_licensed_duplicates(p_items):
    labels = collections.Counter(lab for _, lab, _ in p_items)
    assert labels["wh_movement_skipped"] > 0, "P wh must now be blockable"
    for _, lab, toks in p_items:
        if lab == "neutral" and not any(t.cat == "CAT1PRON" for t in toks):
            assert not has_licensed_duplicate(toks), \
                "screen failed: licensed-distance duplicate in an untransformed string"


def test_h_corpus_binding_invariants(h_items):
    refl = refl_emb = pron = 0
    for _, lab, tree in h_items:
        if lab == "anaphoric_binding_refl":
            for t0 in _type0s(tree):
                subj, obj, _ = _nucleus(t0)
                if obj is not None and obj.label == "CAT1PRON" \
                        and obj.lex.get("subclass") == "Refl":
                    sf = (next(c for c in subj.children if c.role == "head").feats
                          if subj.label == "Type1" else subj.feats)
                    assert obj.feats["INFL1_number"] == sf["INFL1_number"]
                    assert obj.feats["INFL1_gender"] == sf["INFL1_gender"]
                    refl += 1
                    refl_emb += t0 is not tree
                    break
        if lab == "anaphoric_binding_pron":
            terms = []
            def walk(n):
                if not n.children:
                    terms.append(n)
                for c in n.children:
                    walk(c)
            walk(tree)
            p = next(n for n in terms if n.label == "CAT1PRON"
                     and n.lex.get("subclass") == "Pron")
            assert any(n.label == "CAT1"
                       and n.feats.get("INFL1_number") == p.feats["INFL1_number"]
                       and n.feats.get("INFL1_gender") == p.feats["INFL1_gender"]
                       for n in terms), "pron has no feature-matching antecedent"
            pron += 1
    assert refl > 0 and pron > 0
    assert refl_emb > 0, "embedded reflexive nucleus never occurred"


def test_h_corpus_aux_blockable_and_matrix_only(h_items):
    labels = collections.Counter(lab for _, lab, _ in h_items)
    assert labels["auxiliary_movement_skipped"] > 0, \
        "H's structural aux blocking is never exercised"
    for _, lab, tree in h_items:
        if lab == "auxiliary_movement":
            assert tree.children[0].label == "CAT3AUX"


def test_h_corpus_no_nucleus_duplicates(h_items):
    for _, _, tree in h_items:
        assert not has_nucleus_duplicate(tree)


def test_pron_fronting_is_exercised():
    """A background Pron in a licensed position must remain a reachable wh
    target end-to-end (D7). After the clause-level embedding recalibration the
    event is rare (~1e-4/item), so scan in batches with early exit rather
    than reusing the fixed 3k fixtures."""
    for seed in range(SEED, SEED + 8):
        for _, lab, tree in H_meta(2000, seed=seed):
            if lab == "wh_movement" and tree.children[1].label == "CAT1PRON":
                return
        for _, lab, toks in P_meta(2000, seed=seed):
            if lab == "wh_movement" and toks[1].cat == "CAT1PRON":
                return
    pytest.fail("background prons never front (D7 unexercised) in 32k draws")
