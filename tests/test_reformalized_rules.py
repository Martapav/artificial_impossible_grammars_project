"""Unit + corpus-level regression tests for the reformalized transform rules.

Reference: docs/TRANSFORM_RULES.md. Pins:
  * P wh licensing (third CAT1-position / **second CAT1-position after the
    second CAT6** — audit §F with clause (ii) revised 2026-07-30), ordinal
    counting of CAT1PRON, Refl never a target, blockability;
  * H wh licensing (matrix object that HOSTS an embedded clause, proposal E3
    2026-07-30 / inside a Type3 dominated by a Type4 —
    audit §F), matrix subject and embedded subjects never licensed,
    matrix-attached Type3s never licensed, blockability;
  * P binding: distance AND antecedent-CAT1-ordinal licensing (refl rule B,
    pron N9 — 2026-07-30), feature copy from the antecedent,
    subject-substitution agreement fixup, blockability, and the forcing that
    drives the skip rate to 0;
  * H binding: nucleus-local Refl in an EMBEDDED nucleus (rule A′, 2026-07-29 —
    the domain condition itself is pinned in test_refl_embedded_domain.py),
    cross-nucleus Pron, feature copy;
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


def _two_cat6_base():
    """`CAT1 CAT3 CAT6 CAT1 CAT6 CAT1 CAT4 CAT1` — clause (ii)'s geometry.

    CAT6s at indices 2 and 4; the 2nd CAT6 is at 4, so the CAT1-positions after
    it are 5 (1st) and 7 (2nd). Clause (ii) licenses **7**. Clause (i) licenses
    the 3rd CAT1-position, which is index 5 — so the two clauses land on
    different positions here, which is the point of the rule.
    """
    return [T("CAT1"), T("CAT3"), T("CAT6"), T("CAT1"), T("CAT6"),
            T("CAT1"), T("CAT4"), T("CAT1")]


def test_p_wh_clause_ii_second_nominal_after_second_cat6():
    # clause (i) -> index 5 (3rd CAT1-position), clause (ii) -> index 7
    assert p_wh_candidates(_two_cat6_base()) == [5, 7]


def test_p_wh_clause_ii_needs_two_cat6_and_two_following_nominals():
    # only ONE CAT6: clause (ii) licenses nothing (no fallback to the 1st CAT6)
    one = [T("CAT1"), T("CAT3"), T("CAT6"), T("CAT1"), T("CAT1")]
    assert p_wh_candidates(one) == [4]          # clause (i) only
    # two CAT6 but only ONE nominal after the 2nd: no fallback to that nominal
    short = [T("CAT1"), T("CAT6"), T("CAT1"), T("CAT6"), T("CAT1")]
    assert p_wh_candidates(short) == [4]        # index 4 is the 3rd CAT1-pos
    # ...and the 1st nominal after the 2nd CAT6 is NOT licensed on its own:
    # `n=2` is load-bearing, since a CAT6 is always followed by its complement.
    assert 4 not in {i for i, t in enumerate(short) if t.cat == "CAT6"}


def test_p_wh_cat5_no_longer_licenses_anything():
    """The pre-2026-07-30 clause (ii) is gone: CAT5 has no licensing role."""
    toks = [T("CAT1"), T("CAT4"), T("CAT3"), T("CAT5"), T("CAT1")]
    assert p_wh_candidates(toks) == []          # was [4] under the CAT5 rule
    # even two CAT5s license nothing — the anchor category is CAT6 now
    toks = [T("CAT1"), T("CAT3"), T("CAT5"), T("CAT5"), T("CAT1"), T("CAT1")]
    assert p_wh_candidates(toks) == [5]         # clause (i) only


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
    # With only two CAT1-positions and no clause-(ii) geometry, embedding alone
    # no longer makes wh applicable.
    toks = [T("CAT1"), T("CAT9", cid=1), T("CAT1", cid=1),
            T("CAT3", cid=1), T("CAT3")]
    assert p_wh_candidates(toks) == []


def test_p_wh_pron_counts_and_refl_never_fronts():
    # CAT1PRON counts as a CAT1-position for ordinals...
    toks = [T("CAT1"), T("CAT1PRON", subclass="Refl"), T("CAT3"), T("CAT1")]
    assert p_wh_candidates(toks) == [3]  # third position, full CAT1
    # ...a Pron at a licensed position is a target, a Refl is not. Index 7 is
    # clause (ii)'s position in this base; index 5 is clause (i)'s.
    base = _two_cat6_base()
    pron = list(base); pron[7] = T("CAT1PRON", subclass="Pron")
    assert p_wh_candidates(pron) == [5, 7]
    refl = list(base); refl[7] = T("CAT1PRON", subclass="Refl")
    assert p_wh_candidates(refl) == [5]


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


def _emb_clause():
    """A minimal embedded clause, for hanging under a host nominal."""
    return _t4(_t0(_t1("subject", "d"), [_term("CAT3")], root=False))


def test_h_wh_matrix_object_hosting_an_embedded_clause_is_licensed():
    """Clause (i) is E3 (2026-07-30): matrix object AND it hosts a Type4."""
    obj = _t1("object", "b", extra=[_emb_clause()])
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), obj])
    assert [c[0] for c in h_wh_candidates(tree)] == [obj]


def test_h_wh_bare_matrix_object_no_longer_licensed():
    """The second conjunct has teeth: an object hosting nothing is unlicensed."""
    obj = _t1("object", "b")
    tree = _t0(_t1("subject", "a"), [_term("CAT3"), obj])
    assert h_wh_candidates(tree) == []          # was [obj] before E3


def test_h_wh_subject_hosting_an_embedded_clause_is_not_licensed():
    """Only the OBJECT host licenses — the subject host is the natural negative
    that E3's forced geometry deliberately does not produce."""
    subj = _t1("subject", "a", extra=[_emb_clause()])
    obj = _t1("object", "b")
    tree = _t0(subj, [_term("CAT3"), obj])
    assert h_wh_candidates(tree) == []


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
    # A CAT1PRON is a terminal, so it hosts nothing and cannot satisfy clause (i)
    # under E3; it reaches the candidate list via clause (ii) instead.
    pron = _term("CAT1PRON", "m", role="object", subclass="Pron")
    inner = _t0(_t1("subject", "d"), [_term("CAT3"), _t3("vp", _t1("pp", "e"))],
                root=False)
    subj = _t1("subject", "a", extra=[_t4(inner)])
    tree = _t0(subj, [_term("CAT3"), pron])
    assert pron not in [c[0] for c in h_wh_candidates(tree)]
    # ...and a Refl never fronts, from any clause.
    refl = _term("CAT1PRON", "mm", role="object", subclass="Refl")
    tree2 = _t0(_t1("subject", "a"), [_term("CAT3"), refl])
    assert h_wh_candidates(tree2) == []


# ── P binding (unit) ─────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def lex_p():
    return load_lex_P()


def _feats(n, g):
    return {"INFL1_number": n, "INFL1_gender": g}


def _refl_base():
    """A base hosting refl rule B: `CAT1 CAT4 CAT6 CAT1 CAT3 CAT1 CAT4 CAT9 CAT1`.

    CAT1-position ordinals 1 (matrix subject), 2 (PP complement), 3 (matrix
    object — the antecedent), 4 (embedded subject at distance 3 — the target).
    """
    return [T("CAT1", "s", role="subject", feats=_feats("4", "2")),
            T("CAT4"),
            T("CAT6"),
            T("CAT1", "c", role="complement", feats=_feats("4", "2")),
            T("CAT3", feats={**_feats("4", "2"), "INFL3_tense": "8"}),
            T("CAT1", "a", role="object", feats=_feats("3", "1")),   # ordinal 3
            T("CAT4"),
            T("CAT9", cid=1),
            T("CAT1", "b", role="subject", cid=1, feats=_feats("4", "2")),
            T("CAT3", cid=1, feats={**_feats("4", "2"), "INFL3_tense": "7"})]


def test_p_refl_distance_3_at_third_ordinal_copies_and_substitutes(lex_p):
    rng = random.Random(0)
    toks = _refl_base()
    label = p_binding(toks, rng, lex_p, subtype="Refl")
    assert label == "anaphoric_binding_refl"
    assert toks[8].cat == "CAT1PRON" and toks[8].lex["subclass"] == "Refl"
    assert toks[8].feats == _feats("3", "1")   # copied from the antecedent


def test_p_refl_blocked_without_distance_3_pair(lex_p):
    rng = random.Random(0)
    # ordinals 1, 2 (antecedent), 3 (target at distance 2) — a Pron geometry.
    toks = [T("CAT1", "s", feats=_feats("4", "2")), T("CAT6"),
            T("CAT1", "a", feats=_feats("3", "1")), T("CAT3"),
            T("CAT1", "b", feats=_feats("4", "2"))]
    assert p_binding(toks, rng, lex_p, subtype="Refl") == "anaphoric_binding_skipped"
    # ...but it is a licensed Pron geometry (distance 2, antecedent ordinal 2).
    assert p_binding(toks, rng, lex_p, subtype="Pron") == "anaphoric_binding_pron"


def test_p_binding_ordinal_conjunct_is_necessary(lex_p):
    """Rule B / N9 (2026-07-30): the licensed distance alone no longer licenses.

    Same distances as the licensed cases, antecedent at CAT1-ordinal 1 — the
    configuration a distance-only rule would license and the narrowed rules
    leave as blocking evidence.
    """
    rng = random.Random(0)
    d3 = [T("CAT1", "a", role="subject", feats=_feats("3", "1")),
          T("CAT4"), T("CAT3", feats=_feats("3", "1")),
          T("CAT1", "b", role="object", feats=_feats("4", "2"))]
    assert p_binding(d3, rng, lex_p, subtype="Refl") == "anaphoric_binding_skipped"
    d2 = [T("CAT1", "a", feats=_feats("3", "1")), T("CAT3"),
          T("CAT1", "b", feats=_feats("4", "2"))]
    assert p_binding(d2, rng, lex_p, subtype="Pron") == "anaphoric_binding_skipped"
    # Pron licenses ordinals 2 and 3, refl only 3: ordinal 2 at distance 3 is
    # licensed by neither (and distance-disjointness keeps the pair unambiguous).
    ord2_d3 = [T("CAT1", "s", feats=_feats("4", "2")), T("CAT6"),
               T("CAT1", "a", feats=_feats("3", "1")), T("CAT3"), T("CAT4"),
               T("CAT1", "b", feats=_feats("4", "2"))]
    assert p_binding(ord2_d3, rng, lex_p,
                     subtype="Refl") == "anaphoric_binding_skipped"


def test_p_binding_subject_substitution_fixes_agreement(lex_p):
    rng = random.Random(0)
    toks = _refl_base()   # target (index 8) is an embedded SUBJECT
    label = p_binding(toks, rng, lex_p, subtype="Refl")
    assert label == "anaphoric_binding_refl"
    assert toks[8].cat == "CAT1PRON" and toks[8].feats == _feats("3", "1")
    # the embedded verb realigned with the substituted subject's features
    assert toks[9].feats["INFL1_number"] == "3"
    assert toks[9].feats["INFL1_gender"] == "1"
    # the matrix verb untouched
    assert toks[4].feats["INFL1_number"] == "4"


def test_p_duplicate_screen_flags_licensed_configurations_only():
    """The screen narrows with the rules (2026-07-30): distance AND ordinal."""
    a, b = {"form": "z"}, {"form": "z"}
    u, k = {"form": "u"}, {"form": "k"}
    # distance 3, antecedent at ordinal 3 → licensed for refl, must be screened
    licensed = [Token("CAT1", {"form": "s"}), Token("CAT6", u),
                Token("CAT1", {"form": "c"}), Token("CAT3", k),
                Token("CAT1", a), Token("CAT4", u), Token("CAT9", k),
                Token("CAT1", b)]
    assert has_licensed_duplicate(licensed)
    # same distance, antecedent at ordinal 1 → licensed by nothing, so it is
    # NOT screened: it surfaces as a bare duplicate (positional blocking
    # evidence, the analogue of H's matrix same-nucleus pairs under A′)
    ord1 = [Token("CAT1", a), Token("CAT4", u), Token("CAT3", k), Token("CAT1", b)]
    assert not has_licensed_duplicate(ord1)
    # distance 5 from a licensed ordinal: unlicensed distance, also legal
    # (4 would be a licensed Pron distance at this ordinal — hence three fillers)
    far = [Token("CAT1", {"form": "s"}), Token("CAT6", u),
           Token("CAT1", {"form": "c"}), Token("CAT3", k),
           Token("CAT1", a), Token("CAT4", u), Token("CAT5", u),
           Token("CAT5", k), Token("CAT5", u), Token("CAT1", b)]
    assert not has_licensed_duplicate(far)


# ── Corpus-level invariants and divergence directions ────────────────────────

@pytest.fixture(scope="module")
def h_items():
    return H_meta(N, seed=SEED)


@pytest.fixture(scope="module")
def p_items():
    return P_meta(N, seed=SEED)


def _p_pron_anchor_ok(toks, j, subtype):
    """The licensing antecedent of the pron at ``j``, checked on BOTH conjuncts.

    Distance (offsets) and the antecedent's CAT1-position ordinal, plus feature
    identity — i.e. the whole rule as of 2026-07-30, re-derived here from the
    surfaced item rather than read off the licensing module.
    """
    from grammars.grammar_P.binding import ant_ordinals, offsets
    ordinal, n = {}, 0
    for i, t in enumerate(toks):
        if t.cat in ("CAT1", "CAT1PRON"):
            n += 1
            ordinal[i] = n
    for off in offsets(subtype):
        i = j - off
        if i >= 0 and toks[i].cat == "CAT1" \
                and ordinal[i] in ant_ordinals(subtype) \
                and toks[i].feats["INFL1_number"] == toks[j].feats["INFL1_number"] \
                and toks[i].feats["INFL1_gender"] == toks[j].feats["INFL1_gender"]:
            return i
    return None


def test_p_corpus_binding_invariants_and_divergence(p_items):
    refl = pron = refl_cross = pron_svo = 0
    for _, lab, toks in p_items:
        if lab == "anaphoric_binding_refl":
            j = max(i for i, t in enumerate(toks) if t.cat == "CAT1PRON")
            i = _p_pron_anchor_ok(toks, j, "Refl")
            assert i is not None, \
                "refl without an antecedent at -3 whose CAT1-ordinal is 3"
            refl += 1
            if toks[i].clause_id != toks[j].clause_id:
                refl_cross += 1   # P-refl where H demands pron
        if lab == "anaphoric_binding_pron":
            j = max(i for i, t in enumerate(toks) if t.cat == "CAT1PRON"
                    and t.lex.get("subclass") == "Pron")
            i = _p_pron_anchor_ok(toks, j, "Pron")
            assert i is not None, \
                "pron without an antecedent at -2/-4 whose CAT1-ordinal is 2 or 3"
            pron += 1
            if toks[i].clause_id == toks[j].clause_id \
                    and toks[i].role == "subject" and toks[j].role == "object":
                pron_svo += 1     # P-pron where H demands refl
    assert refl > 0 and pron > 0
    assert refl_cross > 0, "cross-clause refl (divergence direction) never occurred"
    assert pron_svo > 0, "same-nucleus pron (divergence direction) never occurred"


def test_p_corpus_binding_geometry_is_forced(p_items):
    """Generation forces the geometry, so binding never skips (2026-07-30).

    Availability is 6.5% (refl B) / 19.9% (pron N9) in a neutral draw, so
    without forcing most binding draws would be dropped and the surviving ones
    would come from an unmatched slice of the base distribution.
    """
    labels = collections.Counter(lab for _, lab, _ in p_items)
    assert labels["anaphoric_binding_skipped"] == 0, \
        f"binding skips must be 0 under forcing, got {labels['anaphoric_binding_skipped']}"
    assert labels["anaphoric_binding_refl"] > 0
    assert labels["anaphoric_binding_pron"] > 0


def test_p_corpus_aux_embedded_fronting_occurs(p_items):
    fronted = [toks for _, lab, toks in p_items if lab == "auxiliary_movement"]
    assert all(t[0].cat == "CAT3AUX" for t in fronted)
    emb = sum(1 for t in fronted if t[0].clause_id != 0)
    assert emb > 0, "embedded-aux fronting (H-skip divergence) never occurred"


def test_p_corpus_wh_geometry_is_forced_and_no_untransformed_duplicates(p_items):
    """wh no longer skips: its geometry is forced (2026-07-30, clause (i) E3).

    Clause (i) is now "the matrix object that HOSTS an embedded clause", which
    occurs in 2.25% of neutral draws, so generation forces it — matrix transitive,
    one embedded clause, hosted by the object. That geometry also guarantees three
    CAT1-positions, so P's own positional clause (i) is satisfied too and the
    phenomenon can never be blocked. wh blockability is therefore RETIRED as a
    property, exactly as refl's was when A′ forced its embedded nucleus; the
    distributional evidence is now the unlicensed nominals that never front.
    """
    labels = collections.Counter(lab for _, lab, _ in p_items)
    assert labels["wh_movement_skipped"] == 0, \
        f"wh geometry is forced; got {labels['wh_movement_skipped']} skips"
    assert labels["wh_movement"] > 0
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
            assert refl_emb == refl, \
                "rule A′: every reflexive nucleus must be EMBEDDED"
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
    assert refl_emb == refl, "rule A′: all reflexive nuclei must be embedded"


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
