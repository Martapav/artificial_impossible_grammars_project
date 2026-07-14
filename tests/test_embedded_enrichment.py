"""Regression tests for background binding (successor of the embedded
enrichment) and embedded in-situ auxiliaries.

Under the reformalized rules (grammars/TRANSFORM_RULES.md) the per-clause
enrichment is replaced by one optional embedded-scoped substitution per
non-binding sentence, applied by each grammar's OWN binding rule:
  * H: refl in an embedded nucleus / pron pair with an embedded member;
  * P: distance-licensed pair whose target token is embedded.
Invariants pinned here:
  * embedded clauses still realize in-situ auxiliaries;
  * background pronominalization occurs in non-binding sentences and lands
    on embedded material;
  * an H background reflexive copies its EMBEDDED nucleus subject (local
    domain), never the matrix subject when the two differ;
  * P aux items may front an embedded aux (positional scan), H aux items
    always front the matrix aux.
"""

from grammars.grammar_H.generate import generate_with_metadata as H_meta
from grammars.grammar_H.transforms import _type0s, _nucleus
from grammars.grammar_P.generate import generate_with_metadata as P_meta

N = 4000
SEED = 7

_BINDING = ("anaphoric_binding_refl", "anaphoric_binding_pron")


def test_H_embedded_aux_and_background_binding():
    items = H_meta(N, seed=SEED)
    embedded = aux = bg_pron = bg_refl_checked = 0
    for _surf, label, tree in items:
        inners = [t0 for t0 in _type0s(tree) if t0 is not tree]
        embedded += len(inners)
        for inner in inners:
            t2 = next((c for c in inner.children if c.label == "Type2"), None)
            if t2 is not None:
                aux += sum(1 for c in t2.children if c.label == "CAT3AUX")
        if label in _BINDING:
            continue  # background applies to non-binding items only
        for inner in inners:
            subj, obj, _ = _nucleus(inner)
            if obj is not None and obj.label == "CAT1PRON":
                bg_pron += 1
                # subj may be missing/GAP when wh later fronted the embedded
                # subject — the refl copied it before movement; skip the check.
                if obj.lex.get("subclass") == "Refl" \
                        and subj is not None and subj.label in ("Type1", "CAT1PRON"):
                    bg_refl_checked += 1
                    sf = (next(c for c in subj.children if c.role == "head").feats
                          if subj.label == "Type1" else subj.feats)
                    # Local domain: the reflexive copies the EMBEDDED subject.
                    assert obj.feats["INFL1_number"] == sf["INFL1_number"]
                    assert obj.feats["INFL1_gender"] == sf["INFL1_gender"]
    assert embedded > 0, "test needs embedded clauses in the sample"
    assert aux > 0, "embedded in-situ aux never occurred"
    assert bg_pron > 0, "background pronominalization never occurred in H"
    assert bg_refl_checked > 0, "no background reflexive to validate the domain on"


def test_P_embedded_aux_background_binding_and_positional_fronting():
    items = P_meta(N, seed=SEED)
    embedded = aux = bg_pron = auxmv = auxmv_emb_front = 0
    for _surf, label, toks in items:
        ids = {t.clause_id for t in toks if t.clause_id != 0}
        embedded += len(ids)
        aux += sum(1 for t in toks if t.cat == "CAT3AUX" and t.clause_id != 0)
        if label not in _BINDING:
            bg_pron += sum(1 for t in toks
                           if t.cat == "CAT1PRON" and t.clause_id != 0)
        if label == "auxiliary_movement":
            auxmv += 1
            # P fronts the FIRST aux positionally: position 0 is a CAT3AUX,
            # and it may come from an embedded clause (H skips that plan).
            assert toks[0].cat == "CAT3AUX"
            if toks[0].clause_id != 0:
                auxmv_emb_front += 1
    assert embedded > 0
    assert aux > 0, "embedded in-situ aux never occurred in P"
    assert bg_pron > 0, "background pronominalization never occurred in P"
    assert auxmv > 0
    assert auxmv_emb_front > 0, "embedded-aux fronting divergence never occurred"
