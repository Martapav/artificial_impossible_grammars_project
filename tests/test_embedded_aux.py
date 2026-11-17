"""Regression tests for embedded in-situ auxiliaries and positional aux fronting.

Under the reformalized rules (docs/TRANSFORM_RULES.md):
  * embedded clauses still realize in-situ auxiliaries;
  * P aux items may front an embedded aux (positional scan), H aux items
    always front the matrix aux.
"""

from grammars.grammar_H.generate import generate_with_metadata as H_meta
from grammars.grammar_H.transforms import _type0s
from grammars.grammar_P.generate import generate_with_metadata as P_meta

N = 4000
SEED = 7

_BINDING = ("anaphoric_binding_refl", "anaphoric_binding_pron")


def test_H_embedded_aux():
    items = H_meta(N, seed=SEED)
    embedded = aux = 0
    for _surf, label, tree in items:
        inners = [t0 for t0 in _type0s(tree) if t0 is not tree]
        embedded += len(inners)
        for inner in inners:
            t2 = next((c for c in inner.children if c.label == "Type2"), None)
            if t2 is not None:
                aux += sum(1 for c in t2.children if c.label == "CAT3AUX")
    assert embedded > 0, "test needs embedded clauses in the sample"
    assert aux > 0, "embedded in-situ aux never occurred"


def test_P_embedded_aux_and_positional_fronting():
    items = P_meta(N, seed=SEED)
    embedded = aux = auxmv = auxmv_emb_front = 0
    for _surf, label, toks in items:
        ids = {t.clause_id for t in toks if t.clause_id != 0}
        embedded += len(ids)
        aux += sum(1 for t in toks if t.cat == "CAT3AUX" and t.clause_id != 0)
        if label == "auxiliary_movement":
            auxmv += 1
            # P fronts the FIRST aux positionally: position 0 is a CAT3AUX,
            # and it may come from an embedded clause (H skips that plan).
            assert toks[0].cat == "CAT3AUX"
            if toks[0].clause_id != 0:
                auxmv_emb_front += 1
    assert embedded > 0
    assert aux > 0, "embedded in-situ aux never occurred in P"
    assert auxmv > 0
    assert auxmv_emb_front > 0, "embedded-aux fronting divergence never occurred"
