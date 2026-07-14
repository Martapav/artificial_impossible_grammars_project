"""Balanced 4-bucket probe construction.

Each grammar/split gets 400 items: 100 each of neutral / binding / aux / wh.

Every bucket is drawn from the grammar's **natural** generation (a single pooled
sample), never hand-built — so probe items carry the full structural richness of
the language at every length (boundedness decisions). The wh
bucket's H'/L' clause-(i) verdicts are recovered from the sampled structure by
reconstructing the BASE string (licensing is evaluated before fronting): the H'
verdict from the target's CAT1-position ordinal, the L' verdict from the
target's generation history (matrix object or not). The generalization split
uses the identical builder with a longer length window — no padding.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from grammars import verdicts as V
from grammars.grammar_H.nodes import Node
from grammars.grammar_H.transforms import (
    _embedded_type0s,
    _collect_wh_candidates as h_wh_candidates,
)
from grammars.grammar_H_prime.transforms import (
    _collect_wh_candidates as hp_wh_candidates,
)
from grammars.grammar_P.transforms import _wh_candidates as p_wh_candidates
from grammars.grammar_L_prime.transforms import _wh_candidates as lp_wh_candidates
from grammars.grammar_P.tokens import Token

from grammars.grammar_H.generate import generate_with_metadata as h_meta
from grammars.grammar_P.generate import generate_with_metadata as p_meta
from grammars.grammar_H_prime.generate import generate_with_metadata as hp_meta
from grammars.grammar_L_prime.generate import generate_with_metadata as lp_meta


N_BUCKET = 100
FAMILY = {"H": "tree", "Hprime": "tree", "P": "token", "Lprime": "token"}
META = {"H": h_meta, "Hprime": hp_meta, "P": p_meta, "Lprime": lp_meta}
MIXED = {"Hprime", "Lprime"}

_POOL_BATCH = 2_000  # small batches so early quota-fill wastes little generation
_POOL_MAX = 4_000_000  # generous ceiling; long-length binding/aux are the scarce case


# ─────────────────────────────────────────────────────────────────────────────
# Row schema
# ─────────────────────────────────────────────────────────────────────────────

def _row(grammar: str, split: str, bucket: str, sentence: str, construction: str,
         **extra) -> Dict:
    row = {
        "sentence": sentence, "grammar_type": grammar, "split": split,
        "bucket": bucket, "construction": construction,
        "length": len(sentence.split()),
        "hprime_verdict": None, "lprime_verdict": None, "divergence": None,
        "target_index": None,
        "blocked_phenomenon": None, "block_reason": None,
    }
    row.update(extra)
    return row


# ─────────────────────────────────────────────────────────────────────────────
# Wh verdict recovery — delegated to grammars.verdicts, which reconstructs the
# BASE string from the fronted item (clause-(i) licensing is evaluated before
# fronting, so ordinals must be counted with the block back at its gap).
# ─────────────────────────────────────────────────────────────────────────────

def _wh_verdicts(grammar: str, struct) -> Optional[Dict]:
    if FAMILY[grammar] == "tree":
        return V.wh_verdicts_tree(struct)
    return V.wh_verdicts_token(struct)


# ─────────────────────────────────────────────────────────────────────────────
# Neutral structural classification
# ─────────────────────────────────────────────────────────────────────────────

# The blocked-wh neutral sub-bucket in the licensing frame: a neutral item
# where the grammar's OWN licensed set is empty (wh could never have applied).
# That is the distributional blocking evidence — per grammar, since the four
# licensed sets differ.
_WH_CANDIDATE_FNS = {
    "H": h_wh_candidates,
    "Hprime": hp_wh_candidates,
    "P": p_wh_candidates,
    "Lprime": lp_wh_candidates,
}


def _tree_neutral_features(grammar: str, tree: Node) -> Dict:
    has_aux = False
    aux_locus = None
    emb_obj_nonrefl = False
    t2 = [c for c in tree.children if c.label == "Type2"]
    if t2 and any(c.label == "CAT3AUX" for c in t2[0].children):
        has_aux, aux_locus = True, "matrix_in_situ"
    for inner in _embedded_type0s(tree):
        it2 = [c for c in inner.children if c.label == "Type2"]
        if not it2:
            continue
        if any(c.label == "CAT3AUX" for c in it2[0].children):
            has_aux = True
            aux_locus = aux_locus or "embedded"
        for c in it2[0].children:
            if c.role == "object" and (
                c.label == "Type1"
                or (c.label == "CAT1PRON" and c.lex and c.lex.get("subclass") == "Pron")
            ):
                emb_obj_nonrefl = True
    return {"has_aux": has_aux, "aux_locus": aux_locus,
            "emb_obj_nonrefl": emb_obj_nonrefl,
            "wh_blocked": not _WH_CANDIDATE_FNS[grammar](tree)}


def _token_neutral_features(grammar: str, toks: List[Token]) -> Dict:
    matrix_aux = any(t.cat == "CAT3AUX" and t.clause_id == 0 for t in toks)
    emb_aux = any(t.cat == "CAT3AUX" and t.clause_id != 0 for t in toks)
    aux_locus = "matrix_in_situ" if matrix_aux else ("embedded" if emb_aux else None)
    emb_obj_nonrefl = any(
        t.cat == "CAT1" and t.role == "object" and t.clause_id != 0 for t in toks)
    return {"has_aux": matrix_aux or emb_aux, "aux_locus": aux_locus,
            "emb_obj_nonrefl": emb_obj_nonrefl,
            "wh_blocked": not _WH_CANDIDATE_FNS[grammar](toks)}


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — one pooled sample fills all four buckets
# ─────────────────────────────────────────────────────────────────────────────

def build_probe_buckets(grammar: str, split: str, seed: int,
                        min_length: int, max_length: int) -> List[Dict]:
    """Return 400 balanced probe rows (4 buckets x 100) sampled from natural output."""
    fam = FAMILY[grammar]

    def features(struct):
        fn = _tree_neutral_features if fam == "tree" else _token_neutral_features
        return fn(grammar, struct)

    is_mixed = grammar in MIXED

    # (bucket, subkey) -> quota
    quota: Dict[Tuple[str, str], int] = {
        ("binding", "any"): N_BUCKET,
        ("aux", "any"): N_BUCKET,
        ("neutral", "blocked-binding"): 33,
        ("neutral", "blocked-aux"): 33,
        ("neutral", "blocked-wh"): 34,
    }
    if is_mixed:
        quota[("wh", "divergence")] = 50
        quota[("wh", "coherent")] = 50
    else:
        quota[("wh", "licit")] = N_BUCKET
    got: Dict[Tuple[str, str], List[Dict]] = {k: [] for k in quota}

    def full(k) -> bool:
        return len(got[k]) >= quota[k]

    def done() -> bool:
        return all(full(k) for k in quota)

    pool_seed = seed
    drawn = 0
    while not done() and drawn < _POOL_MAX:
        for surf, label, struct in META[grammar](_POOL_BATCH, seed=pool_seed):
            drawn += 1
            if not (min_length <= len(surf.split()) <= max_length):
                continue

            if label in ("anaphoric_binding_refl", "anaphoric_binding_pron"):
                k = ("binding", "any")
                if not full(k):
                    got[k].append(_row(grammar, split, "binding", surf, label))

            elif label == "auxiliary_movement":
                k = ("aux", "any")
                if not full(k):
                    got[k].append(_row(grammar, split, "aux", surf, "auxiliary_movement"))

            elif label == "wh_movement":
                if is_mixed:
                    v = _wh_verdicts(grammar, struct)
                    if v is None:
                        continue
                    k = ("wh", "divergence" if v["divergence"] else "coherent")
                    if not full(k):
                        got[k].append(_row(grammar, split, "wh", surf, "wh_movement", **v))
                else:
                    k = ("wh", "licit")
                    if not full(k):
                        got[k].append(_row(grammar, split, "wh", surf, "wh_movement"))

            elif label == "neutral":
                f = features(struct)
                # Priority: scarce sub-buckets first.
                if not full(("neutral", "blocked-binding")) and f["emb_obj_nonrefl"]:
                    got[("neutral", "blocked-binding")].append(_row(
                        grammar, split, "neutral", surf, "neutral",
                        blocked_phenomenon="binding",
                        block_reason="embedded object cannot be bound by the matrix subject (domain)"))
                elif not full(("neutral", "blocked-aux")) and f["has_aux"]:
                    got[("neutral", "blocked-aux")].append(_row(
                        grammar, split, "neutral", surf, "neutral",
                        blocked_phenomenon="aux",
                        block_reason=f"CAT3AUX present but not fronted ({f['aux_locus']})"))
                elif not full(("neutral", "blocked-wh")) and f["wh_blocked"]:
                    got[("neutral", "blocked-wh")].append(_row(
                        grammar, split, "neutral", surf, "neutral",
                        blocked_phenomenon="wh",
                        block_reason="no_licensed_wh_position"))

            if done():
                break
        pool_seed += 1

    if not done():
        short = {f"{b}/{s}": quota[(b, s)] - len(got[(b, s)])
                 for (b, s) in quota if not full((b, s))}
        raise RuntimeError(f"probe buckets for {grammar}/{split}: unmet quotas {short} "
                           f"after {drawn} draws in [{min_length},{max_length}]")

    order = [("neutral", "blocked-binding"), ("neutral", "blocked-aux"),
             ("neutral", "blocked-wh"), ("binding", "any"), ("aux", "any")]
    order += ([("wh", "divergence"), ("wh", "coherent")] if is_mixed else [("wh", "licit")])
    return [row for k in order for row in got[k]]
