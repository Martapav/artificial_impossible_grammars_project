"""Transformations for Grammar L' — identical to Grammar P except one wh clause.

L' keeps Grammar P's binding, auxiliary movement, and wh clause (ii) (the first
CAT1-position after a CAT5 — positional). Only wh clause (i) is swapped:
instead of Grammar P's *third CAT1-position* (positional), L' licenses the
*object of the matrix clause* (structural, = Grammar H's clause (i)), read
from the generation history (``role == "object" and clause_id == 0``) — L''s
designated information channel, with no reference to surface position. The
swap axis is the one licensing notion with no local surface shadow (audit §F):
on a bare transitive L' licenses the object where H'/P block (2nd position,
no adverb), and an embedded or subject-internal nominal at ordinal 3 is
licensed by H'/P but not by L'.

CAT1PRON counts as a CAT1-position for the ordinal scan; as a target, CAT1
and CAT1PRON-Pron are licit, CAT1PRON-Refl never fronts. L' fronts the
nominal block (P granularity).
"""

from __future__ import annotations

import random

from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import cat8_wh
# Everything except wh-movement is Grammar P's own dispatcher.
from grammars.grammar_P.transforms import apply as p_apply, _extract_nominal_block


def apply(toks: list[Token], phenomenon: str, rng: random.Random, lex: dict) -> str:
    """Dispatch to the pre-selected transformation; return the refined label.

    Only wh-movement is L''s own; every other phenomenon (binding subtypes,
    aux, neutral) is delegated verbatim to Grammar P's dispatcher — the
    single-rule difference is enforced right here.
    """
    if phenomenon == "wh_movement":
        return _wh_movement(toks, rng, lex)
    return p_apply(toks, phenomenon, rng, lex)


# ── Wh-movement: structural clause (i'') + positional clause (ii) ─────────────


def _wh_movement(toks: list[Token], rng: random.Random, lex: dict) -> str:
    """Front a LICENSED nominal block to sentence-initial position after CAT8.

    Licensed iff (i'') it is the matrix object (generation history), or (ii)
    it occupies the first CAT1-position after a CAT5. Returns
    "wh_movement_skipped" when no licensed target exists.
    """
    candidates = _wh_candidates(toks)
    if not candidates:
        return "wh_movement_skipped"

    target_idx = rng.choice(candidates)
    block = _extract_nominal_block(toks, target_idx)  # removes block from toks
    gap = Token("GAP", None, role="gap", clause_id=block[0].clause_id)
    toks.insert(target_idx, gap)

    cat8 = Token("CAT8", cat8_wh(lex), role="wh", clause_id=0)
    toks[0:0] = [cat8, *block]
    return "wh_movement"


def _wh_candidates(toks: list[Token]) -> list[int]:
    """Indices of nominal tokens in a licensed L' wh position.

    Licensed positions:
      (i'') the MATRIX OBJECT — ``role == "object" and clause_id == 0``,
            read from the generation history (structural clause, swapped in
            from Grammar H);
      (ii)  the first CAT1-position after a CAT5 (per CAT5) — positional,
            inherited from Grammar P.
    A licensed position is a TARGET iff its token is a CAT1 or a
    CAT1PRON-Pron (a Refl never fronts).
    """
    positions = [i for i, t in enumerate(toks) if t.cat in ("CAT1", "CAT1PRON")]
    licensed: set[int] = set()
    for i in positions:
        t = toks[i]
        if t.role == "object" and t.clause_id == 0:       # clause (i'')
            licensed.add(i)
    for i, t in enumerate(toks):
        if t.cat == "CAT5":
            after = [p for p in positions if p > i]
            if after:
                licensed.add(after[0])                    # clause (ii)
    return sorted(
        p for p in licensed
        if toks[p].cat == "CAT1"
        or (toks[p].lex is not None and toks[p].lex.get("subclass") == "Pron")
    )
