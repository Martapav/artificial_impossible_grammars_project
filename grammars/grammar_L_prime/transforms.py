"""Transformations for Grammar L' — identical to Grammar P except one wh clause.

L' keeps Grammar P's binding, auxiliary movement, and wh clause (ii) (the second
CAT1-position after the second CAT6 — positional, revised 2026-07-30 and imported
from ``grammar_P.wh`` rather than restated). Only wh clause (i) is swapped:
instead of Grammar P's *third CAT1-position* (positional), L' licenses the
*matrix object that HOSTS an embedded clause* (structural, = Grammar H's clause
(i) under proposal E3, 2026-07-30), read from the generation history
(``role``/``clause_id``/``hosts_embedded``) — L''s designated information channel,
with no reference to surface position. Imported from ``grammars.wh_structural``,
which states that rule once for both H and L'. The
swap axis is the one licensing notion with no local surface shadow (audit §F):
L' licenses a matrix object hosting an embedded clause where H'/P block it unless
it happens to sit at ordinal 3, and an embedded or subject-internal nominal at
ordinal 3 is licensed by H'/P but not by L'. (Before the E3 narrowing the first
direction was "on a bare transitive L' licenses the object"; a bare transitive now
hosts nothing, so it is licensed by neither.)

CAT1PRON counts as a CAT1-position for the ordinal scan; as a target, CAT1
and CAT1PRON-Pron are licit, CAT1PRON-Refl never fronts. L' fronts the
nominal block (P granularity).
"""

from __future__ import annotations

import random

from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import cat8_wh
from grammars.grammar_P import wh
from grammars import wh_structural
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

    Licensed iff (i'') it is the matrix object that HOSTS an embedded clause
    (generation history), or (ii) it is the second CAT1-position after the
    second CAT6. Returns
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
      (i'') the MATRIX OBJECT THAT HOSTS AN EMBEDDED CLAUSE — read from the
            generation history (``role``/``clause_id``/``hosts_embedded``), the
            structural clause swapped in from Grammar H. The rule and its
            rationale live in ``grammars/wh_structural.py``, which H reads off
            its tree and L′ off these tokens — one statement, two readers;
      (ii)  the second CAT1-position after the second CAT6 of the sentence —
            positional, inherited **verbatim** from Grammar P via
            ``grammar_P.wh``, which owns the clause. L′ differs from P in
            clause (i) ONLY, so this must never be reimplemented here (it was,
            until 2026-07-30, and drifted-by-copy is exactly the risk).
    A licensed position is a TARGET iff its token is a CAT1 or a
    CAT1PRON-Pron (a Refl never fronts).
    """
    cats = [t.cat for t in toks]
    licensed = wh.licensed_indices(cats, clause_i=False)   # clause (ii) only
    licensed |= wh_structural.licensed_indices_tokens(toks)  # clause (i'')
    return sorted(
        p for p in licensed
        if wh.target_eligible(toks[p].cat, toks[p].lex)
    )
