"""Transformations for Grammar L' — identical to Grammar P except wh-movement.

L' keeps Grammar P's binding, auxiliary movement, and the second-CAT1 and
post-CAT9 wh blocks. Only the complex-NP island is swapped: instead of P's
surface scan it consults the generation history.

L' complex-NP rule: fronting of a CAT1 is blocked iff that CAT1 was inserted as
the complement of a PP-under-NP CAT6 — recorded on the token as
``licensor_attachment == "np"`` (set in grammar_P.build._emit_pp). This makes no
reference to surface position, so it diverges from P's/H's surface rule: it still
blocks when a CAT2 intervenes, and permits a PP-under-VP after the object. Those
are the divergence cases (see grammar_H_prime).
"""

from __future__ import annotations

import random

from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import cat8_wh
# Binding and auxiliary movement are unchanged from Grammar P.
from grammars.grammar_P.transforms import _binding, _aux_movement, _extract_nominal_block


def apply(toks: list[Token], phenomenon: str, rng: random.Random, lex: dict) -> str:
    """Dispatch to the transformation and return the refined label."""
    if phenomenon == "anaphoric_binding":
        return _binding(toks, rng, lex)
    if phenomenon == "auxiliary_movement":
        return _aux_movement(toks)
    if phenomenon == "wh_movement":
        return _wh_movement(toks, rng, lex)
    return phenomenon


# ── Wh-movement (positional second-CAT1 / post-CAT9 + structural complex-NP) ──


def _wh_movement(toks: list[Token], rng: random.Random, lex: dict) -> str:
    """Front a licit CAT1 to sentence-initial position after a CAT8 marker."""
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
    """Indices of CAT1 tokens eligible for fronting. Blocked iff the CAT1 is:

    1. the second CAT1 in the sentence (object) — positional;
    2. the first CAT1 after a CAT9 (subordinate subject) — positional;
    3. the complement of a PP-under-NP CAT6 (``licensor_attachment == "np"``)
       — structural, read from the generation history.
    """
    eligible: list[int] = []
    cat1_count = 0
    pending_after_cat9 = False
    for i, t in enumerate(toks):
        if t.cat == "CAT9":
            pending_after_cat9 = True
            continue
        if t.cat != "CAT1":
            continue
        cat1_count += 1
        is_second = cat1_count == 2                       # condition 1
        is_sub_subject = pending_after_cat9               # condition 2
        is_complex_np = t.licensor_attachment == "np"     # condition 3
        pending_after_cat9 = False
        if not (is_second or is_sub_subject or is_complex_np):
            eligible.append(i)
    return eligible
