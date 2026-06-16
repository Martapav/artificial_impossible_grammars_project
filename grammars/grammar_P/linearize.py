"""Token-list-to-string linearization for Grammar P.

Renders the flat token list in surface order. Transformations have already
reordered the list before this module is called. The output format is
identical to Grammar H's (``form#infl#infl`` words joined by spaces) so the
two grammars share one vocabulary and one tokenizer (§10, §11).

Spec reference: experiment.md §1 (morpheme order), §10 (tokenization).
"""

from __future__ import annotations

from .tokens import Token, is_gap
from .rules import SEPARATOR, NULL_MORPHEME


def to_string(toks: list[Token]) -> str:
    """Return the full surface string (one space between words)."""
    return " ".join(render(t) for t in toks)


def render(tok: Token) -> str:
    """Produce the surface form for one token.

    Inflecting tokens render as ``form # infl_val [# infl_val ...]`` in the
    order given by the lexicon item's ``inflectional_slots``; a slot listed on
    the item but absent from ``feats`` is omitted (this is how a compound-tense
    CAT3 surfaces bare). Non-inflecting tokens (CAT4, CAT5, CAT6, CAT8, CAT9)
    render as their bare form. A GAP renders as the null morpheme.
    """
    if is_gap(tok):
        return NULL_MORPHEME
    form = tok.lex["form"]
    slots = tok.lex.get("inflectional_slots", [])
    parts = [form] + [tok.feats[s] for s in slots if s in tok.feats]
    if len(parts) == 1:
        return form
    return SEPARATOR.join(parts)
