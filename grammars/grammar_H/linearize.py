"""Tree-to-string linearization for Grammar H.

Performs an in-order traversal of the tree and renders each terminal
as its surface morpheme string. Transformations have already reordered
the tree before this module is called.

Spec reference: Morpheme order; separator-based tokenization.
"""

from __future__ import annotations

from .nodes import Node, is_terminal
from .rules import SEPARATOR, NULL_MORPHEME


def to_string(tree: Node) -> str:
    """Return the full surface string (words joined by spaces)."""
    return " ".join(to_tokens(tree))


def to_tokens(node: Node) -> list:
    """Recursive in-order traversal. Returns a flat list of surface token strings."""
    if node.label == "GAP":
        return [NULL_MORPHEME]
    if is_terminal(node):
        return [render(node)]
    tokens = []
    for child in node.children:
        tokens.extend(to_tokens(child))
    return tokens


def render(node: Node) -> str:
    """Produce the surface form for a terminal node.

    Inflecting categories: stem + SEPARATOR + infl_val [+ SEPARATOR + infl_val ...]
    Non-inflecting categories (CAT4, CAT5, CAT6, CAT8, CAT9): stem only.
    Morpheme order (from inflectional_slots in the lexicon item):
      CAT1      → form # INFL1_number # INFL1_gender
      CAT1PRON  → form # INFL1_number # INFL1_gender
      CAT2      → form # INFL1_number # INFL1_gender
      CAT3      → form # INFL1_number # INFL1_gender # INFL3_tense
      CAT3AUX   → form # INFL1_number # INFL1_gender # INFL3_tense

    A slot listed on the lexicon item but absent from node.feats is omitted.
    This is how compound-tense CAT3 surfaces bare: features.py leaves its
    feats empty, and only CAT3AUX carries the inflections (spec §4.1).
    """
    form = node.lex["form"]
    slots = node.lex.get("inflectional_slots", [])
    if not slots:
        return form
    parts = [form] + [node.feats[slot] for slot in slots if slot in node.feats]
    if len(parts) == 1:
        return form
    return SEPARATOR.join(parts)
