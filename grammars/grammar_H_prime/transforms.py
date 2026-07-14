"""Transformations for Grammar H' — identical to Grammar H except one wh clause.

H' keeps Grammar H's binding, auxiliary movement, and wh clause (ii) (a nominal
inside a Type3 dominated by a Type4 — structural). Only wh clause (i) is
swapped: instead of Grammar H's *matrix object* (structural), H' licenses the
*third CAT1-position* of the surface string (positional, = Grammar P's clause
(i)). The swap axis is deliberately the one licensing notion with no local
surface shadow (audit §F): the matrix object's ordinal shifts with
subject-internal complements, embedding, and adverbs, so H's clause (i) and
H''s clause (i') genuinely disagree on natural draws in both directions.

The ordinal scan counts every CAT1 and CAT1PRON terminal as a CAT1-position
(inflection morphemes are transparent). The licensed position maps to the
Type1 constituent whose head occupies it (or to the CAT1PRON itself);
CAT1PRON-Refl never fronts. H' fronts the full constituent (H granularity).
"""

from __future__ import annotations

import random

from grammars.grammar_H.nodes import Node, is_terminal
from grammars.grammar_H.lexicon import cat8_wh
from grammars.grammar_H.build import build_terminal
# Everything except wh-movement is Grammar H's own dispatcher; the annotated
# nominal collector is shared (H' combines its clause-(ii) flag with its own
# positional clause (i')).
from grammars.grammar_H.transforms import (
    apply as h_apply,
    _wh_targetable_nominals,
)


def apply(tree: Node, phenomenon: str, rng: random.Random,
          lex: dict, counter: list, bind_site: str | None = None) -> str:
    """Dispatch to the transformation and return the refined phenomenon label.

    Only wh-movement is H''s own; every other phenomenon (binding with its
    ``bind_site``, aux, neutral) is delegated verbatim to Grammar H's
    dispatcher — the single-rule difference is enforced right here.
    """
    if phenomenon == "wh_movement":
        return _wh_movement(tree, rng, lex, counter)
    return h_apply(tree, phenomenon, rng, lex, counter, bind_site=bind_site)


# ── Wh-movement: positional clause (i') + structural clause (ii) ──────────────


def _wh_movement(tree: Node, rng: random.Random, lex: dict, counter: list) -> str:
    """Front a LICENSED nominal to sentence-initial position after CAT8.

    Licensed iff (i') its head occupies the third CAT1-position of the
    surface string, or (ii) it sits inside a Type3 dominated by a Type4.
    Returns "wh_movement_skipped" when no licensed target exists.
    """
    candidates = _collect_wh_candidates(tree)
    if not candidates:
        return "wh_movement_skipped"

    target, parent, parent_idx = rng.choice(candidates)
    target.role = "fronted"

    gap = Node(
        label="GAP", head_cat="GAP", lex=None, feats={},
        children=[], role="gap", licensor_id=None, node_id=target.node_id,
    )
    parent.children[parent_idx] = gap

    cat8_node = build_terminal("CAT8", cat8_wh(lex), "wh_marker", counter)
    tree.children.insert(0, target)
    tree.children.insert(0, cat8_node)
    return "wh_movement"


def _collect_wh_candidates(tree: Node) -> list[tuple[Node, Node, int]]:
    """All nominals in a licensed H' wh position.

    Clause (i') is evaluated on the surface terminal sequence; clause (ii)
    is Grammar H's structural flag from the shared collector. A nominal
    occupies the third CAT1-position iff its head terminal (Type1) or the
    terminal itself (CAT1PRON) is the third CAT1/CAT1PRON word of the string.
    """
    nominals = _wh_targetable_nominals(tree)
    third = _third_cat1_position_terminal(tree)
    return [
        (c["node"], c["parent"], c["idx"])
        for c in nominals
        if c["in_type3_under_t4"] or (third is not None
                                      and _head_terminal(c["node"]) is third)
    ]


def _third_cat1_position_terminal(tree: Node) -> Node | None:
    """The terminal occupying the third CAT1-position, or None if there is none.

    A CAT1-position is any CAT1 or CAT1PRON terminal, in surface order;
    inflection values live on their word and are not separate positions.
    """
    positions = [
        t for t in _flatten_terminals(tree)
        if t.label in ("CAT1", "CAT1PRON")
    ]
    return positions[2] if len(positions) >= 3 else None


def _head_terminal(nominal: Node) -> Node:
    """The head CAT1 terminal of a Type1; a CAT1PRON is its own head."""
    if nominal.label == "Type1":
        return next(c for c in nominal.children if c.role == "head")
    return nominal


def _flatten_terminals(node: Node) -> list[Node]:
    """In-order list of terminal (word) nodes — the surface word sequence."""
    if is_terminal(node):
        return [node]
    out: list[Node] = []
    for child in node.children:
        out.extend(_flatten_terminals(child))
    return out
