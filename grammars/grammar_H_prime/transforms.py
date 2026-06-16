"""Transformations for Grammar H' — identical to Grammar H except wh-movement.

H' keeps Grammar H's binding, auxiliary movement, and the subject-phrase and
adjunct islands. Only the complex-NP island is swapped: instead of a structural
check it uses a left-to-right surface scan.

H' complex-NP rule: fronting of x is blocked if the word immediately before x is
a CAT6 whose own immediately preceding word is a CAT1. "Word" means a lexical
item; the inflection morphemes attached to each word are transparent to the scan.
This is the deliberately imperfect positional approximation that diverges from
Grammar L's structural rule (see grammar_L_prime).
"""

from __future__ import annotations

import random

from grammars.grammar_H.nodes import Node, is_terminal
from grammars.grammar_H.lexicon import cat8_wh
from grammars.grammar_H.build import build_terminal
# Binding and auxiliary movement are unchanged from Grammar H.
from grammars.grammar_H.transforms import _pronominal_sub, _aux_movement


def apply(tree: Node, phenomenon: str, rng: random.Random,
          lex: dict, counter: list) -> str:
    """Dispatch to the transformation and return the refined phenomenon label."""
    if phenomenon == "anaphoric_binding":
        return _pronominal_sub(tree, rng, lex)
    if phenomenon == "wh_movement":
        return _wh_movement(tree, rng, lex, counter)
    if phenomenon == "auxiliary_movement":
        return _aux_movement(tree)
    return phenomenon  # neutral


# ── Wh-movement (structural subject/adjunct islands + positional complex-NP) ──


def _wh_movement(tree: Node, rng: random.Random, lex: dict, counter: list) -> str:
    """Front a licit Type1 to sentence-initial position after a CAT8 marker.

    A target is eligible unless it is the matrix subject or inside a Type4
    (both structural), or blocked by the positional complex-NP scan. Returns
    "wh_movement_skipped" when nothing is frontable.
    """
    structural = _collect_wh_candidates(tree)  # subject + adjunct islands only
    if not structural:
        return "wh_movement_skipped"

    flat = _flatten_terminals(tree)
    licit = [
        cand for cand in structural
        if not _positional_complex_np_block(cand[0], flat)
    ]
    if not licit:
        return "wh_movement_skipped"

    target, parent, parent_idx = rng.choice(licit)
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
    """Type1 nodes not blocked by the subject-phrase or adjunct (Type4) islands.

    The complex-NP island is NOT applied here — in H' it is positional and
    checked separately by ``_positional_complex_np_block``.
    """
    results: list[tuple[Node, Node, int]] = []

    def walk(node: Node, parent: Node | None, idx_in_parent: int | None,
             inside_type4: bool, is_matrix_subject: bool):
        if node.label == "Type1":
            if not (is_matrix_subject or inside_type4):
                results.append((node, parent, idx_in_parent))

        for i, child in enumerate(node.children):
            child_in_t4 = inside_type4 or (node.label == "Type4")
            child_is_matrix_subj = (
                node.label == "Type0"
                and node.role == "root"
                and child.label == "Type1"
                and child.role == "subject"
            )
            walk(child, node, i, child_in_t4, child_is_matrix_subj)

    walk(tree, parent=None, idx_in_parent=None,
         inside_type4=False, is_matrix_subject=False)
    return results


def _positional_complex_np_block(target: Node, flat: list[Node]) -> bool:
    """True iff the H' positional complex-NP rule blocks fronting of ``target``.

    Let ``i`` be the index of the target's head CAT1 (its first terminal) in the
    word sequence ``flat``. Blocks iff ``flat[i-1]`` is a CAT6 and ``flat[i-2]``
    is a CAT1.
    """
    head = _first_terminal(target)
    if head is None:
        return False
    i = next((k for k, n in enumerate(flat) if n is head), None)
    if i is None or i < 2:
        return False
    return flat[i - 1].label == "CAT6" and flat[i - 2].label == "CAT1"


def _flatten_terminals(node: Node) -> list[Node]:
    """In-order list of terminal (word) nodes — the surface word sequence."""
    if is_terminal(node):
        return [node]
    out: list[Node] = []
    for child in node.children:
        out.extend(_flatten_terminals(child))
    return out


def _first_terminal(node: Node) -> Node | None:
    """Leftmost terminal in the subtree (the head CAT1 of a Type1)."""
    if is_terminal(node):
        return node
    for child in node.children:
        t = _first_terminal(child)
        if t is not None:
            return t
    return None
