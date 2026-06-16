"""Syntactic transformations for Grammar H.

Mutates the tree in place after feature assignment.
All transformations operate on the tree, not the surface string.

Spec reference:
  §7.1 (Anaphoric binding — Principles A and B),
  §7.2 (Auxiliary movement — matrix-only fronting),
  §7.3 (Wh-movement and structural islands: subject, adjunct, complex-NP).
"""

from __future__ import annotations

import random

from .nodes import Node
from .lexicon import cat1pron_item, cat8_wh
from .build import fresh_id, build_terminal


def apply(tree: Node, phenomenon: str, rng: random.Random,
          lex: dict, counter: list) -> str:
    """Dispatch to the transformation and return the refined phenomenon label.

    Returns the input phenomenon unchanged for cases where no tree mutation
    is required ("neutral") or where the transformation cannot apply to the
    drawn structure (label suffixed with "_skipped" so the corpus builder can
    drop it).
    """
    if phenomenon == "anaphoric_binding":
        return _pronominal_sub(tree, rng, lex)
    if phenomenon == "wh_movement":
        return _wh_movement(tree, rng, lex, counter)
    if phenomenon == "auxiliary_movement":
        return _aux_movement(tree)
    # "neutral": no transformation.
    return phenomenon


# Anaphoric binding


def _pronominal_sub(tree: Node, rng: random.Random, lex: dict) -> str:
    """Replace the object Type1 with an inflected CAT1PRON (Refl or Pron).

    CAT1PRON inflects for gender and number only — no person distinction is
    implemented (spec §2.2).
    Refl (Principle A): bound in binding domain; copies subject's gender, number.
    Pron (Principle B): free in binding domain; features drawn independently.
    Binding domain in Grammar H = the matrix clause (Type0).

    Returns "anaphoric_binding_skipped" if there is no object position
    (intransitive VP), since the build phase only forces transitivity when
    the binding phenomenon was pre-selected.
    """
    type2 = _fc(tree, label="Type2")
    obj_idx_obj = _find_indexed_or_none(type2, role="object")
    if obj_idx_obj is None:
        return "anaphoric_binding_skipped"
    obj_idx, obj = obj_idx_obj

    subj = _fc(tree, role="subject")
    subj_head_feats = _fc(subj, role="head").feats

    use_refl = rng.random() < 0.5
    subclass = "Refl" if use_refl else "Pron"
    pron_item = cat1pron_item(lex, subclass)

    if use_refl:
        feats = {
            "INFL1_number": subj_head_feats["INFL1_number"],
            "INFL1_gender": subj_head_feats["INFL1_gender"],
        }
    else:
        feats = {
            "INFL1_number": rng.choice(["3", "4"]),
            "INFL1_gender": rng.choice(["1", "2"]),
        }

    pron_node = Node(
        label="CAT1PRON", head_cat="CAT1PRON", lex=pron_item, feats=feats,
        children=[], role="object", licensor_id=None, node_id=obj.node_id,
    )
    type2.children[obj_idx] = pron_node
    return "anaphoric_binding_refl" if use_refl else "anaphoric_binding_pron"


# Auxiliary movement


def _aux_movement(tree: Node) -> str:
    """Front the matrix CAT3AUX to the position immediately preceding the subject.

    Spec §7.2: only a CAT3AUX in the matrix Type2 is eligible. CAT3AUX items
    inside Type4 (embedded relative clauses) are not fronted.

    Returns "auxiliary_movement_skipped" if the matrix Type2 has no CAT3AUX.
    """
    type2 = _fc(tree, label="Type2")
    aux_idx_aux = _find_indexed_or_none(type2, label="CAT3AUX")
    if aux_idx_aux is None:
        return "auxiliary_movement_skipped"
    aux_idx, aux = aux_idx_aux
    del type2.children[aux_idx]
    tree.children.insert(0, aux)
    return "auxiliary_movement"


# Wh-movement


def _wh_movement(tree: Node, rng: random.Random, lex: dict, counter: list) -> str:
    """Front a licit Type1 to sentence-initial position after a CAT8(wh) marker.

    Eligible targets are Type1 constituents that are NOT blocked by any of
    the three structural islands (spec §7.3):
      1. Subject-phrase extraction (matrix Type1 in Type0).
      2. Adjunct island (Type1 inside Type4).
      3. Complex NP island (Type1 inside a Type3 that is itself a dependent
         of a CAT1 head — i.e. PP-under-NP). PP-under-VP is fine.

    Implementation: collect all Type1 nodes with their parents and walk up to
    classify the position. Pick uniformly from licit targets. If no licit
    target exists, return "wh_movement_skipped".
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

    # Move target to sentence-initial position, immediately preceded by CAT8.
    tree.children.insert(0, target)
    tree.children.insert(0, cat8_node)
    return "wh_movement"


def _collect_wh_candidates(tree: Node) -> list[tuple[Node, Node, int]]:
    """Return all Type1 nodes eligible for wh-fronting (Grammar H).

    Each element is (target, parent, index_in_parent). The walker tracks
    ancestry so that:
      - the matrix subject Type1 is rejected (subject-phrase island);
      - any Type1 strictly inside a Type4 is rejected (adjunct island);
      - any Type1 strictly inside a Type3 with attachment="np" is rejected
        (complex-NP island).
    """
    results: list[tuple[Node, Node, int]] = []

    def walk(node: Node, parent: Node | None, idx_in_parent: int | None,
             inside_type4: bool, inside_pp_np: bool, is_matrix_subject: bool):
        if node.label == "Type1":
            if not (is_matrix_subject or inside_type4 or inside_pp_np):
                results.append((node, parent, idx_in_parent))

        for i, child in enumerate(node.children):
            child_in_t4 = inside_type4 or (node.label == "Type4")
            # A Type1 dominated by a Type3 with attachment="np" is in a
            # complex-NP island. Once inside, all deeper Type1s are blocked too.
            child_in_np = inside_pp_np or (
                node.label == "Type3" and node.feats.get("attachment") == "np"
            )
            # Only the immediate subject of the matrix Type0 counts as the
            # matrix subject; descendents do not inherit the flag.
            child_is_matrix_subj = (
                node.label == "Type0"
                and node.role == "root"
                and child.label == "Type1"
                and child.role == "subject"
            )
            walk(child, node, i, child_in_t4, child_in_np, child_is_matrix_subj)

    walk(tree, parent=None, idx_in_parent=None,
         inside_type4=False, inside_pp_np=False, is_matrix_subject=False)
    return results


# Helpers 


def _fc(node: Node, *, role: str = None, label: str = None) -> Node:
    for c in node.children:
        if (role is None or c.role == role) and (label is None or c.label == label):
            return c
    raise ValueError(f"child not found: role={role!r}, label={label!r} in {node.label!r}")


def _find_indexed_or_none(node: Node, *, role: str = None, label: str = None):
    for i, c in enumerate(node.children):
        if (role is None or c.role == role) and (label is None or c.label == label):
            return i, c
    return None
