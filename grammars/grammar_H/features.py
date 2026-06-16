"""Inflection assignment and head-driven agreement for Grammar H.

Mutates node.feats dicts in place after tree construction.
"""

from __future__ import annotations

import random

from .nodes import Node
from .rules import INFL1_NUMBER_TO_CAT4


def assign(tree: Node, lex: dict, rng: random.Random) -> None:
    """Entry point for inflectional feature assignment."""
    _assign_type0(tree, lex, rng)


def _assign_type0(node: Node, lex: dict, rng: random.Random) -> None:
    subj = _fc(node, role="subject")
    vp = _fc(node, label="Type2")
    _assign_type1(subj, lex, rng)
    _assign_type2(vp, lex, rng, subj_feats=_head_feats(subj))


def _assign_type1(node: Node, lex: dict, rng: random.Random) -> None:
    head = _fc(node, role="head")
    g = head.lex["inherent_gender"]
    n = rng.choice(["3", "4"])
    head.feats = {"INFL1_number": n, "INFL1_gender": g}
    node.feats = {"INFL1_number": n, "INFL1_gender": g}

    for child in node.children:
        if child.label == "CAT2":
            child.feats = {"INFL1_number": n, "INFL1_gender": g}
        elif child.label == "CAT4_SLOT":
            _fill_cat4(child, lex, n)
        elif child.label == "Type3":
            # Recurse into the embedded T1dep (second child of a Type3).
            _assign_type1(child.children[1], lex, rng)
        elif child.label == "Type4":
            # Recurse into the embedded Type0 (second child of a Type4).
            _assign_type0(child.children[1], lex, rng)


def _fill_cat4(node: Node, lex: dict, infl1_number: str) -> None:
    """Replace CAT4_SLOT with a real CAT4 terminal matching the head's number."""
    target_num = INFL1_NUMBER_TO_CAT4[infl1_number]
    candidates = [x for x in lex["cat4"] if x["number"] == target_num]
    node.lex = candidates[0]
    node.label = "CAT4"
    node.head_cat = "CAT4"


def _assign_type2(node: Node, lex: dict, rng: random.Random, subj_feats: dict) -> None:
    """Assign inflections inside a Type2.

    Two cases depending on tense form:
      simple   — INFL1 + INFL3 sit on the CAT3 head.
      compound — CAT3 stays bare; INFL1 + INFL3 sit on CAT3AUX.
    (Spec §4.1.)
    """
    head = _fc(node, role="head")    # CAT3 terminal
    aux = _maybe_fc(node, label="CAT3AUX")
    t = rng.choice(["7", "8", "9"])

    if aux is not None:
        # Compound tense: CAT3 is bare; CAT3AUX carries INFL1 + INFL3.
        head.feats = {}
        aux.feats = {**subj_feats, "INFL3_tense": t}
        node.feats = dict(aux.feats)
    else:
        # Simple tense: CAT3 carries INFL1 + INFL3.
        head.feats = {**subj_feats, "INFL3_tense": t}
        node.feats = dict(head.feats)

    for child in node.children:
        if child.label == "Type1" and child.role == "object":
            _assign_type1(child, lex, rng)
        elif child.label == "Type3":
            # PP-under-VP: recurse into the embedded T1dep.
            _assign_type1(child.children[1], lex, rng)


def _head_feats(type1_node: Node) -> dict:
    """INFL1 features off a Type1's CAT1 head — the agreement source for the VP."""
    return _fc(type1_node, role="head").feats


def _fc(node: Node, *, role: str = None, label: str = None) -> Node:
    """First direct child matching role and/or label; raises on miss (structural bug)."""
    for c in node.children:
        if (role is None or c.role == role) and (label is None or c.label == label):
            return c
    raise ValueError(f"child not found: role={role!r}, label={label!r} in {node.label!r}")


def _maybe_fc(node: Node, *, role: str = None, label: str = None) -> Node | None:
    """Like _fc but returns None instead of raising — for optional children (e.g. CAT3AUX)."""
    for c in node.children:
        if (role is None or c.role == role) and (label is None or c.label == label):
            return c
    return None
