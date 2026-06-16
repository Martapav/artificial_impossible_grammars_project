"""Tree node dataclass for Grammar H constituent trees.

Spec reference: experiment.md — Syntagms (hierarchical grammars).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    label: str              # "Type0","Type1","Type2","Type3","CAT4_SLOT","GAP" or terminal category
    head_cat: str           # dominant lexical category
    lex: dict | None        # lexicon item dict — terminals only; None for non-terminals
    feats: dict             # inflectional features; populated by features.py
    children: list          # List[Node]; empty for terminals
    role: str               # "subject","object","head","modifier","adjunct","pp","gap","aux","wh_marker","fronted"
    licensor_id: int | None # node_id of enclosing Type1 head when this is a Type3; else None
    node_id: int


def is_terminal(node: Node) -> bool:
    return node.lex is not None
