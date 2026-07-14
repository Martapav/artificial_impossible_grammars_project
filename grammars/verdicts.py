"""Clause-(i) wh licensing verdicts for the H′-vs-L′ contrast.

The two mixed grammars differ on exactly one wh clause — clause (i), the
swap axis of the audit's §F design. On any given nominal they decide whether
that clause licenses fronting by different information:

  * **H′ (positional)** — licenses iff the nominal occupies the **third
    CAT1-position** of the surface string (CAT1 and CAT1PRON both count;
    inflection morphemes are transparent).
  * **L′ (structural)** — licenses iff the nominal is the **matrix object**,
    read off the generation history (``role == "object"`` in the clause with
    id 0), with no reference to surface position.

A configuration is a **divergence** case when the two verdicts disagree:
a bare-transitive matrix object (2nd position) is L′-licensed/H′-not; an
embedded or subject-internal nominal at ordinal 3 is H′-licensed/L′-not.
Each grammar's *other* clause (H′: inside a Type3 dominated by a Type4;
L′: first CAT1-position after a CAT5) is inherited from its parent and is
not part of the probe axis.

These functions are category-level and inflection-transparent: they take a
list of bare category labels (``"CAT1"``, ``"CAT6"``, …) plus the target's
index, and the target's generation-history facts. They are the single source
of truth for probe labeling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from grammars.grammar_H.nodes import Node, is_terminal
from grammars.grammar_P.tokens import Token, is_gap

_NOMINAL_CATS = ("CAT1", "CAT1PRON")


def hprime_licenses(categories: List[str], target_index: int) -> bool:
    """True iff H′'s clause (i′) licenses the target at ``target_index``.

    The clause licenses exactly the third CAT1-position of the string
    (CAT1 and CAT1PRON both count as positions).
    """
    if not 0 <= target_index < len(categories):
        raise IndexError(
            f"target_index {target_index} out of range for {len(categories)} categories"
        )
    if categories[target_index] not in _NOMINAL_CATS:
        raise ValueError(
            f"target at {target_index} is {categories[target_index]!r}, not a nominal"
        )
    positions = [i for i, c in enumerate(categories) if c in _NOMINAL_CATS]
    return len(positions) >= 3 and positions[2] == target_index


def lprime_licenses(role: str, clause_id: int) -> bool:
    """True iff L′'s clause (i″) licenses the target: it is the matrix object."""
    return role == "object" and clause_id == 0


def verdicts(
    categories: List[str], target_index: int, role: str, clause_id: int
) -> Dict[str, object]:
    """Return both grammars' clause-(i) verdicts and whether they diverge.

    Keys: ``hprime_verdict`` / ``lprime_verdict`` (``"license"`` | ``"not"``)
    and ``divergence`` (bool). Verdicts are about clause (i) only — a nominal
    neither verdict licenses may still front via the grammar's other clause.
    """
    h_lic = hprime_licenses(categories, target_index)
    l_lic = lprime_licenses(role, clause_id)
    return {
        "hprime_verdict": "license" if h_lic else "not",
        "lprime_verdict": "license" if l_lic else "not",
        "divergence": h_lic != l_lic,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Verdict recovery from a FRONTED item (natural sampling — no hand-built
# probes). Clause-(i) licensing is evaluated on the base string, so the
# fronted block is spliced back at its gap before counting CAT1-positions.
# ─────────────────────────────────────────────────────────────────────────────


def recover_tree(tree: Node) -> Optional[Tuple[List[str], int, str, int]]:
    """(base_categories, target_index, role, clause_id) for a fronted H-family
    tree, or None if the tree is not a fronted wh item.

    The fronted constituent is spliced back at its GAP; the target index is
    its head terminal's base position. ``role``/``clause_id`` encode the L′
    facts: ("object", 0) iff the gap sits in the matrix Type2's object slot
    (the object is the only nominal direct child of a Type2); ("other", -1)
    otherwise.
    """
    top = list(tree.children)
    if len(top) < 2 or top[0].head_cat != "CAT8" or top[1].role != "fronted":
        return None
    fronted = top[1]

    seq: List[str] = []
    target_index: Optional[int] = None

    def emit(n: Node):
        nonlocal target_index
        if n.label == "GAP":
            target_index = len(seq)
            seq.extend(t.head_cat for t in _terminals(fronted))
            return
        if is_terminal(n):
            seq.append(n.head_cat)
            return
        for c in n.children:
            emit(c)

    for child in top[2:]:  # skip CAT8 and the fronted constituent
        emit(child)
    if target_index is None:
        return None

    gap_parent: Optional[Node] = None

    def find_gap_parent(n: Node):
        nonlocal gap_parent
        for c in n.children:
            if c.label == "GAP":
                gap_parent = n
            find_gap_parent(c)

    find_gap_parent(tree)
    matrix_t2 = next((c for c in top if c.label == "Type2"), None)
    is_matrix_object = gap_parent is not None and gap_parent is matrix_t2
    return seq, target_index, ("object" if is_matrix_object else "other"), (
        0 if is_matrix_object else -1
    )


def _terminals(node: Node) -> List[Node]:
    if is_terminal(node):
        return [node]
    out: List[Node] = []
    for c in node.children:
        out.extend(_terminals(c))
    return out


def recover_token(toks: List[Token]) -> Optional[Tuple[List[str], int, str, int]]:
    """(base_categories, target_index, role, clause_id) for a fronted P-family
    token list, or None if it is not a fronted wh item.

    The fronted block (head + contiguous CAT4/CAT2 after CAT8) is spliced
    back at its GAP; role/clause_id are the head token's generation-history
    record.
    """
    if not toks or toks[0].cat != "CAT8" or len(toks) < 2:
        return None
    head = toks[1]
    end = 2
    while end < len(toks) and toks[end].cat in ("CAT4", "CAT2") \
            and toks[end].clause_id == head.clause_id:
        end += 1
    block = toks[1:end]
    rest = toks[end:]
    gi = next((i for i, t in enumerate(rest) if is_gap(t)), None)
    if gi is None:
        return None
    base = rest[:gi] + block + rest[gi + 1:]
    return [t.cat for t in base], gi, head.role, head.clause_id


def wh_verdicts_tree(tree: Node) -> Optional[Dict]:
    """Both clause-(i) verdicts (+ target_index) for a fronted tree, or None."""
    rec = recover_tree(tree)
    if rec is None:
        return None
    cats, ti, role, clause_id = rec
    v = verdicts(cats, ti, role, clause_id)
    v["target_index"] = ti
    return v


def wh_verdicts_token(toks: List[Token]) -> Optional[Dict]:
    """Both clause-(i) verdicts (+ target_index) for a fronted token list, or None."""
    rec = recover_token(toks)
    if rec is None:
        return None
    cats, ti, role, clause_id = rec
    v = verdicts(cats, ti, role, clause_id)
    v["target_index"] = ti
    return v
