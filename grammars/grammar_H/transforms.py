"""Syntactic transformations for Grammar H (reformalized).

Mutates the tree in place after feature assignment.
All transformations operate on the tree, not the surface string.

Reformalization reference: grammars/LINEAR_HIERARCHICAL_TRIVIALITY_AUDIT.md
(assessment section) and grammars/TRANSFORM_RULES.md.

  Anaphoric binding — derivational duplicate-CAT1 formalization:
    Refl: bound within the minimal complete transitive clause NUCLEUS
          (subject head, verb, object head of one Type0). The object of the
          licensing nucleus (matrix or embedded) is substituted, copying the
          nucleus subject's features. Obligatory: no final string may contain
          the same CAT1 stem as subject and object of one nucleus.
    Pron: free within its nucleus — the coreferential pair spans two distinct
          nuclei (or involves a position outside every nucleus, e.g. a PP
          complement). Either occurrence may be substituted (majority: the
          second); features are copied from the surviving antecedent.
          Substitution is optional at the grammar level: cross-nucleus stem
          duplicates MAY surface unsubstituted (background/neutral items).

  Auxiliary movement — matrix-only fronting (unchanged): the matrix CAT3AUX
    fronts; an aux only inside an embedded clause is a structural blocking
    case (skip).

  Wh-movement — LICENSING formalization (audit §F, final): a nominal may
    front iff it is
      (i)  the object of the matrix clause, or
      (ii) inside a Type3 (PP) that is itself dominated by a Type4 (any
           attachment, any depth of nesting).
    Embedded subjects no longer license (their surface shadow — first CAT1
    after CAT9 — was perfect); Type3s outside every Type4 no longer license
    (their shadow — CAT1 after CAT6 — was a bigram). Clause (ii)'s remaining
    shadow requires tracking the unmarked right edge of the Type4.
    CAT1PRON-Pron behaves like a CAT1 (may front from licensed positions);
    CAT1PRON-Refl never fronts (an anaphor stays in its licensing position).
"""

from __future__ import annotations

import random

from .nodes import Node
from .lexicon import cat1pron_item, cat8_wh
from .build import build_terminal
from .rules import P_BG_BIND, P_BIND_EMB, P_PRON_FIRST


def apply(tree: Node, phenomenon: str, rng: random.Random,
          lex: dict, counter: list, bind_site: str | None = None) -> str:
    """Dispatch to the transformation and return the refined phenomenon label.

    Returns the input phenomenon unchanged for cases where no tree mutation
    is required ("neutral") or the label suffixed with "_skipped" when the
    transformation cannot apply to the drawn structure (the corpus builder
    drops those). ``bind_site`` ("matrix"/"embedded"/None) fixes the refl
    nucleus placement decided at generation time.
    """
    if phenomenon == "anaphoric_binding":  # unrefined dispatch (gen items)
        phenomenon = ("anaphoric_binding_refl" if rng.random() < 0.5
                      else "anaphoric_binding_pron")
    if phenomenon == "anaphoric_binding_refl":
        return _refl(tree, rng, lex, site=bind_site)
    if phenomenon == "anaphoric_binding_pron":
        return _pron(tree, rng, lex)
    if phenomenon == "wh_movement":
        return _wh_movement(tree, rng, lex, counter)
    if phenomenon == "auxiliary_movement":
        return _aux_movement(tree)
    # "neutral": no transformation.
    return phenomenon


# Anaphoric binding


def _type0s(node: Node, acc: list | None = None) -> list[Node]:
    """The matrix Type0 (if ``node`` is it) plus every embedded Type0, tree order."""
    if acc is None:
        acc = [node] if node.label == "Type0" else []
    for child in node.children:
        if child.label == "Type4":
            acc.append(child.children[1])  # (CAT9 head, embedded Type0)
        _type0s(child, acc)
    return acc


def _nucleus(t0: Node):
    """(subject_node, object_node_or_None, type2_or_None) of a Type0's nucleus.

    The nucleus is ONLY the subject, the verb, and the object (D2): PP
    complements, modifiers, and anything inside Type3/Type4 are outside every
    nucleus. Subject/object may be CAT1PRON after a prior substitution.
    """
    subj = next((c for c in t0.children if c.role == "subject"), None)
    t2 = next((c for c in t0.children if c.label == "Type2"), None)
    obj = None
    if t2 is not None:
        obj = next((c for c in t2.children if c.role == "object"), None)
    return subj, obj, t2


def _head_feats_of(nominal: Node) -> dict:
    """INFL1 features of a nominal: Type1 → its head CAT1; CAT1PRON → itself."""
    if nominal.label == "Type1":
        head = next(c for c in nominal.children if c.role == "head")
        return head.feats
    return nominal.feats


def _refl(tree: Node, rng: random.Random, lex: dict,
          site: str | None = None) -> str:
    """Reflexive substitution in a licensing nucleus (matrix or embedded).

    Derivationally the nucleus hosts the same CAT1 as subject and object; the
    grammar forces substitution of the SECOND occurrence (the object), so the
    duplicate exists only as an intermediate state. Surface effect: the object
    of the chosen nucleus becomes CAT1PRON-Refl copying the nucleus subject's
    number/gender. The antecedent must be a full CAT1 (no pron-pron chains).

    ``site``: "matrix" / "embedded" restricts the nucleus; None prefers the
    matrix nucleus with probability 1 - P_BIND_EMB when both are available.
    """
    eligible: list[tuple[Node, Node, Node, int, bool]] = []
    for t0 in _type0s(tree):
        subj, obj, t2 = _nucleus(t0)
        if subj is None or t2 is None or obj is None:
            continue
        if subj.label != "Type1" or obj.label != "Type1":
            continue  # no pron antecedent, no double substitution
        obj_idx = t2.children.index(obj)
        eligible.append((t0, subj, obj, obj_idx, t0 is tree))

    if site == "matrix":
        eligible = [e for e in eligible if e[4]]
    elif site == "embedded":
        eligible = [e for e in eligible if not e[4]]
    if not eligible:
        return "anaphoric_binding_skipped"

    if site is None:
        matrix = [e for e in eligible if e[4]]
        embedded = [e for e in eligible if not e[4]]
        if matrix and (not embedded or rng.random() >= P_BIND_EMB):
            eligible = matrix
        else:
            eligible = embedded

    t0, subj, obj, obj_idx, _ = rng.choice(eligible)
    subj_feats = _head_feats_of(subj)
    feats = {"INFL1_number": subj_feats["INFL1_number"],
             "INFL1_gender": subj_feats["INFL1_gender"]}
    t2 = next(c for c in t0.children if c.label == "Type2")
    t2.children[obj_idx] = Node(
        label="CAT1PRON", head_cat="CAT1PRON", lex=cat1pron_item(lex, "Refl"),
        feats=feats, children=[], role="object", licensor_id=None,
        node_id=obj.node_id,
    )
    return "anaphoric_binding_refl"


def _collect_pron_positions(tree: Node) -> list[dict]:
    """Every full-CAT1 Type1 in linear order, annotated for the Pron rule.

    Each entry: node, parent, idx, t0_id + slot (nucleus membership, or None
    if outside every nucleus), embedded (inside any Type4), subtree_t0 (the
    Type0 the node is subject of, for agreement fixup).
    """
    out: list[dict] = []

    def walk(node: Node, parent: Node | None, idx: int | None,
             inside_t4: bool, t0: Node | None):
        if node.label == "Type1":
            slot = None
            owner = None
            if parent is not None and parent.label == "Type0" and node.role == "subject":
                slot, owner = "subj", parent
            elif parent is not None and parent.label == "Type2" and node.role == "object":
                slot, owner = "obj", t0
            out.append({
                "node": node, "parent": parent, "idx": idx,
                "slot": slot, "owner": owner, "embedded": inside_t4,
            })
        for i, child in enumerate(node.children):
            child_t0 = node if node.label == "Type0" else t0
            child_in_t4 = inside_t4 or node.label == "Type4"
            walk(child, node, i, child_in_t4, child_t0)

    walk(tree, None, None, False, None)
    return out


def _dominates(a: Node, b: Node) -> bool:
    if a is b:
        return True
    return any(_dominates(c, b) for c in a.children)


def _pron(tree: Node, rng: random.Random, lex: dict,
          require_embedded: bool = False) -> str:
    """Pronoun substitution over a cross-nucleus coreferential pair.

    Derivationally the pair carries the same CAT1 (same stem, same number) in
    two positions NOT belonging to the same nucleus; one occurrence is
    substituted by CAT1PRON-Pron copying the other's features. The second
    occurrence is substituted in the majority of cases (1 - P_PRON_FIRST).
    Pairs where one nominal dominates the other are excluded (substituting the
    host would delete the antecedent).

    ``require_embedded``: background-binding scope — at least one member of
    the pair must sit inside a Type4.
    """
    positions = _collect_pron_positions(tree)
    pairs: list[tuple[dict, dict]] = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            a, b = positions[i], positions[j]
            same_nucleus = (
                a["owner"] is not None and a["owner"] is b["owner"]
                and {a["slot"], b["slot"]} == {"subj", "obj"}
            )
            if same_nucleus:
                continue
            if _dominates(a["node"], b["node"]) or _dominates(b["node"], a["node"]):
                continue
            if require_embedded and not (a["embedded"] or b["embedded"]):
                continue
            pairs.append((a, b))
    if not pairs:
        return "anaphoric_binding_skipped"

    first, second = rng.choice(pairs)
    target, antecedent = (
        (first, second) if rng.random() < P_PRON_FIRST else (second, first)
    )
    ant_feats = _head_feats_of(antecedent["node"])
    feats = {"INFL1_number": ant_feats["INFL1_number"],
             "INFL1_gender": ant_feats["INFL1_gender"]}
    pron_node = Node(
        label="CAT1PRON", head_cat="CAT1PRON", lex=cat1pron_item(lex, "Pron"),
        feats=feats, children=[], role=target["node"].role, licensor_id=None,
        node_id=target["node"].node_id,
    )
    target["parent"].children[target["idx"]] = pron_node
    # Substituting a clause subject: the clause's verb carrier already copied
    # the old subject's INFL1 values — realign it with the pron's features.
    if target["slot"] == "subj":
        _fix_clause_agreement(target["owner"], feats)
    return "anaphoric_binding_pron"


def _fix_clause_agreement(t0: Node, feats: dict) -> None:
    """Realign a Type0's verb-carrier INFL1 features after subject substitution."""
    t2 = next((c for c in t0.children if c.label == "Type2"), None)
    if t2 is None:
        return
    for c in t2.children:
        if c.label in ("CAT3", "CAT3AUX") and "INFL1_number" in c.feats:
            c.feats["INFL1_number"] = feats["INFL1_number"]
            c.feats["INFL1_gender"] = feats["INFL1_gender"]
    if "INFL1_number" in t2.feats:
        t2.feats["INFL1_number"] = feats["INFL1_number"]
        t2.feats["INFL1_gender"] = feats["INFL1_gender"]


# Background binding (successor of the embedded enrichment)


def apply_background_binding(
    tree: Node, rng: random.Random, lex: dict, p: float = P_BG_BIND,
) -> None:
    """With probability ``p``, add one embedded-scoped pronominal substitution.

    Applied only to sentences whose OWN phenomenon is not binding, before the
    phenomenon transform, so a background CAT1PRON-Pron can be a wh target.
    Refl: licensed nucleus must be embedded. Pron: at least one member of the
    pair must be inside a Type4. A no-op when the geometry is absent. The
    construction label is unchanged; this is background variation.
    """
    if rng.random() >= p:
        return
    if rng.random() < 0.5:
        _refl(tree, rng, lex, site="embedded")
    else:
        _pron(tree, rng, lex, require_embedded=True)


def _embedded_type0s(node: Node) -> list[Node]:
    """Every Type0 embedded under a Type4, in tree order, at all depths."""
    out: list[Node] = []
    for child in node.children:
        if child.label == "Type4":
            out.append(child.children[1])  # (CAT9 head, embedded Type0)
        out.extend(_embedded_type0s(child))
    return out


# Auxiliary movement


def _aux_movement(tree: Node) -> str:
    """Front the matrix CAT3AUX to the position immediately preceding the subject.

    Spec §7.2: only a CAT3AUX in the matrix Type2 is eligible. CAT3AUX items
    inside Type4 (embedded relative clauses) are not fronted — an item whose
    only aux is embedded is a structural blocking case.

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


# Wh-movement (licensing formalization)


def _wh_movement(tree: Node, rng: random.Random, lex: dict, counter: list) -> str:
    """Front a LICENSED nominal to sentence-initial position after CAT8.

    Licensed positions (see module docstring): the matrix object, and any
    nominal inside a Type3 dominated by a Type4. Everything else — matrix
    subject, embedded arguments outside such a Type3, Type3s attached in the
    matrix clause — never fronts. If no licensed target exists, returns
    "wh_movement_skipped" (structural blocking).
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


def _wh_targetable_nominals(tree: Node) -> list[dict]:
    """Every wh-targetable nominal, in tree order, annotated for licensing.

    A nominal is a Type1 or a CAT1PRON-Pron (CAT1PRON-Refl never fronts).
    Each entry: node, parent, idx (index in parent), and the two structural
    licensing facts:
      matrix_object      — role="object" in the MATRIX Type2;
      in_type3_under_t4  — strictly inside a Type3 that is dominated by a
                           Type4 (a Type3 crossed after a Type4 was crossed).
    Shared with Grammar H' (which combines in_type3_under_t4 with its own
    positional clause instead of matrix_object).
    """
    results: list[dict] = []

    def walk(node: Node, parent: Node | None, idx_in_parent: int | None,
             in_licensed_t3: bool, inside_type4: bool, clause_is_matrix: bool):
        is_nominal = node.label == "Type1" or (
            node.label == "CAT1PRON"
            and node.lex is not None
            and node.lex.get("subclass") == "Pron"
        )
        if is_nominal and parent is not None:
            results.append({
                "node": node, "parent": parent, "idx": idx_in_parent,
                "matrix_object": (node.role == "object" and clause_is_matrix
                                  and parent.label == "Type2"),
                "in_type3_under_t4": in_licensed_t3,
            })

        for i, child in enumerate(node.children):
            child_in_t4 = inside_type4 or node.label == "Type4"
            # A Type3 licenses its interior only when a Type4 dominates it.
            child_in_lt3 = in_licensed_t3 or (
                node.label == "Type3" and inside_type4
            )
            # Crossing into a Type4 means every deeper Type0 is embedded.
            child_matrix = clause_is_matrix and node.label != "Type4"
            walk(child, node, i, child_in_lt3, child_in_t4, child_matrix)

    walk(tree, parent=None, idx_in_parent=None,
         in_licensed_t3=False, inside_type4=False, clause_is_matrix=True)
    return results


def _collect_wh_candidates(tree: Node) -> list[tuple[Node, Node, int]]:
    """All nominals in a licensed wh position (Grammar H, licensing rule).

    Each element is (target, parent, index_in_parent). Licensed iff:
      (i)  object of the MATRIX clause, or
      (ii) inside a Type3 dominated by a Type4.
    """
    return [
        (c["node"], c["parent"], c["idx"])
        for c in _wh_targetable_nominals(tree)
        if c["matrix_object"] or c["in_type3_under_t4"]
    ]


# Duplicate screen (grammar constraint, checked by generate.py)


def has_nucleus_duplicate(tree: Node) -> bool:
    """True iff some nucleus has the same CAT1 stem as subject and object.

    This configuration is licensed only as the intermediate state of a
    reflexive derivation and must never surface: generation resamples on it.
    """
    for t0 in _type0s(tree):
        subj, obj, _ = _nucleus(t0)
        if subj is None or obj is None:
            continue
        if subj.label != "Type1" or obj.label != "Type1":
            continue
        s_head = next(c for c in subj.children if c.role == "head")
        o_head = next(c for c in obj.children if c.role == "head")
        if s_head.lex["form"] == o_head.lex["form"]:
            return True
    return False


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
