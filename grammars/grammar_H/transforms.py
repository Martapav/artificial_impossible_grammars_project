"""Syntactic transformations for Grammar H (reformalized).

Mutates the tree in place after feature assignment.
All transformations operate on the tree, not the surface string.

Specification: docs/TRANSFORM_RULES.md (§3 binding, §4 aux, §6 wh).

  Anaphoric binding — derivational duplicate-CAT1 formalization:
    Refl (rule A′): bound within the minimal complete transitive clause NUCLEUS
          (subject head, verb, object head of one Type0) AND that nucleus must be
          EMBEDDED — its Type0 dominated by at least one Type4, at any depth. The
          MATRIX nucleus never licenses. The object of the licensing nucleus is
          substituted, copying the nucleus subject's features. Obligatory: no
          final string may contain the same CAT1 stem as subject and object of one
          EMBEDDED nucleus.
    Pron: free within its nucleus — the coreferential pair spans two distinct
          nuclei (or involves a position outside every nucleus, e.g. a PP
          complement). Either occurrence may be substituted (majority: the
          second); features are copied from the surviving antecedent.
          Substitution is optional at the grammar level: cross-nucleus stem
          duplicates MAY surface unsubstituted (neutral/non-binding items).
    A′ leaves a MATRIX subject-object pair licensed by neither sub-rule — a
    deliberately blocked configuration, the H family's binding-side evidence.

  Auxiliary movement — matrix-only fronting: the CAT3AUX of the matrix Type2
    fronts (there is at most one, so no selection among auxes arises); an aux
    only inside an embedded clause is a structural blocking case (skip).

  Wh-movement — LICENSING formalization (audit §F; clause (i) revised 2026-07-30
  to proposal E3): a nominal may front iff it is
      (i)  the matrix object AND it HOSTS an embedded clause, or
      (ii) inside a Type3 (PP) that is itself dominated by a Type4 (any
           attachment, any depth of nesting).
    Clause (i) is stated once in ``grammars/wh_structural.py``, shared with
    Grammar L′. It was plain "matrix object" until 2026-07-30; objecthood is
    adjacently marked (L1 ∉ {CAT6, CAT9}) and 87.8% of objects are matrix
    objects, so a bounded left observer recovered it at AUC 0.959. Embedded
    subjects no longer license (their surface shadow — first CAT1 after CAT9 —
    was perfect); Type3s outside every Type4 no longer license (their shadow —
    CAT1 after CAT6 — was a bigram). Clause (ii)'s remaining shadow requires
    tracking the unmarked right edge of the Type4.
    CAT1PRON-Pron behaves like a CAT1 (may front from licensed positions);
    CAT1PRON-Refl never fronts (an anaphor stays in its licensing position).

  Granularity (all phenomena): movement and substitution operate on the BLOCK —
  head CAT1/CAT1PRON plus contiguous CAT4/CAT2 — stranding any Type3/Type4 in
  situ as a sibling after the gap or pron, exactly as Grammar P does.
"""

from __future__ import annotations

import random

from grammars.tracing import bernoulli

from .nodes import Node
from .lexicon import cat1pron_item, cat8_wh
from .build import build_terminal
from .rules import P_PRON_FIRST


def apply(tree: Node, phenomenon: str, rng: random.Random,
          lex: dict, counter: list) -> str:
    """Dispatch to the transformation and return the refined phenomenon label.

    Returns the input phenomenon unchanged for cases where no tree mutation
    is required ("neutral") or the label suffixed with "_skipped" when the
    transformation cannot apply to the drawn structure (the corpus builder
    drops those).

    The former ``bind_site`` argument is gone: under rule A′ the refl nucleus is
    always embedded, so there is no placement left to decide at generation time.
    """
    if phenomenon == "anaphoric_binding":  # unrefined dispatch (gen items)
        phenomenon = ("anaphoric_binding_refl"
                      if bernoulli(rng, 0.5, "bind_subtype")
                      else "anaphoric_binding_pron")
    if phenomenon == "anaphoric_binding_refl":
        return _refl(tree, rng, lex)
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
            # Normally [CAT9 head, embedded Type0], but the neutral-positional
            # probe violation `_pos_cat9_displaced` moves the CAT9 into the
            # embedded subject, leaving the Type0 as the sole child — so locate
            # it by label rather than assuming position 1.
            t0 = next((gc for gc in child.children if gc.label == "Type0"), None)
            if t0 is not None:
                acc.append(t0)
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


def _head_form_of(nominal: Node) -> str | None:
    """Surface ``form`` of a nominal's head terminal (Type1 → its head CAT1);
    a CAT1PRON reports its recorded antecedent form (coreference), or its own
    if none. Used to stamp/reconstruct the coreferential duplicate for binding."""
    if nominal.label == "Type1":
        head = next(c for c in nominal.children if c.role == "head")
        return head.lex.get("form") if head.lex is not None else None
    return nominal.coref_form if nominal.coref_form is not None else (
        nominal.lex.get("form") if nominal.lex is not None else None)


# Block granularity (TRANSFORM_RULES §2, ruling 2026-07-19): movement and
# binding operate on the CAT1/CAT1PRON + contiguous CAT4/CAT2 block — exactly
# like Grammar P. A nominal's PP-under-NP (Type3, role "np_adjunct") and
# embedded clause (Type4, role "rel_clause") are NOT part of the block: they
# are STRANDED in place (reattached to the nominal's parent right after the
# gap / the substituted pron), so the transformation's surface effect is
# identical across the four grammars and post-gap material exists for
# surprisal targets.
_BLOCK_ROLES = {"head", "det", "modifier"}


def _strand_nonblock(nominal: Node) -> list[Node]:
    """Remove and return a Type1's non-block children (base order preserved).

    A CAT1PRON leaf has no children — returns []. Callers reinsert the
    returned nodes into the nominal's parent immediately after the gap or the
    substituting CAT1PRON, which preserves the base linear order (the block
    always precedes the strandable material inside a Type1).
    """
    if nominal.label != "Type1":
        return []
    stranded = [c for c in nominal.children if c.role not in _BLOCK_ROLES]
    nominal.children = [c for c in nominal.children if c.role in _BLOCK_ROLES]
    return stranded


# ── Site-directed mechanics ──────────────────────────────────────────────────
# The physical application of each transformation at a caller-chosen site,
# shared by the licensed transforms below (which CHOOSE the site per the
# grammar's rule) and by the structural probes' violating applications
# (grammars/violations.py), so both members of a probe pair go through one
# code path and differ only in where the rule fired.


def front_nominal(tree: Node, target: Node, parent: Node, parent_idx: int,
                  lex: dict, counter: list) -> None:
    """Front a nominal's block to sentence-initial position after CAT8.

    P block granularity: strands the nominal's PP/Type4 at the gap site.
    """
    stranded = _strand_nonblock(target)
    target.role = "fronted"
    gap = Node(
        label="GAP", head_cat="GAP", lex=None, feats={},
        children=[], role="gap", licensor_id=None, node_id=target.node_id,
    )
    parent.children[parent_idx] = gap
    parent.children[parent_idx + 1:parent_idx + 1] = stranded
    cat8_node = build_terminal("CAT8", cat8_wh(lex), "wh_marker", counter)
    tree.children.insert(0, target)
    tree.children.insert(0, cat8_node)


def front_aux(tree: Node, type2: Node, aux_idx: int) -> None:
    """Front the CAT3AUX at ``type2.children[aux_idx]``, leaving a gap trace."""
    aux = type2.children[aux_idx]
    del type2.children[aux_idx]
    gap = Node(
        label="GAP", head_cat="GAP", lex=None, feats={},
        children=[], role="gap", licensor_id=None, node_id=aux.node_id,
    )
    type2.children.insert(aux_idx, gap)
    tree.children.insert(0, aux)


def substitute_nominal(parent: Node, idx: int, nominal: Node,
                       pron_item: dict, feats: dict) -> Node:
    """Replace ``parent.children[idx]`` (a full-CAT1 Type1) with a CAT1PRON.

    P block granularity: the nominal's CAT4/CAT2 vanish with the block; its
    PP/Type4 are stranded in place after the pron. Returns the pron node.
    Callers handle clause-agreement realignment for subject substitution.
    """
    stranded = _strand_nonblock(nominal)
    pron = Node(
        label="CAT1PRON", head_cat="CAT1PRON", lex=pron_item,
        feats=feats, children=[], role=nominal.role, licensor_id=None,
        node_id=nominal.node_id,
    )
    parent.children[idx] = pron
    parent.children[idx + 1:idx + 1] = stranded
    return pron


def _coindex(target: Node, anchor: Node) -> None:
    """Make ``target`` (a full-CAT1 Type1) a genuine coreferential duplicate of
    ``anchor`` (also a full-CAT1 Type1), in the licensing base the binding
    transform then consumes (TRANSFORM_RULES §2, the derivational duplicate).

    The anchor is never touched (it controls verb agreement — §3.1); the target
    is overwritten to the anchor head's lexeme + INFL1_number + INFL1_gender
    (gender is lexeme-inherent, copied explicitly so no stale feat survives) and
    stripped of its direct modifiers (its own CAT4 determiner and CAT2
    modifiers), leaving a bare coindexed head. Type3/Type4 material is left in
    place for the substitution step to strand (block granularity, §5).
    """
    a_head = next(c for c in anchor.children if c.role == "head")
    t_head = next(c for c in target.children if c.role == "head")
    t_head.lex = a_head.lex
    t_head.feats["INFL1_number"] = a_head.feats["INFL1_number"]
    t_head.feats["INFL1_gender"] = a_head.feats["INFL1_gender"]
    target.children = [c for c in target.children
                       if c.role not in ("det", "modifier")]


def refl_licensing_nuclei(tree: Node) -> list[tuple[Node, Node, Node, int]]:
    """Rule A′'s licensing predicate, stated once: every nucleus that licenses
    a reflexive, as ``(type0, subject, object, object_index_in_type2)``.

    A′ (TRANSFORM_RULES §3.1) licenses a nucleus iff it is EMBEDDED — its Type0
    dominated by at least one Type4, at **any depth** (``_embedded_type0s``,
    which excludes the matrix Type0) — and has both a subject and an object,
    both full ``Type1``. That conjunction is the whole rule; nothing about the
    matrix clause, the Type4's host, or the attachment depth enters it.

    Three callers share this one statement, which is what makes them agree by
    construction rather than by review:

    * ``_refl`` below, which chooses among the nuclei it returns;
    * ``grammar_H/generate._one_item``, whose refl conditioning is **rejection
      sampling on this predicate** — draw a base naturally, keep it if this
      returns non-empty (§1c: the acceptance test *is* the rule);
    * the probe builder's ``base_conformity``, which asks whether a probe base
      lies on the corpus's conditioning path for refl — the same question, so
      necessarily the same predicate.
    """
    eligible: list[tuple[Node, Node, Node, int]] = []
    for t0 in _embedded_type0s(tree):
        subj, obj, t2 = _nucleus(t0)
        if subj is None or t2 is None or obj is None:
            continue
        if subj.label != "Type1" or obj.label != "Type1":
            continue  # no pron antecedent, no double substitution
        obj_idx = t2.children.index(obj)
        eligible.append((t0, subj, obj, obj_idx))
    return eligible


def has_refl_licensing_nucleus(tree: Node) -> bool:
    """True iff ``tree`` satisfies rule A′'s licensing predicate (see above)."""
    return bool(refl_licensing_nuclei(tree))


def _refl(tree: Node, rng: random.Random, lex: dict) -> str:
    """Reflexive substitution in a licensing nucleus — which must be EMBEDDED.

    Derivationally the nucleus hosts the same CAT1 as subject and object; the
    grammar forces substitution of the SECOND occurrence (the object), so the
    duplicate exists only as an intermediate state. Surface effect: the object
    of the chosen nucleus becomes CAT1PRON-Refl copying the nucleus subject's
    number/gender. The antecedent must be a full CAT1 (no pron-pron chains).

    Rule A′ (TRANSFORM_RULES §3.1, 2026-07-29): the nucleus must sit inside a
    Type4, at any depth — ``_embedded_type0s`` is exactly that set, and it
    EXCLUDES the matrix Type0, so a matrix nucleus never hosts a reflexive.
    The old matrix/embedded site draw (``P_BIND_EMB``) is retired: there is no
    placement left to choose. A tree with no transitive embedded nucleus skips
    (structural blocking).
    """
    eligible = refl_licensing_nuclei(tree)

    if not eligible:
        return "anaphoric_binding_skipped"

    t0, subj, obj, obj_idx = rng.choice(eligible)
    subj_feats = _head_feats_of(subj)
    feats = {"INFL1_number": subj_feats["INFL1_number"],
             "INFL1_gender": subj_feats["INFL1_gender"]}
    t2 = next(c for c in t0.children if c.label == "Type2")
    # Base coindexation (§2): the object is made a genuine duplicate of the
    # subject (the anchor, which fixes the clause's verb agreement — never the
    # reverse). No agreement realignment: the object controls no verb.
    _coindex(obj, subj)
    pron = substitute_nominal(t2, obj_idx, obj, cat1pron_item(lex, "Refl"), feats)
    # Record the antecedent form so licensing reconstructs the coindexed pair
    # (D3). Surface is unchanged (pron form derives from feats).
    pron.coref_form = _head_form_of(subj)
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


def _pron(tree: Node, rng: random.Random, lex: dict) -> str:
    """Pronoun substitution over a cross-nucleus coreferential pair.

    Derivationally the pair carries the same CAT1 (same stem, same number) in
    two positions NOT belonging to the same nucleus; one occurrence is
    substituted by CAT1PRON-Pron copying the other's features. The second
    occurrence is substituted in the majority of cases (1 - P_PRON_FIRST).
    Pairs where one nominal dominates the other are excluded (substituting the
    host would delete the antecedent).
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
            pairs.append((a, b))
    if not pairs:
        return "anaphoric_binding_skipped"

    first, second = rng.choice(pairs)
    target, antecedent = (
        (first, second) if bernoulli(rng, P_PRON_FIRST, "pron_order")
        else (second, first)
    )
    ant_feats = _head_feats_of(antecedent["node"])
    feats = {"INFL1_number": ant_feats["INFL1_number"],
             "INFL1_gender": ant_feats["INFL1_gender"]}
    # Base coindexation (§2): the target is made a genuine duplicate of the
    # antecedent (the anchor, never touched). If the target is a clause subject,
    # its verb agreed with the target's ORIGINAL number; realign it to the
    # anchor's here, so the coindexed base is agreement-consistent (G2).
    _coindex(target["node"], antecedent["node"])
    if target["slot"] == "subj":
        _fix_clause_agreement(target["owner"], feats)
    pron = substitute_nominal(target["parent"], target["idx"], target["node"],
                              cat1pron_item(lex, "Pron"), feats)
    # Record the antecedent form so licensing reconstructs the cross-nucleus
    # coindexed pair (D3).
    pron.coref_form = _head_form_of(antecedent["node"])
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


def _embedded_type0s(node: Node) -> list[Node]:
    """Every Type0 embedded under a Type4, in tree order, at all depths."""
    out: list[Node] = []
    for child in node.children:
        if child.label == "Type4":
            # See _type0s: a CAT9-displaced Type4 leaves the embedded Type0 as
            # its sole child, so locate it by label rather than by position.
            t0 = next((gc for gc in child.children if gc.label == "Type0"), None)
            if t0 is not None:
                out.append(t0)
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
    aux_idx, _aux = aux_idx_aux
    # front_aux leaves a trace at the vacated slot, mirroring wh-movement so
    # both movements expose a gap for surprisal localization (§8.1). Compound
    # tense is verb-then-aux, so the slot is right AFTER the main verb; the gap
    # renders as the null morpheme -> zero model tokens, so the model's token
    # sequence is unchanged; only the surface string and the metadata gain it.
    front_aux(tree, type2, aux_idx)
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
    front_nominal(tree, target, parent, parent_idx, lex, counter)
    return "wh_movement"


def _wh_targetable_nominals(tree: Node) -> list[dict]:
    """Every wh-targetable nominal, in tree order, annotated for licensing.

    A nominal is a Type1 or a CAT1PRON-Pron (CAT1PRON-Refl never fronts).
    Each entry: node, parent, idx (index in parent), and the structural
    licensing facts:
      matrix_object      — role="object" in the MATRIX Type2;
      hosts_type4        — the nominal has a Type4 child, i.e. it HOSTS an
                           embedded clause (added 2026-07-30 for clause (i)
                           proposal E3; see grammars/wh_structural.py);
      in_type3_under_t4  — strictly inside a Type3 that is dominated by a
                           Type4 (a Type3 crossed after a Type4 was crossed).
    Clause (i) is ``matrix_object AND hosts_type4``; clause (ii) is
    ``in_type3_under_t4``. Shared with Grammar H' (which combines
    in_type3_under_t4 with its own positional clause instead of clause (i)).
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
                "hosts_type4": any(c.label == "Type4" for c in node.children),
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
      (i)  the object of the MATRIX clause that HOSTS an embedded clause
           (proposal E3, 2026-07-30 — the rule and its rationale live in
           ``grammars/wh_structural.py``, shared with Grammar L′), or
      (ii) inside a Type3 dominated by a Type4.
    """
    from grammars.wh_structural import is_licensed_tree_entry
    return [
        (c["node"], c["parent"], c["idx"])
        for c in _wh_targetable_nominals(tree)
        if is_licensed_tree_entry(c) or c["in_type3_under_t4"]
    ]


# Duplicate screen (grammar constraint, checked by generate.py)


def has_nucleus_duplicate(tree: Node) -> bool:
    """True iff some EMBEDDED nucleus has the same CAT1 stem as subject and object.

    That configuration is licensed only as the intermediate state of a reflexive
    derivation and must never surface: generation resamples on it.

    Narrowed to embedded nuclei with rule A′ (2026-07-29): the matrix nucleus no
    longer licenses a reflexive, so a matrix same-stem subject/object pair is not
    a refl intermediate any more — it is an UNLICENSED coreference configuration
    and must be allowed to surface unsubstituted, as the H-family's blocking
    evidence. This mirrors Grammar P's ``has_licensed_duplicate``, which screens
    only the LICENSED offsets and deliberately lets duplicates at every other
    distance through (TRANSFORM_RULES §3.2).
    """
    for t0 in _embedded_type0s(tree):
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
