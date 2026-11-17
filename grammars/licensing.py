"""Grammar-neutral licensing verdicts for binding and auxiliary movement.

Companion to ``verdicts.py`` (the wh clause-(i) H'-vs-L' axis). Together they let
the metadata enumeration (§8.3) assign, for every category-eligible candidate,
each grammar's licensing status on one shared neutral description of the base
sentence.

Key simplification: H' delegates binding AND aux to H, and L' delegates them to P
(only wh is overridden — see the two ``apply`` dispatchers). So for these two
phenomena there are exactly TWO distinct verdicts, an H-FAMILY rule (H = H') and a
P-FAMILY rule (P = L'):

  * Auxiliary movement
      - H-family: the MATRIX CAT3AUX (clause 0) fronts; an aux only inside an
        embedded clause is a blocking case.
      - P-family: the FIRST CAT3AUX in left-to-right scan fronts, any clause.
  * Binding (added in a later increment)
      - H-family (structural): object of a full-CAT1 nucleus (refl) / a member of
        a cross-nucleus CAT1 pair (pron).
      - P-family (positional): a CAT1 at a licensed offset after a coreferential
        CAT1 antecedent that is itself at a licensed CAT1-position ordinal
        (narrowed 2026-07-30 — the rule lives in ``grammar_P.binding``).

The neutral rep carries both positional facts (category sequence + index) and
structural facts (role, clause_id), exactly as ``verdicts.py`` does, so a
positional (P-family) and a structural (H-family) predicate can both read the
same description. Licensing is evaluated on the BASE order: a transformed sentence
has its fronted element spliced back at its gap first.
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional

from grammars.grammar_H.nodes import Node, is_terminal
from grammars.grammar_P.tokens import Token
from grammars.grammar_P import binding as p_binding


class NeutralTok(NamedTuple):
    """One base-order terminal, family-independent."""
    cat: str
    role: str
    clause_id: int


# --------------------------------------------------------------------------
# Neutral rep extraction (base order)
# --------------------------------------------------------------------------

def neutral_from_tokens(toks: List[Token]) -> List[NeutralTok]:
    """Neutral rep of a positional-family token list, in list order."""
    return [NeutralTok(t.cat, t.role, t.clause_id) for t in toks]


def neutral_from_tree(tree: Node) -> List[NeutralTok]:
    """Neutral rep of a hierarchical-family tree, in linear (in-order) order.

    clause_id numbers Type0 clauses in pre-order (matrix Type0 == 0), matching the
    positional family's convention (matrix == 0, embedded clauses increment).
    Terminals inherit their nearest enclosing clause's id; a GAP is emitted as
    ``('GAP', 'gap', clause_id)``.
    """
    out: List[NeutralTok] = []
    counter = [0]

    def walk(node: Node, clause_id: int) -> None:
        cid = clause_id
        if node.label == "Type0":
            cid = counter[0]
            counter[0] += 1
        if node.label == "GAP":
            out.append(NeutralTok("GAP", "gap", clause_id))
            return
        if is_terminal(node):
            out.append(NeutralTok(node.head_cat, node.role, clause_id))
            return
        for child in node.children:
            walk(child, cid)

    walk(tree, 0)
    return out


# --------------------------------------------------------------------------
# Auxiliary movement — grammar-neutral verdicts (base order)
# --------------------------------------------------------------------------

_AUX = "CAT3AUX"


def aux_hfamily_licenses(neutral: List[NeutralTok], idx: int) -> bool:
    """H/H': the aux at ``idx`` fronts iff it is a MATRIX (clause 0) CAT3AUX."""
    return neutral[idx].cat == _AUX and neutral[idx].clause_id == 0


def aux_pfamily_licenses(neutral: List[NeutralTok], idx: int) -> bool:
    """P/L': the aux at ``idx`` fronts iff it is the FIRST CAT3AUX in scan order."""
    if neutral[idx].cat != _AUX:
        return False
    first = next((i for i, t in enumerate(neutral) if t.cat == _AUX), None)
    return idx == first


def aux_candidates(neutral: List[NeutralTok]) -> List[int]:
    """Indices of every category-eligible aux candidate (every CAT3AUX)."""
    return [i for i, t in enumerate(neutral) if t.cat == _AUX]


def recover_aux_base(neutral: List[NeutralTok]) -> Optional[tuple]:
    """Base rep + fronted-aux index for an aux-movement neutral rep, or None.

    Aux fronting puts the aux at absolute position 0 and leaves a GAP at the
    vacated slot (both families). Splicing the aux back at the GAP restores the
    base order; the returned index is the aux's base position (the position the
    licensing predicates evaluate).
    """
    if not neutral or neutral[0].cat != _AUX:
        return None
    aux = neutral[0]
    rest = neutral[1:]
    gi = next((i for i, t in enumerate(rest) if t.cat == "GAP"), None)
    if gi is None:
        return None
    base = rest[:gi] + [aux] + rest[gi + 1:]
    return base, gi


# --------------------------------------------------------------------------
# Binding — own-family exact verdicts (cross-family is 'na', per decision).
#
# Substitution is in place (no linear reorder) but, since the block-
# granularity change (TRANSFORM_RULES §2), it is NOT dominance-preserving:
# a substituted nominal's PP/Type4 material is stranded as a following SIBLING
# of the CAT1PRON rather than deleted or left nested. Physical tree containment
# (``Node.children``) therefore under-counts what the base tree dominated. The
# walk below tracks LOGICAL ancestry instead: a CAT1PRON's immediately
# following stranded Type3/Type4 siblings count as still dominated by it, so
# dominance verdicts match what they would be on the pre-substitution base
# (and are a no-op on a tree with no substitution, i.e. the base itself).
# --------------------------------------------------------------------------

_H_NOMINAL_LABELS = ("Type1", "CAT1PRON")
_STRANDABLE_LABELS = ("Type3", "Type4")


def _h_nominal_positions(tree: Node) -> list:
    """Base nominal positions, mirroring Grammar H's ``_collect_pron_positions``
    but treating a substituted CAT1PRON as the base Type1 nominal it replaced
    (binding evaluates on the base; substitution is in place).

    Each entry: node, slot ("subj"/"obj"/None) + owner (nucleus Type0), embedded,
    ancestor_ids (node_ids of every nominal LOGICALLY dominating this position —
    physical ancestors plus, through stranding, the CAT1PRON a stranded sibling
    was cut from).

    Excludes ``role == "fronted"``: wh-movement relabels+repositions the
    fronted node rather than removing it (a fresh GAP takes its old slot), so
    without this guard a fronted nominal would be double-counted as a second,
    phantom candidate sitting outside any clausal slot — its slot/owner comes
    back None, which spuriously LIFTS the same-nucleus exclusion for its own
    former subject/object partner (verified empirically: this made a fronted
    object's former subject register as newly Pron-licensed with the ghost).
    Needed because binding candidates are now evaluated on off-frame
    (possibly wh-transformed) sentences too, not just binding's own base.
    """
    out = []

    def walk(node, parent, inside_t4, t0, ancestor_ids):
        if (node.label in _H_NOMINAL_LABELS and node.role != "fronted"
                and parent is not None):
            slot = owner = None
            if parent.label == "Type0" and node.role == "subject":
                slot, owner = "subj", parent
            elif parent.label == "Type2" and node.role == "object":
                slot, owner = "obj", t0
            out.append({"node": node, "slot": slot, "owner": owner,
                        "embedded": inside_t4, "ancestor_ids": frozenset(ancestor_ids)})
        child_ancestor_ids = (
            ancestor_ids | {node.node_id} if node.label in _H_NOMINAL_LABELS
            else ancestor_ids
        )
        stranded_owner = None
        for child in node.children:
            extra = (child_ancestor_ids | {stranded_owner}
                     if stranded_owner is not None and child.label in _STRANDABLE_LABELS
                     else child_ancestor_ids)
            walk(child, node, inside_t4 or node.label == "Type4",
                 node if node.label == "Type0" else t0, extra)
            stranded_owner = (child.node_id if child.label == "CAT1PRON"
                              else stranded_owner
                              if child.label in _STRANDABLE_LABELS else None)

    walk(tree, None, False, None, frozenset())
    return out


def binding_hfamily_licensed_ids(tree: Node, subtype: str,
                                 require_coref: bool = False) -> set:
    """node_ids licensed as binding TARGETS of ``subtype`` under the H-family rule,
    evaluated on the base (a substituted CAT1PRON counts as its base CAT1 nominal):

      * Refl → the OBJECT of an **embedded** nucleus whose subject is a full CAT1
        antecedent. "Embedded" = the nucleus's Type0 is dominated by at least one
        Type4, at ANY depth (``_embedded_type0s``); the matrix nucleus never
        licenses (TRANSFORM_RULES §3.1, rule A′ 2026-07-29).
      * Pron → any member of a cross-nucleus coreferential CAT1 pair (not the same
        nucleus, neither dominating the other). UNCHANGED by A′ — so a MATRIX
        subject-object pair is now licensed by neither sub-rule and is a blocked
        configuration (§3.4).

    Why the domain condition exists: scanning ``_type0s`` (every nucleus, matrix
    included) made this predicate return EXACTLY the set of direct objects — 446/446
    over 800 bases — so its rulewise separability cell had zero negatives and the
    rule had no content beyond its own precondition. Objecthood is decidable from
    one adjacent slot (``L1 ∉ {CAT6, CAT9}``, AUC 1.000); embedding is not.

    ``require_coref`` selects between the two legitimately distinct questions this
    predicate answers:

      * ``True`` (annotation, ``metadata``): "is this a genuinely coreferential
        binding target?" — the structural relation AND a genuine *coreferential
        duplicate* (same head form, TRANSFORM_RULES §2). A binding CAT1PRON reports
        its antecedent's form via ``Node.coref_form``; other nominals report their
        own head form. Coreference must be present in the base for a target to
        license.
      * ``False`` (probe site enumeration, ``violations.binding_sites``):
        "WHERE would a coindexed pair license binding?" — the structural relation
        alone. The probe IMPOSES coreference by coindexation at the chosen site
        (``violations.apply_binding`` → ``_coindex``), so site licensing is a
        purely structural question; coreference is not pre-present to require. This
        is by design, not a pending migration.
    """
    from grammars.grammar_H.transforms import (
        _embedded_type0s, _nucleus, _head_form_of,
    )

    if subtype == "Refl":
        out = set()
        # A′: embedded nuclei only — _embedded_type0s is every Type0 under a
        # Type4 at all depths, i.e. exactly "depth >= 1".
        for t0 in _embedded_type0s(tree):
            subj, obj, _ = _nucleus(t0)
            if (subj is not None and obj is not None
                    and subj.label == "Type1"
                    and obj.label in _H_NOMINAL_LABELS):
                if require_coref:
                    sf = _head_form_of(subj)
                    if sf is None or sf != _head_form_of(obj):
                        continue  # coreference required: object must be a duplicate
                out.add(obj.node_id)
        return out

    positions = _h_nominal_positions(tree)
    out = set()
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            a, b = positions[i], positions[j]
            same_nucleus = (
                a["owner"] is not None and a["owner"] is b["owner"]
                and {a["slot"], b["slot"]} == {"subj", "obj"}
            )
            if same_nucleus:
                continue
            dominates = (a["node"].node_id in b["ancestor_ids"]
                        or b["node"].node_id in a["ancestor_ids"])
            if dominates:
                continue
            if require_coref:
                fa = _head_form_of(a["node"])
                if fa is None or fa != _head_form_of(b["node"]):
                    continue  # coreference required: the pair must be a duplicate
            out.add(a["node"].node_id)
            out.add(b["node"].node_id)
    return out


def binding_pfamily_licensed_indices(base_cats: List[str], subtype: str,
                                     base_forms: Optional[List[Optional[str]]] = None) -> set:
    """Base-order indices licensed as binding TARGETS of ``subtype`` under the
    P-family rule: a CAT1 sitting a licensed offset after a **coreferential**
    CAT1 antecedent that is itself at a licensed CAT1-position ordinal — i.e. a
    genuine *duplicated* CAT1 (TRANSFORM_RULES §2) in the geometry of §3.2.

    Thin wrapper over ``grammar_P.binding``, which owns the rule (both conjuncts,
    one copy shared with generation, the screen and the probes).

    ``base_cats`` is the category sequence with the substituted CAT1PRON reverted
    to its base CAT1. ``base_forms`` selects between the two legitimately distinct
    questions this predicate answers (mirrors the H-family predicate above):

      * given (annotation, ``metadata``): the parallel surface-form list, each
        CAT1PRON reverted to its antecedent's form via ``Token.coref_form`` —
        coreference is ENFORCED (antecedent and target must share a form). This is
        the necessary condition for binding; the distance geometry alone is not
        sufficient.
      * ``None`` (probe site enumeration, ``violations.binding_sites``): the
        distance geometry alone — "WHERE would a coindexed pair license binding?".
        The probe imposes coreference by coindexation at the chosen site, so this
        is by design, not a pending migration.
    """
    return p_binding.licensed_targets(base_cats, subtype, base_forms)
