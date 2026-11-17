"""Structural wh clause (i) for Grammar H and Grammar L′ — the rule, stated once.

Companion to ``grammar_P/wh.py``, which owns the *positional* clauses. Together
they cover the wh licensing 2x2 of TRANSFORM_RULES §6:

    H  = (structural i, structural ii)      H′ = (positional i, structural ii)
    L′ = (structural i, positional ii)      P  = (positional i, positional ii)

Clause (i) is the **swap axis**, and its structural form is shared by **H and L′**.
Those two grammars read it off different objects — H off the constituent tree, L′
off the flat token list — so this module holds one statement of the rule and two
representation-specific readers. Before 2026-07-30 the rule was written out
separately in ``grammar_H/transforms``, ``grammar_L_prime/transforms``,
``violations.wh_sites`` and ``metadata._wh_facts_tokens``; that is the same
four-copy drift risk ``grammar_P/wh.py`` was created to remove on the positional
side.

THE RULE (proposal E3, adopted 2026-07-30 — assessment:
TRANSFORM_RULES.md §6; full derivation in git history, commit 191e573):

> A nominal is licensed by clause (i) iff it is the **object of the matrix clause**
> **and it hosts an embedded clause**.

It is a conjunctive **narrowing** of the previous rule ("matrix object", full stop):
every E3-licensed position was already licensed, nothing becomes newly licensed —
the same shape as refl rule B ⊂ refl-current and A′ ⊂ its predecessor.

*Why the second conjunct exists.* "Matrix object" alone was **bounded-local**: a
bounded left observer recovers it at U 0.751 / **AUC 0.959 at τ=0.5**, because
objecthood is adjacently marked (``L1 ∉ {CAT6, CAT9}`` — the same L1 cue that made
H refl degenerate before A′) and 87.8% of objects are matrix objects, so the
non-local residue carried almost no negative mass. The other arm of the same swap
axis (the positional "third CAT1-position") sits at U 0.018 / AUC 0.543, i.e. chance
at bounded memory: the two arms were not epistemically comparable. Under E3 the
structural arm reads U 0.040 / **AUC 0.677** at τ=0.5, and its AUC does not climb
with memory (0.677 → 0.669 at τ=∞) where the old rule's went 0.959 → 0.998.

*The licensor stays at the extraction site.* Fronting moves only the block — head
CAT1/CAT1PRON plus its CAT4 determiner and contiguous CAT2 modifiers — and leaves
the hosted embedded clause **stranded in situ immediately after the gap** (§6,
unchanged). So the thing that licenses the movement never moves: the fronted phrase
does not carry the evidence for its own licensing, the **gap site** does. Clause (i)
is therefore a rule about the gap, not about the filler.

*Known and accepted (TRANSFORM_RULES §6): the difficulty is directional.*
Lookahead is +0.163 (AUC 0.673 → 0.833 once right context is allowed) against the
positional arm's +0.002, precisely because the licensing clause sits to the target's
right. E3 is hard for a **left-to-right** observer, not hard in both directions.
That is defensible for a causal LM, which never sees the right context, but it is a
weaker claim than the positional arm supports and should be stated as such.

*THE UNION FORCING — the one documented exception to §1c.* Availability is 2.25%
in a neutral draw, so every wh item forces a transitive matrix clause plus one
embedded clause, in both families. This is **not** the same kind of thing as refl
A′'s forcing, and the two must not be read as parallel: A′ forces exactly its own
licensing predicate and is now implemented as rejection sampling on it, whereas
this forces the **union of four grammars' predicates**, which is strictly more
than any one of them requires.

The reason is the matched-procedure constraint. The four clause-(i) predicates are
not co-satisfiable at a common minimum — H and L′ require a matrix object hosting
an embedded clause, H′ and P require only a third CAT1-position, entailing
neither — and one base plan must serve all four. Matrix-transitive-plus-embedding
is strictly the stronger predicate, so the union is what gets forced. It is the
minimum compatible with matching, and the only place in the grammar family where
forcing exceeds a grammar's own licensing predicate (TRANSFORM_RULES §1c "One
documented exception", §6 "Forcing: the union case").

**Named consequence — H′ and P are over-conditioned, and analysis must carry it.**
For H′ and P, wh items are conditioned on matrix transitivity and embedding, which
their clause (i) does not require. Any analysis of H′ or P wh items must treat
matrix transitivity as a **constant of the slice**, not as a property the
positional rule could be exploiting, and must **check whether it correlates with
anything the positional clause can read**. That check is outstanding (D10).
Reporting an H′ or P wh result without it states more than the design supports.

*The host must stay MIXED.* It is **drawn subject-or-object, never pinned to the
object**, and that is load-bearing: pinning it makes every matrix object a host, so
the second conjunct becomes vacuous inside the wh slice and the rule collapses back
into plain "matrix object" — measured 3000/3000 matrix objects licensed, 0
non-hosting negatives, bounded-memory recoverability back to U 0.565 / AUC 0.917.
Mixing restores the contrast and returns the rulewise bounded observer to chance
(U −0.007 / AUC 0.500), at the cost of clause (i) being available in ~53% of wh
items rather than all. See ``grammar_H/generate._one_item``
(``attach_forced_type4(mix_host=True)``) and ``grammar_P/generate._one_item``
(``build_sentence(force_embed_here=True, force_matrix_transitive=True)``, host left
natural).

**General lesson: forcing a conjunct makes that conjunct vacuous unless it still
varies across candidates within one sentence.** Refl rule B's ordinal conjunct
does; E3's host conjunct does not, which is why it may not be forced.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set


# ---------------------------------------------------------------------------
# Hierarchical reader (Grammar H) — constituent tree
# ---------------------------------------------------------------------------

def licensed_ids_tree(tree) -> Set[int]:
    """``node_id`` of every nominal licensed by clause (i) on an H-family tree.

    Reads the two facts ``_wh_targetable_nominals`` already annotates:
    ``matrix_object`` (role "object" in the MATRIX Type2) and ``hosts_type4``
    (the nominal has a Type4 child, i.e. it hosts an embedded clause). Because
    the host is a matrix object, that Type4 is necessarily *first-level* —
    "first-level" is entailed by the rule, not an extra conjunct.
    """
    from grammars.grammar_H.transforms import _wh_targetable_nominals
    return {c["node"].node_id for c in _wh_targetable_nominals(tree)
            if c["matrix_object"] and c["hosts_type4"]}


def is_licensed_tree_entry(entry: Dict) -> bool:
    """Clause (i) applied to one ``_wh_targetable_nominals`` entry."""
    return bool(entry["matrix_object"] and entry["hosts_type4"])


# ---------------------------------------------------------------------------
# Positional reader (Grammar L′) — flat token list
# ---------------------------------------------------------------------------

def licensed_indices_tokens(toks: Sequence) -> Set[int]:
    """Indices of tokens licensed by clause (i) on a P-family token list.

    The same rule in the flat representation: ``role == "object"`` (the clause's
    object), ``clause_id == 0`` (that clause is the matrix), and
    ``hosts_embedded`` — the generation-history flag recording that this nominal
    is the one the embedded clause attached to.

    ``hosts_embedded`` is written by ``grammar_P.build._maybe_attach_cat9``, which
    already chose the host in order to compute its insertion point and previously
    discarded the choice. It is generation history, not surface: L′'s designated
    information channel, exactly like ``licensor_attachment``.
    """
    return {i for i, t in enumerate(toks) if is_licensed_token(t)}


def is_licensed_token(tok) -> bool:
    """Clause (i) applied to one P-family token."""
    return bool(getattr(tok, "role", None) == "object"
                and getattr(tok, "clause_id", None) == 0
                and getattr(tok, "hosts_embedded", False))
