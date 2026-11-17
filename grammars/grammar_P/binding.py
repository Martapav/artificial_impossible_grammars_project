"""Positional binding licensing for the P family (Grammar P and Grammar L').

**One copy of the geometry.** Generation (``transforms._binding``), the surfacing
screen (``transforms.has_licensed_duplicate``), the annotation predicate
(``licensing.binding_pfamily_licensed_indices``) and the structural probes
(``violations.binding_sites`` / ``violations.apply_binding``) all read the rule
from here, so they cannot drift — before the 2026-07-30 narrowing each carried its
own offset loop.

The rules (TRANSFORM_RULES §3.2, decided 2026-07-30; full derivation in git
history, commit 191e573). A pair (antecedent, target) is licensed for
``subtype`` iff **all** of:

  1. both are CAT1 tokens of the base string, and they are a coreferential
     duplicate — same lexeme (checked here via surface form when ``forms`` is
     supplied; ``None`` asks the geometry question alone, see below);
  2. the target sits exactly one of the subtype's OFFSETS in surface items after
     the antecedent (Refl: 3; Pron: 2 or 4) — every whitespace token counts,
     ``#``-attached inflection values do not (D4);
  3. the antecedent's CAT1-POSITION ORDINAL, counted 1-based from the sentence
     start, is one of the subtype's licensed ordinals (Refl: exactly 3; Pron:
     2 or 3 — the first nominal never licenses pron).

Clause 3 is the 2026-07-30 narrowing (refl "rule B", pron "N9"). It is what makes
each rule a *count* from the sentence beginning rather than a fixed-window
distance, and the two reference points — one absolute (the ordinal), one relative
to it (the offset) — are the only ones the P design constraint admits.

**A CAT1-position is any CAT1 or CAT1PRON** (D7): pronouns are counted by the
ordinal scan, but are never antecedents or targets themselves (no pron-pron
chains). Callers that hold a *transformed* token list revert each binding
CAT1PRON to its base CAT1 first, so the target is eligible again.

**The two questions ``forms`` selects between** (mirrors the H-family predicate):

  * ``forms`` given — coreference is ENFORCED: "is this pair a licensed binding
    configuration?" Used by annotation, where the duplicate either exists or does
    not.
  * ``forms=None`` — geometry alone: "WHERE would a coindexed pair be licensed?"
    Used by generation and by probe-site enumeration, both of which *create* the
    coreference by coindexing the site they pick.

Offset/ordinal distance-disjointness (Refl {3} vs Pron {2,4}) is what preserves
form determinism D5 now that both rules admit antecedent ordinal 3: no single
pair is ever licensed for both subtypes. ``check_disjoint()`` asserts it.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set, Tuple

from .rules import (
    PRON_ANT_ORDINALS,
    PRON_OFFSETS,
    REFL_ANT_ORDINALS,
    REFL_OFFSET,
)

#: Categories counted by the ordinal scan (D7).
NOMINAL_CATS = ("CAT1", "CAT1PRON")

SUBTYPES = ("Refl", "Pron")


def rule(subtype: str) -> Tuple[Tuple[int, ...], frozenset]:
    """``(offsets, licensed antecedent ordinals)`` for ``subtype``."""
    if subtype == "Refl":
        return (REFL_OFFSET,), frozenset(REFL_ANT_ORDINALS)
    if subtype == "Pron":
        return tuple(PRON_OFFSETS), frozenset(PRON_ANT_ORDINALS)
    raise ValueError(f"unknown binding subtype: {subtype!r}")


def offsets(subtype: str) -> Tuple[int, ...]:
    """The subtype's licensed antecedent→target surface distances."""
    return rule(subtype)[0]


def ant_ordinals(subtype: str) -> frozenset:
    """The subtype's licensed antecedent CAT1-position ordinals (1-based)."""
    return rule(subtype)[1]


def cat1_ordinals(cats: Sequence[str]) -> dict:
    """``{index: 1-based CAT1-position ordinal}`` over CAT1 **and** CAT1PRON."""
    out, n = {}, 0
    for i, c in enumerate(cats):
        if c in NOMINAL_CATS:
            n += 1
            out[i] = n
    return out


def _iter_licensed_pairs(cats, subtype, forms):
    """The rule itself, in ONE place. Everything else in this module wraps it.

    A generator so ``has_licensed_pair`` keeps its early exit without needing a
    second copy of the loop — which is the failure mode this module exists to
    prevent, and which it briefly reproduced internally.
    """
    offs, ords = rule(subtype)
    ordinal = cat1_ordinals(cats)
    for i, c in enumerate(cats):
        # The antecedent must be a full CAT1 at a licensed CAT1-position ordinal.
        # A CAT1PRON is counted by ``cat1_ordinals`` but can never be an
        # antecedent (no pron-pron chains).
        if c != "CAT1" or ordinal[i] not in ords:
            continue
        for off in offs:
            j = i + off
            if j >= len(cats) or cats[j] != "CAT1":
                continue   # the target must be a full CAT1 too
            if forms is not None and (forms[i] is None or forms[i] != forms[j]):
                continue   # coreference enforced
            yield i, j


def licensed_pairs(
    cats: Sequence[str],
    subtype: str,
    forms: Optional[Sequence[Optional[str]]] = None,
) -> List[Tuple[int, int]]:
    """Every licensed ``(antecedent_index, target_index)`` pair, in scan order.

    ``cats`` is the BASE category sequence (fronted material spliced back, each
    binding CAT1PRON reverted to CAT1). ``forms`` is the parallel surface-form
    list and enforces coreference when given; see the module docstring.
    """
    return list(_iter_licensed_pairs(cats, subtype, forms))


def licensed_targets(
    cats: Sequence[str],
    subtype: str,
    forms: Optional[Sequence[Optional[str]]] = None,
) -> Set[int]:
    """Base indices licensed as binding TARGETS of ``subtype``."""
    return {j for _i, j in licensed_pairs(cats, subtype, forms)}


def antecedent_of(
    cats: Sequence[str],
    subtype: str,
    target: int,
    forms: Optional[Sequence[Optional[str]]] = None,
) -> Optional[int]:
    """The licensing antecedent index for ``target``, or None if unlicensed."""
    for i, j in licensed_pairs(cats, subtype, forms):
        if j == target:
            return i
    return None


def has_licensed_pair(
    cats: Sequence[str],
    subtype: str,
    forms: Optional[Sequence[Optional[str]]] = None,
) -> bool:
    """Does the base offer at least one licensed pair for ``subtype``?

    With ``forms=None`` this is the availability question generation forces on
    (`generate._one_item`): the geometry must exist before the binding transform
    can coindex and substitute.
    """
    return any(True for _pair in _iter_licensed_pairs(cats, subtype, forms))


def cats_and_forms(toks) -> Tuple[List[str], List[Optional[str]]]:
    """Base categories + surface forms for a P-family Token list.

    Binding CAT1PRONs revert to CAT1 carrying their antecedent's
    ``coref_form``, which reconstructs the coreferential duplicate the transform
    consumed (D3). Pre-transform bases hold no CAT1PRON, so this is the identity
    there.
    """
    cats, forms = [], []
    for t in toks:
        if t.cat == "CAT1PRON":
            cats.append("CAT1")
            forms.append(t.coref_form)
        else:
            cats.append(t.cat)
            forms.append(t.lex.get("form") if t.lex is not None else None)
    return cats, forms


def check_disjoint() -> None:
    """Assert no (antecedent, target) pair can be licensed for both subtypes.

    Form determinism D5: with both rules now carrying an ordinal conjunct, the
    guarantee rests entirely on the offset sets being disjoint — antecedent
    ordinal 3 is licit for Refl *and* Pron.
    """
    if set(offsets("Refl")) & set(offsets("Pron")):
        raise AssertionError(
            "P binding offsets overlap: a single (antecedent, target) pair "
            "would be licensed for both subtypes, breaking form determinism D5")


check_disjoint()
