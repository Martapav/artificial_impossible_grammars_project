"""Positional wh licensing for the P family (Grammar P and Grammar L′).

**One copy of the two positional clauses.** Before 2026-07-30 the "third
CAT1-position" and "after a CAT5" tests were written out separately in
``grammar_P/transforms._wh_candidates``, ``grammar_L_prime/transforms._wh_candidates``,
``violations.wh_sites`` and ``metadata._wh_facts_tokens`` — four copies of one rule.
They all read it from here now, exactly as the binding rules read
``grammar_P/binding``.

The clauses (TRANSFORM_RULES §6). Licensing is evaluated on the **base** string,
before fronting; a **CAT1-position** is any CAT1 **or** CAT1PRON (D7):

  (i)  the **third** CAT1-position of the sentence.
  (ii) the **second** CAT1-position after the **second CAT6** of the sentence.

Grammar P uses both. Grammar L′ swaps clause (i) for the structural "matrix object"
test and keeps clause (ii) verbatim — clause (ii) is the shared, non-swapped clause,
which is why it lives in one module rather than in each grammar.

**Clause (ii) was changed 2026-07-30** (assessment:
TRANSFORM_RULES §6; derivation in git history, commit 191e573). It used to be "the first CAT1-position
after a CAT5, once per CAT5". That is a **bigram**: CAT5 is a category the window
sees directly and the licensed nominal is the very next one, so a 5-slot left window
recovered it at U 0.981 → 0.947 / AUC 0.993 — the rule carried no evidence about
counting at all.

Three things fix it, and all three are load-bearing:

  * **an ORDINAL over the anchors** ("the second CAT6") instead of per-anchor
    quantification, so the anchor must be *counted from the sentence beginning* —
    which is also what keeps the clause inside the P design constraint (one
    reference point at the sentence start, the rest relative to it);
  * **CAT6 rather than CAT5** as the anchor category. This is the part that is
    about the corpus, not the rule: ``build._insert_cat5`` appends a clause's CAT5s
    *contiguously*, so 92% of two-CAT5 bases have them adjacent and "the second
    CAT5" degenerates back into the bigram ``CAT5 CAT5``. CAT6s are never
    contiguous (each heads its own PP), so their ordinal has to be counted. CAT6 is
    also commoner: 17.3% of bases carry two, against CAT5's 10.3%.
  * **the SECOND nominal after it, not the first.** A CAT6 is immediately followed
    by its own complement nominal, so n=1 would reinstate the adjacency bigram.

Measured on 12k train / 2k OOD bases: availability 9.12%, window U 0.298 → 0.100 at
AUC 0.714 (against the audit's 0.6 recoverability cutoff), and count-solvable —
counts-only U 0.61 at tree depth 3, 0.69 at depth 6 and 10; window+counts 0.67 /
0.91 / 0.96 at depths 3 / 6 / 10. The depth-3 readings understate it, which is the
recorded capacity trap, not a property of the rule. **100% of its licensed sites are
ones clause (i) does not already license**, so it is a genuine second licensing
route rather than a decorative one.

Known and accepted cost: every "second nominal after X" clause implies the sentence
has at least three nominals, which is clause (i)'s own condition, so clause (ii) can
only add sites *inside* bases clause (i) already covers and never rescue one it
misses. P's wh skip rate is therefore 60.41% rather than the old 52.72% — the price
of a clause (ii) that is not a bigram (TRANSFORM_RULES §6).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Set

from .rules import (
    WH_I_NOMINAL_ORDINAL,
    WH_II_ANCHOR_CAT,
    WH_II_ANCHOR_ORDINAL,
    WH_II_NOMINAL_ORDINAL,
)

#: Categories counted as CAT1-positions by every ordinal scan (D7).
NOMINAL_CATS = ("CAT1", "CAT1PRON")


def nominal_positions(cats: Sequence[str]) -> List[int]:
    """Indices of the CAT1-positions, in base order."""
    return [i for i, c in enumerate(cats) if c in NOMINAL_CATS]


def clause_i_index(cats: Sequence[str]) -> Optional[int]:
    """Clause (i): the third CAT1-position of the sentence, or None."""
    pos = nominal_positions(cats)
    return (pos[WH_I_NOMINAL_ORDINAL - 1]
            if len(pos) >= WH_I_NOMINAL_ORDINAL else None)


def clause_ii_index(cats: Sequence[str]) -> Optional[int]:
    """Clause (ii): the 2nd CAT1-position after the 2nd CAT6, or None.

    At most one licensed position per sentence — the anchor ordinal picks exactly
    one CAT6, and the nominal ordinal exactly one position after it. There is no
    fallback to an earlier nominal when fewer than two follow, and none to the
    first CAT6 when only one exists.
    """
    anchors = [i for i, c in enumerate(cats) if c == WH_II_ANCHOR_CAT]
    if len(anchors) < WH_II_ANCHOR_ORDINAL:
        return None
    anchor = anchors[WH_II_ANCHOR_ORDINAL - 1]
    after = [i for i in nominal_positions(cats) if i > anchor]
    return (after[WH_II_NOMINAL_ORDINAL - 1]
            if len(after) >= WH_II_NOMINAL_ORDINAL else None)


def licensed_indices(cats: Sequence[str], *, clause_i: bool = True) -> Set[int]:
    """The P-family positional licensed set.

    ``clause_i=False`` returns clause (ii) alone — Grammar L′ asks for that, since
    it supplies its own structural clause (i).
    """
    out = set()
    if clause_i:
        i1 = clause_i_index(cats)
        if i1 is not None:
            out.add(i1)
    i2 = clause_ii_index(cats)
    if i2 is not None:
        out.add(i2)
    return out


def target_eligible(cat: str, lex: Optional[dict]) -> bool:
    """Is a token at a licensed position a licit wh TARGET? (D7)

    CAT1 and CAT1PRON-**Pron** front; CAT1PRON-**Refl** never does (an anaphor
    stays in its licensing configuration). Note the ordinal scans above count
    every CAT1PRON regardless of subclass — being counted and being frontable are
    different questions.
    """
    if cat == "CAT1":
        return True
    if cat == "CAT1PRON":
        return lex is not None and lex.get("subclass") == "Pron"
    return False
