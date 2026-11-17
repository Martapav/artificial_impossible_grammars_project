"""ℓ(x) for structural-probe items, under π_corpus (A2.2).

The quantity
------------
A probe item's ℓ(x) is NOT the probe builder's own sampling cost. The probe
builder forces construction sites, filters candidate lists to the licensed ones,
and rejects on uniqueness; ``−log π_probe(d)`` is therefore defined over a
different measure than the one the model was trained on, and
``neg_logq − (−log π_probe)`` would subtract two incommensurable quantities.

What we compute instead is the *same derivation* scored under the *corpus*
generator's sampling distributions: forced sites are charged the probability
π_corpus would have assigned to that choice. Rejection loops — the probe's
length window, movement-target guard, uniqueness and exclusion screens — affect
acceptance, not the derivation, and are excluded from the product.

How
---
:mod:`grammars.scoring` supplies the mechanism: replay the derivation through the
corpus generator's own code path, which is what defines π_corpus at each site.
This module supplies the *mapping* — the probe builder and the corpus generator
reach the same item by different routes, and their derivations must be aligned
before one can be scored under the other:

    corpus:  [phenomenon][bind_subtype?][aux_site?][insitu_aux][build][assign][transform]
    probe:   [build][insitu_aux][assign][transform]              (tree family)
             [insitu_aux][build][assign][transform]              (token family)

So the corpus script is the probe's base script with the pre-selection choices
PREPENDED (the probe never draws them — it is told which subcase to build) and
the ``insitu_aux`` decision moved to corpus position. Under
``auxiliary_movement`` with a matrix site the corpus does not draw ``insitu_aux``
at all — ``has_aux`` is implied by the site — so that factor is dropped rather
than moved, which is the difference between π_corpus and π_probe rather than a
bookkeeping detail.

Coverage, and what is deliberately not claimed
----------------------------------------------
Verified exact for ``neutral_agreement`` and ``neutral_positional`` on all four
grammars, at both probe length windows: the replay reproduces the probe item's
own surface with a non-skipped construction label.

``aux`` maps exactly only when the item's licensed aux site is the MATRIX one,
because :func:`corpus_script` asserts ``aux_site = matrix``. In the token family
P's rule fronts the FIRST CAT3AUX in left-to-right scan, so an item whose first
aux is the EMBEDDED one has a corpus derivation running through
``_force_embedded_aux`` — a block of draws this splice does not supply. Those
fall out as ``surface_mismatch`` rather than as a number, which is the guard
below working as intended; extending the mapping to the embedded site is
deliberately not done (probe coverage is not the objective).

⚠ **This limit grew with C7.1** (embedded in-situ aux raised from 0.30 to
``P_AUX_EMBEDDED = 0.60``, TRANSFORM_RULES §4, 2026-08-05), and the earlier
claim here — that the matrix-site mapping "covers every sampled IOD item on all
four grammars" — was already an artifact of a small sample. Measured at
n=30/subcase, IOD, all fallouts ``surface_mismatch``:

===========  ==============  ==============
subcase      at 0.30         at 0.60
===========  ==============  ==============
P aux        29/30 (96.7%)   26/30 (86.7%)
L′ aux       29/30 (96.7%)   27/30 (90.0%)
H / H′ aux   30/30           30/30
neutrals     30/30           30/30
===========  ==============  ==============

So the embedded-fronting case reached IOD *before* C7.1 — the old test simply
sampled 8 records per subcase and did not see it. C7.1 roughly tripled it in the
token family. H and H′ are structurally immune: H's rule reads the matrix Type2,
so the matrix-site script always holds. The density was **not** adjusted back to
restore coverage; probe coverage is not the objective, and a documented gap beats
a hidden one.

NOT mapped, by measurement rather than by assumption:

``pron``   the corpus's ``_pron`` makes its own transform draws (the coreferent
           pair, and ``pron_order``) which the probe's forced site does not
           supply. Mapping needs a translation from the probe's chosen site into
           the corpus's own candidate enumeration — bounded work, not done here.

``refl``,  the corpus FORCES geometry the probe's base does not have: a
``wh_i``,  transitive matrix (``force_transitive``, which narrows the verb pool
``wh_ii``  at the same label and position) and a forced embedded clause
           (``attach_forced_type4``, an entire block of draws with no counterpart
           in an unforced base). No splice of the probe's script can produce
           those factors; recovering them means re-deriving the item from its
           tree — a parser, not a replay.

Both cases return an explicit reason, never a number and never an infinity. The
distinction matters downstream: a *reason* is a statement about this pipeline's
coverage, while ``out_of_support`` is a statement about the corpus distribution
(π_corpus really is zero there — e.g. an intransitive matrix verb under the refl
path, which forces transitivity).

Verification contract
---------------------
A score is only returned when the replay reproduces the probe item's own
surface. A replay that consumes the script but yields a different string has
scored a different sentence, and a finite wrong number is far worse than an
explicit null — so surface mismatch is a failure, not a result.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from grammars.scoring import (
    Choice, Derivation, OutOfSupport, ScoringRandom, StructuralDivergence,
)

# Subcase → the corpus generator's construction label.
SUBCASE_CONSTRUCTION = {
    "refl": "anaphoric_binding",
    "pron": "anaphoric_binding",
    "wh_i": "wh_movement",
    "wh_ii": "wh_movement",
    "aux": "auxiliary_movement",
    "neutral_agreement": "neutral",
    "neutral_positional": "neutral",
}

# Subcases whose probe derivation aligns with the corpus derivation exactly.
# Measured, not assumed — see the module docstring and tests/test_probe_scoring.
MAPPED_SUBCASES = ("aux", "neutral_agreement", "neutral_positional")

# --- reasons for an undefined ℓ(x) -------------------------------------------
# Statements about the CORPUS DISTRIBUTION:
REASON_UNGRAMMATICAL = "ungrammatical_member"
REASON_OUT_OF_SUPPORT = "out_of_support"
# Statements about THIS PIPELINE's coverage:
REASON_FORCED_STRUCTURE = "forced_structure_unmapped"
REASON_TRANSFORM_CHOICE = "transform_choice_unmapped"
REASON_DIVERGENCE = "structural_divergence"
REASON_SURFACE_MISMATCH = "surface_mismatch"
REASON_SKIPPED = "construction_skipped"
REASON_NO_DERIVATION = "no_derivation_recorded"

_UNMAPPED_REASON = {
    "pron": REASON_TRANSFORM_CHOICE,
    "refl": REASON_FORCED_STRUCTURE,
    "wh_i": REASON_FORCED_STRUCTURE,
    "wh_ii": REASON_FORCED_STRUCTURE,
}


@dataclass
class ProbeScore:
    """ℓ(x) for one probe member, or an explicit reason it is undefined."""

    pair_id: str
    variant: str
    subcase: str
    neg_logp: Optional[float] = None
    n_choices: Optional[int] = None
    reason: Optional[str] = None
    detail: Optional[str] = None

    @property
    def item_uid(self) -> str:
        """The position-independent key probe records carry (A1/A4)."""
        return f"{self.pair_id}/{self.variant}"

    @property
    def defined(self) -> bool:
        return self.reason is None


def corpus_script(base_derivation: Derivation, subcase: str) -> Derivation:
    """The probe's base derivation, rewritten into corpus order.

    Prepends the pre-selection the probe never draws and repositions
    ``insitu_aux``; see the module docstring for why the aux factor is dropped
    outright under a matrix ``auxiliary_movement`` site.
    """
    construction = SUBCASE_CONSTRUCTION[subcase]
    aux = [c for c in base_derivation.choices if c.label == "insitu_aux"]
    rest = [c for c in base_derivation.choices if c.label != "insitu_aux"]

    head: List[Choice] = [Choice("phenomenon", "categorical", construction)]
    if construction == "anaphoric_binding":
        head.append(Choice("bind_subtype", "bernoulli", subcase == "refl"))
    if construction == "auxiliary_movement":
        # aux_site = matrix. The corpus sets has_aux from the site rather than
        # drawing it, so the probe's insitu_aux factor has no counterpart here.
        head.append(Choice("aux_site", "bernoulli", True))
    else:
        head.extend(aux)
    return Derivation(tuple(head + rest))


def score_probe_member(record: Dict[str, Any], variant: str,
                       one_item: Callable, lex: dict) -> ProbeScore:
    """ℓ(x) under π_corpus for one member of one probe pair.

    ``record`` is an entry from ``build_structural_probe(derivations_out=...)``.
    ``one_item`` must be the CORPUS generator's item function for this grammar —
    it is what supplies π_corpus at each site.
    """
    subcase = record["subcase"]
    pair_id = record["pair_id"]
    out = ProbeScore(pair_id=pair_id, variant=variant, subcase=subcase)

    if variant == "ungrammatical":
        # p_corpus = 0 by construction: the corpus generator emits only
        # grammatical items, so no derivation of this string exists under it.
        out.reason = REASON_UNGRAMMATICAL
        return out

    base = record.get("base_derivation")
    if base is None:
        out.reason = REASON_NO_DERIVATION
        return out

    if subcase not in MAPPED_SUBCASES:
        out.reason = _UNMAPPED_REASON[subcase]
        return out

    rng = ScoringRandom(corpus_script(base, subcase))
    try:
        surface, phenomenon, _ = one_item(rng, lex)
    except OutOfSupport as e:
        out.reason, out.detail = REASON_OUT_OF_SUPPORT, str(e)
        return out
    except StructuralDivergence as e:
        out.reason, out.detail = REASON_DIVERGENCE, str(e)
        return out

    if not rng.exhausted:
        out.reason = REASON_DIVERGENCE
        out.detail = f"{rng.remaining} scripted choices left over"
        return out
    if phenomenon.endswith("_skipped"):
        # The corpus builder drops these, so the item is not a corpus item and
        # has no ℓ(x) under the corpus distribution.
        out.reason = REASON_SKIPPED
        return out
    if surface != record["grammatical"]:
        # Consumed the script but produced a different sentence: the mapping is
        # wrong, and a finite wrong number is worse than a null.
        out.reason = REASON_SURFACE_MISMATCH
        out.detail = f"replay produced {surface!r}"
        return out

    out.neg_logp = rng.trace.neg_logp
    out.n_choices = rng.trace.n_draws
    return out


def score_probe_records(records: List[Dict[str, Any]], one_item: Callable,
                        lex: dict) -> List[ProbeScore]:
    """Both members of every captured pair, in probe emission order."""
    scores: List[ProbeScore] = []
    for record in records:
        for variant in ("grammatical", "ungrammatical"):
            scores.append(score_probe_member(record, variant, one_item, lex))
    return scores


def coverage(scores: List[ProbeScore]) -> Dict[str, int]:
    """Tally of defined scores and of each reason, for reporting (A4)."""
    out: Dict[str, int] = {"defined": 0}
    for s in scores:
        if s.defined:
            out["defined"] += 1
        else:
            out[s.reason] = out.get(s.reason, 0) + 1
    return out
