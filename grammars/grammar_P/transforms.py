"""Syntactic transformations for Grammar P (positional formalizations).

Every transformation operates on the flat surface token list by a
left-to-right scan; none consults constituent structure, c-command, or
generation history (that is what distinguishes P from H). Each function
mutates the list in place and returns the refined phenomenon label.

Specification: docs/TRANSFORM_RULES.md (§3 binding, §4 aux, §6 wh).

  Anaphoric binding — purely positional, derivational duplicate-CAT1 rule.
  Distances count surface items (every whitespace token; #-attached
  inflection values are not separate items); CAT1-position ordinals are
  1-based from the sentence start over CAT1 and CAT1PRON alike (D7). The
  geometry lives in ONE place, binding.py — see it for the full statement:
    Refl: licensed iff the coreferential CAT1 sits exactly REFL_OFFSET (3)
          items after its antecedent CAT1 in the base string AND that
          antecedent is the THIRD CAT1-position; the second occurrence is
          obligatorily substituted by CAT1PRON-Refl copying the
          antecedent's number/gender.
    Pron: licensed iff the coreferential CAT1 sits exactly 2 or 4 items
          after its antecedent AND the antecedent's CAT1-ordinal is 2 or 3
          (the first nominal never licenses); the second occurrence is
          obligatorily substituted by CAT1PRON-Pron copying the
          antecedent's features.
    Duplicates at any other distance, or at a licensed distance with an
    antecedent at an unlicensed ordinal, are unlicensed: they surface
    unsubstituted (the positional blocking evidence). Duplicates in a
    licensed configuration never surface (generation screens them).

  Auxiliary movement — front the FIRST CAT3AUX in scan order, regardless of
    clause (unchanged). Never blocked when an aux exists anywhere.

  Wh-movement — LICENSING formalization (audit §F; clause (ii) revised
  2026-07-30): a nominal may front iff it is
      (i)  the third CAT1-position of the sentence, or
      (ii) the second CAT1-position after the second CAT6 of the sentence.
    Clause (ii) was "the first CAT1-position after a CAT5, per CAT5" until
    2026-07-30; that is a CAT5→nominal bigram, window-recoverable at U 0.981
    → 0.947, so it tested adjacency rather than counting. The rule and the
    reasons for each of its three parts live in ``wh.py``. The
    second-CAT1-after-CAT9 condition was removed earlier (audit §F).
    CAT1PRON counts as a CAT1-position for the ordinal scans; as a TARGET,
    CAT1 and CAT1PRON-Pron are licit, CAT1PRON-Refl never fronts. If no
    licensed target exists, the phenomenon is blocked — positional
    blockability.
"""

from __future__ import annotations

import random

from grammars.tracing import bernoulli

from .tokens import Token
from .lexicon import cat1pron_item, cat8_wh
from . import binding, wh


def apply(toks: list[Token], phenomenon: str, rng: random.Random, lex: dict) -> str:
    """Dispatch to the pre-selected transformation; return the refined label.

    Returns the phenomenon suffixed with ``_skipped`` when it cannot apply to
    the drawn string (no licensed geometry), so the corpus builder can drop
    it. ``neutral`` returns unchanged.
    """
    if phenomenon == "anaphoric_binding":  # legacy dispatch (L', gen items)
        phenomenon = ("anaphoric_binding_refl"
                      if bernoulli(rng, 0.5, "bind_subtype")
                      else "anaphoric_binding_pron")
    if phenomenon == "anaphoric_binding_refl":
        return _binding(toks, rng, lex, subtype="Refl")
    if phenomenon == "anaphoric_binding_pron":
        return _binding(toks, rng, lex, subtype="Pron")
    if phenomenon == "auxiliary_movement":
        return _aux_movement(toks)
    if phenomenon == "wh_movement":
        return _wh_movement(toks, rng, lex)
    return phenomenon


# Anaphoric binding (positional)


def _binding(toks: list[Token], rng: random.Random, lex: dict,
             subtype: str | None = None) -> str:
    """Positional pronominal substitution (spec §7.1 rev., narrowed 2026-07-30).

    Collects every licensed (antecedent, target) CAT1 pair — the subtype's
    offset AND its antecedent-ordinal conjunct, ``binding.licensed_pairs`` —
    picks one uniformly, and substitutes the SECOND occurrence (always — P's
    substitution is deterministic) with the CAT1PRON of the subtype, copying the
    antecedent's number/gender. CAT1PRON tokens count as intervening items and
    are counted by the ordinal scan, but are never antecedents or targets (no
    pron-pron coreference).

    Coreference is not required of the drawn pair: ``_coindex`` below *creates*
    it, so the geometry question is the right one here (binding.py's ``forms``
    contract).

    Returns ``anaphoric_binding_skipped`` when no licensed pair exists in the
    drawn string. Under phenomenon-driven generation the geometry is forced
    (``generate._one_item``), so this is the L'/legacy and probe path only.
    """
    if subtype is None:  # legacy path (Grammar L')
        subtype = "Refl" if bernoulli(rng, 0.5, "bind_subtype_tok") else "Pron"

    pairs = binding.licensed_pairs([t.cat for t in toks], subtype)
    if not pairs:
        return "anaphoric_binding_skipped"

    i, j = rng.choice(pairs)
    anchor = toks[i]
    feats = {"INFL1_number": anchor.feats["INFL1_number"],
             "INFL1_gender": anchor.feats["INFL1_gender"]}
    # Base coindexation (TRANSFORM_RULES §2): make the target a genuine
    # duplicate of the anchor (never touch the anchor — it fixes verb agreement)
    # before the transform consumes it. Realign a subject target's verb here so
    # the coindexed base is agreement-consistent (G2).
    _coindex(toks, i, j)
    if toks[j].role == "subject":
        _fix_clause_agreement(toks, toks[j].clause_id, feats)
    # Record the antecedent form so licensing reconstructs the coindexed pair
    # (D3). Surface is unchanged (the pron form derives from feats).
    _substitute_pron(toks, j, cat1pron_item(lex, subtype), feats,
                     coref_form=anchor.lex["form"])
    return ("anaphoric_binding_refl" if subtype == "Refl"
            else "anaphoric_binding_pron")


def _fix_clause_agreement(toks: list[Token], clause_id: int, feats: dict) -> None:
    """Realign a clause's verb-carrier INFL1 features after subject substitution."""
    for t in toks:
        if t.cat in ("CAT3", "CAT3AUX") and t.clause_id == clause_id \
                and "INFL1_number" in t.feats:
            t.feats["INFL1_number"] = feats["INFL1_number"]
            t.feats["INFL1_gender"] = feats["INFL1_gender"]


def _substitute_pron(toks: list[Token], idx: int, pron_item: dict, feats: dict,
                     coref_form: str | None = None) -> None:
    """Replace the CAT1 at ``idx`` with a CAT1PRON and delete its CAT4/CAT2.

    ``coref_form`` is the antecedent CAT1's surface form, stamped on the pron so
    licensing can reconstruct the coreferential duplicate (see ``_binding``).
    """
    host = toks[idx]
    # Delete contiguous CAT4/CAT2 (+ their inflections, which live on the same
    # token) immediately following the host CAT1.
    j = idx + 1
    while j < len(toks) and toks[j].cat in ("CAT4", "CAT2") \
            and toks[j].clause_id == host.clause_id:
        del toks[j]
    toks[idx] = Token("CAT1PRON", pron_item, feats=feats,
                      role=host.role, clause_id=host.clause_id,
                      coref_form=coref_form)


def _coindex(toks: list[Token], anchor_idx: int, target_idx: int) -> None:
    """Make ``toks[target_idx]`` a genuine coreferential duplicate of
    ``toks[anchor_idx]`` in the licensing base the binding transform consumes
    (TRANSFORM_RULES §2). The anchor (first occurrence) is never touched — it
    fixes verb agreement; the target (second) is overwritten to the anchor's
    lexeme + INFL1_number + INFL1_gender (gender copied explicitly so no stale
    feat survives) and stripped of its direct modifiers (the contiguous CAT4/
    CAT2 immediately following it)."""
    anchor, target = toks[anchor_idx], toks[target_idx]
    target.lex = anchor.lex
    target.feats["INFL1_number"] = anchor.feats["INFL1_number"]
    target.feats["INFL1_gender"] = anchor.feats["INFL1_gender"]
    k = target_idx + 1
    while k < len(toks) and toks[k].cat in ("CAT4", "CAT2") \
            and toks[k].clause_id == target.clause_id:
        del toks[k]


def has_licensed_duplicate(toks: list[Token]) -> bool:
    """True iff a same-stem CAT1 pair sits in a LICENSED binding configuration.

    Licensed = the offset *and* the antecedent-ordinal conjunct of either
    subtype (``binding.has_licensed_pair`` with coreference enforced). That
    configuration exists only as the intermediate state of a binding derivation
    and must never surface unsubstituted: generation resamples on it.

    Narrowed 2026-07-30 with the rules themselves. A same-stem pair at a
    licensed *distance* whose antecedent sits at an unlicensed ordinal is no
    longer licensed, so it is no longer screened — it surfaces as a bare
    duplicate and becomes additional positional blocking evidence, exactly as
    matrix same-nucleus pairs do for the H family under rule A′.
    """
    cats, forms = binding.cats_and_forms(toks)
    return any(binding.has_licensed_pair(cats, s, forms)
               for s in binding.SUBTYPES)


# Auxiliary movement


def front_aux_at(toks: list[Token], aux_idx: int) -> None:
    """Front the CAT3AUX at ``aux_idx``, leaving a gap trace at its slot.

    Site-directed mechanics shared by the licensed transform (which picks the
    first-in-scan aux) and the structural probes' violating application
    (grammars/violations.py). The trace mirrors wh-movement (§8.1): compound
    tense is verb-then-aux, so the slot is right AFTER the main verb; the null
    morpheme strips to zero model tokens, so the model input is unchanged.
    """
    aux = toks.pop(aux_idx)
    gap = Token("GAP", None, role="gap", clause_id=aux.clause_id)
    toks.insert(aux_idx, gap)
    toks.insert(0, aux)


def _aux_movement(toks: list[Token]) -> str:
    """Front the FIRST CAT3AUX in left-to-right scan to absolute string-initial
    position, regardless of clausal domain (spec §7.2)."""
    aux_idx = _first(toks, cat="CAT3AUX")
    if aux_idx is None:
        return "auxiliary_movement_skipped"
    front_aux_at(toks, aux_idx)
    return "auxiliary_movement"


# Wh-movement (licensing formalization)


def _wh_movement(toks: list[Token], rng: random.Random, lex: dict) -> str:
    """Front a LICENSED nominal to sentence-initial position after CAT8.

    Licensed positions (see module docstring): third CAT1-position overall, or the
    second CAT1-position after the second CAT6 — evaluated on the base string. If no
    licensed target exists, returns
    "wh_movement_skipped" (positional blocking). The whole nominal block
    (CAT1/CAT1PRON plus contiguous CAT4/CAT2) moves; a null morpheme is left
    in place and CAT8 is prefixed.
    """
    candidates = _wh_candidates(toks)
    if not candidates:
        return "wh_movement_skipped"

    front_block(toks, rng.choice(candidates), lex)
    return "wh_movement"


def front_block(toks: list[Token], target_idx: int, lex: dict) -> None:
    """Front the nominal block at ``target_idx``, leaving a gap and prefixing
    CAT8. Site-directed mechanics shared by the licensed transform and the
    structural probes' violating application (grammars/violations.py)."""
    block = _extract_nominal_block(toks, target_idx)  # removes block from toks
    gap = Token("GAP", None, role="gap", clause_id=block[0].clause_id)
    toks.insert(target_idx, gap)

    cat8 = Token("CAT8", cat8_wh(lex), role="wh", clause_id=0)
    # Prepend: CAT8 then the fronted block, at absolute string-initial position.
    toks[0:0] = [cat8, *block]


def _wh_candidates(toks: list[Token]) -> list[int]:
    """Indices of nominal tokens in a licensed positional wh position.

    Both clauses come from ``grammar_P.wh``, which owns them (one copy, shared with
    L′, the probes and the annotation path):
      (i)  the third CAT1-position of the sentence;
      (ii) the second CAT1-position after the second CAT6 of the sentence.
    A CAT1-position is any CAT1 or CAT1PRON (D7); a licensed position is a TARGET
    iff its token is a CAT1 or a CAT1PRON-Pron (a Refl never fronts).
    """
    cats = [t.cat for t in toks]
    return sorted(
        p for p in wh.licensed_indices(cats)
        if wh.target_eligible(toks[p].cat, toks[p].lex)
    )


def _extract_nominal_block(toks: list[Token], idx: int) -> list[Token]:
    """Remove and return the nominal block starting at ``idx``.

    Block = the CAT1/CAT1PRON plus its contiguous CAT4 and CAT2 morphemes
    (same clause).
    """
    host = toks[idx]
    end = idx + 1
    while end < len(toks) and toks[end].cat in ("CAT4", "CAT2") \
            and toks[end].clause_id == host.clause_id:
        end += 1
    block = toks[idx:end]
    del toks[idx:end]
    return block


def _first(toks: list[Token], *, cat: str, role: str | None = None,
           clause_id: int | None = None) -> int | None:
    for i, t in enumerate(toks):
        if t.cat != cat:
            continue
        if role is not None and t.role != role:
            continue
        if clause_id is not None and t.clause_id != clause_id:
            continue
        return i
    return None
