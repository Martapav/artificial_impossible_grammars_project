"""Syntactic transformations for Grammar P (positional formalizations).

Every transformation operates on the flat surface token list by a
left-to-right scan; none consults constituent structure, c-command, or
generation history (that is what distinguishes P from H). Each function
mutates the list in place and returns the refined phenomenon label.

Reformalization reference: grammars/LINEAR_HIERARCHICAL_TRIVIALITY_AUDIT.md
(assessment section) and grammars/TRANSFORM_RULES.md.

  Anaphoric binding — purely positional, derivational duplicate-CAT1 rule.
  Distances count surface items (every whitespace token; #-attached
  inflection values are not separate items):
    Refl: licensed iff the coreferential CAT1 sits exactly REFL_OFFSET (3)
          items after its antecedent CAT1 in the base string; the second
          occurrence is obligatorily substituted by CAT1PRON-Refl copying
          the antecedent's number/gender.
    Pron: licensed iff the coreferential CAT1 sits exactly 2 or 4 items
          after its antecedent; the second occurrence is obligatorily
          substituted by CAT1PRON-Pron copying the antecedent's features.
    Duplicates at any other distance are unlicensed: they surface
    unsubstituted (the positional blocking evidence). Duplicates at a
    licensed distance never surface (generation screens them).

  Auxiliary movement — front the FIRST CAT3AUX in scan order, regardless of
    clause (unchanged). Never blocked when an aux exists anywhere.

  Wh-movement — LICENSING formalization (audit §F, final): a nominal may
  front iff it is
      (i)  the third CAT1-position of the sentence, or
      (ii) the first CAT1-position after a CAT5.
    The second-CAT1-after-CAT9 condition is removed (audit §F: rule-mass
    reduction, and it kept wh–embedding correlations parallel with H after H
    dropped embedded subjects). CAT1PRON counts as a CAT1-position for the
    ordinal scan; as a TARGET, CAT1 and CAT1PRON-Pron are licit,
    CAT1PRON-Refl never fronts. If no licensed target exists (e.g. bare SVO
    with no adverb), the phenomenon is blocked — positional blockability.
"""

from __future__ import annotations

import random

from .tokens import Token
from .lexicon import cat1pron_item, cat8_wh
from .rules import REFL_OFFSET, PRON_OFFSETS


def apply(toks: list[Token], phenomenon: str, rng: random.Random, lex: dict) -> str:
    """Dispatch to the pre-selected transformation; return the refined label.

    Returns the phenomenon suffixed with ``_skipped`` when it cannot apply to
    the drawn string (no licensed geometry), so the corpus builder can drop
    it. ``neutral`` returns unchanged.
    """
    if phenomenon == "anaphoric_binding":  # legacy dispatch (L', gen items)
        phenomenon = ("anaphoric_binding_refl" if rng.random() < 0.5
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
             subtype: str | None = None, embedded_only: bool = False) -> str:
    """Positional pronominal substitution (distance-licensed, spec §7.1 rev.).

    Collects every (antecedent, target) CAT1 pair at the subtype's licensed
    offsets, picks one uniformly, and substitutes the SECOND occurrence
    (always — P's substitution is deterministic) with the CAT1PRON of the
    subtype, copying the antecedent's number/gender. CAT1PRON tokens count as
    intervening items but are never antecedents or targets (no pron-pron
    coreference).

    ``embedded_only`` restricts targets to embedded tokens (background
    binding). Returns ``anaphoric_binding_skipped`` when no licensed pair
    exists in the drawn string.
    """
    if subtype is None:  # legacy path (Grammar L')
        subtype = "Refl" if rng.random() < 0.5 else "Pron"
    offsets = (REFL_OFFSET,) if subtype == "Refl" else PRON_OFFSETS

    pairs: list[tuple[int, int]] = []
    for i, t in enumerate(toks):
        if t.cat != "CAT1":
            continue
        for off in offsets:
            j = i + off
            if j >= len(toks) or toks[j].cat != "CAT1":
                continue
            if embedded_only and toks[j].clause_id == 0:
                continue
            pairs.append((i, j))
    if not pairs:
        return "anaphoric_binding_skipped"

    i, j = rng.choice(pairs)
    anchor, target = toks[i], toks[j]
    feats = {"INFL1_number": anchor.feats["INFL1_number"],
             "INFL1_gender": anchor.feats["INFL1_gender"]}
    # Substituting a subject: its clause's verb carrier copied the old
    # subject's INFL1 values at build time — realign with the pron's features.
    if target.role == "subject":
        _fix_clause_agreement(toks, target.clause_id, feats)
    _substitute_pron(toks, j, cat1pron_item(lex, subtype), feats)
    return ("anaphoric_binding_refl" if subtype == "Refl"
            else "anaphoric_binding_pron")


def _fix_clause_agreement(toks: list[Token], clause_id: int, feats: dict) -> None:
    """Realign a clause's verb-carrier INFL1 features after subject substitution."""
    for t in toks:
        if t.cat in ("CAT3", "CAT3AUX") and t.clause_id == clause_id \
                and "INFL1_number" in t.feats:
            t.feats["INFL1_number"] = feats["INFL1_number"]
            t.feats["INFL1_gender"] = feats["INFL1_gender"]


def _substitute_pron(toks: list[Token], idx: int, pron_item: dict, feats: dict) -> None:
    """Replace the CAT1 at ``idx`` with a CAT1PRON and delete its CAT4/CAT2."""
    host = toks[idx]
    # Delete contiguous CAT4/CAT2 (+ their inflections, which live on the same
    # token) immediately following the host CAT1.
    j = idx + 1
    while j < len(toks) and toks[j].cat in ("CAT4", "CAT2") \
            and toks[j].clause_id == host.clause_id:
        del toks[j]
    toks[idx] = Token("CAT1PRON", pron_item, feats=feats,
                      role=host.role, clause_id=host.clause_id)


def has_licensed_duplicate(toks: list[Token]) -> bool:
    """True iff two same-stem CAT1s sit at a licensed binding offset (2/3/4).

    That configuration is licensed only as the intermediate state of a
    binding derivation and must never surface unsubstituted: generation
    resamples on it. Duplicates at any other distance are legal (they are the
    positional blocking evidence).
    """
    all_offsets = (REFL_OFFSET,) + tuple(PRON_OFFSETS)
    for i, t in enumerate(toks):
        if t.cat != "CAT1":
            continue
        for off in all_offsets:
            j = i + off
            if j < len(toks) and toks[j].cat == "CAT1" \
                    and toks[j].lex["form"] == t.lex["form"]:
                return True
    return False


# Auxiliary movement


def _aux_movement(toks: list[Token]) -> str:
    """Front the FIRST CAT3AUX in left-to-right scan to absolute string-initial
    position, regardless of clausal domain (spec §7.2)."""
    aux_idx = _first(toks, cat="CAT3AUX")
    if aux_idx is None:
        return "auxiliary_movement_skipped"
    aux = toks.pop(aux_idx)
    toks.insert(0, aux)
    return "auxiliary_movement"


# Wh-movement (licensing formalization)


def _wh_movement(toks: list[Token], rng: random.Random, lex: dict) -> str:
    """Front a LICENSED nominal to sentence-initial position after CAT8.

    Licensed positions (see module docstring): third CAT1-position overall,
    first CAT1-position after a CAT5 — evaluated on the base string. If no
    licensed target exists, returns
    "wh_movement_skipped" (positional blocking). The whole nominal block
    (CAT1/CAT1PRON plus contiguous CAT4/CAT2) moves; a null morpheme is left
    in place and CAT8 is prefixed.
    """
    candidates = _wh_candidates(toks)
    if not candidates:
        return "wh_movement_skipped"

    target_idx = rng.choice(candidates)
    block = _extract_nominal_block(toks, target_idx)  # removes block from toks
    gap = Token("GAP", None, role="gap", clause_id=block[0].clause_id)
    toks.insert(target_idx, gap)

    cat8 = Token("CAT8", cat8_wh(lex), role="wh", clause_id=0)
    # Prepend: CAT8 then the fronted block, at absolute string-initial position.
    toks[0:0] = [cat8, *block]
    return "wh_movement"


def _wh_candidates(toks: list[Token]) -> list[int]:
    """Indices of nominal tokens in a licensed positional wh position.

    A CAT1-position is any CAT1 or CAT1PRON token (the ordinal scan does not
    distinguish them). Licensed positions:
      (i)  the third CAT1-position of the sentence;
      (ii) the first CAT1-position after a CAT5 (per CAT5).
    A licensed position is a TARGET iff its token is a CAT1 or a
    CAT1PRON-Pron (a Refl never fronts).
    """
    positions = [i for i, t in enumerate(toks) if t.cat in ("CAT1", "CAT1PRON")]
    licensed: set[int] = set()
    if len(positions) >= 3:
        licensed.add(positions[2])                        # condition (i)
    for i, t in enumerate(toks):
        if t.cat == "CAT5":
            after = [p for p in positions if p > i]
            if after:
                licensed.add(after[0])                    # condition (ii)
    return sorted(
        p for p in licensed
        if toks[p].cat == "CAT1"
        or (toks[p].lex is not None and toks[p].lex.get("subclass") == "Pron")
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
