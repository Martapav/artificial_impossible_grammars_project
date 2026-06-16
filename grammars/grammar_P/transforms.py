"""Syntactic transformations for Grammar P (positional formalizations).

Every transformation operates on the flat surface token list by a
left-to-right scan; none consults constituent structure, c-command, or
generation history (that is what distinguishes P from H, and what Grammar L'
will override for the island rule). Each function mutates the list in place
and returns the refined phenomenon label.

Spec reference:
  experiment.md §7.1 (binding — positional Principles A/B),
  §7.2 (auxiliary movement — front the first CAT3AUX in scan order),
  §7.3 (wh-movement — positional blocking conditions).
"""

from __future__ import annotations

import random

from .tokens import Token
from .lexicon import cat1pron_item, cat8_wh
from .rules import INFL1_NUMBER_VALUES, INFL1_GENDER_VALUES


def apply(toks: list[Token], phenomenon: str, rng: random.Random, lex: dict) -> str:
    """Dispatch to the pre-selected transformation; return the refined label.

    Returns the phenomenon suffixed with ``_skipped`` when it cannot apply to
    the drawn string (e.g. binding/aux with no licensing item), so the corpus
    builder can drop it. ``neutral`` returns unchanged.
    """
    if phenomenon == "anaphoric_binding":
        return _binding(toks, rng, lex)
    if phenomenon == "auxiliary_movement":
        return _aux_movement(toks)
    if phenomenon == "wh_movement":
        return _wh_movement(toks, rng, lex)
    return phenomenon


# Anaphoric binding


def _binding(toks: list[Token], rng: random.Random, lex: dict) -> str:
    """Pronominal substitution by surface position (spec §7.1).

    Refl (Principle A): substituted at the object position, copies subject's number/gender.
    Pron (Principle B): substituted at a non-object CAT1 position.
    The host CAT1's contiguous CAT4/CAT2 morphemes are deleted on substitution.
    """
    subj_idx = _first(toks, cat="CAT1", role="subject", clause_id=0)

    if subj_idx is None:
        return "anaphoric_binding_skipped"
    subj = toks[subj_idx]

    use_refl = rng.random() < 0.5
    use_pron = rng.random() < 0.5

    if use_refl:
        obj_idx = _first(toks, cat="CAT1", role="object", clause_id=0)
        if obj_idx is None:
            return "anaphoric_binding_skipped"
        feats = {"INFL1_number": subj.feats["INFL1_number"],
                 "INFL1_gender": subj.feats["INFL1_gender"]}
        _substitute_pron(toks, obj_idx, cat1pron_item(lex, "Refl"), feats)
        return "anaphoric_binding_refl"

    # Pron: substitute the cat1 in non-object position
    if use_pron:
        no_obj_idx = _other_pos_cat1(toks, cat="CAT1", role="object", clause_id=0, rng=rng)
        if no_obj_idx is not None:
            feats = {"INFL1_number": toks[no_obj_idx].feats["INFL1_number"],
                    "INFL1_gender": toks[no_obj_idx].feats["INFL1_gender"]}
            _substitute_pron(toks, no_obj_idx, cat1pron_item(lex, "Pron"), feats)
        if not use_pron or no_obj_idx is None: 
            return "anaphoric_binding_skipped"
    return "anaphoric_binding_pron"


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


# Wh-movement


def _wh_movement(toks: list[Token], rng: random.Random, lex: dict) -> str:
    """Front a licit CAT1 to sentence-initial position after a CAT8 marker.

    A CAT1 is blocked from fronting iff (spec §7.3, positional):
      1. it is the second CAT1 to occur in the sentence (= object in SVO base);
      2. it is the first CAT1 occurring after a CAT9 (= subordinate subject);
      3. it is the CAT1 immediately preceded by a CAT6 (= PP complement).
    Condition 3 is the surface-scan rule that Grammar L' replaces with a
    generation-history rule.

    The matrix subject (first CAT1) is NOT blocked by any condition, so it is
    almost always frontable — this is the deliberate mirror image of Grammar H,
    where the matrix subject is an island. The whole nominal block (CAT1, its
    inflections, CAT4 and CAT2 modifiers) moves; a null morpheme is left in its
    place and CAT8 is prefixed.
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
    """Return indices of CAT1 tokens eligible for positional wh-fronting."""
    eligible: list[int] = []
    cat1_count = 0
    pending_after_cat9 = False
    for i, t in enumerate(toks):
        if t.cat == "CAT9":
            pending_after_cat9 = True
            continue
        if t.cat != "CAT1":
            continue
        cat1_count += 1
        is_second = cat1_count == 2                      # condition 1
        is_sub_subject = pending_after_cat9              # condition 2
        prev_is_cat6 = i > 0 and toks[i - 1].cat == "CAT6"   # condition 3
        pending_after_cat9 = False
        if not (is_second or is_sub_subject or prev_is_cat6):
            eligible.append(i)
    return eligible


def _extract_nominal_block(toks: list[Token], idx: int) -> list[Token]:
    """Remove and return the CAT1 nominal block starting at ``idx``.

    Block = the CAT1 plus its contiguous CAT4 and CAT2 morphemes (same clause).
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

def _other_pos_cat1(toks: list[Token], *, cat: str, role: str | None = None,
           clause_id: int | None = None, rng: random.Random) -> int | None:
    count = []
    for i, t in enumerate(toks):
        if t.cat != cat:
            continue
        if role is not None and t.role == role:
            continue
        if clause_id is not None and t.clause_id != clause_id:
            continue
        count.append(i)
    if not count:
        return None
    return rng.choice(count)
