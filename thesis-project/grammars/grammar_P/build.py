"""Positional sentence construction for Grammar P.

Builds a flat list of Token objects by a left-to-right, scan-based procedure.
No constituent tree is ever created; every insertion step operates only on
the string state produced by prior steps, and no step looks ahead to items
not yet inserted (experiment.md §5 "Constraint on scan-based generation").

Pipeline (experiment.md §8.2), per clause:
  2. base sentence        CAT1 (subject) + CAT3 (verb) [+ CAT1 (object)]
  3. inflection           INFL1 on subject; INFL1 (copied) + INFL3 on verb;
                          INFL1 on object
  4. obligatory items     CAT3AUX (compound tense), CAT4 (countable CAT1)
  5. first expansion       CAT2 (after CAT1), CAT5 (after CAT3)
  6. second expansion      CAT6 (PP), CAT9 (new substring → recurse)

Feature assignment happens inline (no separate features module) because
positional inflection is a fixed-position copy operation.

Spec reference: experiment.md §5, §8.2.
"""

from __future__ import annotations

import random

from .tokens import Token
from .lexicon import (
    cat1_items,
    cat3_items,
    cat6_np_items,
    cat6_vp_items,
    cat9_items,
    cat3aux_item,
    compatible_cat4,
    pick,
)
from .rules import (
    P_CAT2,
    P_CAT5,
    P_AUX,
    P_CAT6_NP,
    P_CAT6_NP_COMP,
    P_CAT6_VP,
    P_CAT9,
    INFL1_NUMBER_VALUES,
    INFL3_TENSE_VALUES,
)


def next_clause_id(counter: list) -> int:
    """Allocate a fresh clause id (counter[0] is the running clause index)."""
    counter[0] += 1
    return counter[0]


def build_sentence(
    rng: random.Random,
    lex: dict,
    counter: list,
    phenomenon: str | None,
    has_aux: bool,
    force_cat2_subject: int | None = None,
    force_cat9_depth: int | None = None,
) -> list[Token]:
    """Build a full positional sentence (matrix clause + any CAT9 substrings).

    Parameters
    ----------
    has_aux:
        Whether the matrix clause is compound tense (a CAT3AUX is inserted).
    force_cat2_subject:
        If set, the matrix subject receives exactly this many CAT2 modifiers
        (used by generalization items to control dependency length).
    force_cat9_depth:
        If set, exactly this many CAT9 substrings are nested (used by
        generalization items to control embedding depth). When ``None``,
        CAT9 nesting is drawn probabilistically.
    """
    return _build_clause(
        rng, lex, counter,
        phenomenon=phenomenon,
        clause_id=0,
        depth=0,
        has_aux=has_aux,
        force_cat2_main=force_cat2_subject,
        force_cat9_depth=force_cat9_depth,
    )

def _build_clause(
    rng: random.Random,
    lex: dict,
    counter: list,
    phenomenon: str | None,
    clause_id: int,
    depth: int,
    has_aux: bool,
    force_cat2_main: int | None = None,
    force_cat9_depth: int | None = None,
) -> list[Token]:
    """Build one positional clause and recursively any CAT9 substrings."""
    is_matrix = clause_id == 0

    # ── Step 2: base sentence ────────────────────────────────────────────────
    # A transitive object CAT1 is needed for reflexive binding (object slot)
    # and useful for wh-movement; force transitivity for those matrix clauses.
    force_transitive = is_matrix and phenomenon in ("anaphoric_binding", "wh_movement")
    transitivity = "transitive" if force_transitive else None

    subj_item = pick(rng, cat1_items(lex))
    verb_item = pick(rng, cat3_items(lex, transitivity=transitivity))
    is_transitive = verb_item["transitivity"] == "transitive"

    subj = Token("CAT1", subj_item, role="subject", clause_id=clause_id)
    verb = Token("CAT3", verb_item, role="verb", clause_id=clause_id)
    toks: list[Token] = [subj, verb]
    obj = None
    if is_transitive:
        obj = Token("CAT1", pick(rng, cat1_items(lex)), role="object", clause_id=clause_id)
        toks.append(obj)

    # ── Step 4a: CAT3AUX (compound tense) — inserted before inflection so the
    #    inflection step can route INFL1/INFL3 onto the auxiliary (spec §5.1.3).
    aux = None
    if has_aux:
        aux = Token("CAT3AUX", cat3aux_item(lex), role="aux", clause_id=clause_id)
        toks.insert(_index(toks, verb) + 1, aux)

    # ── Step 3: inflection (fixed-position copy) ─────────────────────────────
    n = rng.choice(INFL1_NUMBER_VALUES)
    subj.feats = {"INFL1_number": n, "INFL1_gender": subj_item["inherent_gender"]}
    tense = rng.choice(INFL3_TENSE_VALUES)
    if aux is not None:
        # Compound: CAT3 stays bare; INFL1 (copied from subject) + INFL3 on aux.
        verb.feats = {}
        aux.feats = {"INFL1_number": n, "INFL1_gender": subj_item["inherent_gender"],
                     "INFL3_tense": tense}
    else:
        verb.feats = {"INFL1_number": n, "INFL1_gender": subj_item["inherent_gender"],
                      "INFL3_tense": tense}
    if obj is not None:
        obj.feats = {"INFL1_number": rng.choice(INFL1_NUMBER_VALUES),
                     "INFL1_gender": obj.lex["inherent_gender"]}

    # ── Step 4b: CAT4 insertion after each countable base CAT1 ───────────────
    for cat1_tok in [t for t in toks if t.cat == "CAT1"]:
        _insert_cat4(toks, cat1_tok, lex)

    # ── Step 5: first expansion — CAT2 then CAT5 ─────────────────────────────
    for cat1_tok in [t for t in toks if t.cat == "CAT1" and t.role in ("subject", "object")]:
        forced = force_cat2_main if (cat1_tok.role == "subject") else None
        _insert_cat2(toks, cat1_tok, lex, rng, forced=forced)
    _insert_cat5(toks, verb, aux, lex, rng)

    # ── Step 6a: second expansion — CAT6 prepositional phrases ───────────────
    _insert_cat6(toks, subj, obj, verb, aux, lex, rng, counter, clause_id)

    # ── Step 6b: second expansion — CAT9 substrings (recurse) ────────────────
    _maybe_attach_cat9(
        toks, lex, rng, counter, depth, clause_id,
        force_cat9_depth=force_cat9_depth,
    )

    return toks


# ── Insertion helpers ─────────────────────────────────────────────────────────


def _index(toks: list[Token], tok: Token) -> int:
    """Index of a token by object identity."""
    for i, t in enumerate(toks):
        if t is tok:
            return i
    raise ValueError("token not in list")


def _insert_cat4(toks: list[Token], cat1_tok: Token, lex: dict) -> None:
    """Insert a CAT4 determiner immediately after a countable CAT1 (spec §5.1.2).

    The CAT4 form is fixed by the CAT1's INFL1_number value. Non-countable
    (proper-like) CAT1 items take no determiner.
    """
    if not cat1_tok.lex.get("cat4_required"):
        return
    cat4 = compatible_cat4(lex, cat1_tok.feats["INFL1_number"])
    pos = _index(toks, cat1_tok) + 1
    toks.insert(pos, Token("CAT4", cat4, role="det", clause_id=cat1_tok.clause_id))


def _insert_cat2(
    toks: list[Token], cat1_tok: Token, lex: dict, rng: random.Random,
    forced: int | None = None,
) -> None:
    """Insert CAT2 modifiers after a CAT1 (or after its CAT4), spec §5.1.4.

    Only countable CAT1 items admit CAT2 (parallel with Grammar H). CAT2 copies
    the host CAT1's INFL1 values positionally.
    """
    if not cat1_tok.lex.get("cat4_required"):
        return  # proper-like CAT1: no CAT2 (parallel with Grammar H)
    if forced is not None:
        n_cat2 = forced
    else:
        n_cat2 = 0
        while rng.random() < P_CAT2:
            n_cat2 += 1
    if n_cat2 == 0:
        return
    # Insertion point: after the CAT1, skipping its CAT4 determiner if present.
    pos = _index(toks, cat1_tok) + 1
    if pos < len(toks) and toks[pos].cat == "CAT4" and toks[pos].clause_id == cat1_tok.clause_id:
        pos += 1
    feats = {"INFL1_number": cat1_tok.feats["INFL1_number"],
             "INFL1_gender": cat1_tok.feats["INFL1_gender"]}
    for _ in range(n_cat2):
        cat2 = pick(rng, lex["cat2"])
        toks.insert(pos, Token("CAT2", cat2, feats=dict(feats),
                               role="modifier", clause_id=cat1_tok.clause_id))
        pos += 1


def _insert_cat5(
    toks: list[Token], verb: Token, aux: Token | None, lex: dict, rng: random.Random,
) -> None:
    """Insert CAT5 adverbs after CAT3 (or after CAT3AUX if present), spec §5.1.5."""
    anchor = aux if aux is not None else verb
    pos = _index(toks, anchor) + 1
    n_cat5 = 0
    while rng.random() < P_CAT5:
        n_cat5 += 1
    for _ in range(n_cat5):
        cat5 = pick(rng, lex["cat5"])
        toks.insert(pos, Token("CAT5", cat5, role="adverb", clause_id=verb.clause_id))
        pos += 1


def _nominal_block_end(toks: list[Token], cat1_tok: Token) -> int:
    """Return the index just past the CAT1's nominal block (CAT1 [CAT4] [CAT2]*)."""
    i = _index(toks, cat1_tok) + 1
    while i < len(toks) and toks[i].cat in ("CAT4", "CAT2") \
            and toks[i].clause_id == cat1_tok.clause_id:
        i += 1
    return i


def _insert_cat6(
    toks: list[Token], subj: Token, obj: Token | None, verb: Token, aux: Token | None,
    lex: dict, rng: random.Random, counter: list, clause_id: int,
) -> None:
    """Insert CAT6 prepositional phrases in two positional zones (spec §5.1.6).

    Nominal zone (PP-under-NP): lands after the first CAT2 if present, else at
    the end of the CAT1's block. Verbal zone (PP-under-VP): lands after the last
    CAT5, else after CAT3/CAT3AUX. Zone is recorded on the CAT6 token for Grammar L'.
    """
    n_added = 0

    # Nominal-zone PPs
    for cat1_tok in [subj] + ([obj] if obj is not None else []):
        if rng.random() >= P_CAT6_NP:  # inverted for early-exit inside the loop
            continue
        # Insertion point inside the nominal block (spec §5.1.6 first bullet).
        start = _index(toks, cat1_tok) + 1
        end = _nominal_block_end(toks, cat1_tok)
        first_cat2 = next((j for j in range(start, end) if toks[j].cat == "CAT2"), None)
        pos = (first_cat2 + 1) if first_cat2 is not None else end
        _emit_pp(toks, pos, zone="np", lex=lex, rng=rng, clause_id=clause_id)
        n_added += 1

    # Verbal-zone PP
    if rng.random() < P_CAT6_VP:
        anchor = aux if aux is not None else verb
        pos = _index(toks, anchor) + 1
        while pos < len(toks) and toks[pos].cat == "CAT5" \
                and toks[pos].clause_id == clause_id:
            pos += 1
        _emit_pp(toks, pos, zone="vp", lex=lex, rng=rng, clause_id=clause_id)
        n_added += 1


def _emit_pp(
    toks: list[Token], pos: int, zone: str, lex: dict, rng: random.Random, clause_id: int,
) -> None:
    """Insert a PP (``CAT6 CAT1 [CAT4] [CAT2]* [nested-PP]?``) at ``pos``.

    The complement CAT1 is expanded by the same steps as any other CAT1:
      (A) CAT4 if countable
      (B) CAT2 modifiers, geometric draw (P_CAT2)
      (C) one optional nested PP-under-NP, geometric gate (P_CAT6_NP_COMP)

    Step (C) is a recursive call to _emit_pp() with zone="np" — only NP-zone
    PPs can nest inside a complement NP.  Termination is almost sure because
    P_CAT6_NP_COMP < 1.

    licensor_attachment on the complement records the zone of the immediately
    licensing CAT6 so Grammar L' can apply its structural island rule without
    surface scanning.  Nested complements each get their own licensor_attachment
    set to "np" by their own recursive _emit_pp() call, so L' island logic
    works correctly at any depth without modification.
    """
    pool = cat6_np_items(lex) if zone == "np" else cat6_vp_items(lex)
    cat6 = Token("CAT6", pick(rng, pool), role="prep", clause_id=clause_id, attachment=zone)
    comp_item = pick(rng, cat1_items(lex))
    comp = Token("CAT1", comp_item, role="complement", clause_id=clause_id,
                 licensor_attachment=zone)
    comp.feats = {"INFL1_number": rng.choice(INFL1_NUMBER_VALUES),
                  "INFL1_gender": comp_item["inherent_gender"]}
    toks.insert(pos, cat6)
    toks.insert(pos + 1, comp)

    _insert_cat4(toks, comp, lex)                    # (A) determiner if countable
    _insert_cat2(toks, comp, lex, rng)               # (B) zero or more CAT2 modifiers

    if rng.random() < P_CAT6_NP_COMP:               # (C) optional nested PP-under-NP
        nested_pos = _nominal_block_end(toks, comp)  # end of comp's CAT1[CAT4][CAT2]* block
        _emit_pp(toks, nested_pos, zone="np", lex=lex, rng=rng, clause_id=clause_id)


def _maybe_attach_cat9(
    toks: list[Token], lex: dict, rng: random.Random,
    counter: list, depth: int, clause_id: int,
    force_cat9_depth: int | None = None,
) -> None:
    """Append a CAT9 + fresh positional clause and recurse (spec §5.1.7).

    Insertion point: after the last CAT2, else after the last CAT4, else after CAT1.
    """
    if force_cat9_depth is not None:
        do_attach = force_cat9_depth > 0
        child_force = force_cat9_depth - 1
    else:
        do_attach = rng.random() < P_CAT9
        child_force = None
    if not do_attach:
        return

    # Scan toks for the insertion point, restricted to the current clause.
    clause_indices = [i for i, t in enumerate(toks) if t.clause_id == clause_id]
    last_cat2 = next((i for i in reversed(clause_indices) if toks[i].cat == "CAT2"), None)
    last_cat4 = next((i for i in reversed(clause_indices) if toks[i].cat == "CAT4"), None)
    cat1_idx  = next((i for i in clause_indices if toks[i].cat == "CAT1"), None)
    if last_cat2 is not None:
        pos = last_cat2 + 1
    elif last_cat4 is not None:
        pos = last_cat4 + 1
    else:
        pos = cat1_idx + 1  # cat1_idx is always present (subject)

    cid = next_clause_id(counter)
    cat9 = Token("CAT9", pick(rng, cat9_items(lex)), role="sub", clause_id=cid)
    sub = _build_clause(
        rng, lex, counter,
        phenomenon=None, clause_id=cid, depth=depth + 1,
        has_aux=False, force_cat9_depth=child_force,
    )
    toks[pos:pos] = [cat9, *sub]
