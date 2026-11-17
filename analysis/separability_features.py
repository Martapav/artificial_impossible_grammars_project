"""Candidate-level feature tables for the separability audit.

This is the data layer of the descriptive "separability" analysis. The question
it feeds is uniform across all four grammars and all four phenomena:

    Can a WEAK, LOCAL, surface-CATEGORY-only classifier — looking only at a
    small window of neighbouring token categories around a candidate — predict
    whether that candidate sits in a position LICENSED by the phenomenon's rule?

Read as REASSURING for the positional families (P, L′ — the rule *is* local, so
predictability is expected) and as a WARNING for the hierarchical families
(H, H′ — local predictability means the "structural" phenomenon has an
accidental n-gram shortcut; TRANSFORM_RULES §1(b) "non-reducible divergence").

Design decisions (confirmed with Marta, 2026-07-26 — see
``docs/SEPARABILITY.md``):

  * **One observation per candidate**, centred on the candidate's own
    **canonical (base) position**. The window CENTRE is excluded from features
    (it is the target; its category is constant within a phenomenon anyway).

  * **Base order, via replay** — not the derived-string sidecar. Every base is
    drawn fresh from the grammar's natural generation (``_draw_base``; same
    rules/probabilities as training) and the transform is NEVER applied: we
    audit the *base* configuration, before any fronting/substitution mutates the
    surface. This (a) removes the confound that sank the old binding panel — a
    reflexive/pronoun's stripped-modifier neighbourhood leaking "this is the
    licensed one" — because on the base every candidate is a plain CAT1 with an
    intact neighbourhood; and (b) makes the licensed target category-identical
    to the distractors, so any predictive power is genuinely CONTEXT.

  * **Licensing = the rule's own site geometry**, read straight from the
    site-enumeration machinery the probe builder uses
    (``violations.wh_sites`` / ``aux_sites`` / ``binding_sites``). For binding
    this is the structural notion that already requires an antecedent to
    corefer with (``has_antecedent``) — "no coreference, no binding".

  * **All eligible positions, every sentence, all constructions.** The site
    tables enumerate EVERY eligible token (all CAT1 for wh/binding, all CAT3AUX
    for aux) with its licensed flag, so both classes are addressed in every
    drawn base.

  * **Regimes = length window on the drawn base** (TRANSFORM_RULES §9's
    "length is the only difference"): ``train`` (natural lengths, [2, 25]) as
    the descriptive baseline / IOD anchor, and ``ood`` ([26, 48]) for the
    length-stability comparison. Same draw machinery, same rules; only the
    length window differs.

Features are CATEGORY-only (``grammars.metadata._ref_cat``) — no lexical stem,
no tree access beyond identity mapping — i.e. exactly what a surface n-gram
observer could see.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

from grammars import metadata as M
from grammars import violations as X
from corpusgen.structural_probes import _draw_base, _registry

GRAMMARS = ("H", "P", "Hprime", "Lprime")
PAD = "PAD"

# Widest window built into every frame. Covers the full left/right context of any
# train sentence (len ≤ 25); a cell subselects a narrower ±w from it. Exact
# whole-side COUNTS (below) are computed independently of the window, so the
# full-reach reference is exact even for OOD sentences longer than WMAX.
WMAX = 25

# Base-order terminal categories counted per side (the counting-rule reference).
CATEGORY_VOCAB = ("CAT1", "CAT2", "CAT3", "CAT3AUX", "CAT4", "CAT5", "CAT6",
                  "CAT9")

# Regime -> (min_len, max_len) on the BASE word count.
REGIMES: Dict[str, Tuple[int, int]] = {
    "train": (2, 25),
    "ood": (26, 48),
}

# Phenomena and the eligible-token category each ranges over.
PHENOMENA = ("wh_i", "wh_ii", "aux", "refl", "pron")

# RULEWISE band (2026-07-27). ``in_band`` (the "comparable" band) controls for a
# precondition SHARED by all grammars; it is deliberately uniform, which is what
# makes cells comparable ACROSS grammars. But a rule whose domain is narrower than
# that precondition still gets a freebie: for H-family refl the negatives left in
# the comparable band are exactly the PP complements (L1=CAT6) and embedded
# subjects (L1=CAT9), each perfectly marked by its selecting head — so U measures
# objecthood, not the rule.
#
# The rulewise band instead controls for the RULE'S OWN precondition, so U measures
# only what the rule decides. It defaults to ``in_band``; the cells below override
# it with objecthood, because their rule's domain is the direct objects:
#   * H/H' refl  — the rule's domain is the direct objects
#   * H/L' wh_i  — the rule's domain is the direct objects: clause (i) is E3,
#     "matrix object AND it hosts an embedded clause" (2026-07-30), so the
#     licensed set is a strict SUBSET of the matrix objects. Banding on
#     objecthood therefore leaves the non-hosting matrix objects as negatives,
#     which is exactly the contrast E3 introduced — the same shape as A′ below.
#     (Before E3 the rule WAS "matrix object", and this cell had the degenerate
#     problem A′ fixed for refl.)
#
# UPDATED 2026-07-29 (rule A′). This band used to be EMPTY of negatives for H/H'
# refl, because ``binding_hfamily_licensed_ids`` returned exactly the set of direct
# objects (446/446 over 800 bases) — a rule with no negatives inside its own
# precondition class is a definition, not a rule. A′ adds the domain condition
# "the nucleus must be embedded", so the matrix objects are now the negatives and
# the cell is populated and informative: rulewise U runs
# −0.004 / 0.035 / 0.309 / 0.572 / 0.823 across τ = 0.5/1/2/4/∞ (memory-graded,
# τ_recover ≈ 4), against 0.966 flat under the old rule.
#
# NOT comparable across grammars by construction — read a rulewise U within a
# grammar, against ``comparable`` for cross-grammar claims.
OBJECT_DOMAIN_RULES = {("H", "refl"), ("Hprime", "refl"),
                       ("H", "wh_i"), ("Lprime", "wh_i")}


# ---------------------------------------------------------------------------
# Base-order linearisation + candidate-node -> linear-index mapping
# ---------------------------------------------------------------------------

NOMINAL_CATS = ("CAT1", "CAT1PRON")


def _infl(word: str):
    """The agreement tuple ``(number, gender)`` of any inflected surface word
    ``stem#number#gender[#…]``; ``None`` for uninflected function words.

    Used stem-free (NOT a coreference tautology). Covers ALL inflected tokens —
    nominals (binding coreference copies ``INFL1_number``+``INFL1_gender``, grammar_H
    ``_coindex``) AND modifiers/verbs (whose agreement is what the neutral
    agreement-violation control breaks)."""
    parts = word.split("#")
    return (parts[1], parts[2]) if len(parts) >= 3 else None


def _linearise(base, family: str):
    """Return (categories, infls, ref_index_by_id) for a base in base order.

    ``categories[i]`` is the surface category; ``infls[i]`` the (number,gender)
    agreement tuple for inflected tokens (else None); ``ref_index_by_id`` maps a
    terminal/token id to its linear index (to centre the window on a site handle).
    """
    refs, sep, _null = M._refs_and_constants(base, family)
    cats = [M._ref_cat(r) for _w, r in refs]
    infls = [_infl(w) for w, _r in refs]
    ref_index = {id(r): i for i, (_w, r) in enumerate(refs)}
    return cats, infls, ref_index


def _wh_center(base, family, site, ref_index) -> Optional[int]:
    if family == M.FAMILY_TOKENS:
        return site  # site is already the token/linear index
    from grammars.grammar_H.transforms import _wh_targetable_nominals
    from grammars.grammar_H_prime.transforms import _head_terminal
    node = _wh_targetable_nominals(base)[site]["node"]
    return ref_index.get(id(_head_terminal(node)))


def _aux_center(base, family, site, ref_index) -> Optional[int]:
    if family == M.FAMILY_TOKENS:
        return site
    from grammars.violations import _aux_positions_tree
    t2, idx = _aux_positions_tree(base)[site]
    return ref_index.get(id(t2.children[idx]))


def _binding_center(base, family, site, ref_index) -> Optional[int]:
    if family == M.FAMILY_TOKENS:
        return site
    from grammars.violations import _full_nominals_tree
    node = _full_nominals_tree(base)[site]
    return ref_index.get(id(node["head"]))


def _object_flags(base, family: str) -> Tuple[Optional[List[bool]], Optional[List[bool]]]:
    """``(binding_site_is_object, wh_site_is_object)`` indexed by site handle, or
    ``(None, None)`` for the tokens family (whose handles are token indices, so
    objecthood is read straight off ``base[i].role``).

    Objecthood is the precondition of the ``OBJECT_DOMAIN_RULES`` cells. Computed
    ONCE per base here rather than per candidate — the ``_*_center`` helpers each
    rebuild their site table per call, and doing the same for roles would make
    frame building quadratic in the candidate count.
    """
    if family == M.FAMILY_TOKENS:
        return None, None
    from grammars.violations import _full_nominals_tree
    from grammars.grammar_H.transforms import _wh_targetable_nominals
    return ([n["node"].role == "object" for n in _full_nominals_tree(base)],
            [c["node"].role == "object" for c in _wh_targetable_nominals(base)])


# ---------------------------------------------------------------------------
# Per-base candidate extraction (all phenomena)
# ---------------------------------------------------------------------------

def _candidates_for_base(base, grammar: str, family: str,
                         cats: List[str], ref_index: Dict[int, int]
                         ) -> List[Dict]:
    """One dict per (phenomenon, candidate) on the base struct.

    Each dict carries ``phenomenon``, ``center`` (linear base index),
    ``category`` (the candidate's base category), ``is_licensed`` (bool), and
    ``in_band`` — whether the candidate belongs to the COMPARABLE band that
    isolates the grammar-specific rule from a precondition cue shared by all
    grammars (option 2, confirmed 2026-07-26). Banding trims only the unlicensed
    pool; every licensed candidate is in-band by construction.

      * binding (refl/pron): in_band = ``has_antecedent`` — the candidate has a
        preceding CAT1 that could be its antecedent. Removes the free
        "sentence-initial subject has no antecedent → unlicensed" giveaway that
        every grammar shares ("no coreference, no binding").
      * wh (wh_i/wh_ii): in_band = the candidate is NOT the first eligible
        nominal — a wh target always has structure before it (a matrix object
        follows its subject and verb; the third CAT1-position follows two others;
        the second nominal after the second CAT6 follows both CAT6s), so the
        first nominal is a shared freebie.
      * aux: in_band = True (candidates are homogeneous CAT3AUX; the P-family
        rule licenses the FIRST aux, so a "non-initial" band would wrongly drop
        the positive class — no shared freebie to remove).
    """
    per_phen: Dict[str, List[Dict]] = {p: [] for p in PHENOMENA}
    bind_obj, wh_obj = _object_flags(base, family)

    def _is_object(site, flags) -> bool:
        if flags is None:                       # tokens family: site == token index
            return getattr(base[site], "role", None) == "object"
        return bool(flags[site])

    for s in X.wh_sites(base, family, grammar):
        c = _wh_center(base, family, s["site"], ref_index)
        if c is None:
            continue
        obj = _is_object(s["site"], wh_obj)
        per_phen["wh_i"].append({"center": c, "category": cats[c],
                                 "is_licensed": bool(s["licensed_i"]),
                                 "_is_object": obj})
        per_phen["wh_ii"].append({"center": c, "category": cats[c],
                                  "is_licensed": bool(s["licensed_ii"]),
                                  "_is_object": obj})

    for s in X.aux_sites(base, family):
        c = _aux_center(base, family, s["site"], ref_index)
        if c is None:
            continue
        per_phen["aux"].append({"center": c, "category": cats[c],
                                "is_licensed": bool(s["licensed"]),
                                "_in_band": True, "_is_object": False})

    for subtype, phen in (("Refl", "refl"), ("Pron", "pron")):
        for s in X.binding_sites(base, family, subtype):
            c = _binding_center(base, family, s["site"], ref_index)
            if c is None:
                continue
            per_phen[phen].append({
                "center": c, "category": cats[c],
                "is_licensed": bool(s["licensed"] and s["has_antecedent"]),
                "_in_band": bool(s["has_antecedent"]),
                "_is_object": _is_object(s["site"], bind_obj)})

    out: List[Dict] = []
    for phen, cands in per_phen.items():
        if not cands:
            continue
        if phen in ("wh_i", "wh_ii"):
            first = min(x["center"] for x in cands)
            for x in cands:
                x["_in_band"] = x["center"] != first
        object_domain = (grammar, phen) in OBJECT_DOMAIN_RULES
        for x in cands:
            # licensed candidates are always in-band; banding trims negatives.
            in_band = bool(x["is_licensed"] or x["_in_band"])
            rule_raw = x["_is_object"] if object_domain else x["_in_band"]
            out.append({"phenomenon": phen, "center": x["center"],
                        "category": x["category"],
                        "is_licensed": x["is_licensed"], "in_band": in_band,
                        "in_rule_band": bool(x["is_licensed"] or rule_raw)})
    return out


def _window(cats: List[str], center: int, w: int) -> Dict[str, str]:
    """2*w position-tagged category slots around ``center`` (centre excluded),
    plus ``realized_len`` = number of non-PAD slots."""
    out: Dict[str, str] = {}
    realized = 0
    for k in range(1, w + 1):
        li, ri = center - k, center + k
        lv = cats[li] if 0 <= li < len(cats) else PAD
        rv = cats[ri] if 0 <= ri < len(cats) else PAD
        out[f"L{k}"] = lv
        out[f"R{k}"] = rv
        realized += (lv != PAD) + (rv != PAD)
    out["realized_len"] = realized
    return out


def _counts(cats: List[str], center: int) -> Dict[str, int]:
    """EXACT whole-side category counts (the full-reach counting reference,
    independent of the window width): ``nL_<cat>`` over positions [0, center),
    ``nR_<cat>`` over (center, end]. This is what a counting rule (nth-position,
    first-in-scan) needs — recoverability from it distinguishes 'reach-limited
    counting' from 'genuinely structural' rules."""
    left = cats[:center]
    right = cats[center + 1:]
    out: Dict[str, int] = {}
    for c in CATEGORY_VOCAB:
        out[f"nL_{c}"] = left.count(c)
        out[f"nR_{c}"] = right.count(c)
    return out


def _featmatch(infls: List, center: int, w: int) -> Dict[str, str]:
    """Per-window-position agreement-match indicator: ``fmL{k}``/``fmR{k}`` ∈
    {match, diff, na} — does the neighbour k slots away share the candidate's
    (number,gender)? ``na`` if the candidate or neighbour is uninflected / out of
    range. NOT stem-based (that would be a coreference tautology). Drives the
    binding coreference cue AND the neutral agreement-violation control.
    """
    own = infls[center] if 0 <= center < len(infls) else None
    out: Dict[str, str] = {}
    for k in range(1, w + 1):
        for tag, idx in (("L", center - k), ("R", center + k)):
            nb = infls[idx] if 0 <= idx < len(infls) else None
            if own is None or nb is None:
                out[f"fm{tag}{k}"] = "na"
            else:
                out[f"fm{tag}{k}"] = "match" if nb == own else "diff"
    return out


# ---------------------------------------------------------------------------
# Frame builder
# ---------------------------------------------------------------------------

def build_candidate_frame(
    grammar: str,
    regime: str,
    n_bases: int,
    seed: int,
    w: int = WMAX,
    max_attempts_factor: int = 600,
) -> pd.DataFrame:
    """Draw ``n_bases`` natural bases in ``regime``'s length window and return a
    tidy candidate frame (one row per phenomenon-candidate) with a ±``w`` base-order
    category window PLUS exact whole-side counts (``nL_*``/``nR_*``)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}; choose from {list(REGIMES)}")
    lo, hi = REGIMES[regime]
    family, load_lex = _registry()[grammar]
    lex = load_lex()
    rng = random.Random(seed)

    rows: List[Dict] = []
    kept = 0
    attempts = 0
    cap = n_bases * max_attempts_factor
    while kept < n_bases and attempts < cap:
        attempts += 1
        base = _draw_base(family, rng, lex)
        cats, infls, ref_index = _linearise(base, family)
        if not (lo <= len(cats) <= hi):
            continue
        base_id = f"{grammar}-{regime}-{kept:06d}"
        for cand in _candidates_for_base(base, grammar, family, cats, ref_index):
            row = {
                "grammar": grammar,
                "regime": regime,
                "base_id": base_id,
                "base_len": len(cats),
                "phenomenon": cand["phenomenon"],
                "category": cand["category"],
                "is_licensed": cand["is_licensed"],
                "in_band": cand["in_band"],
                "in_rule_band": cand["in_rule_band"],
                "center": cand["center"],
                **_window(cats, cand["center"], w),
                **_counts(cats, cand["center"]),
                **_featmatch(infls, cand["center"], w),
            }
            rows.append(row)
        kept += 1

    if kept < n_bases:
        raise RuntimeError(
            f"{grammar}/{regime}: only {kept}/{n_bases} bases in [{lo},{hi}] "
            f"after {attempts} attempts")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Neutral control cells — cross-member, from the structural-probe splits
# ---------------------------------------------------------------------------

PROBE_SPLITS = {"iod": "structural_probe_IOD", "ood": "structural_probe_OOD"}


def build_neutral_frame(probe_regime: str = "iod") -> pd.DataFrame:
    """Neutral-violation control cells from the probe pairs.

    Positive control for the whole method: the candidate is the EDITED token of a
    neutral minimal pair; label ``is_licensed`` = the *grammatical* member (vs the
    ungrammatical one). Read cross-member on each member's surface string.

      * ``neutral_agreement`` — one inflection flipped: the category window is
        IDENTICAL across members (only agreement changed), so it should be
        detectable ONLY via the feature-match (``fmL/fmR``) channel — the control
        that validates that channel.
      * ``neutral_positional`` — one token relocated: the category window differs,
        so it should be detectable from categories alone.
    """
    import json
    from analysis.lexicon_map import categorize_sentence, load_shared_stem_map
    from models.dataload import CORPORA_ROOT as corpora
    split = PROBE_SPLITS[probe_regime]
    stem_map = load_shared_stem_map()
    rows: List[Dict] = []
    for grammar in GRAMMARS:
        cp = corpora / split / f"{grammar}.jsonl"
        mp = corpora / split / f"{grammar}.meta.jsonl"
        members: Dict[str, Dict] = {}
        with cp.open() as cf, mp.open() as mf:
            for cl, ml in zip(cf, mf):
                c, m = json.loads(cl), json.loads(ml)
                if m.get("construction") != "neutral":
                    continue
                members.setdefault(m["pair_id"], {})[c["variant"]] = m
        for pair_id, mem in members.items():
            g_m, u_m = mem.get("grammatical"), mem.get("ungrammatical")
            if not (g_m and u_m):
                continue
            subcase = g_m.get("subcase")
            gw, uw = g_m["sentence"].split(), u_m["sentence"].split()
            diff = [i for i in range(min(len(gw), len(uw))) if gw[i] != uw[i]]
            if not diff:
                continue
            center = diff[0]
            for m_rec, words, variant in ((g_m, gw, "grammatical"),
                                          (u_m, uw, "ungrammatical")):
                cats = [c[0] for c in categorize_sentence(m_rec["sentence"], stem_map)]
                infls = [_infl(w) for w in words]
                rows.append({
                    "grammar": grammar, "regime": f"probe_{probe_regime}",
                    "base_id": pair_id, "base_len": len(words),
                    "phenomenon": subcase, "category": cats[center]
                    if center < len(cats) else PAD,
                    "is_licensed": variant == "grammatical", "in_band": True,
                    "in_rule_band": True,   # neutral controls have no rule domain
                    "center": center,
                    **_window(cats, center, WMAX),
                    **_counts(cats, center),
                    **_featmatch(infls, center, WMAX),
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":  # smoke test
    for g in ("H", "P"):
        df = build_candidate_frame(g, "train", n_bases=200, seed=20260726)
        wcols = [c for c in df.columns if c.startswith(("L", "R")) and c[1:].isdigit()]
        ncols = [c for c in df.columns if c.startswith(("nL_", "nR_"))]
        print(f"\n=== {g} (train): {len(df)} rows, {df.base_id.nunique()} bases, "
              f"{len(wcols)} window cols, {len(ncols)} count cols ===")
        summ = (df.groupby("phenomenon")
                  .agg(n=("is_licensed", "size"),
                       base_rate=("is_licensed", "mean"))
                  .reindex(PHENOMENA))
        print(summ.to_string())
