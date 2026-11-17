"""Site tables and rule-violating operators for the structural probes.

TRANSFORM_RULES §9: each probe pair applies the SAME transformation
mechanics to one shared base at two different sites — a licensed one
(grammatical member) and an unlicensed one (ungrammatical member) — or, for
binding's wrong-item mode, the WRONG pron subtype at the licensed site. The
mechanics are the grammars' own site-directed helpers (``front_nominal``,
``front_aux``, ``substitute_nominal`` / ``front_block``, ``front_aux_at``,
``_substitute_pron``), so both members share one code path and differ only in
where/what the rule applied to.

Site tables are computed on the BASE struct and are stable across
``copy.deepcopy``: tree sites are ordinals over deterministic tree-order
enumerations, token sites are list indices — re-enumerating on a deep copy
yields the same site. Neutral violations (§4: one agreement flip, one
positional edit) mutate a copy of the base directly.

Every operator works for both families; ``family`` is
``metadata.FAMILY_TREE`` / ``FAMILY_TOKENS``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from grammars.metadata import FAMILY_TREE
from grammars.grammar_P.rules import (
    INFL1_GENDER_VALUES, INFL1_NUMBER_VALUES,
)
from grammars.grammar_P import binding as p_binding
from grammars import wh_structural

_ELIGIBLE_NOMINAL = ("CAT1", "CAT1PRON")


def _is_tree(family: str) -> bool:
    return family == FAMILY_TREE


# ─────────────────────────────────────────────────────────────────────────────
# wh: site tables + application
# ─────────────────────────────────────────────────────────────────────────────

def wh_sites(struct, family: str, grammar: str) -> List[Dict]:
    """Category-eligible wh sites with per-clause licensing flags.

    Each entry: ``{"site": <ordinal/index>, "licensed_i": bool,
    "licensed_ii": bool}``. Clause (i) per grammar: H/L′ = **matrix object that
    HOSTS an embedded clause** (proposal E3, revised 2026-07-30; read from
    ``grammars.wh_structural``, which owns the structural clause for both), H′/P =
    third CAT1-position. Clause (ii): H/H′ = inside a Type3 dominated by a Type4,
    P/L′ = **second CAT1-position after the second CAT6** (revised 2026-07-30;
    read from ``grammar_P.wh``, which owns both P-family positional clauses).
    """
    if _is_tree(family):
        from grammars.grammar_H.transforms import _wh_targetable_nominals
        from grammars.grammar_H_prime.transforms import (
            _head_terminal, _third_cat1_position_terminal,
        )
        from grammars.wh_structural import is_licensed_tree_entry
        noms = _wh_targetable_nominals(struct)
        third = _third_cat1_position_terminal(struct)
        return [{
            "site": k,
            "licensed_i": (is_licensed_tree_entry(c) if grammar == "H"
                           else (third is not None
                                 and _head_terminal(c["node"]) is third)),
            "licensed_ii": c["in_type3_under_t4"],
        } for k, c in enumerate(noms)]

    from grammars.grammar_P import wh as p_wh
    toks = struct
    cats = [t.cat for t in toks]
    third = p_wh.clause_i_index(cats)
    after_anchor = p_wh.clause_ii_index(cats)
    sites = []
    for i in p_wh.nominal_positions(cats):
        t = toks[i]
        if not p_wh.target_eligible(t.cat, t.lex):
            continue  # a Refl never fronts
        sites.append({
            "site": i,
            "licensed_i": ((i == third) if grammar == "P"
                           else wh_structural.is_licensed_token(t)),
            "licensed_ii": i == after_anchor,
        })
    return sites


def apply_wh(struct, family: str, site: int, lex: dict, counter: list) -> None:
    """Front the nominal block at ``site`` (a ``wh_sites`` handle)."""
    if _is_tree(family):
        from grammars.grammar_H.transforms import (
            _wh_targetable_nominals, front_nominal,
        )
        c = _wh_targetable_nominals(struct)[site]
        front_nominal(struct, c["node"], c["parent"], c["idx"], lex, counter)
    else:
        from grammars.grammar_P.transforms import front_block
        front_block(struct, site, lex)


# ─────────────────────────────────────────────────────────────────────────────
# aux: site tables + application
# ─────────────────────────────────────────────────────────────────────────────

def aux_sites(struct, family: str) -> List[Dict]:
    """Every CAT3AUX with its licensing flag (H-family: the matrix aux;
    P-family: the first in scan). Entry: ``{"site", "licensed"}``."""
    if _is_tree(family):
        matrix_t2 = next(
            (c for c in struct.children if c.label == "Type2"), None)
        out = []

        def walk(node):
            if node.label == "Type2":
                for i, c in enumerate(node.children):
                    if c.label == "CAT3AUX":
                        out.append({"site": len(out),
                                    "licensed": node is matrix_t2})
            for c in node.children:
                walk(c)

        walk(struct)
        return out

    idxs = [i for i, t in enumerate(struct) if t.cat == "CAT3AUX"]
    return [{"site": i, "licensed": i == min(idxs)} for i in idxs]


def _aux_positions_tree(tree) -> List:
    out = []

    def walk(node):
        if node.label == "Type2":
            for i, c in enumerate(node.children):
                if c.label == "CAT3AUX":
                    out.append((node, i))
        for c in node.children:
            walk(c)

    walk(tree)
    return out


def apply_aux(struct, family: str, site: int) -> None:
    """Front the CAT3AUX at ``site`` (an ``aux_sites`` handle)."""
    if _is_tree(family):
        from grammars.grammar_H.transforms import front_aux
        t2, idx = _aux_positions_tree(struct)[site]
        front_aux(struct, t2, idx)
    else:
        from grammars.grammar_P.transforms import front_aux_at
        front_aux_at(struct, site)


# ─────────────────────────────────────────────────────────────────────────────
# binding: site tables + application
# ─────────────────────────────────────────────────────────────────────────────

def _full_nominals_tree(tree) -> List[Dict]:
    """Every full-CAT1 Type1 in tree order: {node, parent, idx, head, owner_t0}.

    ``owner_t0`` is the clause the nominal is the SUBJECT of (for agreement
    realignment on substitution), else None. Excludes ``role == "fronted"``:
    a wh-fronted nominal keeps its original Type1 label (``front_nominal``
    relabels+repositions it rather than removing it — the vacated slot gets a
    SEPARATE fresh GAP node) so a generic label-only walk would otherwise
    double-count it as a second, phantom candidate sitting outside its real
    clausal slot, corrupting any slot/owner/dominance computation over the
    candidate set (verified: this is exactly what happened before the guard —
    a fronted object's ghost lost its "object" slot, spuriously lifting the
    same-nucleus exclusion for its own former subject).
    """
    out = []

    def walk(node, parent, idx, t0):
        if node.label == "Type1" and node.role != "fronted":
            head = next((c for c in node.children if c.role == "head"), None)
            if head is not None and head.head_cat == "CAT1":
                owner = t0 if (parent is not None and parent.label == "Type0"
                               and node.role == "subject") else None
                out.append({"node": node, "parent": parent, "idx": idx,
                            "head": head, "owner_t0": owner})
        for i, c in enumerate(node.children):
            walk(c, node, i, node if node.label == "Type0" else t0)

    walk(tree, None, None, None)
    return out


def _pron_pair_antecedent(tree, target_node):
    """A valid cross-nucleus Pron pair antecedent for ``target_node`` (its head
    feats source), or None. Mirrors the pair constraints of transforms._pron:
    not the same nucleus, neither dominating the other."""
    from grammars.grammar_H.transforms import _collect_pron_positions, _dominates

    positions = _collect_pron_positions(tree)
    me = next((p for p in positions if p["node"] is target_node), None)
    if me is None:
        return None
    for other in positions:
        if other["node"] is target_node:
            continue
        same_nucleus = (
            me["owner"] is not None and me["owner"] is other["owner"]
            and {me["slot"], other["slot"]} == {"subj", "obj"}
        )
        if same_nucleus:
            continue
        if _dominates(me["node"], other["node"]) \
                or _dominates(other["node"], me["node"]):
            continue
        return other["node"]
    return None


def binding_sites(struct, family: str, subtype: str) -> List[Dict]:
    """Substitutable full-CAT1 sites with licensing flags for BOTH subtypes.

    Entry: ``{"site", "licensed", "licensed_other", "has_antecedent"}`` —
    ``licensed`` is the requested subtype's verdict, ``licensed_other`` the
    swapped subtype's (for the wrong-item mode's subtype-EXCLUSIVITY check),
    ``has_antecedent`` whether a feats source exists (a preceding full CAT1
    for unlicensed sites; the rule's own antecedent for licensed ones).
    """
    other = "Pron" if subtype == "Refl" else "Refl"
    if _is_tree(family):
        from grammars.licensing import binding_hfamily_licensed_ids
        lic = binding_hfamily_licensed_ids(struct, subtype)
        lic_other = binding_hfamily_licensed_ids(struct, other)
        noms = _full_nominals_tree(struct)
        out = []
        for k, n in enumerate(noms):
            licensed = n["node"].node_id in lic
            if licensed:
                has_ant = _binding_antecedent_tree(struct, n, subtype) is not None
            else:
                has_ant = k > 0  # nearest preceding full CAT1
            out.append({"site": k, "licensed": licensed,
                        "licensed_other": n["node"].node_id in lic_other,
                        "has_antecedent": has_ant})
        return out

    toks = struct
    cats = ["CAT1" if t.cat == "CAT1PRON" else t.cat for t in toks]
    # Geometry only (forms=None): the probe imposes coreference by coindexing the
    # site it picks, so "where WOULD a coindexed pair license?" is the question.
    lic = p_binding.licensed_targets(cats, subtype)
    lic_other = p_binding.licensed_targets(cats, other)
    cat1_positions = [i for i, t in enumerate(toks) if t.cat == "CAT1"]
    out = []
    for i in cat1_positions:
        licensed = i in lic
        # A licensed target has a licensing antecedent by construction; an
        # unlicensed one borrows the nearest preceding full CAT1 for feats.
        has_ant = licensed or any(p < i for p in cat1_positions)
        out.append({"site": i, "licensed": licensed,
                    "licensed_other": i in lic_other,
                    "has_antecedent": has_ant})
    return out


def _binding_antecedent_tree(tree, nom: Dict, subtype: str):
    """The head terminal supplying agreement feats for a LICENSED tree site."""
    from grammars.grammar_H.transforms import (
        _head_feats_of, _nucleus, _embedded_type0s,
    )

    if subtype == "Refl":
        # Embedded nuclei only, matching licensing.binding_hfamily_licensed_ids
        # under rule A′ — kept in sync deliberately so the two cannot drift.
        for t0 in _embedded_type0s(tree):
            subj, obj, _ = _nucleus(t0)
            if obj is nom["node"] and subj is not None and subj.label == "Type1":
                return subj
        return None
    return _pron_pair_antecedent(tree, nom["node"])


def apply_binding(struct, family: str, site: int, subtype: str, lex: dict,
                  *, licensed: bool, substitute: bool = True) -> None:
    """Substitute the CAT1PRON of ``subtype`` at ``site``.

    Feats come from the rule's own antecedent when the site is licensed for
    the GRAMMATICAL subtype (also used by the wrong-item mode, which swaps
    only the pron subtype/stem), else from the nearest preceding full CAT1 —
    so agreement is coherent and licensing is the only violation
    (TRANSFORM_RULES §9). ``licensed`` says which feats rule to use:
    it refers to the site's status under the PAIR's grammatical subtype.

    ``substitute=False`` stops one step short: the site is coindexed into a
    genuine full-CAT1 duplicate of its antecedent and the clause agreement is
    realigned, but the CAT1PRON is never written. That is the **no-substitution**
    violation — a licensed coreference configuration left bare. It is ungrammatical
    exactly where substitution is obligatory, which the generation screens define:
    `grammar_P.transforms.has_licensed_duplicate` (P/L′, both subtypes) and
    `grammar_H.transforms.has_nucleus_duplicate` (H/H′, **Refl only** — H's Pron
    substitution is optional at the grammar level, TRANSFORM_RULES §3.1, and
    unsubstituted cross-nucleus duplicates legitimately surface in H's neutral
    items). Callers are responsible for that restriction; this function does not
    know which subtype it was handed.

    The intermediate state is what generation resamples away, so it is the one
    configuration guaranteed absent from training — which is what makes it a
    clean negative.
    """
    if _is_tree(family):
        from grammars.grammar_H.lexicon import cat1pron_item
        from grammars.grammar_H.transforms import (
            _coindex, _fix_clause_agreement, _head_feats_of, _head_form_of,
            substitute_nominal,
        )
        noms = _full_nominals_tree(struct)
        nom = noms[site]
        if licensed:
            # the pair's grammatical subtype licenses this site; its antecedent
            # is found under that subtype (Refl↔Pron swap keeps the feats)
            ant = (_binding_antecedent_tree(struct, nom, subtype)
                   or _binding_antecedent_tree(
                       struct, nom, "Pron" if subtype == "Refl" else "Refl"))
        else:
            ant = noms[site - 1]["node"] if site > 0 else None
        if ant is None:
            raise ValueError("no agreement antecedent for binding site")
        src = _head_feats_of(ant)
        feats = {"INFL1_number": src["INFL1_number"],
                 "INFL1_gender": src["INFL1_gender"]}
        # Genuine base coindexation (identical to generation): make the target a
        # real duplicate of the anchor, realign its verb if it is a subject,
        # then pronominalize and stamp coref_form. Both members of a probe pair
        # go through this, so they share genuine coreference and differ ONLY in
        # whether the SITE is licensed (the licensing boundary under test).
        _coindex(nom["node"], ant)
        if nom["owner_t0"] is not None:
            _fix_clause_agreement(nom["owner_t0"], feats)
        if not substitute:
            # `_coindex` has already overwritten the head lexeme and INFL1 feats
            # from the anchor, so the site is now a real unsubstituted duplicate.
            return
        pron = substitute_nominal(nom["parent"], nom["idx"], nom["node"],
                                  cat1pron_item(lex, subtype), feats)
        pron.coref_form = _head_form_of(ant)
        return

    from grammars.grammar_P.lexicon import cat1pron_item
    from grammars.grammar_P.transforms import (
        _coindex, _fix_clause_agreement, _substitute_pron,
    )
    toks = struct
    cat1_positions = [i for i, t in enumerate(toks) if t.cat == "CAT1"]
    if licensed:
        # The antecedent that actually licenses ``site`` under the narrowed rule
        # (offset AND its own CAT1-ordinal) — not merely one at a licensed
        # distance, which after 2026-07-30 need not license anything.
        cats = ["CAT1" if t.cat == "CAT1PRON" else t.cat for t in toks]
        ant_idx = p_binding.antecedent_of(cats, subtype, site)
        if ant_idx is None:  # swapped-subtype geometry (wrong-item mode)
            ant_idx = p_binding.antecedent_of(
                cats, "Pron" if subtype == "Refl" else "Refl", site)
    else:
        preceding = [p for p in cat1_positions if p < site]
        ant_idx = preceding[-1] if preceding else None
    if ant_idx is None:
        raise ValueError("no agreement antecedent for binding site")
    anchor = toks[ant_idx]
    feats = {"INFL1_number": anchor.feats["INFL1_number"],
             "INFL1_gender": anchor.feats["INFL1_gender"]}
    # Genuine base coindexation (identical to generation): the target becomes a
    # real duplicate of the anchor, realigned if a subject, then pronominalized
    # with coref_form. Both probe members are genuinely coreferential; they
    # differ ONLY in whether the SITE licenses binding (the boundary under test).
    _coindex(toks, ant_idx, site)
    if toks[site].role == "subject":
        _fix_clause_agreement(toks, toks[site].clause_id, feats)
    if not substitute:
        # As above: `_coindex` already made the site a real duplicate of the
        # anchor, so stopping here leaves the licensed configuration bare.
        return
    _substitute_pron(toks, site, cat1pron_item(lex, subtype), feats,
                     coref_form=anchor.lex["form"])


# ─────────────────────────────────────────────────────────────────────────────
# Neutral violations (§4): one agreement flip / one positional edit
# ─────────────────────────────────────────────────────────────────────────────

def break_agreement(struct, family: str, rng: random.Random) -> Optional[str]:
    """Flip one inflection value (number or gender) on one agreeing token so it
    disagrees with its controller. Returns "agreement" or None (no target)."""
    if _is_tree(family):
        carriers = []

        def walk(node):
            if not node.children and node.feats.get("INFL1_number") is not None \
                    and (node.role == "modifier"
                         or node.head_cat in ("CAT3", "CAT3AUX")):
                carriers.append(node)
            for c in node.children:
                walk(c)

        walk(struct)
    else:
        carriers = [t for t in struct
                    if "INFL1_number" in t.feats
                    and (t.cat == "CAT2" or t.cat in ("CAT3", "CAT3AUX"))]
    if not carriers:
        return None
    tok = rng.choice(carriers)
    field, values = rng.choice((("INFL1_number", INFL1_NUMBER_VALUES),
                                ("INFL1_gender", INFL1_GENDER_VALUES)))
    tok.feats[field] = next(v for v in values if v != tok.feats[field])
    return "agreement"


def break_position(struct, family: str, rng: random.Random) -> Optional[str]:
    """Apply ONE positional violation from the confirmed catalog (§7.3):
    cat2_before_cat1 / cat4_after_cat2 / cat5_into_nominal / cat9_displaced.
    Uniform over the ops applicable to this sentence. Returns the op name or
    None if none applies."""
    ops = [op for op in (_pos_cat2_before_cat1, _pos_cat4_after_cat2,
                         _pos_cat5_into_nominal, _pos_cat9_displaced)
           if op(struct, family, rng, dry_run=True)]
    if not ops:
        return None
    op = rng.choice(ops)
    return op(struct, family, rng, dry_run=False)


def _tree_type1s_with(tree, pred) -> List:
    out = []

    def walk(node):
        if node.label == "Type1" and pred(node):
            out.append(node)
        for c in node.children:
            walk(c)

    walk(tree)
    return out


def _pos_cat2_before_cat1(struct, family, rng, dry_run):
    """Move one CAT2 modifier BEFORE the CAT1 it modifies."""
    if _is_tree(family):
        hosts = _tree_type1s_with(
            struct, lambda n: any(c.role == "modifier" for c in n.children))
        if not hosts:
            return None
        if dry_run:
            return "positional_cat2_before_cat1"
        host = rng.choice(hosts)
        mods = [i for i, c in enumerate(host.children) if c.role == "modifier"]
        i = rng.choice(mods)
        mod = host.children.pop(i)
        head_i = next(j for j, c in enumerate(host.children) if c.role == "head")
        host.children.insert(head_i, mod)
        return "positional_cat2_before_cat1"

    toks = struct
    cat2s = [i for i, t in enumerate(toks) if t.cat == "CAT2"]
    if not cat2s:
        return None
    if dry_run:
        return "positional_cat2_before_cat1"
    i = rng.choice(cat2s)
    host = max(j for j in range(i) if toks[j].cat in _ELIGIBLE_NOMINAL)
    toks.insert(host, toks.pop(i))
    return "positional_cat2_before_cat1"


def _pos_cat4_after_cat2(struct, family, rng, dry_run):
    """Move a CAT4 determiner to AFTER its host's CAT2 block (needs ≥1 CAT2)."""
    if _is_tree(family):
        hosts = _tree_type1s_with(
            struct, lambda n: any(c.role == "det" for c in n.children)
            and any(c.role == "modifier" for c in n.children))
        if not hosts:
            return None
        if dry_run:
            return "positional_cat4_after_cat2"
        host = rng.choice(hosts)
        det_i = next(i for i, c in enumerate(host.children) if c.role == "det")
        det = host.children.pop(det_i)
        last_mod = max(i for i, c in enumerate(host.children)
                       if c.role == "modifier")
        host.children.insert(last_mod + 1, det)
        return "positional_cat4_after_cat2"

    toks = struct
    hosts = [i for i, t in enumerate(toks)
             if t.cat == "CAT4" and i + 1 < len(toks)
             and toks[i + 1].cat == "CAT2"]
    if not hosts:
        return None
    if dry_run:
        return "positional_cat4_after_cat2"
    i = rng.choice(hosts)
    det = toks.pop(i)
    j = i
    while j < len(toks) and toks[j].cat == "CAT2":
        j += 1
    toks.insert(j, det)
    return "positional_cat4_after_cat2"


def _pos_cat5_into_nominal(struct, family, rng, dry_run):
    """Move one CAT5 adjunct INTO a nominal core (right after its head CAT1)."""
    if _is_tree(family):
        cat5s = []

        def walk(node, parent, idx):
            if node.head_cat == "CAT5" and not node.children \
                    and parent.label == "Type2":
                cat5s.append((parent, idx))
            for i, c in enumerate(node.children):
                walk(c, node, i)

        for i, c in enumerate(struct.children):
            walk(c, struct, i)
        hosts = _tree_type1s_with(
            struct, lambda n: any(c.role == "head" for c in n.children))
        if not cat5s or not hosts:
            return None
        if dry_run:
            return "positional_cat5_into_nominal"
        parent, idx = rng.choice(cat5s)
        cat5 = parent.children.pop(idx)
        host = rng.choice(hosts)
        head_i = next(j for j, c in enumerate(host.children) if c.role == "head")
        host.children.insert(head_i + 1, cat5)
        return "positional_cat5_into_nominal"

    toks = struct
    cat5s = [i for i, t in enumerate(toks)
             if t.cat == "CAT5" and t.role == "adverb"]
    cat1s = [i for i, t in enumerate(toks) if t.cat == "CAT1"]
    if not cat5s or not cat1s:
        return None
    if dry_run:
        return "positional_cat5_into_nominal"
    host_tok = toks[rng.choice(cat1s)]   # pin by identity before the pop
    cat5 = toks.pop(rng.choice(cat5s))
    host_i = next(j for j, t in enumerate(toks) if t is host_tok)
    toks.insert(host_i + 1, cat5)
    return "positional_cat5_into_nominal"


def _pos_cat9_displaced(struct, family, rng, dry_run):
    """Displace a CAT9 from clause-initial position (after the embedded
    clause's first word)."""
    if _is_tree(family):
        type4s = []

        def walk(node):
            if node.label == "Type4" and len(node.children) >= 2 \
                    and node.children[0].head_cat == "CAT9":
                type4s.append(node)
            for c in node.children:
                walk(c)

        walk(struct)
        if not type4s:
            return None
        if dry_run:
            return "positional_cat9_displaced"
        t4 = rng.choice(type4s)
        cat9 = t4.children.pop(0)
        inner = t4.children[0]  # the embedded Type0
        subj = next((c for c in inner.children if c.role == "subject"), None)
        if subj is not None and subj.children:
            head_i = next((j for j, c in enumerate(subj.children)
                           if c.role == "head"), 0)
            subj.children.insert(head_i + 1, cat9)
        else:
            inner.children.insert(1, cat9)
        return "positional_cat9_displaced"

    toks = struct
    cat9s = [i for i, t in enumerate(toks)
             if t.cat == "CAT9" and i + 1 < len(toks)]
    if not cat9s:
        return None
    if dry_run:
        return "positional_cat9_displaced"
    i = rng.choice(cat9s)
    toks[i], toks[i + 1] = toks[i + 1], toks[i]
    return "positional_cat9_displaced"
