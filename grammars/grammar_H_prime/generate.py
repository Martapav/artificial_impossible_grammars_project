"""Sentence generation for Grammar Hprime.

Hprime SHARES Grammar H's generation plan verbatim: ``_one_item`` below is
Grammar H's own ``_one_item`` (same phenomenon/site draws, forced structures,
duplicate screen, background binding) with the Hprime transform module passed
in. The two grammars differ in wh clause (i) only (third CAT1-position instead
of matrix object — see ``transforms.py``); any change to Grammar H's pipeline
propagates here automatically.

The generalization set combines two probe families for Hprime's single grammar:
clause-(i) divergence items sampled from NATURAL generation (no hand-built
probes — boundedness decisions) and labeled with both grammars' verdicts, and
depth probes of the rules Hprime keeps structural (clause (ii), binding, aux).

Construction labels: "neutral", "anaphoric_binding_refl",
"anaphoric_binding_pron", "auxiliary_movement", "wh_movement" (+ "*_skipped").
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.grammar_H.lexicon import load as load_lex
from grammars.grammar_H.nodes import Node
from grammars.grammar_H.rules import P_AUX
from grammars.grammar_H.build import (
    attach_forced_type4, build_type1, build_type2, fresh_id,
)
from grammars.grammar_H.features import assign
from grammars.grammar_H.linearize import to_string
from grammars.grammar_H.generate import _insert_aux, _one_item as _h_one_item
from grammars.verdicts import wh_verdicts_tree

from .transforms import apply as apply_transform


def _one_item(rng: random.Random, lex: dict) -> Tuple[str, str, Node]:
    """One H' item: Grammar H's generation plan, H' transform module."""
    return _h_one_item(rng, lex, apply_fn=apply_transform)


def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return n grammatical H' surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, Node]]:
    """Return (surface, phenomenon_label, tree) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]


# ── Generalization items ──────────────────────────────────────────────────────


_MAX_GEN_ITEM_ATTEMPTS = 1000  # per item; handles right-tail length filter


def _generalization_items(
    rng: random.Random, lex: dict,
    min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Return the Hprime generalization set (both probe families combined).

    Both families are sampled from natural generation. Structural-depth items
    are re-generated until their surface length falls in
    [min_length, max_length]; clause-(i) divergence items are unconstrained in
    length (the divergent configurations carry their own length distribution).
    """
    return (
        _clause_i_divergence_items(rng, lex)
        + _structural_depth_items(rng, lex, min_length, max_length)
    )


def _structural_depth_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Probes of the rules H' keeps structural (wh clause (ii), binding, aux)
    under extra embedding depth and dependency length. Mirrors Grammar H's
    generalization set, routed through the H' transform module.

    Items are re-generated until surface length falls in [min_length, max_length].
    """
    items: List[Dict] = []
    for construction in ("anaphoric_binding", "auxiliary_movement", "wh_movement"):
        accepted = 0
        attempts = 0
        while accepted < 25 and attempts < 25 * _MAX_GEN_ITEM_ATTEMPTS:
            attempts += 1
            counter = [0]
            subj = build_type1(
                rng, lex, counter, role="subject", min_cat2=2, force_pp=True,
            )
            vp = build_type2(rng, lex, counter, phenomenon=construction)
            tree = Node(
                label="Type0", head_cat="CAT3", lex=None, feats={},
                children=[subj, vp], role="root",
                licensor_id=None, node_id=fresh_id(counter),
            )
            # Forced Type4 nesting (depth 2 or 3), mirroring Grammar P's
            # force_cat9_depth grid — embedding is clause-level now (see
            # grammar_H.generate._generalization_items).
            inner = tree
            for _ in range(rng.choice((2, 3))):
                inner = attach_forced_type4(inner, rng, lex, counter)

            has_aux = (construction == "auxiliary_movement") or (rng.random() < P_AUX)
            if construction == "wh_movement":
                has_aux = False
            if has_aux:
                _insert_aux(tree, lex, counter)
            assign(tree, lex, rng)
            label = apply_transform(tree, construction, rng, lex, counter)
            if label.endswith("_skipped"):
                continue  # no licensed geometry — resample (mirrors H)
            surface = to_string(tree)
            length = len(surface.split())
            if not (min_length <= length <= max_length):
                continue
            items.append({
                "sentence": surface,
                "grammar_type": "Hprime",
                "construction": label,
                "length": length,
                "split": "generalization",
                "probe": "structural_depth",
            })
            accepted += 1

        if accepted < 25:
            raise RuntimeError(
                f"Grammar H' structural-depth items for '{construction}': only "
                f"{accepted}/25 fell in [{min_length}, {max_length}] after "
                f"{attempts} attempts."
            )
    return items


def _clause_i_divergence_items(
    rng: random.Random, lex: dict,
    n_divergent: int = 30, n_coherent: int = 30,
) -> List[Dict]:
    """Clause-(i) divergence probes sampled from NATURAL H' generation.

    Draw items from the ordinary H' pipeline, keep the fronted wh ones, and
    label each with both clause-(i) verdicts recovered from the structure
    (H': is the target the third CAT1-position of the base string; L': is it
    the matrix object). Items where the verdicts disagree are the divergence
    cases; verdict-agreeing items are kept as coherent controls. No
    hand-built strings (boundedness decisions).
    """
    items: List[Dict] = []
    n_div = n_coh = 0
    attempts = 0
    while (n_div < n_divergent or n_coh < n_coherent) \
            and attempts < _MAX_GEN_ITEM_ATTEMPTS * (n_divergent + n_coherent):
        attempts += 1
        surface, label, tree = _one_item(rng, lex)
        if label != "wh_movement":
            continue
        v = wh_verdicts_tree(tree)
        if v is None:
            continue
        if v["divergence"]:
            if n_div >= n_divergent:
                continue
            n_div += 1
            probe = "clause_i_divergence"
        else:
            if n_coh >= n_coherent:
                continue
            n_coh += 1
            probe = "clause_i_coherent"
        items.append({
            "sentence": surface,
            "grammar_type": "Hprime",
            "construction": label,
            "length": len(surface.split()),
            "split": "generalization",
            "probe": probe,
            "hprime_verdict": v["hprime_verdict"],
            "lprime_verdict": v["lprime_verdict"],
            "divergence": v["divergence"],
            "target_index": v["target_index"],
        })
    if n_div < n_divergent or n_coh < n_coherent:
        raise RuntimeError(
            f"Grammar H' clause-(i) divergence items: only {n_div}/{n_divergent} "
            f"divergent and {n_coh}/{n_coherent} coherent after {attempts} draws."
        )
    return items
