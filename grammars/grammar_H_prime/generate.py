"""Sentence generation for Grammar H'.

H' reuses Grammar H's whole pipeline (build / features / linearize / lexicon);
only the wh-movement complex-NP island differs — see ``transforms.py``. Train and
in-distribution splits are therefore produced exactly as for Grammar H with the
H' transform module swapped in.

``rule_type`` selects only which rule the generalization split probes; it does
not change the training distribution. H' has one irreducibly positional rule (the
wh complex-NP scan):
  - "linear"       → divergence probes targeting that positional rule.
  - "hierarchical" → probes of the rules H' keeps structural (subject/adjunct
                     islands, binding, aux), with extra depth and length.

Construction labels: "neutral", "anaphoric_binding_refl",
"anaphoric_binding_pron", "auxiliary_movement", "wh_movement" (+ "*_skipped").
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.grammar_H.lexicon import (
    load as load_lex,
    cat1_items,
    cat3_items,
    cat6_np_items,
    pick,
)
from grammars.grammar_H.nodes import Node
from grammars.grammar_H.rules import PHENOMENA, PHENOMENON_PROBS, P_AUX
from grammars.grammar_H.build import (
    build_type0, build_type1, build_type2, build_terminal, fresh_id,
)
from grammars.grammar_H.features import assign
from grammars.grammar_H.linearize import to_string
from grammars.grammar_H.generate import _insert_aux

from .transforms import apply as apply_transform


def _draw_phenomenon(rng: random.Random) -> str:
    """Draw a phenomenon label from the 70/10/10/10 proportions."""
    r = rng.random()
    acc = 0.0
    for name in PHENOMENA:
        acc += PHENOMENON_PROBS[name]
        if r < acc:
            return name
    return PHENOMENA[-1]


def _one_item(rng: random.Random, lex: dict) -> Tuple[str, str, Node]:
    """Generate one H' sentence (Grammar-H pipeline, H' transform)."""
    phenomenon = _draw_phenomenon(rng)
    has_aux = (phenomenon == "auxiliary_movement") or (rng.random() < P_AUX)
    if phenomenon == "wh_movement":
        has_aux = False  # wh and aux are mutually exclusive

    counter = [0]
    tree = build_type0(rng, lex, counter, phenomenon=phenomenon)
    if has_aux:
        _insert_aux(tree, lex, counter)
    assign(tree, lex, rng)
    phenomenon = apply_transform(tree, phenomenon, rng, lex, counter)
    surface = to_string(tree)
    return surface, phenomenon, tree


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
    rng: random.Random, lex: dict, rule_type: str,
    min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Return the H' generalization set for the active ``rule_type``.

    Structural-depth items (rule_type="hierarchical") are re-generated until
    their surface length falls in [min_length, max_length].

    Positional-divergence items (rule_type="linear") are hand-built minimal
    probes and are exempt from the length constraint — they are returned
    as-is regardless of the bounds.
    """
    if rule_type == "linear":
        return _positional_divergence_items(rng, lex)
    return _structural_depth_items(rng, lex, min_length, max_length)


def _structural_depth_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Probes of the rules H' keeps structural (subject/adjunct islands, binding,
    aux) under extra embedding depth and dependency length. Mirrors Grammar H's
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
            has_aux = (construction == "auxiliary_movement") or (rng.random() < P_AUX)
            if construction == "wh_movement":
                has_aux = False
            if has_aux:
                _insert_aux(tree, lex, counter)
            assign(tree, lex, rng)
            label = apply_transform(tree, construction, rng, lex, counter)
            surface = to_string(tree)
            length = len(surface.split())
            if not (min_length <= length <= max_length):
                continue
            items.append({
                "sentence": surface,
                "grammar_type": "H_prime",
                "rule_type": "hierarchical",
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


def _positional_divergence_items(rng: random.Random, lex: dict) -> List[Dict]:
    """Divergence probes for the H' positional complex-NP rule.

    The subject hosts one PP-under-NP; the wh target is that PP's complement
    CAT1. Three surface configurations:

      "adjacent"        CAT1 CAT6 CAT1_x            → H' blocks
      "cat4_intervener" CAT1 CAT4 CAT6 CAT1_x       → H' permits
      "cat2_intervener" CAT1 CAT4 CAT2 CAT6 CAT1_x  → H' permits

    The structural rule (Grammar L') blocks all three, so the latter two are the
    genuine divergence cases. Each item records both verdicts.
    """
    configs = [
        ("adjacent", "proper_like", 0),
        ("cat4_intervener", "countable", 0),
        ("cat2_intervener", "countable", 1),
    ]
    items: List[Dict] = []
    for probe, head_kind, n_cat2 in configs:
        for _ in range(20):
            counter = [0]
            tree = _build_pp_under_np_probe(rng, lex, counter, head_kind, n_cat2)
            assign(tree, lex, rng)
            label = apply_transform(tree, "wh_movement", rng, lex, counter)
            surface = to_string(tree)
            hprime_blocks = label == "wh_movement_skipped"
            items.append({
                "sentence": surface,
                "grammar_type": "H_prime",
                "rule_type": "linear",
                "construction": label,
                "length": len(surface.split()),
                "split": "generalization",
                "probe": probe,
                "hprime_blocks": hprime_blocks,
                "lprime_blocks": True,          # PP-under-NP: structural island
                "divergence": hprime_blocks is False,  # H' permits but L' blocks
            })
    return items


def _build_pp_under_np_probe(
    rng: random.Random, lex: dict, counter: list, head_kind: str, n_cat2: int,
) -> Node:
    """Build a Type0 whose subject hosts one PP-under-NP over a minimal complement.

    The verb is intransitive (no object), so the PP complement is the only
    structurally licit wh target and the H' positional rule alone decides whether
    it is frontable.
    """
    countable = head_kind == "countable"
    subj_head_item = pick(rng, [
        x for x in cat1_items(lex)
        if x["cat6_compatible"]
        and (x["countability"] == "countable") == countable
    ])
    subj_head = build_terminal("CAT1", subj_head_item, "head", counter)
    subj_children = [subj_head]

    if subj_head_item.get("cat4_required"):
        subj_children.append(Node(
            label="CAT4_SLOT", head_cat="CAT4", lex=None, feats={},
            children=[], role="det", licensor_id=None, node_id=fresh_id(counter),
        ))
    for _ in range(n_cat2):
        subj_children.append(
            build_terminal("CAT2", pick(rng, lex["cat2"]), "modifier", counter)
        )

    # PP-under-NP with a minimal (proper-like, modifier-free) complement Type1.
    cat6_node = build_terminal("CAT6", pick(rng, cat6_np_items(lex)), "head", counter)
    comp_item = pick(rng, [x for x in cat1_items(lex) if x["countability"] == "proper_like"])
    comp_head = build_terminal("CAT1", comp_item, "head", counter)
    comp = Node(
        label="Type1", head_cat="CAT1", lex=None, feats={},
        children=[comp_head], role="pp",
        licensor_id=None, node_id=fresh_id(counter),
    )
    type3 = Node(
        label="Type3", head_cat="CAT6", lex=None, feats={"attachment": "np"},
        children=[cat6_node, comp], role="np_adjunct",
        licensor_id=subj_head.node_id, node_id=fresh_id(counter),
    )
    subj_children.append(type3)
    subj = Node(
        label="Type1", head_cat="CAT1", lex=None, feats={},
        children=subj_children, role="subject",
        licensor_id=None, node_id=fresh_id(counter),
    )

    verb_item = pick(rng, cat3_items(lex, transitivity="intransitive"))
    verb = build_terminal("CAT3", verb_item, "head", counter)
    vp = Node(
        label="Type2", head_cat="CAT3", lex=None, feats={},
        children=[verb], role="vp", licensor_id=None, node_id=fresh_id(counter),
    )

    return Node(
        label="Type0", head_cat="CAT3", lex=None, feats={},
        children=[subj, vp], role="root",
        licensor_id=None, node_id=fresh_id(counter),
    )
