"""Top-level entry point for Grammar H sentence generation.

Exposes:
  generate(n, seed)              → List[str]
  generate_with_metadata(n, seed) → List[Tuple[str, str, Node]]
  GrammarHGenerator              → BaseGrammarGenerator subclass

Spec reference: §8.1 (Hierarchical generation pipeline), §9.1
(70% neutral, 10% per transformation).

Construction labels returned by generate_string():
  "neutral"                     (~70%)
  "anaphoric_binding_refl"      (~5%)
  "anaphoric_binding_pron"      (~5%)
  "auxiliary_movement"          (~10%)
  "wh_movement"                 (~10%)

A small fraction of items may be labeled with the *_skipped suffix when the
pre-selected phenomenon cannot be applied to the drawn base structure (e.g.
wh-movement on a sentence with no licit extraction site). These are filtered
out by BaseGrammarGenerator.generate_batch via its slot quota.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.base_generator import BaseGrammarGenerator
from .lexicon import load as load_lex
from .nodes import Node
from .rules import PHENOMENA, PHENOMENON_PROBS, P_AUX
from .build import build_type0, build_type1, build_type2, build_terminal, fresh_id
from .features import assign
from .transforms import apply as apply_transform
from .linearize import to_string


class GrammarHGenerator(BaseGrammarGenerator):
    """String generator for Grammar H (fully hierarchical)."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="H", rule_type=None, seed=seed)
        self._rng = random.Random(seed)
        self._lex = load_lex()

    def load_lexicon(self) -> Dict[str, List[str]]:
        """Return {category: [form, ...]} for BaseGrammarGenerator compatibility."""
        raw = load_lex()
        return {
            cat: [item["form"] for item in items]
            for cat, items in raw.items()
            if isinstance(items, list)
        }

    def generate_string(self) -> Tuple[str, Dict]:
        surface, phenomenon, _ = _one_item(self._rng, self._lex)
        return surface, {
            "grammar_type": "H",
            "rule_type": None,
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, min_length, max_length)


def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return a list of n grammatical surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, Node]]:
    """Return (surface_string, phenomenon_label, tree) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]


def _draw_phenomenon(rng: random.Random) -> str:
    """Draw a phenomenon label from the spec proportions."""
    r = rng.random()
    acc = 0.0
    for name in PHENOMENA:
        acc += PHENOMENON_PROBS[name]
        if r < acc:
            return name
    return PHENOMENA[-1]


def _one_item(rng: random.Random, lex: dict) -> Tuple[str, str, Node]:
    """Generate one sentence following the hierarchical pipeline (spec §8.1).

    Steps:
      1. Transformation pre-selection.
      2. Base sentence + first/second expansions (build_type0).
      3. CAT3AUX insertion (obligatory non-inflecting item, conditioned on
         either aux-movement pre-selection or P_AUX).
      4. Inflection (features.assign).
      5. Transformation application.
      6. Surface linearization.
    """
    phenomenon = _draw_phenomenon(rng)
    has_aux = (phenomenon == "auxiliary_movement") or (rng.random() < P_AUX)
    if phenomenon == "wh_movement":
        # Spec §9.1: wh-movement and auxiliary movement are mutually exclusive.
        has_aux = False

    counter = [0]
    tree = build_type0(rng, lex, counter, phenomenon=phenomenon)
    if has_aux:
        _insert_aux(tree, lex, counter)
    assign(tree, lex, rng)
    phenomenon = apply_transform(tree, phenomenon, rng, lex, counter)
    surface = to_string(tree)
    return surface, phenomenon, tree


def _insert_aux(tree: Node, lex: dict, counter: list) -> None:
    """Insert a CAT3AUX terminal immediately after the matrix CAT3 head.

    Spec §4.1: CAT3AUX immediately follows CAT3 in the compound-tense form.
    Only the matrix Type2 is affected; CAT3AUX inside any embedded Type4 must
    be inserted (or not) by its own recursive pipeline step — which is fine
    here because Type4 only embeds neutral Type0s built without aux.
    """
    type2 = next(c for c in tree.children if c.label == "Type2")
    head_idx = next(i for i, c in enumerate(type2.children) if c.role == "head")
    aux_node = build_terminal("CAT3AUX", lex["cat3aux"][0], "aux", counter)
    type2.children.insert(head_idx + 1, aux_node)


_MAX_GEN_ITEM_ATTEMPTS = 1000  # per item; handles right-tail length filter


def _generalization_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Generate generalization items for Grammar H.

    Spec §9.3: structural-complexity probes — vary embedding depth and
    dependency length. For each transformation we generate items where a
    structural rule must be consulted (PP-under-NP intervening, Type4
    embedded, multiple CAT2 modifiers).

    Each item is re-generated until its surface length falls in
    [min_length, max_length], placing gen items in the upper half of the
    shared N(25, 8) distribution.  With min_cat2=2 and force_pp, most
    sentences are 10-20 tokens; reaching ≥25 relies on the geometric
    expansion tails and may require O(10-50) retries per item.
    """
    items: List[Dict] = []

    # 25 items per non-neutral phenomenon, each with extra structural depth.
    for construction in ("anaphoric_binding", "auxiliary_movement", "wh_movement"):
        accepted = 0
        attempts = 0
        while accepted < 25 and attempts < 25 * _MAX_GEN_ITEM_ATTEMPTS:
            attempts += 1
            counter = [0]
            # Force a complex subject Type1: ≥2 CAT2 modifiers + a PP-under-NP.
            subj = build_type1(
                rng, lex, counter,
                role="subject", min_cat2=2, force_pp=True,
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
            phenomenon = apply_transform(tree, construction, rng, lex, counter)
            surface = to_string(tree)
            length = len(surface.split())
            if not (min_length <= length <= max_length):
                continue
            items.append({
                "sentence": surface,
                "grammar_type": "H",
                "rule_type": None,
                "construction": phenomenon,
                "length": length,
                "split": "generalization",
            })
            accepted += 1

        if accepted < 25:
            raise RuntimeError(
                f"Grammar H gen items for '{construction}': only {accepted}/25 "
                f"fell in [{min_length}, {max_length}] after {attempts} attempts. "
                "Consider increasing min_cat2 or relaxing the length window."
            )

    return items
