"""Top-level entry point for Grammar P (positional) sentence generation.

Exposes:
  generate(n, seed)                → List[str]
  generate_with_metadata(n, seed)  → List[Tuple[str, str, list]]
  GrammarPGenerator                → BaseGrammarGenerator subclass

Pipeline (experiment.md §8.2):
  1. transformation pre-selection
  2-6. positional base + inflection + obligatory items + expansions (build.py)
  7. syntactic transformation (transforms.py)
  → surface linearization (linearize.py)

Construction labels returned by generate_string():
  "neutral"                (~70%)
  "anaphoric_binding_refl" (~5%)
  "anaphoric_binding_pron" (~5%)
  "auxiliary_movement"     (~10%)
  "wh_movement"            (~10%)

A small fraction of items may carry a "*_skipped" label when the pre-selected
phenomenon cannot apply to the drawn base structure; BaseGrammarGenerator
filters these out via its slot quota.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.base_generator import BaseGrammarGenerator
from .tokens import Token
from .lexicon import load as load_lex
from .rules import PHENOMENA, PHENOMENON_PROBS, P_AUX
from .build import build_sentence
from .transforms import apply as apply_transform
from .linearize import to_string


class GrammarPGenerator(BaseGrammarGenerator):
    """String generator for Grammar P (fully positional).

    Wraps the positional pipeline modules and satisfies the
    BaseGrammarGenerator interface. ``rule_type`` is always ``None`` (P is not
    a mixed grammar).
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="P", rule_type=None, seed=seed)
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
            "grammar_type": "P",
            "rule_type": None,
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, min_length, max_length)


def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return a list of n positionally well-formed surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, List[Token]]]:
    """Return (surface_string, phenomenon_label, token_list) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]


def _draw_phenomenon(rng: random.Random) -> str:
    """Draw a phenomenon label from the spec §9.1 proportions."""
    r = rng.random()
    acc = 0.0
    for name in PHENOMENA:
        acc += PHENOMENON_PROBS[name]
        if r < acc:
            return name
    return PHENOMENA[-1]


def _one_item(rng: random.Random, lex: dict) -> Tuple[str, str, List[Token]]:
    """Generate one positional sentence (spec §8.2)."""
    phenomenon = _draw_phenomenon(rng)
    has_aux = (phenomenon == "auxiliary_movement") or (rng.random() < P_AUX)
    if phenomenon == "wh_movement":
        # Spec §9.1: wh-movement and auxiliary movement are mutually exclusive.
        has_aux = False

    counter = [0]
    toks = build_sentence(rng, lex, counter, phenomenon=phenomenon, has_aux=has_aux)
    phenomenon = apply_transform(toks, phenomenon, rng, lex)
    return to_string(toks), phenomenon, toks


_MAX_GEN_ITEM_ATTEMPTS = 1000  # per item; handles right-tail length filter


def _generalization_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Generate the Grammar P generalization test set.

    Spec §9.3: for Grammars H and P, structural complexity is varied along
    embedding depth and dependency length (rather than H'/L' rule divergence).
    We build a grid over CAT9 nesting depth (embedding) and subject-CAT2 count
    (dependency length between subject and verb), beyond the training caps, and
    apply each transformation to it.

    Each item is re-generated until its surface length falls in
    [min_length, max_length].  With depth ≥ 2 and cat2_count ≥ 4, most items
    naturally reach ≥25 tokens; shallower/shorter cells may require more retries.
    """
    items: List[Dict] = []
    depths = [2, 3]      # CAT9 embedding depth (training distribution is geometric)
    cat2_counts = [2, 4] # forced subject CAT2 modifiers (dependency length)

    for construction in ("anaphoric_binding", "auxiliary_movement", "wh_movement"):
        for depth in depths:
            for cat2_count in cat2_counts:
                accepted = 0
                attempts = 0
                while accepted < 5 and attempts < 5 * _MAX_GEN_ITEM_ATTEMPTS:
                    attempts += 1
                    counter = [0]
                    has_aux = construction == "auxiliary_movement"
                    toks = build_sentence(
                        rng, lex, counter,
                        phenomenon=construction, has_aux=has_aux,
                        force_cat2_subject=cat2_count, force_cat9_depth=depth,
                    )
                    label = apply_transform(toks, construction, rng, lex)
                    surface = to_string(toks)
                    tok_len = len(surface.split())
                    if not (min_length <= tok_len <= max_length):
                        continue
                    items.append({
                        "sentence": surface,
                        "grammar_type": "P",
                        "rule_type": None,
                        "construction": label,
                        "length": tok_len,
                        "split": "generalization",
                        "embedding_depth": depth,
                        "dependency_length": cat2_count,
                    })
                    accepted += 1

                if accepted < 5:
                    raise RuntimeError(
                        f"Grammar P gen items (depth={depth}, cat2={cat2_count}, "
                        f"construction='{construction}'): only {accepted}/5 fell in "
                        f"[{min_length}, {max_length}] after {attempts} attempts."
                    )
    return items
