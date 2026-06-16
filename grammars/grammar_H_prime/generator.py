"""Generator for Grammar H' — hierarchical base with one positional wh rule.

H' equals Grammar H except the wh-movement complex-NP island, which becomes a
positional surface scan (see ``transforms.py``). Rule logic lives in
``transforms.py`` and ``generate.py``.

``rule_type`` does not change the train/test distribution; it selects the
generalization probe family:
  - "linear"       → the positional complex-NP rule (divergence cases).
  - "hierarchical" → the rules H' keeps structural, with extra depth.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.grammar_H.lexicon import load as load_lex

from .generate import _one_item, _generalization_items


class GrammarHPrimeGenerator(BaseGrammarGenerator):
    """String generator for Grammar H'.

    Parameters
    ----------
    rule_type:
        ``"hierarchical"`` or ``"linear"`` — selects the generalization probe
        family (required for the mixed grammars).
    seed:
        Random seed.
    """

    def __init__(self, rule_type: str, seed: int = 42) -> None:
        super().__init__(grammar_type="H_prime", rule_type=rule_type, seed=seed)
        self._rng = random.Random(seed)
        self._lex = load_lex()

    def load_lexicon(self) -> Dict[str, List[str]]:
        """Return {category: [form, ...]} from the shared lexicon."""
        raw = load_lex()
        return {
            cat: [item["form"] for item in items]
            for cat, items in raw.items()
            if isinstance(items, list)
        }

    def generate_string(self) -> Tuple[str, Dict]:
        surface, phenomenon, _ = _one_item(self._rng, self._lex)
        return surface, {
            "grammar_type": "H_prime",
            "rule_type": self.rule_type,
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, self.rule_type, min_length, max_length)
