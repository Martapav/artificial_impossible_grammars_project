"""Generator for Grammar Hprime — hierarchical base with one positional wh rule.

Hprime equals Grammar H except wh clause (i), which is swapped from the matrix
object (structural) to the third CAT1-position of the string (positional; see
``transforms.py``). Rule logic lives in ``transforms.py`` and ``generate.py``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.grammar_H.lexicon import load as load_lex

from .generate import _one_item, _generalization_items


class GrammarHPrimeGenerator(BaseGrammarGenerator):
    """String generator for Grammar Hprime.

    Parameters
    ----------
    seed:
        Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="Hprime", seed=seed)
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
            "grammar_type": "Hprime",
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, min_length, max_length)

