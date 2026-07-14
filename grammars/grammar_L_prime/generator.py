"""Generator for Grammar Lprime — positional base with one structural wh rule.

Lprime is the mirror image of Hprime: it equals Grammar P except wh clause (i),
which is swapped from the third CAT1-position (positional) to the matrix object,
read from the generation history (see ``transforms.py``). Rule logic lives in
``transforms.py`` and ``generate.py``.
"""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.grammar_P.lexicon import load as load_lex

from .generate import _one_item, _generalization_items


class GrammarLPrimeGenerator(BaseGrammarGenerator):
    """String generator for Grammar Lprime.

    Parameters
    ----------
    seed:
        Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="Lprime", seed=seed)
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
            "grammar_type": "Lprime",
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, min_length, max_length)

