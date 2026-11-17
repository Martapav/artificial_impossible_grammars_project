"""Generator for Grammar Hprime — hierarchical base with one positional wh rule.

Hprime equals Grammar H except wh clause (i), which is swapped from the matrix
object (structural) to the third CAT1-position of the string (positional; see
``transforms.py``). Rule logic lives in ``transforms.py`` and ``generate.py``.
"""

from __future__ import annotations

import random

from grammars.tracing import TracedRandom, derivation_record, reset_trace
from typing import Dict, List, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.grammar_H.lexicon import load as load_lex
from grammars.metadata import FAMILY_TREE

from .generate import _one_item


class GrammarHPrimeGenerator(BaseGrammarGenerator):
    """String generator for Grammar Hprime.

    Parameters
    ----------
    seed:
        Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="Hprime", seed=seed)
        # TracedRandom is stream-identical to random.Random (tests/
        # test_tracing.py), so this records ell(x) without changing
        # a single generated character.
        self._rng = TracedRandom(seed)
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
        # One call = one generation ATTEMPT. generate_batch may reject it
        # (quota/length/skip), so each attempt starts a fresh record.
        reset_trace(self._rng)
        sites: Dict = {}
        surface, phenomenon, tree = _one_item(self._rng, self._lex,
                                              sites_out=sites)
        return surface, {
            "grammar_type": "Hprime",
            "length": len(surface.split()),
            "construction": phenomenon,
            # Transient bundle for the sidecar meta record (§8.5);
            # popped by metadata.finalize_item on accepted items, never written.
            "_payload": {"struct": tree, "family": FAMILY_TREE, "sites": sites,
                          "derivation": derivation_record(self._rng)},
        }

