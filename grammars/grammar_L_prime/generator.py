"""Generator for Grammar Lprime — positional base with one structural wh rule.

Lprime is the mirror image of Hprime: it equals Grammar P except wh clause (i),
which is swapped from the third CAT1-position (positional) to the matrix object,
read from the generation history (see ``transforms.py``). Rule logic lives in
``transforms.py`` and ``generate.py``.
"""

from __future__ import annotations

import random

from grammars.tracing import TracedRandom, derivation_record, reset_trace
from typing import Dict, List, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.grammar_P.lexicon import load as load_lex
from grammars.metadata import FAMILY_TOKENS

from .generate import _one_item


class GrammarLPrimeGenerator(BaseGrammarGenerator):
    """String generator for Grammar Lprime.

    Parameters
    ----------
    seed:
        Random seed.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="Lprime", seed=seed)
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
        surface, phenomenon, toks = _one_item(self._rng, self._lex,
                                              sites_out=sites)
        return surface, {
            "grammar_type": "Lprime",
            "length": len(surface.split()),
            "construction": phenomenon,
            # Transient bundle for the sidecar meta record (§8.5);
            # popped by metadata.finalize_item on accepted items, never written.
            "_payload": {"struct": toks, "family": FAMILY_TOKENS, "sites": sites,
                          "derivation": derivation_record(self._rng)},
        }

