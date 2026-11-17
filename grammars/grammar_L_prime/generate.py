"""Sentence generation for Grammar Lprime.

Lprime SHARES Grammar P's generation plan verbatim: ``_one_item`` below is
Grammar P's own ``_one_item`` (same phenomenon/site draws, forced structures,
duplicate screen) with the Lprime transform module passed
in. The two grammars differ in wh clause (i) only (matrix object from
generation history instead of third CAT1-position — see ``transforms.py``);
any change to Grammar P's pipeline propagates here automatically.

Construction labels: "neutral", "anaphoric_binding_refl",
"anaphoric_binding_pron", "auxiliary_movement", "wh_movement" (+ "*_skipped").
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import load as load_lex
from grammars.grammar_P.generate import _one_item as _p_one_item

from .transforms import apply as apply_transform


def _one_item(rng: random.Random, lex: dict,
              sites_out: Optional[Dict] = None,
              base_out: Optional[Dict] = None) -> Tuple[str, str, List[Token]]:
    """One L' item: Grammar P's generation plan, L' transform module."""
    return _p_one_item(rng, lex, apply_fn=apply_transform,
                       sites_out=sites_out, base_out=base_out)


def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return n positionally well-formed L' surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, List[Token]]]:
    """Return (surface, phenomenon_label, token_list) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]
