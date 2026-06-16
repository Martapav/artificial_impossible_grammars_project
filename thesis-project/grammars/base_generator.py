"""Grammar-agnostic infrastructure for artificial grammar generation.

All four grammar generators (H, H-prime, L-prime, P) inherit from
BaseGrammarGenerator. 
"""

from __future__ import annotations

import json
import random
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


MIXED_GRAMMARS = {"H_prime", "L_prime"}  # grammars that have two rule_type variants
VALID_RULE_TYPES = {"hierarchical", "linear"}


class BaseGrammarGenerator(ABC):
    """Abstract base for all four grammar generators."""

    def __init__(
        self,
        grammar_type: str,
        rule_type: Optional[str] = None,
        seed: int = 42,
    ) -> None:
        if grammar_type in MIXED_GRAMMARS and rule_type is None:
            raise ValueError(
                f"Grammar '{grammar_type}' requires a rule_type "
                f"('hierarchical' or 'linear')."
            )
        if grammar_type not in MIXED_GRAMMARS and rule_type is not None:
            raise ValueError(
                f"Grammar '{grammar_type}' does not accept a rule_type."
            )
        if rule_type is not None and rule_type not in VALID_RULE_TYPES:
            raise ValueError(
                f"rule_type must be one of {VALID_RULE_TYPES}, got '{rule_type}'."
            )

        self.grammar_type = grammar_type
        self.rule_type = rule_type
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.lexicon: Dict[str, List[str]] = self.load_lexicon()


    @abstractmethod
    def load_lexicon(self) -> Dict[str, List[str]]:
        """Return the shared lexicon as {category: [form, ...]}."""

    @abstractmethod
    def generate_string(self) -> Tuple[str, Dict]:
        """Generate one sentence; return (surface_string, metadata_dict)."""

    @abstractmethod
    def get_generalization_items(
        self,
        min_length: int = 25,
        max_length: int = 48,
    ) -> List[Dict]:
        """Return the grammar's generalization test set.

        Depth/length probe items are filtered to [min_length, max_length];
        hand-built divergence probes are returned as-is.
        """


    def sample_length(
        self,
        mu: float = 25.0,
        sigma: float = 8.0,
        min_len: int = 2,
        max_len: int = 48,
    ) -> int:
        """Sample a sentence length from N(mu, sigma) truncated to [min_len, max_len].

        Default window [2, 48] = full support; callers narrow it to [2, 25]
        for train/test or [25, 48] for generalization.
        """
        while True:
            length = int(np.random.normal(mu, sigma))
            if min_len <= length <= max_len:
                return length

    def generate_batch(
        self,
        n: int,
        constructions: Optional[List[str]] = None,
        show_progress: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[Dict]:
        """Generate n sentences, optionally balanced over construction types.

        If constructions is given, each type receives n // len(constructions)
        slots; draws outside the quota or length window are rejected and
        regenerated.
        """
        items: List[Dict] = []
        _filter = min_length is not None or max_length is not None

        def _length_ok(meta: Dict) -> bool:
            if not _filter:
                return True
            ln = meta.get("length", 0)
            if min_length is not None and ln < min_length:
                return False
            if max_length is not None and ln > max_length:
                return False
            return True

        if constructions is None:
            if not _filter:
                for i in range(n):
                    sentence, meta = self.generate_string()
                    items.append({"sentence": sentence, **meta})
                    if show_progress and (i + 1) % 1000 == 0:
                        print(f"  {i + 1}/{n}")
                return items
            max_attempts = n * 50
            attempts = 0
            while len(items) < n and attempts < max_attempts:
                attempts += 1
                sentence, meta = self.generate_string()
                if not _length_ok(meta):
                    continue
                items.append({"sentence": sentence, **meta})
                if show_progress and len(items) % 1000 == 0:
                    print(f"  {len(items)}/{n}")
            if len(items) < n:
                raise RuntimeError(
                    f"Length-filtered sampling exhausted after {max_attempts} attempts "
                    f"with only {len(items)}/{n} items in "
                    f"[{min_length}, {max_length}]. "
                    "Check that the natural sentence length distribution overlaps the window."
                )
            return items

        # Slot-controlled path.
        quota = n // len(constructions)
        counts: Dict[str, int] = {c: 0 for c in constructions}
        attempts = 0
        max_attempts = n * (50 if _filter else 20)  # higher when length filtering active

        while len(items) < n and attempts < max_attempts:
            attempts += 1
            sentence, meta = self.generate_string()
            construction = meta.get("construction")
            if construction not in counts:
                continue  # *_skipped or unknown label — discard
            if counts[construction] >= quota:
                continue
            if not _length_ok(meta):
                continue
            counts[construction] += 1
            items.append({"sentence": sentence, **meta})
            if show_progress and len(items) % 1000 == 0:
                print(f"  {len(items)}/{n}")

        if len(items) < n:
            raise RuntimeError(
                f"Slot-controlled sampling exhausted after {max_attempts} "
                f"attempts with only {len(items)}/{n} items accepted. "
                "Check that generate_string covers all construction labels "
                "and that the length window overlaps the natural distribution."
            )

        return items
