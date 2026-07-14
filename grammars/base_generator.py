"""Grammar-agnostic infrastructure for artificial grammar generation.

All four grammar generators (H, Hprime, Lprime, P) inherit from
BaseGrammarGenerator.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import numpy as np

# Canonical corpus label mix (spec §9.1, TRANSFORM_RULES.md §1): 70% neutral,
# 10% per transformation, binding refined to refl/pron at 5%/5%. Passed by
# CorpusBuilder as the proportional construction quota for every corpus split,
# so the trained-on mix is exact by construction (skips and length rejections
# are resampled within each label's quota).
CONSTRUCTION_MIX: Dict[str, float] = {
    "neutral": 0.70,
    "anaphoric_binding_refl": 0.05,
    "anaphoric_binding_pron": 0.05,
    "auxiliary_movement": 0.10,
    "wh_movement": 0.10,
}


class BaseGrammarGenerator(ABC):
    """Abstract base for all four grammar generators."""

    def __init__(
        self,
        grammar_type: str,
        seed: int = 42,
    ) -> None:
        self.grammar_type = grammar_type
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
        constructions: Optional[List[str] | Dict[str, float]] = None,
        show_progress: bool = False,
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> List[Dict]:
        """Generate n sentences, optionally balanced over construction types.

        ``constructions`` may be a list (equal slots of n // len each) or a
        dict of proportions (e.g. CONSTRUCTION_MIX → exact 70/5/5/10/10
        quotas). Draws outside the quota or length window — including
        ``*_skipped`` draws — are rejected and regenerated.
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
            # ``*_skipped`` items never enter a corpus (TRANSFORM_RULES.md §1:
            # blocked draws are discarded and regenerated; blockability is
            # realized distributionally). Without construction quotas the
            # label mix is the natural proportions renormalized over applied
            # items.
            max_attempts = n * 50
            attempts = 0
            while len(items) < n and attempts < max_attempts:
                attempts += 1
                sentence, meta = self.generate_string()
                if str(meta.get("construction", "")).endswith("_skipped"):
                    continue
                if not _length_ok(meta):
                    continue
                items.append({"sentence": sentence, **meta})
                if show_progress and len(items) % 1000 == 0:
                    print(f"  {len(items)}/{n}")
            if len(items) < n:
                raise RuntimeError(
                    f"Sampling exhausted after {max_attempts} attempts with only "
                    f"{len(items)}/{n} applied items in window "
                    f"[{min_length}, {max_length}]. Check skip rates and that the "
                    "natural length distribution overlaps the window."
                )
            return items

        # Slot-controlled path: per-label quotas summing exactly to n.
        if isinstance(constructions, dict):
            total = sum(constructions.values())
            exact = {c: n * w / total for c, w in constructions.items()}
            quota = {c: int(v) for c, v in exact.items()}
            # Distribute the rounding remainder by largest fractional part.
            for c in sorted(exact, key=lambda c: exact[c] - quota[c],
                            reverse=True)[: n - sum(quota.values())]:
                quota[c] += 1
        else:
            quota = {c: n // len(constructions) for c in constructions}
        counts: Dict[str, int] = {c: 0 for c in quota}
        attempts = 0
        max_attempts = n * (50 if _filter else 20)  # higher when length filtering active

        while len(items) < n and attempts < max_attempts:
            attempts += 1
            sentence, meta = self.generate_string()
            construction = meta.get("construction")
            if construction not in counts:
                continue  # *_skipped or unknown label — discard
            if counts[construction] >= quota[construction]:
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
