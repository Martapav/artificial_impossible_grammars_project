"""Grammar-agnostic corpus builder.

Writes JSONL to:
  <output_root>/train/<grammar_type>.jsonl
  <output_root>/validation/<grammar_type>.jsonl
  <output_root>/test_indistribution/<grammar_type>.jsonl
  <output_root>/test_generalization/<grammar_type>.jsonl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from grammars.base_generator import BaseGrammarGenerator, CONSTRUCTION_MIX


class CorpusBuilder:
    """Orchestrates three-split corpus generation for one grammar condition.

    Receives an instantiated generator and an output directory; each build_*
    method drives the generator to produce one split and writes a JSONL file.
    """

    def __init__(self, generator: BaseGrammarGenerator, output_root: Path) -> None:
        self.generator = generator
        self.output_root = Path(output_root)

        # One file per grammar; the grammar_type is the filename stem.
        self._suffix = generator.grammar_type

    # ------------------------------------------------------------------
    # Split builders
    # ------------------------------------------------------------------

    def build_train(
        self,
        n: int,
        constructions: List[str] | Dict[str, float] | None = None,
        min_length: int = 2,
        max_length: int = 25,
    ) -> Path:
        """Generate and write the training split; return the output path."""
        items = self.generator.generate_batch(
            n, constructions=constructions,
            min_length=min_length, max_length=max_length,
        )
        for item in items:
            item["split"] = "train"
        return self._write(items, "train")

    def build_validation(
        self,
        n: int,
        constructions: List[str] | Dict[str, float] | None = None,
        min_length: int = 2,
        max_length: int = 25,
    ) -> Path:
        """Generate and write the validation split; return the output path.

        Uses seed + 50_000 to keep the validation RNG trajectory independent
        from both training (seed) and test (seed + 100_000).
        """
        import random
        import numpy as np

        val_seed = self.generator.seed + 50_000
        random.seed(val_seed)
        np.random.seed(val_seed)

        items = self.generator.generate_batch(
            n, constructions=constructions,
            min_length=min_length, max_length=max_length,
        )
        for item in items:
            item["split"] = "validation"

        random.seed(self.generator.seed)
        np.random.seed(self.generator.seed)

        return self._write(items, "validation")

    def build_test_indistribution(
        self,
        n: int,
        constructions: List[str] | Dict[str, float] | None = None,
        min_length: int = 2,
        max_length: int = 25,
    ) -> Path:
        """Generate and write the in-distribution test split; return the output path.

        Uses seed + 100_000 to keep the test RNG trajectory independent from training.
        """
        import random
        import numpy as np

        held_out_seed = self.generator.seed + 100_000
        random.seed(held_out_seed)
        np.random.seed(held_out_seed)

        items = self.generator.generate_batch(
            n, constructions=constructions,
            min_length=min_length, max_length=max_length,
        )
        for item in items:
            item["split"] = "test_indistribution"

        # Restore original seed so subsequent calls are not affected.
        random.seed(self.generator.seed)
        np.random.seed(self.generator.seed)

        return self._write(items, "test_indistribution")

    def build_test_generalization(
        self,
        min_length: int = 25,
        max_length: int = 48,
    ) -> Path:
        """Generate and write the generalization split via get_generalization_items()."""
        items = self.generator.get_generalization_items(
            min_length=min_length, max_length=max_length,
        )
        return self._write(items, "test_generalization")

    def build_all(
        self,
        train_n: int,
        val_n: int,
        test_n: int,
        train_min_length: int = 2,
        train_max_length: int = 25,
        gen_min_length: int = 25,
        gen_max_length: int = 48,
    ) -> Dict[str, Path]:
        """Build all splits and return {split_name: Path}.

        Length windows: train/val/test use [2, 25]; generalization uses [25, 48].
        Every split is quota-balanced to the canonical CONSTRUCTION_MIX
        (exact 70/5/5/10/10 over neutral/refl/pron/aux/wh; ``*_skipped``
        draws are discarded and resampled).
        """
        return {
            "train": self.build_train(
                train_n, constructions=CONSTRUCTION_MIX,
                min_length=train_min_length, max_length=train_max_length,
            ),
            "validation": self.build_validation(
                val_n, constructions=CONSTRUCTION_MIX,
                min_length=train_min_length, max_length=train_max_length,
            ),
            "test_indistribution": self.build_test_indistribution(
                test_n, constructions=CONSTRUCTION_MIX,
                min_length=train_min_length, max_length=train_max_length,
            ),
            "test_generalization": self.build_test_generalization(
                min_length=gen_min_length, max_length=gen_max_length,
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _write(self, items: List[Dict], split_name: str) -> Path:
        """Serialize items as JSONL to <output_root>/<split_name>/<suffix>.jsonl."""
        out_dir = self.output_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self._suffix}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")
        return out_path
