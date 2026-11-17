"""Grammar-agnostic corpus builder.

Five-split architecture (TRANSFORM_RULES §10 — nothing else):
  <output_root>/train/<grammar_type>.jsonl                 70/5/5/10/10, [2,25]
  <output_root>/validation/<grammar_type>.jsonl            same
  <output_root>/test_indistribution/<grammar_type>.jsonl   same (the "normal test set")
  <output_root>/structural_probe_IOD/<grammar_type>.jsonl  4200 pair items, [2,25]
  <output_root>/structural_probe_OOD/<grammar_type>.jsonl  4200 pair items, [26,48]
  <output_root>/structural_probe_SELECT/<grammar_type>.jsonl 4200 items, [2,25]

Each corpus file gets a line-aligned metadata sidecar
``<split>/<grammar_type>.meta.jsonl`` (TRANSFORM_RULES §8.5) carrying the
per-sentence structural fields, surprisal targets, and annotated tokens computed
at generation time. Corpus and sidecar records share an explicit stable ``id``
(``{grammar}-{split}-{index}``) — never rely on line order alone. The training
loader reads only ``sentence``, so the sidecar leaves training untouched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from grammars import tracing
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

    def build_structural_probe_iod(self, n_per_subcase: int = 300) -> Path:
        """Generate and write the IOD structural probe (TRANSFORM_RULES §9).

        Grammatical/ungrammatical minimal pairs on training-matched structural
        configurations, lengths [2, 25]. Uses seed + 200_000 via a dedicated
        ``random.Random`` — independent of every other split and of build order.
        """
        from corpusgen.structural_probes import build_structural_probe

        items = build_structural_probe(
            self.generator.grammar_type, seed=self.generator.seed + 200_000,
            min_length=2, max_length=25, n_per_subcase=n_per_subcase,
        )
        for item in items:
            item["split"] = "structural_probe_IOD"
        return self._write(items, "structural_probe_IOD")

    def build_structural_probe_ood(self, n_per_subcase: int = 300) -> Path:
        """Generate and write the OOD structural probe: identical builder and
        generation rules, only longer strings (lengths [26, 48]; §2/§7.1).
        Uses seed + 250_000 via a dedicated ``random.Random``."""
        from corpusgen.structural_probes import build_structural_probe

        items = build_structural_probe(
            self.generator.grammar_type, seed=self.generator.seed + 250_000,
            min_length=26, max_length=48, n_per_subcase=n_per_subcase,
        )
        for item in items:
            item["split"] = "structural_probe_OOD"
        return self._write(items, "structural_probe_OOD")

    def build_structural_probe_select(self, n_per_subcase: int = 300) -> Path:
        """The recipe-SELECTION probe: same builder, fresh seed, same n.

        Kept disjoint from the reporting probes on purpose. Taking the argmax
        over a dozen recipes on the same items that later report the result
        biases that number upward and lets the recipe choice shape the H-vs-P
        contrast, so selection gets its own draw (seed + 300_000, distinct from
        IOD's +200_000 and OOD's +250_000). ``corpusgen/probe_checks.py`` enforces
        that disjointness after the build — it is not guaranteed by the seeds.

        **IOD length window only.** There is deliberately no OOD selection
        probe: OOD measures structural generalization, which is the finding, and
        tuning a recipe against it would convert the result into an artifact.

        Built last, and explicitly excluding the reporting probes' grammatical
        members: at production quotas the seeds alone collide in the short tail
        of the length distribution (see ``build_structural_probe``).
        """
        from corpusgen.structural_probes import build_structural_probe

        items = build_structural_probe(
            self.generator.grammar_type, seed=self.generator.seed + 300_000,
            min_length=2, max_length=25, n_per_subcase=n_per_subcase,
            exclude_grammatical=self._reporting_probe_members(),
        )
        for item in items:
            item["split"] = "structural_probe_SELECT"
        return self._write(items, "structural_probe_SELECT")

    def build_all(
        self,
        train_n: int,
        val_n: int,
        test_n: int,
        train_min_length: int = 2,
        train_max_length: int = 25,
        probe_n_per_subcase: int = 300,
        select_n_per_subcase: int = 300,
    ) -> Dict[str, Path]:
        """Build the **six** locked splits and return {split_name: Path}.

        Train/val/test are quota-balanced to the canonical CONSTRUCTION_MIX
        (exact 70/5/5/10/10 over neutral/refl/pron/aux/wh; ``*_skipped`` draws
        are discarded and resampled), lengths [2, 25]. The **three** structural
        probes are grammatical/ungrammatical minimal-pair corpora
        (TRANSFORM_RULES §10): IOD at [2, 25], OOD at [26, 48], and SELECT at
        [2, 25] for recipe selection only.

        "Five" was wrong wherever it appeared: SELECT has been one of the locked
        splits since it was introduced, and this method has always returned six
        paths. Corrected 2026-08-06 (A3).
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
            "structural_probe_IOD": self.build_structural_probe_iod(
                probe_n_per_subcase),
            "structural_probe_OOD": self.build_structural_probe_ood(
                probe_n_per_subcase),
            "structural_probe_SELECT": self.build_structural_probe_select(
                select_n_per_subcase),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _reporting_probe_members(self) -> set:
        """Grammatical members of this grammar's already-written IOD/OOD probes.

        Read from disk rather than threaded through the call chain so that
        SELECT is protected however it is invoked — ``build_all``, the CLI's
        ``--probes-only`` path, or a bare method call in a notebook. A probe
        that has not been written yet contributes nothing, and the post-build
        check in ``probe_checks`` is what catches that case.
        """
        members: set = set()
        for split in ("structural_probe_IOD", "structural_probe_OOD"):
            path = self.output_root / split / f"{self._suffix}.jsonl"
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    rec = json.loads(line)
                    if rec.get("variant") == "grammatical":
                        members.add(rec["sentence"])
        return members

    def _write(self, items: List[Dict], split_name: str) -> Path:
        """Serialize items as JSONL to <output_root>/<split_name>/<suffix>.jsonl.

        Splits each item into its corpus record and its ``_meta`` sidecar
        record, stamps the shared stable ``id`` on both, and writes the
        line-aligned ``<suffix>.meta.jsonl`` next to the corpus file. A
        partially annotated batch (some items with ``_meta``, some without)
        is an alignment bug and raises.
        """
        out_dir = self.output_root / split_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{self._suffix}.jsonl"

        metas: List[Dict] = []
        seen_uids: Dict[str, str] = {}
        self.uid_duplicates: List[Dict] = []
        with out_path.open("w", encoding="utf-8") as fh:
            for i, item in enumerate(items):
                item.pop("_payload", None)  # never serialize a stray payload
                meta = item.pop("_meta", None)
                # `id` is a dense index INTO THIS FILE. It is the line-alignment
                # key between the corpus and its sidecar and nothing more: it
                # means "the i-th item of this build", so a rebuild that changes
                # the item order re-points it without changing its value. Anything
                # joining model scores back to generation records must key on
                # `item_uid`.
                item_id = f"{self._suffix}-{split_name}-{i:05d}"
                item["id"] = item_id
                # `item_uid` is CONTENT-DERIVED here (grammars.tracing.item_uid,
                # A3) for every split whose generator does not assign one of its
                # own. The structural-probe builder does assign one
                # (`{pair_id}/{variant}`, keyed to the pair it belongs to rather
                # than to a line), and that value is left alone: it is the join
                # key `evaluation/score_pairs.py` and `grammars/probe_scoring.py`
                # already use.
                if item.get("item_uid") is None:
                    item["item_uid"] = tracing.item_uid(
                        grammar=item.get("grammar_type", self._suffix),
                        split=split_name,
                        sentence=item["sentence"],
                        derivation=(meta or {}).get("derivation"),
                    )
                # A repeat means both items agree on grammar, split, surface
                # string AND derivation signature — a genuine duplicate, not a
                # digest accident. Flagged and counted; never merged, never
                # dropped (the caller decides what a duplicate means).
                #
                # The flag goes on the SIDECAR ONLY, never on the corpus record.
                # The corpus record is the training input and its schema must be
                # identical across all six splits: `models.dataload.tokenize_corpus`
                # takes `remove_columns` from SPLITS[0] and applies it to every
                # split, so a column present on train and absent on the probes
                # (which have no duplicates) raises before a single step runs.
                # Generation metadata belongs in the sidecar; this is metadata.
                uid = item["item_uid"]
                duplicate_of = seen_uids.get(uid)
                if duplicate_of is not None:
                    self.uid_duplicates.append(
                        {"item_uid": uid, "id": item_id,
                         "first_id": duplicate_of,
                         "sentence": item["sentence"]})
                else:
                    seen_uids[uid] = item_id
                if meta is not None:
                    meta["item_uid"] = uid
                    if duplicate_of is not None:
                        meta["uid_duplicate_of"] = duplicate_of
                    metas.append({"id": item_id, "sentence": item["sentence"],
                                  **meta})
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Acceptance record for this split: the rejection-sampling correction
        # that turns the sidecar's raw ell(x) into -log p_corpus(x).
        acceptance = getattr(self.generator, "acceptance", None)
        if acceptance is not None and acceptance.attempts:
            (out_dir / f"{self._suffix}.accept.json").write_text(
                json.dumps(acceptance.summary(), indent=2), encoding="utf-8")

        if metas:
            if len(metas) != len(items):
                raise RuntimeError(
                    f"{split_name}/{self._suffix}: {len(metas)}/{len(items)} "
                    "items carried a meta record — sidecar would misalign"
                )
            meta_path = out_dir / f"{self._suffix}.meta.jsonl"
            with meta_path.open("w", encoding="utf-8") as fh:
                for meta in metas:
                    fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
        return out_path
