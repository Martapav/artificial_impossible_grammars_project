"""Training-recipe grid search for the artificial-grammar LMs.

Sweeps a small, coarse grid to pick a training recipe (learning rate, batch
size, model size) before committing to the full 100k-sentence training runs.
Everything is centred on the default architecture point in `models.model`.

The grid (edit the constants below to change it):

    LEARNING_RATES  x  MODEL_SIZES  x  BATCH_SIZES   = 2 x 3 x 2 = 12 configs

run over `GRAMMARS` (4) and `SEEDS` (2) -> 96 runs by default. Warmup and
weight decay are held fixed as low-sensitivity axes. Each run trains on the FULL
train split until the probe score stops improving (or until the `--epochs` cap)
and is scored two ways at every epoch end: loss / perplexity on the full
validation split, and per-item minimal-pair surprisal on the SELECTION probe.

**Why training runs to a plateau instead of a fixed budget.** A stopping rule
read off a fixed-budget run is only valid at that budget, and the thesis needs
one rule applied to every grammar. Two things make the plateau meaningful:

  * the schedule is `constant_with_warmup`, not cosine. Cosine anneals over
    `num_train_epochs`, so under it the learning rate — and therefore the
    plateau — is a function of the cap, which is the same coupling in disguise;
    worse, a cosine run flattens near its end simply because the learning rate
    is going to zero. Under a constant rate the learning rate at step t does not
    depend on the total, so "no improvement for `--patience` epochs" means the
    same thing in a run that stops at 22 and one that stops at 47.
  * stopping reads the STOP half of the probe pairs and selection reads the
    SCORE half (`evaluation.aggregate.pair_half`). Sharing them would let every
    run stop at its own best moment on the items it is judged by.

**Why both, and why the probe is the one that selects.** Validation loss puts
~1.3% of its weight on the transformation-critical positions (3,000 of 237,530
targets on H: the split is 70% neutral, and even a phenomenon item has one
critical token). It cannot resolve a difference in how well a recipe learned the
rules, and the 2026-08-02 sweep bore that out — 12 configs inside a 0.07-nat
band, eight inside 0.01. Recipe selection therefore runs on the minimal-pair
statistic (`evaluation/aggregate.py`); loss is kept as an adequacy gate against
the n-gram baseline. Full reasoning in `docs/RECIPE_SELECTION.md`.

Probe scores are written per ITEM, not summarized: `S[r]` z-scores each term
across the recipes being compared, so it is a cross-run quantity that cannot be
computed inside one run, and keeping the raw rows means the objective can be
re-run without retraining. They are sharded one parquet per run under
`probe_scores/<run_id>.parquet`, so every file names its own grammar and no run
can overwrite another's scores; `evaluation.aggregate` takes globs.

Output layout (under --output-dir; a bare NAME resolves under results/):

    run_manifest.jsonl              # one line per invocation: grid, corpora +
                                    # row counts, probe split, device, git SHA
    results.jsonl                   # one line per finished run (source of truth)
    results.csv                     # the same, flattened (curves dropped)
    summary_by_grammar.csv          # per recipe, ranked WITHIN each grammar
                                    # (no pooled table: the four corpora have
                                    #  different entropies, so a loss averaged
                                    #  over grammars is not interpretable —
                                    #  selection is analysis/recipe_selection.py)
    probe_scores/<run_id>.parquet   # per-item probe scores, one shard per run

Pipeline integration:
  * vocabulary / tokenizer  -> models.tokenize
  * corpora + tokenization  -> models.dataload (load_corpus, tokenize_corpus)
  * model instantiation     -> models.model  (build_model, same defaults)

The run is resumable: each finished run is appended to `results.jsonl` in the
output directory and skipped on re-invocation, so an interrupted sweep can be
restarted (or chunked with `--limit`) without repeating work. No checkpoints
are kept (save_strategy="no"); recipe selection only needs the final metrics.

Presets:
  * `--preset full` (default) — 4 grammars x 2 seeds x 12 configs = 96 runs on
    the full train split. Cost now depends on where the runs plateau; at the
    `--epochs` cap it is ~9 min per epoch across a 24-run grid on a 3090.
  * `--preset reduced` — 2 grammars (one per FAMILY) x 1 seed x 12 configs = 24
    runs on a 10k subset: ~12x less compute. See PRESETS for the reasoning.

Every axis (`--grammars --seeds --sizes --learning-rates --batch-sizes
--subset-size`) can also be set explicitly and overrides the preset. Widening
later costs nothing: runs are keyed by `run_id` and already-completed ones are
skipped, so a `full` sweep reuses every `reduced` result in results.jsonl.

Device: `--device auto|cpu|cuda` (default auto). Requesting `cuda` when it is
unavailable is an ERROR, not a silent CPU fallback — this file previously
hardcoded `use_cpu=True`, which would have run a GPU machine on CPU with no
visible symptom. Add `--fp16` or `--bf16` on a GPU.

Examples:
    python -m models.grid_search --preset reduced --dry-run     # list the 24 runs
    python -m models.grid_search --preset reduced --device cuda --bf16
    python -m models.grid_search --device cuda --bf16            # the full sweep
    python -m models.grid_search --limit 8                       # next 8 pending
    python -m models.grid_search --patience 0                    # train to the cap
    python -m models.grid_search --summary                       # print summary

The pilot (one recipe, three sizes, trained to plateau):
    python -m models.grid_search --preset full \\
        --sizes small medium large --learning-rates 3e-4 --batch-sizes 64 \\
        --seeds 100 101 --epochs 60 --patience 5 --device cuda --bf16 \\
        --output-dir pilot_search
    python -m evaluation.aggregate 'results/pilot_search/probe_scores/*.parquet' \\
        --select --half score --out results/eval/pilot_terms.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import socket
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from evaluation import probe_callback
# Shared with `models.train`, which scores the same probe for its learning curve
# with early stopping switched off. Re-exported here (rather than only imported)
# because `_Patience`/`stopping_metric` are the sweep's stopping rule and are
# tested through this module's namespace.
from evaluation.probe_callback import (  # noqa: F401
    STOP_MIN_DELTA,
    ProbeScoringCallback,
    _Patience,
    stopping_metric,
)

from models.dataload import (CORPORA_ROOT, corpus_path, corpus_set_name,
                            load_corpus, tokenize_corpus)
from models.train import (
    git_sha,
    planned_total_steps,
    resolve_device,
    resolve_warmup_steps,
)
from models.model import (
    DEFAULT_N_EMBD,
    DEFAULT_N_HEAD,
    DEFAULT_N_LAYER,
    build_model,
    count_parameters,
)
from models.tokenize import tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
# Every sweep's artefacts live under results/, beside results/grid_search/ and
# results/eval/, so "where did that table come from" is answerable from the path
# alone. A bare --output-dir NAME resolves here (see `resolve_output_dir`).
RESULTS_ROOT = REPO_ROOT / "results"

# --- The grid -------------------------------------------------------------
# Three model sizes centred on models.model's default point ("medium").
MODEL_SIZES: dict[str, dict[str, int]] = {
    "small": dict(n_layer=2, n_embd=128, n_head=4),
    "medium": dict(n_layer=DEFAULT_N_LAYER, n_embd=DEFAULT_N_EMBD, n_head=DEFAULT_N_HEAD),
    "large": dict(n_layer=6, n_embd=256, n_head=8),
}
LEARNING_RATES: tuple[float, ...] = (3e-4, 1e-3)   # coarse LR band
BATCH_SIZES: tuple[int, ...] = (32, 64)
# -> 3 x 2 x 2 = 12 configurations.

# Held fixed (low-sensitivity axes).
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
# Budget-independent by default, which is what makes the stopping epoch a
# property of the run rather than of the cap. See the module docstring.
DEFAULT_LR_SCHEDULER = "constant_with_warmup"

# The four base grammars.
GRAMMARS: tuple[str, ...] = ("H", "P", "Hprime", "Lprime")
SEEDS: tuple[int, ...] = (0, 1)

# Fixed subsample of the train split, shared across every run for comparability.
SUBSET_SEED = 12345

# None = the full train split (the production budget). An int subsamples it.
DEFAULT_SUBSET_SIZE: int | None = None

# --- Presets ---------------------------------------------------------------
# `full` is the sweep as originally specified. `reduced` narrows three axes on
# the grounds that recipe selection does not need the resolution the final
# experiment does:
#   * grammars -> one per FAMILY. H'/L' share their generation plans with H/P
#     (they differ only in the transform module), so their optimal recipe is not
#     independent evidence. Both families are still represented, since the tree
#     and token families have different sequence statistics.
#   * seeds -> 1. Seed variance is a property of the trained model, not of the
#     recipe ranking; 2 seeds halves the noise but doubles the cost, and the
#     production run uses 5 seeds anyway.
#   * subset -> 10k. Recipe ranking is a relative comparison; it needs enough
#     data to separate configs, not enough to converge.
# 96 runs -> 24, and each run is 3x shorter: ~12x less compute.
# Widening later is free: runs are keyed by `run_id`, so a `full` sweep reuses
# every `reduced` result already in results.jsonl.
PRESETS: dict[str, dict] = {
    "full": dict(grammars=GRAMMARS, seeds=SEEDS, sizes=tuple(MODEL_SIZES),
                 learning_rates=LEARNING_RATES, batch_sizes=BATCH_SIZES,
                 subset_size=DEFAULT_SUBSET_SIZE),
    "reduced": dict(grammars=("H", "P"), seeds=(0,), sizes=tuple(MODEL_SIZES),
                    learning_rates=LEARNING_RATES, batch_sizes=BATCH_SIZES,
                    subset_size=10_000),
}
# `--epochs` is now a CAP, not a target: early stopping normally ends a run
# first, and a run that reaches the cap has NOT demonstrated a plateau (the
# record says so in `hit_cap`). Sized from measured wall times — across a 24-run
# pilot at bs64 one epoch costs ~9 min, so 60 is ~9 h, one night.
DEFAULT_EPOCHS = 60
# Epochs without improvement before a run stops. 3 fires on noise in curves this
# flat; 0 disables early stopping and trains every run to the cap.
DEFAULT_PATIENCE = 5
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_OUTPUT_DIR = RESULTS_ROOT / "pilot_search"
RESULTS_FILENAME = "results.jsonl"
# Probe scores are SHARDED, one parquet per run_id, rather than accumulated into
# a single file. Three reasons: `run_id` carries the grammar, so a shard's
# provenance is readable from its filename; each run writes its own file exactly
# once, so a crash or a second job pointed at the same directory cannot destroy
# earlier runs (the previous read-concat-rewrite could); and at 4,200 probe items
# x up to 60 epochs x 24 runs the single table reaches millions of rows that were
# being rewritten in full after every run. `evaluation.aggregate` already accepts
# globs, so nothing downstream changes.
SCORES_DIRNAME = "probe_scores"
# One JSON line per INVOCATION (not per run): what was asked for, on which
# corpora, from which commit. Append-only, so a resumed sweep records every
# invocation that contributed to the directory instead of overwriting the first.
MANIFEST_FILENAME = "run_manifest.jsonl"

# The production train split (README, TRANSFORM_RULES §10). A sweep launched on
# anything else is not the experiment, and the difference is invisible in the
# output until the run is over — so it is checked before training, not after.
EXPECTED_TRAIN_N = 100_000

# The probe the recipe is SELECTED on — never the reporting probes, and never
# OOD (docs/RECIPE_SELECTION.md §1.7). Re-exported from
# `evaluation.probe_callback`, which the production run shares.
SELECT_SPLIT = probe_callback.SELECT_SPLIT


@dataclass(frozen=True)
class RunSpec:
    """One point in the full grammar x config x seed grid."""

    grammar: str
    size_name: str
    learning_rate: float
    batch_size: int
    seed: int

    @property
    def run_id(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("-0", "-")
        return f"{self.grammar}__{self.size_name}__lr{lr}__bs{self.batch_size}__s{self.seed}"

    @property
    def size(self) -> dict[str, int]:
        return MODEL_SIZES[self.size_name]


def all_run_specs(
    grammars: tuple[str, ...] = GRAMMARS,
    seeds: tuple[int, ...] = SEEDS,
    sizes: tuple[str, ...] = tuple(MODEL_SIZES),
    learning_rates: tuple[float, ...] = LEARNING_RATES,
    batch_sizes: tuple[int, ...] = BATCH_SIZES,
) -> list[RunSpec]:
    """The grid, grouped by grammar so each corpus is loaded once.

    Every axis is a parameter so the sweep can be narrowed from the CLI without
    editing constants — a reduced grid stays reproducible and self-describing in
    the run ids, and `results.jsonl` from a narrow sweep is still a valid subset
    of a wider one (runs are keyed by `run_id`, so a later widening reuses them).
    """
    specs = []
    for grammar in grammars:
        for size_name, lr, batch_size in product(sizes, learning_rates, batch_sizes):
            for seed in seeds:
                specs.append(RunSpec(grammar, size_name, lr, batch_size, seed))
    return specs


# --- Data ------------------------------------------------------------------

def prepare_datasets(grammar: str, subset_size: int | None) -> tuple[Dataset, Dataset]:
    """Tokenized (train-subset, full-validation) for one grammar.

    `subset_size=None` uses the full train split — the production budget. An
    int shuffles with a fixed seed and truncates, so every config/seed sees the
    identical training subset. Only these two splits are tokenized (the test and
    probe splits are left untouched; probes are scored by the callback, which
    reads them directly).
    """
    raw = load_corpus(grammar)
    n = len(raw["train"])
    if subset_size is not None and subset_size < n:
        train_raw = raw["train"].shuffle(seed=SUBSET_SEED).select(range(subset_size))
    else:
        train_raw = raw["train"]  # full split, or corpus smaller than requested
    tokenized = tokenize_corpus(DatasetDict(train=train_raw, validation=raw["validation"]))
    return tokenized["train"], tokenized["validation"]


def count_rows(path: Path) -> int:
    """Rows in a .jsonl file, counted without parsing it (~0.02 s for 100k)."""
    total = 0
    last = b"\n"
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            total += chunk.count(b"\n")
            last = chunk[-1:]
    return total + (1 if last != b"\n" else 0)   # file may lack a final newline


def corpus_inventory(grammars: tuple[str, ...],
                     probe_split: str = SELECT_SPLIT) -> dict[str, dict[str, int]]:
    """Row counts for the three splits this sweep actually reads, per grammar.

    Reported in the banner and recorded in the manifest. The sweep loads its
    corpora through `models.dataload`, which honours `THESIS_CORPORA_ROOT` and
    may resolve anywhere — so
    "which files did this run read, and how big were they" is not something the
    launch command can be trusted to answer. Counting them costs a second.
    """
    return {
        grammar: {
            split: count_rows(corpus_path(grammar, split))
            for split in ("train", "validation", probe_split)
        }
        for grammar in grammars
    }


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of one file (~0.4 s for a 100k-line split)."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def corpus_digests(grammars: tuple[str, ...],
                   probe_split: str = SELECT_SPLIT) -> dict[str, dict[str, str]]:
    """SHA-256 of every split file this sweep reads, plus its metadata sidecar.

    Row counts already catch a mis-pointed corpora root; they do not catch the
    failure that actually costs a result — a corpus REBUILT between two runs of
    the same sweep. The item `id` in a probe file is a dense index into that
    file (`corpusgen/corpus_builder._write`), so a rebuild that changes the item
    order leaves every previously written score shard joining onto the wrong
    generation records, with matching row counts and no error anywhere. The
    digest is what makes that detectable after the fact: two invocations whose
    manifest lines disagree on a hash were not scoring the same items, whatever
    else they agree on.

    The sidecar is hashed too, and separately. It carries the critical spans and
    every label column, so a sidecar regenerated against an unchanged corpus
    changes the scores without changing the corpus hash.
    """
    out: dict[str, dict[str, str]] = {}
    for grammar in grammars:
        entry: dict[str, str] = {}
        for split in ("train", "validation", probe_split):
            path = corpus_path(grammar, split)
            entry[split] = sha256_file(path)
            sidecar = path.with_suffix(".meta.jsonl")
            if sidecar.exists():
                entry[f"{split}.meta"] = sha256_file(sidecar)
        out[grammar] = entry
    return out


def check_corpora(inventory: dict[str, dict[str, int]], subset_requested: int | None,
                  expect_train_n: int) -> None:
    """Refuse to start when the train split is not the production budget.

    The failure this catches is a wrong corpora location (a mis-pointed
    root, a partial copy, probe data sitting in the train slot): the sweep
    would train happily on a fraction of the data, and nothing in its output says
    so until the runs are finished. `expect_train_n=0` disables the check; a
    requested `--subset-size` disables it too, since then the smaller split is
    the deliberate ask.
    """
    if not expect_train_n or subset_requested is not None:
        return
    wrong = {g: n["train"] for g, n in inventory.items() if n["train"] != expect_train_n}
    if wrong:
        detail = ", ".join(f"{g}={n}" for g, n in sorted(wrong.items()))
        raise SystemExit(
            f"\nERROR: train split is not the production budget ({expect_train_n} "
            f"rows): {detail}.\n"
            f"Corpora root: {CORPORA_ROOT}\n"
            f"Check that THESIS_CORPORA_ROOT points at the corpora you audited "
            f"(`ls -l $THESIS_CORPORA_ROOT` and `sha256sum -c "
            f"CHECKSUMS.sha256` inside it), or pass --expect-train-n to override "
        )


def check_corpus_set_binding() -> None:
    """Refuse to start when the resolved corpus is not the set that was asked for.

    `THESIS_CORPUS_SET` names a generation; `CORPORA_ROOT` is where that name
    resolved. If the resolved directory is not named for a set, the run is about
    to read a corpus nobody addressed — e.g. a stale flat `corpora/train/`
    shadowing `corpora/set_1/`, which would score the whole sweep against the
    wrong generation. `dataload.resolve_corpora_root` refuses that shadowing
    outright; this is the second line, because the cost of being wrong is an
    overnight sweep.

    A deliberately flat root is still reachable — point `THESIS_CORPORA_ROOT`
    straight at it and leave `THESIS_CORPUS_SET` unset.
    """
    from models.dataload import (CORPUS_SET, CORPUS_SET_IS_EXPLICIT,
                                 resolve_corpora_root)

    # Re-resolve rather than trust the module-level CORPORA_ROOT: `_initial_root`
    # swallows a FileNotFoundError at import time and falls back to `base/<set>`,
    # which may not exist. Comparing only that path's last component would then
    # "pass" on a directory that is not there at all.
    try:
        resolved = resolve_corpora_root()
    except FileNotFoundError as exc:
        raise SystemExit(f"\nERROR: {exc}")

    if not resolved.is_dir():
        raise SystemExit(
            f"\nERROR: the corpora resolved to {resolved}, which does not exist.")
    if resolved != CORPORA_ROOT:
        raise SystemExit(
            f"\nERROR: corpora root is inconsistent — imported as {CORPORA_ROOT}, "
            f"re-resolves to {resolved}. Something changed the environment after "
            f"import; refusing to guess which corpus this sweep is about.")
    if CORPUS_SET_IS_EXPLICIT and resolved.name != CORPUS_SET:
        raise SystemExit(
            f"\nERROR: THESIS_CORPUS_SET={CORPUS_SET!r} but the corpora resolved "
            f"to\n  {resolved}\nwhich is not that set. A corpus directory at the "
            f"parent level shadows the set you shipped, and the sweep would train "
            f"on a generation nobody chose (this cost a full pilot on "
            f"2026-08-05).\n"
            f"Point THESIS_CORPORA_ROOT at the set itself, or move the flat "
            f"splits into a set directory."
        )
    print(f"corpus set: {resolved}")


# --- Per-epoch probe scoring ------------------------------------------------

def write_scores(scores_dir: Path, run_id: str, frame) -> Path:
    """Writes one run's probe scores to its own shard, `<run_id>.parquet`.

    One file per run, written once, named by a `run_id` that carries the grammar
    — so no run can overwrite another's scores, a half-written shard costs only
    the cell that produced it, and the grammar is readable from the filename
    without opening anything. `evaluation.aggregate` globs the directory.
    """
    from evaluation.provenance import write_parquet
    from evaluation.score_pairs import validate_scores
    from models.dataload import corpus_provenance

    # Checked here, not inside the callback: the invariants that span epochs
    # (one row per item-epoch, the same item set every epoch) can only be
    # evaluated once the run's frames are concatenated. A run that fails leaves
    # no shard, which is the right outcome — a shard `aggregate.py` would silently
    # mis-pair is worse than a missing one, and `run_manifest.jsonl` records that
    # the run was attempted.
    validate_scores(frame, context=f"probe scores for {run_id}")

    scores_dir.mkdir(parents=True, exist_ok=True)
    path = scores_dir / f"{run_id}.parquet"
    # Stamped with the corpus these scores describe, exactly as
    # `evaluation.score_pairs` does — `evaluation.aggregate` refuses to join a
    # shard it cannot bind to a generation, and these shards are the ones the
    # recipe is selected from.
    grammars = sorted(frame["grammar"].unique()) if "grammar" in frame else []
    splits = sorted(frame["split"].unique()) if "split" in frame else []
    write_parquet(frame, path, corpus_provenance(grammars, splits))
    return path


def write_manifest(manifest_path: Path, record: dict) -> None:
    """Appends one invocation's description to the sweep manifest."""
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# --- One run ---------------------------------------------------------------

def _perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def run_one(
    spec: RunSpec,
    train_ds: Dataset,
    eval_ds: Dataset,
    *,
    epochs: int,
    max_steps: int | None,
    eval_batch_size: int,
    report_to: str,
    disable_tqdm: bool,
    device: str = "cpu",
    fp16: bool = False,
    bf16: bool = False,
    score_probe: bool = True,
    probe_split: str = SELECT_SPLIT,
    patience: int = DEFAULT_PATIENCE,
    lr_scheduler: str = DEFAULT_LR_SCHEDULER,
) -> tuple[dict, object]:
    """Train one configuration; return (metrics record, per-item probe scores).

    `epochs` is the cap; the run normally ends earlier, on `patience`.
    """
    model = build_model(
        n_layer=spec.size["n_layer"],
        n_embd=spec.size["n_embd"],
        n_head=spec.size["n_head"],
        seed=spec.seed,
    )
    total_params, _ = count_parameters(model)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # `warmup_ratio` is deprecated (removal announced for transformers v5.2).
    # The ratio is resolved per CELL, against that cell's own planned step count
    # — the grid varies batch size, so the cells do not share one. See
    # `models.train.resolve_warmup_steps` for why a fixed count is not the
    # equivalent migration here.
    total_steps = planned_total_steps(
        len(train_ds), spec.batch_size, epochs, max_steps)
    warmup_steps = resolve_warmup_steps(WARMUP_RATIO, total_steps)

    with tempfile.TemporaryDirectory(prefix="gridrun_") as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=spec.batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=spec.learning_rate,
            num_train_epochs=epochs,
            max_steps=max_steps if max_steps is not None else -1,
            warmup_steps=warmup_steps,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=lr_scheduler,
            # Per-epoch eval: without it there is no learning curve at all, which
            # is exactly why the 2026-08-02 sweep could not answer whether its
            # runs had converged. It is the cheapest fix in this file.
            eval_strategy="epoch",
            save_strategy="no",     # recipe selection needs metrics, not weights
            logging_strategy="epoch",
            seed=spec.seed,
            data_seed=spec.seed,
            report_to=report_to,
            disable_tqdm=disable_tqdm,
            # Device is resolved by the caller (models.train.resolve_device),
            # which errors rather than silently falling back to CPU. This used to
            # be hardcoded `use_cpu=True`, which would have quietly run a GPU box
            # on CPU and made the sweep ~10x slower with no visible symptom.
            use_cpu=(device == "cpu"),
            fp16=fp16,
            bf16=bf16,
            dataloader_num_workers=0,
        )
        probe_cb = (
            ProbeScoringCallback(spec.grammar, spec.run_id, split=probe_split,
                                 device=device, batch_size=eval_batch_size,
                                 patience=patience)
            if score_probe else None
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
            callbacks=[probe_cb] if probe_cb else None,
        )
        start = time.time()
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        wall = time.time() - start

    eval_loss = eval_metrics["eval_loss"]
    # The validation learning curve, so "did it converge?" is answerable from
    # the artefacts alone rather than needing the run to be repeated.
    # Deduplicated on epoch: the explicit `trainer.evaluate()` above re-scores
    # the final epoch, which eval_strategy="epoch" has already logged, so the
    # raw log_history carries the last epoch twice.
    curve_by_epoch: dict[float, float] = {}
    for e in trainer.state.log_history:
        if "eval_loss" in e:
            curve_by_epoch[round(e["epoch"], 4)] = e["eval_loss"]
    curve = [{"epoch": ep, "eval_loss": loss}
             for ep, loss in sorted(curve_by_epoch.items())]
    train_curve = [
        {"epoch": e["epoch"], "loss": e["loss"]}
        for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e
    ]
    # Where the run actually ended, and whether it got there on its own. A run
    # that reached the cap has NOT shown a plateau, so anything concluded from
    # its stopping epoch is provisional — the sweep must say which is which.
    # `hit_cap` is left None under --max-steps, where the cap is not in epochs.
    stopped_epoch = round(trainer.state.epoch or 0.0, 4)
    hit_cap = None if max_steps is not None else bool(stopped_epoch >= epochs - 1e-6)
    record = {
        "run_id": spec.run_id,
        "grammar": spec.grammar,
        "size_name": spec.size_name,
        "n_layer": spec.size["n_layer"],
        "n_embd": spec.size["n_embd"],
        "n_head": spec.size["n_head"],
        "n_params": total_params,
        "learning_rate": spec.learning_rate,
        "batch_size": spec.batch_size,
        "seed": spec.seed,
        "epochs": epochs,
        "stopped_epoch": stopped_epoch,
        "hit_cap": hit_cap,
        "patience": patience if score_probe else None,
        "max_steps": max_steps,
        "subset_size": len(train_ds),
        "warmup_ratio": WARMUP_RATIO,
        "planned_total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "weight_decay": WEIGHT_DECAY,
        "lr_scheduler": lr_scheduler,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_loss,
        "perplexity": _perplexity(eval_loss),
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "wall_time_s": round(wall, 1),
        "eval_curve": curve,
        "train_curve": train_curve,
        "probe_split": probe_split if score_probe else None,
    }
    return record, (probe_cb.scores() if probe_cb else None)


# --- Results I/O + resume --------------------------------------------------

def load_completed(results_path: Path) -> dict[str, tuple]:
    """run_id -> the (epochs, subset_size) budget it was trained at.

    The budget is returned, not discarded, because `run_id` does **not** encode
    it: a `reduced` cell and a `full` cell with the same grammar/size/lr/bs/seed
    share a run_id. Resuming on the id alone would silently accept a 10k-subset
    result into a full-budget sweep and rank recipes across two different
    budgets — the exact confound this sweep exists to remove. `check_budget`
    turns that into a loud failure.
    """
    if not results_path.exists():
        return {}
    done: dict[str, tuple] = {}
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done[rec["run_id"]] = (rec.get("epochs"),
                                       rec.get("subset_requested"))
    return done


def check_budget(completed: dict[str, tuple], specs: list[RunSpec],
                 epochs: int, subset_requested: int | None,
                 results_path: Path) -> None:
    """Refuse to resume into results trained at a different budget.

    Compares the REQUESTED subset (`None` = full split), not the realized row
    count, so the check needs no corpus load and can therefore run before the
    pending list is computed. That ordering is the point: when every cell is
    already present at the wrong budget the pending list is empty, so a guard
    placed inside the training loop would never execute — which is exactly the
    case it exists to catch.
    """
    want = (epochs, subset_requested)
    mismatched = [
        (s.run_id, completed[s.run_id]) for s in specs
        if s.run_id in completed and completed[s.run_id] != want
    ]
    if mismatched:
        run_id, (had_epochs, had_subset) = mismatched[0]
        raise SystemExit(
            f"\nERROR: {results_path} already holds {len(mismatched)} run(s) "
            f"trained at a DIFFERENT budget — e.g. {run_id} at "
            f"epochs={had_epochs}, subset={had_subset}, but this invocation is "
            f"epochs={epochs}, subset={subset_requested}.\n"
            f"run_id does not encode the budget, so resuming would mix them and "
            f"rank recipes across two budgets.\n"
            f"Use a fresh --output-dir for this budget."
        )


def append_result(results_path: Path, record: dict) -> None:
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- Summary ---------------------------------------------------------------

def _sd(values: list[float]) -> float:
    """Population SD; 0.0 for a single value."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def _recipe_rows(records: list[dict]) -> list[dict]:
    """Recipe aggregates over whatever runs are passed, best mean loss first.

    Callers pass either every record (the headline table) or one grammar's
    records (the per-grammar table), so the two views are computed by the same
    code and cannot drift apart.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["size_name"], r["learning_rate"], r["batch_size"])
        groups.setdefault(key, []).append(r)

    rows = []
    for (size_name, lr, bs), runs in groups.items():
        losses = [r["eval_loss"] for r in runs]
        mean_loss = sum(losses) / len(losses)
        rows.append({
            "size_name": size_name,
            "learning_rate": lr,
            "batch_size": bs,
            # Which grammars are pooled into this row. The pooling is deliberate
            # (recipe selection wants it), but a row that does not say what it
            # pooled cannot be told apart from a single-grammar row.
            "grammars": "+".join(sorted({r["grammar"] for r in runs})),
            "n_runs": len(runs),
            "n_seeds": len({r["seed"] for r in runs}),
            "mean_eval_loss": mean_loss,
            "sd_eval_loss": _sd(losses),
            "min_eval_loss": min(losses),
            "max_eval_loss": max(losses),
            "mean_perplexity": _perplexity(mean_loss),
        })
    rows.sort(key=lambda x: x["mean_eval_loss"])
    return rows


def _recipe_key(row: dict) -> tuple:
    return (row["size_name"], row["learning_rate"], row["batch_size"])


def _recipe_label(row: dict) -> str:
    return f"{row['size_name']}/lr{row['learning_rate']:.0e}/bs{row['batch_size']}"


def summarize(results_path: Path, output_dir: Path,
              total_expected: int | None = None) -> None:
    """Aggregate per recipe (size x lr x batch) PER GRAMMAR, over seeds.

    Writes the full per-run table and a per-grammar recipe table to CSV, and
    prints the per-grammar rankings.

    **The pooled table is deliberately absent.** This function used to lead with
    recipes ranked by validation loss averaged over the four grammars, and to
    write it to `summary.csv`. That number is not interpretable: H, P, H' and L'
    are different corpora with different true entropies (which is what
    `analysis/entropy_baseline.py` measures), so averaging their losses ranks
    recipes partly by which grammars are intrinsically easier, and lets a recipe
    win the average while being the worst available choice on one grammar. It
    was removed rather than caveated — a footnoted wrong number still gets read
    off the top of the table.

    Recipe SELECTION does not happen here. It is
    `analysis/recipe_selection.py`: admissibility by convergence on all four
    grammars, then a ranking on the worst grammar's normalized loss.
    """
    records = read_results(results_path)
    if not records:
        print("No results yet.")
        return

    # Full per-run table. The learning curves are nested lists, which do not
    # belong in a CSV — they stay in results.jsonl, which is the source of truth.
    nested = ("eval_curve", "train_curve")
    full_csv = output_dir / "results.csv"
    fields = [k for k in records[0].keys() if k not in nested]
    with open(full_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # Aggregate by recipe WITHIN each grammar. Seeds are pooled (they are
    # replicates of the same cell); grammars are not (they are different
    # corpora — see the docstring).
    grammars = sorted({r["grammar"] for r in records})
    by_grammar: dict[str, list[dict]] = {
        g: _recipe_rows([r for r in records if r["grammar"] == g])
        for g in grammars
    }
    fieldnames = list(next(iter(by_grammar.values()))[0].keys())

    per_grammar_csv = output_dir / "summary_by_grammar.csv"
    with open(per_grammar_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["grammar", "rank", *fieldnames])
        writer.writeheader()
        for g in grammars:
            for rank, row in enumerate(by_grammar[g], start=1):
                writer.writerow({"grammar": g, "rank": rank, **row})

    # Default to the FULL grid only when the caller does not say what it
    # configured; otherwise a reduced sweep would report progress against
    # 96 runs it was never going to do.
    if total_expected is None:
        total_expected = len(all_run_specs())
    print(f"\n{len(records)}/{total_expected} runs complete. "
          f"Recipes ranked WITHIN each grammar (top 3 each).")
    print("Losses are not comparable across grammars, so there is no pooled "
          "column and no cross-grammar mean.\n")
    sub = (f"{'grammar':8s} {'size':7s} {'lr':>7s} {'bs':>4s} {'runs':>5s} "
           f"{'mean_loss':>10s} {'sd':>7s}")
    print(sub)
    print("-" * len(sub))
    for g in grammars:
        for i, row in enumerate(by_grammar[g][:3]):
            print(f"{g if i == 0 else '':8s} {row['size_name']:7s} "
                  f"{row['learning_rate']:>7.0e} {row['batch_size']:>4d} "
                  f"{row['n_runs']:>5d} {row['mean_eval_loss']:>10.4f} "
                  f"{row['sd_eval_loss']:>7.4f}")

    # Whether the grammars even agree on a winner is a fact about the grid worth
    # surfacing — but it is reported as disagreement, not resolved by averaging.
    winners = {g: _recipe_key(by_grammar[g][0]) for g in grammars}
    if len(set(winners.values())) > 1:
        print("\nNOTE: the grammars do not agree on a best recipe "
              f"({'; '.join(f'{g}: ' + _recipe_label(by_grammar[g][0]) for g in grammars)}). "
              "Selecting between them is `analysis/recipe_selection.py`'s job — "
              "admissibility by convergence, then a ranking on the worst "
              "grammar. Do not resolve it by averaging the losses.")
    elif len(grammars) > 1:
        print(f"\nEvery grammar's best recipe is the same one: "
              f"{_recipe_label(by_grammar[grammars[0]][0])}. That is agreement, "
              "not a selection — admissibility still has to be checked.")
    print("\nSelect a recipe with:\n"
          f"  python -m analysis.recipe_selection {results_path}")

    print(f"\nFull table: {full_csv}\nPer grammar: {per_grammar_csv}")


# --- CLI -------------------------------------------------------------------

def resolve_output_dir(value: Path) -> Path:
    """A bare name lands under `results/`; a path is taken as given.

    `--output-dir pilot_search` -> `<repo>/results/pilot_search`, so local sweeps
    cannot land beside the repo root by accident. Anything with a separator or a
    root — `/abs/path/pilot`, `results/x` — is used
    unchanged, since that caller has already said where it wants the files.
    """
    return value if (value.is_absolute() or len(value.parts) > 1) else RESULTS_ROOT / value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=("full", "reduced"), default="full",
                        help="'reduced' narrows the sweep to one grammar per "
                             "FAMILY (H, P), one seed, and a 10k subset — see the "
                             "module docstring for the rationale. Explicit axis "
                             "flags below override the preset.")
    parser.add_argument("--grammars", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--sizes", nargs="+", choices=list(MODEL_SIZES), default=None,
                        help="model sizes to sweep (default: all three)")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=None)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                        help="'cuda' errors if unavailable rather than falling back")
    parser.add_argument("--fp16", action="store_true", help="mixed precision (GPU)")
    parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision (GPU)")
    parser.add_argument("--subset-size", type=int, default=None,
                        help="subsample the train split; omit for the full "
                             "split (the production budget)")
    parser.add_argument("--probe-split", default=SELECT_SPLIT,
                        help="probe scored at each epoch end. Do NOT point this "
                             "at a reporting probe or at OOD — see "
                             "docs/RECIPE_SELECTION.md")
    parser.add_argument("--no-probe", action="store_true",
                        help="skip probe scoring (loss-only sweep). Also "
                             "disables early stopping, which reads the probe.")
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                        help="stop after this many epochs with no improvement "
                             "in the probe score on the STOP half of the pairs; "
                             "0 trains every run to --epochs")
    parser.add_argument("--lr-scheduler", default=DEFAULT_LR_SCHEDULER,
                        help="'constant_with_warmup' keeps the stopping epoch "
                             "independent of the cap; 'cosine' does not — see "
                             "the module docstring before changing it")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                        help="epoch CAP; early stopping normally ends a run "
                             "first (see --patience)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap steps per run (overrides --epochs); useful for smoke tests")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="a bare NAME lands in results/NAME; a path (or an "
                             "absolute path) is "
                             f"used as given. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--expect-train-n", type=int, default=EXPECTED_TRAIN_N,
                        help="refuse to start unless every grammar's train split "
                             "has this many rows (0 disables). Guards against a "
                             "wrong corpora location, which is otherwise "
                             "invisible until the sweep has finished.")
    parser.add_argument("--limit", type=int, default=None,
                        help="run at most N pending configurations, then stop")
    parser.add_argument("--report-to", default="none",
                        help='e.g. "wandb"; default "none" keeps the run self-contained')
    parser.add_argument("--progress", action="store_true",
                        help="show the per-run training progress bar")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the pending runs and exit without training")
    parser.add_argument("--summary", action="store_true",
                        help="(re)compute and print the summary from existing results, then exit")
    args = parser.parse_args()

    # Preset supplies defaults; any explicit axis flag wins over it.
    preset = PRESETS[args.preset]
    grammars = tuple(args.grammars if args.grammars is not None else preset["grammars"])
    seeds = tuple(args.seeds if args.seeds is not None else preset["seeds"])
    sizes = tuple(args.sizes if args.sizes is not None else preset["sizes"])
    lrs = tuple(args.learning_rates if args.learning_rates is not None
                else preset["learning_rates"])
    bss = tuple(args.batch_sizes if args.batch_sizes is not None
                else preset["batch_sizes"])
    subset_size = (args.subset_size if args.subset_size is not None
                   else preset["subset_size"])

    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME
    scores_dir = output_dir / SCORES_DIRNAME
    manifest_path = output_dir / MANIFEST_FILENAME

    # Build the spec list BEFORE the --summary early exit so both paths report
    # progress against the grid that was actually configured.
    specs = all_run_specs(grammars, seeds, sizes, lrs, bss)

    if args.summary:
        summarize(results_path, output_dir, total_expected=len(specs))
        return

    device = resolve_device(args.device)
    completed = load_completed(results_path)
    # Before `pending`: when every cell is already present at another budget the
    # pending list is empty and nothing downstream would ever run.
    check_budget(completed, specs, args.epochs, subset_size, results_path)
    pending = [s for s in specs if s.run_id not in completed]

    # What is actually on disk, before anything is trained on it.
    # Before anything reads a row: the corpus must be the SET that was asked
    # for, not merely a corpus that happens to sit where one was expected.
    check_corpus_set_binding()
    inventory = corpus_inventory(grammars, args.probe_split)
    check_corpora(inventory, subset_size, args.expect_train_n)

    print(f"Preset: {args.preset}  |  device: {device}"
          f"{'  fp16' if args.fp16 else ''}{'  bf16' if args.bf16 else ''}")
    print(f"Output:  {output_dir}")
    print(f"Corpora: {CORPORA_ROOT}")
    for grammar, counts in inventory.items():
        print("  " + f"{grammar:8s} " + "  ".join(
            f"{split.replace('structural_probe_', 'probe:')}={n}"
            for split, n in counts.items()))
    print(f"Grid: {len(sizes)} sizes x {len(lrs)} LRs x {len(bss)} batch sizes = "
          f"{len(sizes) * len(lrs) * len(bss)} configs")
    print(f"Grammars: {list(grammars)}  seeds: {list(seeds)}  "
          f"subset: {subset_size if subset_size is not None else 'FULL'}  "
          f"epoch cap: {args.epochs}  schedule: {args.lr_scheduler}")
    print(f"Probe: {'none' if args.no_probe else args.probe_split} "
          f"(scored at every epoch end)")
    stops = (not args.no_probe) and args.patience > 0
    print(f"Early stopping: {f'patience {args.patience} on the STOP half' if stops else 'off'}")
    print(f"Total runs: {len(specs)}  |  completed: {len(completed)}  |  pending: {len(pending)}")

    if args.dry_run:
        for s in pending:
            print("  pending:", s.run_id)
        return

    if args.limit is not None:
        pending = pending[:args.limit]
        print(f"--limit: running {len(pending)} of the pending runs this invocation")

    # Resolved warmup, per pending run. Computed from the inventory rather than
    # from a loaded Dataset so the manifest can still be written before any
    # corpus is tokenized; `run_one` recomputes it from `len(train_ds)` and the
    # two agree because the subset is a fixed-size slice of the counted split.
    warmup_steps_by_run = {
        s.run_id: resolve_warmup_steps(
            WARMUP_RATIO,
            planned_total_steps(
                subset_size if subset_size is not None
                else inventory[s.grammar]["train"],
                s.batch_size, args.epochs, args.max_steps),
        )
        for s in pending
    }

    # The manifest is written BEFORE training, so an interrupted sweep still
    # says what it was asked to do and on which corpora. One line per
    # invocation, appended: a resumed sweep records both.
    write_manifest(manifest_path, {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "preset": args.preset,
        "grammars": list(grammars),
        "seeds": list(seeds),
        "sizes": list(sizes),
        "learning_rates": list(lrs),
        "batch_sizes": list(bss),
        "subset_requested": subset_size,
        "epoch_cap": args.epochs,
        "max_steps": args.max_steps,
        "patience": None if args.no_probe else args.patience,
        "lr_scheduler": args.lr_scheduler,
        "probe_split": None if args.no_probe else args.probe_split,
        "device": device,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "corpora_root": str(CORPORA_ROOT),
        "corpus_set": corpus_set_name(),
        # WHICH MACHINE ran this. When code and data live on different
        # filesystems, a sweep's code and its data can come from different
        # machines — and without this, nothing in the output says where either
        # came from.
        "hostname": socket.gethostname(),
        "corpus_rows": inventory,
        "corpus_sha256": corpus_digests(grammars, args.probe_split),
        "warmup_ratio": WARMUP_RATIO,
        # Resolved per run: the grid's batch sizes give the cells different step
        # counts, so one number would not describe the invocation.
        "warmup_steps": warmup_steps_by_run,
        "run_ids_this_invocation": [s.run_id for s in pending],
        "n_total": len(specs),
        "n_completed_before": len(completed),
        "git": git_sha(),
    })
    print(f"Manifest: {manifest_path}")

    # Group pending runs by grammar so each corpus is prepared once.
    by_grammar: dict[str, list[RunSpec]] = {}
    for s in pending:
        by_grammar.setdefault(s.grammar, []).append(s)

    done = 0
    for grammar, grammar_specs in by_grammar.items():
        print(f"\n=== {grammar}: preparing data "
              f"(subset {subset_size if subset_size is not None else 'FULL'}) ===")
        train_ds, eval_ds = prepare_datasets(grammar, subset_size)
        for spec in grammar_specs:
            done += 1
            print(f"[{done}/{len(pending)}] {spec.run_id} ...", flush=True)
            record, scores = run_one(
                spec, train_ds, eval_ds,
                epochs=args.epochs,
                max_steps=args.max_steps,
                eval_batch_size=args.eval_batch_size,
                report_to=args.report_to,
                disable_tqdm=not args.progress,
                device=device,
                fp16=args.fp16,
                bf16=args.bf16,
                score_probe=not args.no_probe,
                probe_split=args.probe_split,
                patience=args.patience,
                lr_scheduler=args.lr_scheduler,
            )
            # The REQUESTED budget (None = full split), which is what
            # `check_budget` compares against — `subset_size` in the record is
            # the realized row count and cannot distinguish "full" from an int
            # that happens to equal it.
            record["subset_requested"] = subset_size
            # Which corpora this cell actually read, recorded per run rather than
            # only in the manifest: results.csv travels on its own.
            record["corpora_root"] = str(CORPORA_ROOT)
            record["train_split_n"] = inventory[grammar]["train"]
            # Scores first: a crash between the two would otherwise mark the run
            # complete (resume keys on results.jsonl) with its scores missing.
            if scores is not None:
                write_scores(scores_dir, spec.run_id, scores)
            append_result(results_path, record)
            print(f"    eval_loss={record['eval_loss']:.4f} "
                  f"ppl={record['perplexity']:.2f} "
                  f"stopped@{record['stopped_epoch']}"
                  f"{' (CAP)' if record['hit_cap'] else ''} "
                  f"({record['wall_time_s']}s)")

    if done == 0:
        print("\nNothing to do; all requested runs already complete.")
    summarize(results_path, output_dir, total_expected=len(specs))


if __name__ == "__main__":
    main()
