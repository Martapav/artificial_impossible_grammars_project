"""Training-recipe grid search for the artificial-grammar LMs.

Sweeps a small, coarse grid to pick a training recipe (learning rate, batch
size, model size) before committing to the full 100k-sentence training runs.
Everything is centred on the default architecture point in `models.model`.

The grid (edit the constants below to change it):

    LEARNING_RATES  x  MODEL_SIZES  x  BATCH_SIZES   = 2 x 3 x 2 = 12 configs

run over `GRAMMARS` (4) and `SEEDS` (2) -> 96 runs by default. Warmup and
weight decay are held fixed as low-sensitivity axes. Each run trains on a
`--subset-size` (default 30k) sample of the grammar's train split and is scored
by loss / perplexity on the full validation split.

Pipeline integration:
  * vocabulary / tokenizer  -> models.tokenize
  * corpora + tokenization  -> models.dataload (load_corpus, tokenize_corpus)
  * model instantiation     -> models.model  (build_model, same defaults)

The run is resumable: each finished run is appended to `results.jsonl` in the
output directory and skipped on re-invocation, so an interrupted sweep can be
restarted (or chunked with `--limit`) without repeating work. No checkpoints
are kept (save_strategy="no"); recipe selection only needs the final metrics.

Examples:
    python -m models.grid_search --dry-run            # list the 96 runs
    python -m models.grid_search                      # run the full sweep
    python -m models.grid_search --limit 8            # run the next 8 pending
    python -m models.grid_search --summary            # (re)print the summary
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import tempfile
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from models.dataload import load_corpus, tokenize_corpus
from models.model import (
    DEFAULT_N_EMBD,
    DEFAULT_N_HEAD,
    DEFAULT_N_LAYER,
    build_model,
    count_parameters,
)
from models.tokenize import tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

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
LR_SCHEDULER = "cosine"

# The four base grammars.
GRAMMARS: tuple[str, ...] = ("H", "P", "Hprime", "Lprime")
SEEDS: tuple[int, ...] = (0, 1)

# Fixed subsample of the train split, shared across every run for comparability.
SUBSET_SEED = 12345

DEFAULT_SUBSET_SIZE = 30_000
DEFAULT_EPOCHS = 3
DEFAULT_EVAL_BATCH_SIZE = 64
DEFAULT_OUTPUT_DIR = REPO_ROOT / "grid_search"
RESULTS_FILENAME = "results.jsonl"


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
) -> list[RunSpec]:
    """The full grid, grouped by grammar so each corpus is loaded once."""
    specs = []
    for grammar in grammars:
        for size_name, lr, batch_size in product(MODEL_SIZES, LEARNING_RATES, BATCH_SIZES):
            for seed in seeds:
                specs.append(RunSpec(grammar, size_name, lr, batch_size, seed))
    return specs


# --- Data ------------------------------------------------------------------

def prepare_datasets(grammar: str, subset_size: int) -> tuple[Dataset, Dataset]:
    """Tokenized (train-subset, full-validation) for one grammar.

    The train split is shuffled with a fixed seed and truncated to
    `subset_size`, so every config/seed sees the identical training subset.
    Only these two splits are tokenized (the test splits are left untouched).
    """
    raw = load_corpus(grammar)
    n = len(raw["train"])
    if subset_size < n:
        train_raw = raw["train"].shuffle(seed=SUBSET_SEED).select(range(subset_size))
    else:
        train_raw = raw["train"]  # corpus smaller than the requested subset
    tokenized = tokenize_corpus(DatasetDict(train=train_raw, validation=raw["validation"]))
    return tokenized["train"], tokenized["validation"]


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
) -> dict:
    """Train one configuration and return a flat metrics record."""
    model = build_model(
        n_layer=spec.size["n_layer"],
        n_embd=spec.size["n_embd"],
        n_head=spec.size["n_head"],
        seed=spec.seed,
    )
    total_params, _ = count_parameters(model)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    with tempfile.TemporaryDirectory(prefix="gridrun_") as tmp:
        args = TrainingArguments(
            output_dir=tmp,
            per_device_train_batch_size=spec.batch_size,
            per_device_eval_batch_size=eval_batch_size,
            learning_rate=spec.learning_rate,
            num_train_epochs=epochs,
            max_steps=max_steps if max_steps is not None else -1,
            warmup_ratio=WARMUP_RATIO,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER,
            eval_strategy="no",     # single evaluate() at the end
            save_strategy="no",     # recipe selection needs metrics, not weights
            logging_strategy="epoch",
            seed=spec.seed,
            data_seed=spec.seed,
            report_to=report_to,
            disable_tqdm=disable_tqdm,
            use_cpu=True,           # GPU unavailable in this environment
            dataloader_num_workers=0,
        )
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            processing_class=tokenizer,
        )
        start = time.time()
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        wall = time.time() - start

    eval_loss = eval_metrics["eval_loss"]
    return {
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
        "max_steps": max_steps,
        "subset_size": len(train_ds),
        "warmup_ratio": WARMUP_RATIO,
        "weight_decay": WEIGHT_DECAY,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_loss,
        "perplexity": _perplexity(eval_loss),
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "wall_time_s": round(wall, 1),
    }


# --- Results I/O + resume --------------------------------------------------

def load_completed(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    done = set()
    with open(results_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                done.add(json.loads(line)["run_id"])
    return done


def append_result(results_path: Path, record: dict) -> None:
    with open(results_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def read_results(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with open(results_path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# --- Summary ---------------------------------------------------------------

def summarize(results_path: Path, output_dir: Path) -> None:
    """Aggregate per recipe (size x lr x batch) over grammars x seeds.

    Writes the full table and the aggregate to CSV and prints the recipes
    ranked by mean validation loss (the recipe-selection metric).
    """
    records = read_results(results_path)
    if not records:
        print("No results yet.")
        return

    # Full per-run table.
    full_csv = output_dir / "results.csv"
    fields = list(records[0].keys())
    with open(full_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    # Aggregate by recipe.
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
            "n_runs": len(runs),
            "mean_eval_loss": mean_loss,
            "min_eval_loss": min(losses),
            "max_eval_loss": max(losses),
            "mean_perplexity": _perplexity(mean_loss),
        })
    rows.sort(key=lambda x: x["mean_eval_loss"])

    agg_csv = output_dir / "summary.csv"
    with open(agg_csv, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    total_expected = len(all_run_specs())
    print(f"\n{len(records)}/{total_expected} runs complete. "
          f"Recipes ranked by mean validation loss:\n")
    header = f"{'size':7s} {'lr':>7s} {'bs':>4s} {'runs':>5s} {'mean_loss':>10s} {'mean_ppl':>9s}"
    print(header)
    print("-" * len(header))
    for row in rows:
        print(f"{row['size_name']:7s} {row['learning_rate']:>7.0e} {row['batch_size']:>4d} "
              f"{row['n_runs']:>5d} {row['mean_eval_loss']:>10.4f} {row['mean_perplexity']:>9.2f}")
    print(f"\nFull table: {full_csv}\nAggregate:  {agg_csv}")


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grammars", nargs="+", default=list(GRAMMARS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--subset-size", type=int, default=DEFAULT_SUBSET_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap steps per run (overrides --epochs); useful for smoke tests")
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / RESULTS_FILENAME

    if args.summary:
        summarize(results_path, output_dir)
        return

    specs = all_run_specs(tuple(args.grammars), tuple(args.seeds))
    completed = load_completed(results_path)
    pending = [s for s in specs if s.run_id not in completed]

    print(f"Grid: {len(MODEL_SIZES)} sizes x {len(LEARNING_RATES)} LRs x "
          f"{len(BATCH_SIZES)} batch sizes = "
          f"{len(MODEL_SIZES) * len(LEARNING_RATES) * len(BATCH_SIZES)} configs")
    print(f"Total runs: {len(specs)}  |  completed: {len(completed)}  |  pending: {len(pending)}")

    if args.dry_run:
        for s in pending:
            print("  pending:", s.run_id)
        return

    if args.limit is not None:
        pending = pending[:args.limit]
        print(f"--limit: running {len(pending)} of the pending runs this invocation")

    # Group pending runs by grammar so each corpus is prepared once.
    by_grammar: dict[str, list[RunSpec]] = {}
    for s in pending:
        by_grammar.setdefault(s.grammar, []).append(s)

    done = 0
    for grammar, grammar_specs in by_grammar.items():
        print(f"\n=== {grammar}: preparing data (subset {args.subset_size}) ===")
        train_ds, eval_ds = prepare_datasets(grammar, args.subset_size)
        for spec in grammar_specs:
            done += 1
            print(f"[{done}/{len(pending)}] {spec.run_id} ...", flush=True)
            record = run_one(
                spec, train_ds, eval_ds,
                epochs=args.epochs,
                max_steps=args.max_steps,
                eval_batch_size=args.eval_batch_size,
                report_to=args.report_to,
                disable_tqdm=not args.progress,
            )
            append_result(results_path, record)
            print(f"    eval_loss={record['eval_loss']:.4f} "
                  f"ppl={record['perplexity']:.2f} "
                  f"({record['wall_time_s']}s)")

    if done == 0:
        print("\nNothing to do; all requested runs already complete.")
    summarize(results_path, output_dir)


if __name__ == "__main__":
    main()
