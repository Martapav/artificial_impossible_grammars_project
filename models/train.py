"""Full-split training of the artificial-grammar LMs (the models DAS runs on).

This is the *production* training run, distinct from `models.grid_search`:
the grid search picks a recipe and throws the weights away; this script takes
that fixed recipe and trains the models we keep. It trains one model per
(grammar, seed), on the grammar's full train split, and writes a self-contained
checkpoint bundle per run.

Design points:
  * Device-parametrized (`--device auto|cpu|cuda`); errors loudly if `cuda`
    is requested but unavailable, rather than silently falling back to CPU.
  * Resumable at two levels. Run level: a finished run writes
    `run_metadata.json` with status "complete" and is skipped on re-invocation.
    Step level: intermediate checkpoints (save_strategy, default per-epoch) let
    an interrupted run resume from the last checkpoint (`get_last_checkpoint`).
  * Reproducible + auditable: every run records the git SHA (+ dirty flag), the
    lexicon fingerprint (sha256 + the lexicon's own content_hash), the resolved
    recipe/architecture, library versions, and the device it ran on.

DAS comparability: architecture and recipe are held FIXED across all grammars
and seeds (they are global CLI options, not per-run), so the trained models
differ only in their training grammar and init seed. Do not vary architecture
per grammar.

Output layout (per run, under --output-dir):
    <grammar>__s<seed>/
        final_model/            # config + weights + tokenizer (save_pretrained)
        checkpoint-*/           # intermediate checkpoints (for resume/dynamics)
        run_metadata.json       # git SHA, lexicon hash, recipe, versions, status
        training_log.jsonl      # full Trainer log history (loss/lr per epoch)
        metrics.csv             # same, flattened for quick plotting
        eval_final.json         # final validation loss + perplexity
    training_manifest.jsonl     # one summary line per completed run

Examples:
    python -m models.train --dry-run                       # list the runs
    python -m models.train                                 # full run, all grammars x seeds
    python -m models.train --grammars H --seeds 0 \
        --subset-size 500 --max-steps 20 --epochs 1        # local smoke test
    python -m models.train --device cuda --fp16            # server GPU run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from models.dataload import load_corpus, tokenize_corpus
from models.model import (
    DEFAULT_DROPOUT,
    DEFAULT_N_EMBD,
    DEFAULT_N_HEAD,
    DEFAULT_N_LAYER,
    DEFAULT_N_POSITIONS,
    build_model,
    count_parameters,
)
from models.tokenize import LEXICON_PATH, tokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

# The four grammars we keep trained models for.
GRAMMARS: tuple[str, ...] = ("H", "P", "Hprime", "Lprime")
# Five init seeds per grammar, mirroring Kallini et al. (2024, "Mission:
# Impossible Language Models"), who train 5 GPT-2 models per language. 4 x 5 = 20
# kept models. (grid_search stays at 2 seeds — it only selects a recipe.)
SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)

DEFAULT_OUTPUT_DIR = REPO_ROOT / "models" / "checkpoints"
MANIFEST_FILENAME = "training_manifest.jsonl"

# Recipe defaults. These mirror `models.model`'s architecture point and a
# conservative training recipe; OVERRIDE them from the grid-search winner
# before the real run.
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 10
DEFAULT_WARMUP_RATIO = 0.05
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LR_SCHEDULER = "cosine"
DEFAULT_EVAL_BATCH_SIZE = 64


# --- Run identity + recipe -------------------------------------------------

@dataclass(frozen=True)
class RunSpec:
    """One trained model: a grammar trained under one init seed."""

    grammar: str
    seed: int

    @property
    def run_id(self) -> str:
        return f"{self.grammar}__s{self.seed}"


@dataclass(frozen=True)
class Recipe:
    """The fixed training recipe + architecture, shared across all runs."""

    n_layer: int
    n_embd: int
    n_head: int
    n_positions: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    max_steps: int | None
    warmup_ratio: float
    weight_decay: float
    lr_scheduler: str
    subset_size: int | None  # None -> full train split (production default)


# --- Provenance ------------------------------------------------------------

def git_sha() -> dict:
    """Current commit and whether the working tree is dirty (best effort)."""
    def _run(args: list[str]) -> str | None:
        try:
            return subprocess.check_output(
                args, cwd=REPO_ROOT, stderr=subprocess.DEVNULL, text=True
            ).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None

    sha = _run(["git", "rev-parse", "HEAD"])
    status = _run(["git", "status", "--porcelain"])
    return {"sha": sha, "dirty": bool(status) if status is not None else None}


def lexicon_fingerprint(path: Path = LEXICON_PATH) -> dict:
    """sha256 of the lexicon file plus the lexicon's own recorded content_hash.

    A trained model is only meaningful with the exact lexicon/tokenizer that
    produced its token ids, so we pin both here.
    """
    raw = path.read_bytes()
    file_sha = hashlib.sha256(raw).hexdigest()
    try:
        content_hash = json.loads(raw)["metadata"].get("content_hash")
    except (json.JSONDecodeError, KeyError):
        content_hash = None
    return {"file_sha256": file_sha, "content_hash": content_hash}


def resolve_device(requested: str) -> str:
    """Map auto/cpu/cuda to a concrete device, failing loudly on a bad ask.

    Requesting `cuda` when it is unavailable is an error (we never want the
    long production run to silently drop to CPU); `auto` picks cuda if present.
    """
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: --device cuda requested but torch.cuda.is_available() "
                "is False (driver/toolkit mismatch?). Refusing to fall back to "
                "CPU for a production run."
            )
        return "cuda"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- Data ------------------------------------------------------------------

def prepare_datasets(grammar: str, subset_size: int | None) -> tuple[Dataset, Dataset]:
    """Tokenized (train, validation) for one grammar.

    Uses the FULL train split by default; `subset_size` (for smoke tests only)
    shuffles with a fixed seed and truncates. The test splits are never touched
    here — evaluation is a separate, post-hoc pass on the saved models.
    """
    raw = load_corpus(grammar)
    train_raw = raw["train"]
    if subset_size is not None and subset_size < len(train_raw):
        train_raw = train_raw.shuffle(seed=12345).select(range(subset_size))
    tokenized = tokenize_corpus(
        DatasetDict(train=train_raw, validation=raw["validation"])
    )
    return tokenized["train"], tokenized["validation"]


# --- One run ---------------------------------------------------------------

def _perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def _write_training_log(run_dir: Path, log_history: list[dict]) -> None:
    """Persist the Trainer log history as jsonl and a flat CSV for plotting."""
    (run_dir / "training_log.jsonl").write_text(
        "".join(json.dumps(entry) + "\n" for entry in log_history),
        encoding="utf-8",
    )
    if log_history:
        fields = sorted({k for entry in log_history for k in entry})
        with open(run_dir / "metrics.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(log_history)


def train_one(
    spec: RunSpec,
    recipe: Recipe,
    train_ds: Dataset,
    eval_ds: Dataset,
    *,
    run_dir: Path,
    device: str,
    eval_batch_size: int,
    save_strategy: str,
    save_total_limit: int | None,
    fp16: bool,
    bf16: bool,
    report_to: str,
    disable_tqdm: bool,
) -> dict:
    """Train one model, save its checkpoint bundle, return a summary record."""
    run_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(
        n_layer=recipe.n_layer,
        n_embd=recipe.n_embd,
        n_head=recipe.n_head,
        n_positions=recipe.n_positions,
        dropout=recipe.dropout,
        seed=spec.seed,
    )
    total_params, trainable_params = count_parameters(model)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Eval + log per epoch so we get a validation learning curve for free.
    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=recipe.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=recipe.learning_rate,
        num_train_epochs=recipe.epochs,
        max_steps=recipe.max_steps if recipe.max_steps is not None else -1,
        warmup_ratio=recipe.warmup_ratio,
        weight_decay=recipe.weight_decay,
        lr_scheduler_type=recipe.lr_scheduler,
        eval_strategy="epoch",
        save_strategy=save_strategy,
        save_total_limit=save_total_limit,
        logging_strategy="epoch",
        seed=spec.seed,
        data_seed=spec.seed,
        report_to=report_to,
        disable_tqdm=disable_tqdm,
        use_cpu=(device == "cpu"),
        fp16=fp16,
        bf16=bf16,
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

    # Step-level resume: pick up the last checkpoint in this run dir if any.
    last_ckpt = get_last_checkpoint(str(run_dir)) if save_strategy != "no" else None
    if last_ckpt is not None:
        print(f"    resuming from {Path(last_ckpt).name}")

    start = time.time()
    train_result = trainer.train(resume_from_checkpoint=last_ckpt)
    eval_metrics = trainer.evaluate()
    wall = time.time() - start

    # --- Save the checkpoint bundle ---------------------------------------
    final_dir = run_dir / "final_model"
    trainer.save_model(str(final_dir))          # config + weights
    tokenizer.save_pretrained(str(final_dir))   # exact tokenizer alongside
    _write_training_log(run_dir, trainer.state.log_history)

    eval_loss = eval_metrics["eval_loss"]
    eval_record = {
        "eval_loss": eval_loss,
        "perplexity": _perplexity(eval_loss),
        "train_loss": train_result.training_loss,
    }
    (run_dir / "eval_final.json").write_text(
        json.dumps(eval_record, indent=2), encoding="utf-8"
    )

    summary = {
        "run_id": spec.run_id,
        "grammar": spec.grammar,
        "seed": spec.seed,
        "n_params": total_params,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "epochs": recipe.epochs,
        "max_steps": recipe.max_steps,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_loss,
        "perplexity": _perplexity(eval_loss),
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "wall_time_s": round(wall, 1),
    }

    metadata = {
        "run_id": spec.run_id,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "recipe": asdict(recipe),
        "architecture": {
            "n_layer": recipe.n_layer,
            "n_embd": recipe.n_embd,
            "n_head": recipe.n_head,
            "n_positions": recipe.n_positions,
            "dropout": recipe.dropout,
            "vocab_size": len(tokenizer),
            "n_params_total": total_params,
            "n_params_trainable": trainable_params,
        },
        "device": device,
        "git": git_sha(),
        "lexicon": lexicon_fingerprint(),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metrics": eval_record,
        "wall_time_s": round(wall, 1),
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return summary


# --- Run-level resume + manifest -------------------------------------------

def is_complete(run_dir: Path) -> bool:
    """A run is done iff its metadata exists and is marked complete."""
    meta = run_dir / "run_metadata.json"
    if not meta.exists():
        return False
    try:
        return json.loads(meta.read_text())["status"] == "complete"
    except (json.JSONDecodeError, KeyError):
        return False


def append_manifest(manifest_path: Path, record: dict) -> None:
    with open(manifest_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


# --- CLI -------------------------------------------------------------------

def build_recipe(args: argparse.Namespace) -> Recipe:
    return Recipe(
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_positions=args.n_positions,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        epochs=args.epochs,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        lr_scheduler=args.lr_scheduler,
        subset_size=args.subset_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # What to run.
    parser.add_argument("--grammars", nargs="+", default=list(GRAMMARS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="root for checkpoint bundles; keep this on LOCAL disk")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    # Architecture (fixed across runs for DAS comparability).
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-positions", type=int, default=DEFAULT_N_POSITIONS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    # Recipe.
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--lr-scheduler", default=DEFAULT_LR_SCHEDULER)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    # Checkpointing.
    parser.add_argument("--save-strategy", choices=("epoch", "steps", "no"),
                        default="epoch",
                        help="intermediate checkpoints (needed for step-level resume)")
    parser.add_argument("--save-total-limit", type=int, default=None,
                        help="cap kept intermediate checkpoints (None = keep all)")
    # Smoke-test / performance knobs.
    parser.add_argument("--subset-size", type=int, default=None,
                        help="train on a fixed subset instead of the full split (smoke tests)")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="cap total steps (overrides --epochs); for smoke tests")
    parser.add_argument("--fp16", action="store_true", help="mixed precision (GPU)")
    parser.add_argument("--bf16", action="store_true", help="bf16 mixed precision (GPU)")
    # Misc.
    parser.add_argument("--report-to", default="none",
                        help='e.g. "wandb"; default "none" keeps the run self-contained')
    parser.add_argument("--progress", action="store_true",
                        help="show the per-run training progress bar")
    parser.add_argument("--force", action="store_true",
                        help="retrain runs already marked complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="list the runs and exit without training")
    args = parser.parse_args()

    device = resolve_device(args.device)
    recipe = build_recipe(args)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_FILENAME

    specs = [
        RunSpec(grammar, seed)
        for grammar in args.grammars
        for seed in args.seeds
    ]
    pending = [
        s for s in specs
        if args.force or not is_complete(output_dir / s.run_id)
    ]

    print(f"Device: {device}  |  output: {output_dir}")
    print(f"Architecture: n_layer={recipe.n_layer} n_embd={recipe.n_embd} "
          f"n_head={recipe.n_head} n_positions={recipe.n_positions}")
    print(f"Recipe: lr={recipe.learning_rate} bs={recipe.batch_size} "
          f"epochs={recipe.epochs} "
          f"train={'full' if recipe.subset_size is None else recipe.subset_size}")
    print(f"Runs: {len(specs)} total  |  pending: {len(pending)}")

    if args.dry_run:
        for s in pending:
            print("  pending:", s.run_id)
        return

    # Group by grammar so each corpus is prepared once.
    by_grammar: dict[str, list[RunSpec]] = {}
    for s in pending:
        by_grammar.setdefault(s.grammar, []).append(s)

    done = 0
    for grammar, grammar_specs in by_grammar.items():
        print(f"\n=== {grammar}: preparing data "
              f"({'full' if recipe.subset_size is None else recipe.subset_size}) ===")
        train_ds, eval_ds = prepare_datasets(grammar, recipe.subset_size)
        for spec in grammar_specs:
            done += 1
            print(f"[{done}/{len(pending)}] {spec.run_id} ...", flush=True)
            summary = train_one(
                spec, recipe, train_ds, eval_ds,
                run_dir=output_dir / spec.run_id,
                device=device,
                eval_batch_size=args.eval_batch_size,
                save_strategy=args.save_strategy,
                save_total_limit=args.save_total_limit,
                fp16=args.fp16,
                bf16=args.bf16,
                report_to=args.report_to,
                disable_tqdm=not args.progress,
            )
            append_manifest(manifest_path, summary)
            print(f"    eval_loss={summary['eval_loss']:.4f} "
                  f"ppl={summary['perplexity']:.2f} "
                  f"({summary['wall_time_s']}s)  -> {spec.run_id}/final_model")

    if done == 0:
        print("\nNothing to do; all requested runs already complete "
              "(use --force to retrain).")


if __name__ == "__main__":
    main()
