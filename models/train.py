"""Full-split training of the artificial-grammar LMs (the models the probes and held-out evaluation score).

This is the *production* training run, distinct from `models.grid_search`:
the grid search picks a recipe and throws the weights away; this script takes
that fixed recipe and trains the models we keep. The recipe is medium (4/256/8),
lr 3e-4, batch 64, `constant_with_warmup` — the size/lr/batch axes frozen
2026-08-06 from the pilot sweep and reconfirmed 2026-08-08 on the full 96-run
grid (`results/eval/set_2/selected_recipe.json`). The schedule was corrected
2026-09-08: 12 epochs (was 30), with warmup pinned to the grid's 3.0-epoch
(4689-step) warmup so the cap is a pure truncation. Every model is reported at
its final (epoch-12) checkpoint — see the `DEFAULT_EPOCHS` / `FROZEN_RECIPE`
comments below for the why. The header comment on `FROZEN_RECIPE` records what
selected each value and what was traded away.
It trains one model per
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

Cross-grammar comparability: architecture and recipe are held FIXED across all grammars
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
        probe_curve.parquet     # per-item minimal-pair scores at EVERY epoch end
                                # (the surprisal learning curve; --no-probe-curve
                                #  skips it)
    training_manifest.jsonl     # one summary line per completed run

Examples:
    python -m models.train --dry-run                       # list the runs
    python -m models.train                                 # full run, all grammars x seeds
    python -m models.train --grammars H --seeds 0 \
        --subset-size 500 --max-steps 20 --epochs 1        # local smoke test
    python -m models.train --device cuda --fp16            # GPU run
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

from evaluation.probe_callback import SELECT_SPLIT, ProbeScoringCallback

from models.dataload import (
    corpus_path,
    load_corpus,
    resolve_corpora_root,
    tokenize_corpus,
)
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

# --- The frozen recipe -----------------------------------------------------
# FROZEN 2026-08-06 from the pilot sweep; RECONFIRMED 2026-08-08 on the full
# 96-run grid `results/grid96/set_2/` (12 recipes = 3 sizes x 2 lrs x 2 batch
# sizes, x 4 grammars x 2 seeds; seeds 100/101, disjoint from this script's 0-4).
# The grid varied lr and batch size, which the pilot had pinned, and selected
# the same three values. Changing one invalidates the comparability of every
# kept model, so change them only by re-running the selection.
#
# The machine-readable record of the 2026-08-08 selection — objective, gates,
# tied set, input hashes, sweep git SHA — is
# `results/eval/set_2/selected_recipe.json`, written by
# `python -m analysis.recipe_selection <sweep>/results.jsonl --terms <terms.csv>
#  --record <path>`.
#
# What selected them, and on what:
#   * Objective `S[r]` = `margin_corrected` (= margin - margin_b1) on the SELECT
#     probe, **scoring half only**, z-scored within (grammar, term) across the
#     compared recipes and pooled symmetrically over the four grammars. Term
#     table: `results/eval/set_2/grid96_terms.csv`; methodology:
#     docs/RECIPE_SELECTION.md Step 3/5.1.
#   * Admissibility is Step 3's two gates only — `neutral_agreement` accuracy
#     >= 0.90 on the worst grammar, and validation loss beating `b_tok` (2.26
#     nats/token). They reject exactly the two recipes that actually diverged
#     (medium and large at lr 1e-3 / bs 32, final loss 3.7 and 6.6). The B2
#     validation-curve check is NOT applied: this sweep stops when the probe
#     plateaus, a median of 10 epochs after validation loss bottoms out, so the
#     loss curve is rising at the stopping epoch by design.
#   * lr=3e-4: every 3e-4 recipe outranks every 1e-3 recipe on S[r], and the
#     only two divergent recipes are 1e-3. This was pinned in the pilot on loss
#     evidence and is now confirmed on the probe.
#   * Size and batch: the top four recipes (medium and large, both batch sizes,
#     all at lr 3e-4) score S = 0.731 / 0.688 / 0.661 / 0.636 against a seed
#     noise of 0.081, so they are A TIE INSIDE SEED NOISE. Step 5.4's
#     pre-registered tiebreak applies — smallest model, then largest batch —
#     giving **medium/bs64**. medium = `models.model`'s default architecture
#     point (4 layers / 256 / 8 heads, 4.41M params); no arch flags are needed,
#     `--n-layer/--n-embd/--n-head` defaults suffice.
#     This is a throughput convention, NOT a claim that medium/bs64 is the best
#     of those four. It is not distinguishable from them. What IS separated is
#     the gap down to `small`, which is ~6x the seed noise: all four `small`
#     recipes sit at S = 0.21-0.25.
#
# The two schedule choices are NOT independent of each other:
#   * `constant_with_warmup`, not cosine. Cosine anneals over `num_train_epochs`,
#     which makes the trajectory a function of the budget; a constant rate does
#     not, so the epoch cap becomes a pure truncation and the pilot/grid evidence
#     transfers to any cap. This is load-bearing for the warmup pin below.
#   * `warmup_steps` pinned to 4689 (= the grid's 3.0-epoch warmup), NOT a ratio.
#     `warmup_ratio` is a fraction of total steps, and total steps scale with the
#     epoch cap, so a ratio silently changes the warmup length when the cap moves
#     (cap 60 -> 3.0 ep, cap 30 -> 1.5 ep, cap 7 -> 0.35 ep). Because the schedule
#     never anneals, the cap otherwise reshapes the trajectory ONLY through warmup;
#     pinning warmup absolutely makes training to cap E reproduce the first E
#     epochs of the grid-selected schedule exactly. See `DEFAULT_WARMUP_STEPS`.
#   * `epochs = 12`, applied identically to every (grammar, seed) — ONE number for
#     all 20 runs (docs/RECIPE_SELECTION.md Step 5.5), so no grammar is read at its
#     own best moment (which would confound the H-vs-P contrast the thesis rests
#     on).
#
#     REVISED 2026-09-08 from 30 to 12, and the reporting rule reversed. The
#     30-epoch budget was set from the probe curve on the premise that the probe
#     keeps improving past the loss plateau and that training there is harmless.
#     The set_3 selection analysis (`analysis/selection_epoch.py`) showed it is
#     not: past the dev-loss minimum the model keeps sharpening confidence (margin,
#     a confidence gap, rises monotonically and never plateaus) while the harder
#     string-pooling ACCURACY decays — the 30-epoch models are overfit for the
#     probe task, and testing a model past the point it learned the task is not
#     defensible. Dev loss bottoms at ~ep7 in both the grid and the training runs.
#     The cap is now 12 (headroom above ~7 to see the minimum on the real
#     retrained curves). The 30-epoch run was additionally OFF the grid
#     schedule (its ratio gave a 1.5-ep warmup, not 3.0) — a second reason its
#     curves are not the reference.
#
#     The per-epoch probe curve is recorded (`probe_curve.parquet`).
#
# Report the final (epoch-12) model. On the 12-epoch retrain the overfitting is
# gone: dev loss bottoms at ~ep7 (range 5-9) and flattens rather than turning up
# (0/20 runs end > 0.02 nats above their own minimum), while probe margin and
# string-pooling accuracy are still improving at the cap. Epoch 12 is therefore
# the best probe checkpoint, still climbing — reported as the cap, not as a
# converged point.
DEFAULT_LEARNING_RATE = 3e-4
DEFAULT_BATCH_SIZE = 64
DEFAULT_EPOCHS = 12
DEFAULT_WARMUP_RATIO = 0.05
# Absolute warmup, pinned so the epoch cap is a pure truncation (see the schedule
# note above). 4689 = ceil(0.05 * 60 * ceil(100000/64)) = the grid's 3.0-epoch
# warmup at the 100k/64 production corpus. RECOMPUTE if the train split size or
# batch size changes: 3.0-epoch warmup is ceil(3 * ceil(n_train / batch_size)).
DEFAULT_WARMUP_STEPS = 4689
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_LR_SCHEDULER = "constant_with_warmup"
DEFAULT_EVAL_BATCH_SIZE = 64


# --- Warmup ----------------------------------------------------------------

def planned_total_steps(
    n_examples: int,
    batch_size: int,
    epochs: float,
    max_steps: int | None = None,
    *,
    gradient_accumulation_steps: int = 1,
) -> int:
    """Optimizer steps a run is SCHEDULED for, by Trainer's own arithmetic.

    Mirrors `Trainer._inner_training_loop`: a non-drop-last dataloader has
    ``ceil(n / batch_size)`` batches, which floor-divides by the accumulation
    factor into update steps per epoch (at least one), and the total is that
    times the epoch count, rounded up.

    "Scheduled", not "taken": early stopping ends most runs before the cap, and
    Trainer builds the LR schedule from the cap regardless. Resolving against the
    cap is what reproduces the schedule the runs have been getting.
    """
    if max_steps is not None and max_steps > 0:
        return max_steps
    batches_per_epoch = math.ceil(n_examples / batch_size)
    updates_per_epoch = max(batches_per_epoch // gradient_accumulation_steps, 1)
    return math.ceil(epochs * updates_per_epoch)


def resolve_warmup_steps(ratio: float, total_steps: int) -> int:
    """Absolute warmup steps from a ratio, by `TrainingArguments.get_warmup_steps`.

    Why this migration resolves the ratio instead of adopting a fixed absolute
    count: the sweep varies `batch_size` (32 vs 64) and `subset_size`, so its
    cells differ ~2x in total steps. A fixed count would hand those cells
    different warmup FRACTIONS, changing the recipe comparison itself — and the
    comparison is the entire purpose of the sweep. Resolving reproduces what
    `warmup_ratio` computed, cell by cell, to the step.

    `math.ceil` (not `round`) is Trainer's rule; matching it is what makes this a
    migration rather than a new setting.

    Note that `warmup_steps` is overloaded in transformers >= 5: a float below 1
    is re-read as a ratio. Always pass the resolved `int`.
    """
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"warmup ratio must be in [0, 1); got {ratio}")
    return math.ceil(total_steps * ratio)


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
    warmup_steps: int | None  # absolute pin; None -> derive from warmup_ratio
    weight_decay: float
    lr_scheduler: str
    subset_size: int | None  # None -> full train split (production default)


# The frozen recipe as a mapping, so a run can say out loud whether it is the
# selected recipe. Every kept model must be trained under the same
# one; a stray flag is otherwise invisible until the models are compared.
# Architecture is included because `--n-layer/--n-embd/--n-head` are per-run CLI
# options and cross-grammar comparability depends on their being identical everywhere.
FROZEN_RECIPE: dict[str, object] = {
    "n_layer": DEFAULT_N_LAYER,
    "n_embd": DEFAULT_N_EMBD,
    "n_head": DEFAULT_N_HEAD,
    "learning_rate": DEFAULT_LEARNING_RATE,
    "batch_size": DEFAULT_BATCH_SIZE,
    "epochs": DEFAULT_EPOCHS,
    "warmup_ratio": DEFAULT_WARMUP_RATIO,
    "warmup_steps": DEFAULT_WARMUP_STEPS,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "lr_scheduler": DEFAULT_LR_SCHEDULER,
    "subset_size": None,
}


def recipe_deviations(recipe: Recipe) -> dict[str, tuple]:
    """Fields where `recipe` departs from `FROZEN_RECIPE`, as {field: (frozen, actual)}.

    Empty for a production run. Non-empty is legitimate for smoke tests, which
    is why this reports rather than raises — but it is recorded in every run's
    metadata, so a model trained off-recipe can be identified after the fact.
    """
    actual = asdict(recipe)
    return {k: (v, actual[k]) for k, v in FROZEN_RECIPE.items() if actual[k] != v}


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


def corpus_fingerprint(grammar: str, root: Path | None = None) -> dict:
    """Which corpus bytes this run trained on: the set, and per-split sha256.

    Recorded for the same reason as the lexicon hash. Corpora are regenerated
    into a NEW set rather than over the old one, so the set name usually
    identifies them — but a name is a label and a hash is the data, and only the
    hash catches a set that was rebuilt with `--force` or copied from a
    different machine. Only the two splits training actually reads are hashed;
    the test and probe splits are hashed by whatever scores them.
    """
    root = root or resolve_corpora_root()
    files = {}
    for split in ("train", "validation"):
        path = corpus_path(grammar, split, root)
        files[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"corpus_set": root.name, "corpora_root": str(root), "files": files}


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


def _write_probe_curve(run_dir: Path, callback, probe_split: str | None) -> dict:
    """Persist the per-epoch minimal-pair scores as `probe_curve.parquet`.

    Validated with the same `validate_scores` the sweep's shards go through: the
    cross-epoch invariants (one row per item-epoch, the same item set every
    epoch) can only be checked once the run's frames are concatenated, and a
    curve that silently mis-pairs is worse than no curve.

    Returns a small descriptor for the run record — the file is per-item and
    per-epoch, so the summary should say what is in it without opening it.
    """
    if callback is None:
        return {"probe_split": None, "probe_curve_rows": 0, "probe_curve_epochs": 0}

    from evaluation.score_pairs import validate_scores

    frame = callback.scores()
    if frame is None or frame.empty:
        raise ValueError(
            f"{run_dir.name}: the probe callback produced no rows. Training "
            f"ran, so this is a scoring failure, not an empty run — and a "
            f"missing curve is exactly the failure that cost the 2026-08-04 "
            f"pilot its H shard."
        )
    validate_scores(frame, context=f"probe curve for {run_dir.name}")
    frame.to_parquet(run_dir / "probe_curve.parquet", index=False)
    return {
        "probe_split": probe_split,
        "probe_curve_rows": int(len(frame)),
        "probe_curve_epochs": int(frame["epoch"].nunique()),
    }


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
    probe_split: str | None = SELECT_SPLIT,
    probe_batch_size: int = 128,
) -> dict:
    """Train one model, save its checkpoint bundle, return a summary record.

    `probe_split=None` skips the per-epoch probe curve (smoke tests); anything
    else is scored at every epoch end with early stopping OFF — the curve is a
    record of this run, never a decision about when it ends.
    """
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

    # Warmup is passed to Trainer as an absolute `warmup_steps` count (the
    # `warmup_ratio` API is deprecated for removal in transformers v5.2). A recipe
    # may pin that count directly — the production recipe does, so the epoch cap is
    # a pure truncation of the grid-selected schedule rather than a rescaling of
    # its warmup (see the schedule note by `DEFAULT_WARMUP_STEPS`). When it is not
    # pinned, the ratio is resolved against the scheduled step count, unchanged.
    total_steps = planned_total_steps(
        len(train_ds), recipe.batch_size, recipe.epochs, recipe.max_steps)
    if recipe.warmup_steps is not None:
        warmup_steps = recipe.warmup_steps
    else:
        warmup_steps = resolve_warmup_steps(recipe.warmup_ratio, total_steps)
    if warmup_steps >= total_steps:
        raise ValueError(
            f"warmup_steps ({warmup_steps}) >= scheduled total steps "
            f"({total_steps}); warmup would consume the whole run and the LR would "
            f"never reach {recipe.learning_rate}. Lower the warmup or raise the "
            f"epoch cap.")

    # Eval + log per epoch so we get a validation learning curve for free.
    args = TrainingArguments(
        output_dir=str(run_dir),
        per_device_train_batch_size=recipe.batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=recipe.learning_rate,
        num_train_epochs=recipe.epochs,
        max_steps=recipe.max_steps if recipe.max_steps is not None else -1,
        warmup_steps=warmup_steps,
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
    # The learning curve validation loss cannot give: per-item minimal-pair
    # surprisal at every epoch end. `patience=0` is load-bearing — it disables
    # the sweep's early stopping, so every kept model sees the same number of
    # epochs and the curve records the run rather than truncating it.
    probe_cb = (
        ProbeScoringCallback(spec.grammar, spec.run_id, split=probe_split,
                             device=device, batch_size=probe_batch_size,
                             patience=0)
        if probe_split else None
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
    curve_info = _write_probe_curve(run_dir, probe_cb, probe_split)

    summary = {
        "run_id": spec.run_id,
        "grammar": spec.grammar,
        "seed": spec.seed,
        "n_params": total_params,
        "n_train": len(train_ds),
        "n_eval": len(eval_ds),
        "epochs": recipe.epochs,
        "max_steps": recipe.max_steps,
        # The resolved schedule, not just the ratio that produced it: the ratio
        # alone cannot be turned back into steps without knowing the corpus size
        # and batch size this run actually saw.
        "planned_total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "train_loss": train_result.training_loss,
        "eval_loss": eval_loss,
        "perplexity": _perplexity(eval_loss),
        "train_runtime_s": train_result.metrics.get("train_runtime"),
        "wall_time_s": round(wall, 1),
        **curve_info,
    }

    metadata = {
        "run_id": spec.run_id,
        "status": "complete",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "spec": asdict(spec),
        "recipe": asdict(recipe),
        # {} for a production run; a smoke test or a hand-overridden flag shows
        # up here as {field: [frozen, actual]} rather than being unrecoverable
        # from the weights later.
        "recipe_deviations": {k: list(v) for k, v in recipe_deviations(recipe).items()},
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
        "corpus": corpus_fingerprint(spec.grammar),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "metrics": eval_record,
        "probe_curve": curve_info,
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
        # 0 is the opt-out sentinel: fall back to the ratio (smoke tests / legacy).
        warmup_steps=(args.warmup_steps if args.warmup_steps else None),
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
    # Architecture (fixed across runs for cross-grammar comparability).
    parser.add_argument("--n-layer", type=int, default=DEFAULT_N_LAYER)
    parser.add_argument("--n-embd", type=int, default=DEFAULT_N_EMBD)
    parser.add_argument("--n-head", type=int, default=DEFAULT_N_HEAD)
    parser.add_argument("--n-positions", type=int, default=DEFAULT_N_POSITIONS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    # Recipe.
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--warmup-ratio", type=float, default=DEFAULT_WARMUP_RATIO,
                        help="used only when --warmup-steps is 0")
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS,
                        help="absolute warmup, overriding --warmup-ratio; pinned so "
                             "the epoch cap is a pure truncation (constant_with_warmup "
                             "does not anneal). Pass 0 to derive warmup from the ratio "
                             "instead (smoke tests / legacy).")
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
    # The per-epoch probe curve. On by default: it is the only record of WHEN
    # each phenomenon was learned, and it cannot be reconstructed later without
    # re-scoring every saved checkpoint.
    parser.add_argument("--probe-split", default=SELECT_SPLIT,
                        help="probe scored at every epoch end for the learning "
                             "curve (default: the SELECT probe, which has "
                             "already done its selection work — scoring a "
                             "REPORTED split 30x per run invites reading the "
                             "curve as a result)")
    parser.add_argument("--no-probe-curve", action="store_true",
                        help="skip the per-epoch probe scoring (smoke tests)")
    parser.add_argument("--probe-batch-size", type=int, default=128,
                        help="batch size for the per-epoch probe scoring")
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
    # The corpus set is a silent failure mode: the wrong one trains perfectly
    # well on the wrong data, so it is named up front and recorded per run.
    print(f"Corpora: {resolve_corpora_root()}")
    print(f"Architecture: n_layer={recipe.n_layer} n_embd={recipe.n_embd} "
          f"n_head={recipe.n_head} n_positions={recipe.n_positions}")
    warmup_desc = (f"{recipe.warmup_steps} steps (pinned)"
                   if recipe.warmup_steps is not None
                   else f"ratio {recipe.warmup_ratio}")
    print(f"Recipe: lr={recipe.learning_rate} bs={recipe.batch_size} "
          f"epochs={recipe.epochs} sched={recipe.lr_scheduler} "
          f"warmup={warmup_desc} wd={recipe.weight_decay} "
          f"train={'full' if recipe.subset_size is None else recipe.subset_size}")
    deviations = recipe_deviations(recipe)
    if deviations:
        print("Recipe: OFF the frozen recipe --")
        for field, (frozen, actual) in deviations.items():
            print(f"  {field}: frozen={frozen} -> {actual}")
        print("  (fine for a smoke test; NOT for models you intend to keep)")
    else:
        print("Recipe: frozen (grid96/set_2, reconfirmed 2026-08-08) — "
              "see models/train.py header")
    if args.no_probe_curve:
        print("Probe curve: OFF (no per-epoch minimal-pair scores will be kept)")
    else:
        print(f"Probe curve: {args.probe_split} scored at every epoch end "
              f"(early stopping OFF; every run trains {recipe.epochs} epochs)")
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
                probe_split=None if args.no_probe_curve else args.probe_split,
                probe_batch_size=args.probe_batch_size,
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
