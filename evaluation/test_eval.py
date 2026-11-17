"""Held-out evaluation of the kept models on `test_indistribution`.

The final language-modelling number for each of the 20 production models. It is
deliberately NOT the one `models/train.py` writes: that one is validation loss,
computed at every epoch and therefore part of how the run was watched, while
this is a single pass over data no model, recipe or epoch choice ever saw.

Why not `Trainer.evaluate` on the test split
--------------------------------------------
Two reasons, both about comparability with everything else in `evaluation/`.

* **Same scorer as the probes.** `score_pairs.score_sentences` is what produces
  every ``neg_logq`` in this project — same nats assertion, same BOS handling,
  same ``t``/``t+1`` offset. A test-set surprisal computed by a second code path
  would be a second convention, and the two are compared (a model's test-set
  surprisal and its probe margins are read together).
* **Exactness.** ``eval_loss`` is a mean of per-batch means, so with variable
  sentence lengths it weights short batches like long ones. Corpus perplexity
  here is ``exp(Σ −log q / Σ tokens)``, summed over the split — the quantity
  that is actually well defined, and reported alongside the mean per-sentence
  surprisal, which is a different (also useful) number rather than a rounding
  difference.

Scoring runs in fp32 whatever precision the model trained in: a held-out figure
should not depend on the eval dtype.

Per-item output, and the join key
---------------------------------
``--per-item`` writes one row per sentence, keyed on ``item_uid`` — the
generator-assigned identifier, stable across corpus rebuilds. ``id`` is carried
too but is a LINE index (``{grammar}-{split}-{n}``) and must never be joined on
(`analysis/differential_surprisal.py`). This table is the model side of A4:
``Δ(x) = neg_logq(x) − ℓ_corpus(x)``, which is why it exists in per-item form at
all rather than as a summary.

Grammar-matched by construction: the model trained on H is scored on H's test
split. Nothing here scores a model against another grammar's corpus.

Output
------
    <run_dir>/test_eval.json                 # per run, beside eval_final.json
    <out-dir>/<split>_eval.csv|.jsonl          # all runs, one row each
    <out-dir>/<split>_items/<run_id>.parquet   # --per-item, one shard per run

Usage
-----
    python -m evaluation.test_eval --checkpoints <checkpoints-dir> \\
        --device cuda --per-item --out-dir <eval-dir>

    python -m evaluation.test_eval --checkpoints <checkpoints-dir> \\
        --split validation            # cross-check against eval_final.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Sequence

from transformers import AutoModelForCausalLM

from models.dataload import CORPORA_ROOT as CORPORA_DIR
from models.train import resolve_device
from evaluation.score_pairs import score_sentences

DEFAULT_SPLIT = "test_indistribution"
DEFAULT_CHECKPOINTS = Path(__file__).resolve().parent.parent / "models" / "checkpoints"

# Item fields carried into the per-item table: exactly the ones the sentence
# `.jsonl` carries, so a stratification by construction or by length needs no
# second pass over the weights. The deeper stratifiers (embedding_depth,
# n_morphemes, family, phenomenon_family) live in the `.meta.jsonl` sidecar —
# join them on `item_uid` afterwards rather than emitting all-null columns here
# that read like measured absences.
_PASSTHROUGH = ("item_uid", "construction", "length")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_split_items(grammar: str, split: str = DEFAULT_SPLIT,
                     corpora_dir: Path = CORPORA_DIR) -> list[dict]:
    """Rows of `corpora/<split>/<grammar>.jsonl`, in file order.

    The `.meta.jsonl` sidecar is NOT read: the sentence file already carries
    ``item_uid``, and the sidecars are large (~1 GB), so requiring them here
    would make this module unrunnable on a GPU machine that holds only the
    weights and the sentence files.
    """
    path = corpora_dir / split / f"{grammar}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no {split} split for {grammar}: {path}")
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def discover_runs(checkpoints: Path) -> list[Path]:
    """Run directories with a saved model, as `<grammar>__s<seed>/final_model`."""
    return sorted(d for d in checkpoints.glob("*__s*")
                  if (d / "final_model").is_dir())


def parse_run_id(run_dir: Path) -> tuple[str, int]:
    grammar, _, seed = run_dir.name.partition("__s")
    return grammar, int(seed)


def score_run(run_dir: Path, *, split: str = DEFAULT_SPLIT, device: str = "cpu",
              batch_size: int = 64, corpora_dir: Path = CORPORA_DIR):
    """Score one saved model on its own grammar's split.

    Returns ``(summary, rows)``: the run-level record, and one dict per sentence.
    """
    grammar, seed = parse_run_id(run_dir)
    items = load_split_items(grammar, split, corpora_dir)
    model = AutoModelForCausalLM.from_pretrained(run_dir / "final_model")

    scored = score_sentences(
        model,
        [it["sentence"] for it in items],
        [[] for _ in items],          # no critical spans: whole-string −log q
        device=device, batch_size=batch_size,
    )

    rows = []
    for item, s in zip(items, scored):
        row = {"run_id": run_dir.name, "grammar": grammar, "seed": seed,
               "split": split, "id": item.get("id")}
        for key in _PASSTHROUGH:
            row[key] = item.get(key)
        row["n_tokens"] = s["n_target_tokens"]
        row["neg_logq"] = s["neg_logq"]
        row["neg_logq_per_token"] = (
            s["neg_logq"] / s["n_target_tokens"] if s["n_target_tokens"] else float("nan")
        )
        rows.append(row)

    total_nats = sum(r["neg_logq"] for r in rows)
    total_tokens = sum(r["n_tokens"] for r in rows)
    if not total_tokens:
        raise ValueError(f"{run_dir.name}: scored 0 tokens on {split}")
    nats_per_token = total_nats / total_tokens

    metadata = {}
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    summary = {
        "run_id": run_dir.name,
        "grammar": grammar,
        "seed": seed,
        "split": split,
        "n_items": len(rows),
        "n_tokens": total_tokens,
        # The corpus-level figure: one sum over the whole split, not a mean of
        # per-batch means. `perplexity` is its exponential and nothing else.
        "nats_per_token": nats_per_token,
        "perplexity": math.exp(nats_per_token),
        # Per-SENTENCE, which weights every item equally regardless of length.
        # Reported because the corpus figure is dominated by long sentences.
        "mean_neg_logq_per_item": total_nats / len(rows),
        "total_neg_logq": total_nats,
        # Provenance: which weights, produced by which code, on which bytes.
        # `corpus_set` names the generation; the sha256 is what actually pins
        # it, and `run_metadata.json`'s `corpus` block says which set the model
        # was TRAINED on — the two must agree or the test is cross-set.
        "corpus_set": corpora_dir.name,
        "corpus_sha256": sha256(corpora_dir / split / f"{grammar}.jsonl"),
        "trained_on_set": ((metadata.get("corpus") or {}).get("corpus_set")),
        "git_sha": (metadata.get("git") or {}).get("sha"),
        "lexicon_sha256": (metadata.get("lexicon") or {}).get("sha256"),
        "recipe_deviations": metadata.get("recipe_deviations"),
        "epochs": (metadata.get("recipe") or {}).get("epochs"),
        "val_loss": (metadata.get("metrics") or {}).get("eval_loss"),
    }
    if summary["trained_on_set"] and summary["trained_on_set"] != summary["corpus_set"]:
        print(f"  WARNING: {run_dir.name} trained on {summary['trained_on_set']} "
              f"but is being scored on {summary['corpus_set']}. Held out means "
              f"held out from THIS model's training corpus — set THESIS_CORPUS_SET.")
    return summary, rows


def evaluate_all(checkpoints: Path, *, split: str = DEFAULT_SPLIT,
                 device: str = "cpu", batch_size: int = 64,
                 grammars: Sequence[str] | None = None,
                 corpora_dir: Path = CORPORA_DIR,
                 out_dir: Path | None = None, per_item: bool = False) -> list[dict]:
    """Score every run under `checkpoints`; write the tables; return the summaries."""
    import pandas as pd

    runs = discover_runs(checkpoints)
    if grammars:
        runs = [r for r in runs if parse_run_id(r)[0] in set(grammars)]
    if not runs:
        raise FileNotFoundError(
            f"no <grammar>__s<seed>/final_model under {checkpoints}")

    summaries = []
    for run_dir in runs:
        summary, rows = score_run(run_dir, split=split, device=device,
                                  batch_size=batch_size, corpora_dir=corpora_dir)
        summaries.append(summary)
        (run_dir / "test_eval.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        print(f"{summary['run_id']:>16}  n={summary['n_items']:>6}  "
              f"nats/token={summary['nats_per_token']:.4f}  "
              f"ppl={summary['perplexity']:7.3f}  "
              f"per-item={summary['mean_neg_logq_per_item']:.3f}")

        if per_item and out_dir is not None:
            shard_dir = out_dir / f"{split}_items"
            shard_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows).to_parquet(
                shard_dir / f"{summary['run_id']}.parquet", index=False)

    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame(summaries)
        frame.to_csv(out_dir / f"{split}_eval.csv", index=False)
        (out_dir / f"{split}_eval.jsonl").write_text(
            "".join(json.dumps(s) + "\n" for s in summaries), encoding="utf-8")
        print(f"\nwrote {out_dir / f'{split}_eval.csv'}")
        print(frame.groupby("grammar")[["nats_per_token", "perplexity"]]
              .agg(["mean", "std"]).to_string())
        print("\nA loss is only comparable ACROSS grammars against each "
              "grammar's own entropy floor (analysis.entropy_baseline).")

    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS,
                        help="directory of <grammar>__s<seed>/ run dirs")
    parser.add_argument("--split", default=DEFAULT_SPLIT,
                        help="corpus split to score (default the test set)")
    parser.add_argument("--grammars", nargs="+", default=None,
                        help="restrict to these grammars (default: all found)")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="where the combined tables go (default: no tables)")
    parser.add_argument("--per-item", action="store_true",
                        help="also write one parquet of per-sentence −log q per "
                             "run (the model side of the A4 join)")
    args = parser.parse_args()

    if args.per_item and args.out_dir is None:
        parser.error("--per-item needs --out-dir to write the shards into")

    device = resolve_device(args.device)
    print(f"Device:  {device}")
    print(f"Corpora: {args.corpora_dir}")
    print(f"Split:   {args.split}\n")

    evaluate_all(args.checkpoints, split=args.split, device=device,
                 batch_size=args.batch_size, grammars=args.grammars,
                 corpora_dir=args.corpora_dir, out_dir=args.out_dir,
                 per_item=args.per_item)


if __name__ == "__main__":
    main()
