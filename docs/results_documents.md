# Pipeline outputs — what training and evaluation write

A field guide to the files the pipeline produces, using the reported run as the
concrete example: **`set_3_ep12`**, 4 grammars (H, Hprime, Lprime, P) × 5 seeds
(0–4) = **20 models**, trained on corpus set `set_3` with the frozen recipe:

> medium GPT-2 (4 layers / n_embd 256 / 8 heads; 4,414,464 params; vocab 4,646),
> lr 3e-4, batch 64, `constant_with_warmup` with warmup pinned to 4,689 steps,
> weight decay 0.01, **12 epochs**, full 100k train split (18,756 optimizer steps).
> See `models/train.py` `FROZEN_RECIPE` and [RECIPE_SELECTION.md](RECIPE_SELECTION.md).

Every model is reported at its final (epoch-12) checkpoint. Output lives in two places:

- **`models/checkpoints/set_3_ep12/`**: the training bundle (`models/train.py`).
- **`results/eval/set_3_ep12/`**: the post-hoc evaluation of the final models (the
  `evaluation/` scripts). Training never touches the test split.

```
models/checkpoints/set_3_ep12/
├── <grammar>__s<seed>/          # 20 run dirs: H__s0 … P__s4
│   ├── final_model/             # config + weights + tokenizer
│   ├── run_metadata.json        # full provenance + recipe + metrics + status
│   ├── training_log.jsonl       # HF Trainer log history (per epoch)
│   ├── metrics.csv              # same, flattened for plotting
│   ├── eval_final.json          # final VALIDATION loss + perplexity
│   ├── test_eval.json           # held-out test number (evaluation/test_eval.py)
│   └── probe_curve.parquet      # per-item SELECT-probe surprisal at every epoch end
└── training_manifest.jsonl      # one summary line per completed run

results/eval/set_3_ep12/
├── test_indistribution_eval.csv / .jsonl        # held-out LM number, one row per model
├── test_indistribution_items/<run_id>.parquet   # per-sentence −log q (20 shards)
├── probe_scores/<run_id>.parquet                # per-item IOD+OOD probe surprisal (20 shards)
├── baseline_pairs.parquet                       # n-gram margins (frequency/length control)
└── terms.csv                                    # aggregated per-(run,grammar,split,term) table
```

## A. Training bundle — `models/checkpoints/set_3_ep12/`

Producer: **`models/train.py`** (`train_one`). Per run directory `<grammar>__s<seed>/`:

| File | What it contains |
|---|---|
| `final_model/` | The saved model (`save_model`: config + weights) and the tokenizer (`save_pretrained`). |
| `run_metadata.json` | Full provenance: `git` sha + dirty flag, `lexicon` fingerprint (`file_sha256` + the lexicon's own `content_hash`), `corpus` (`corpus_set`, `corpora_root`, per-split train/validation sha256), resolved `recipe` and `architecture`, library `versions`, `device`, `metrics`, a `probe_curve` descriptor, `wall_time_s`, and `status`. |
| `training_log.jsonl` | The HF Trainer `log_history` verbatim: per-epoch train rows (`loss`, `grad_norm`, `learning_rate`, `step`), per-epoch eval rows (`eval_loss`), and a final summary row (`total_flos`, `train_runtime`, `train_loss`). |
| `metrics.csv` | The same log history flattened to one CSV, for plotting. |
| `eval_final.json` | Final **validation** loss, `perplexity` and `train_loss`. This is validation, **not** the held-out test number (§C). |
| `probe_curve.parquet` | Per-item minimal-pair surprisal scored at **every epoch end** on the **SELECT** probe: the learning curve (`ProbeScoringCallback`, `patience=0`, so every run trains all 12 epochs). 4,200 probe items × 12 epochs, with `item_uid`, `pair_id`, `variant`, `violation`, `subcase`, `length`, `embedding_depth`, `neg_logq`, `neg_logq_crit` and related columns. |

Top-level **`training_manifest.jsonl`**: one summary line per completed run
(`run_id, grammar, seed, n_params, n_train, n_eval, epochs, planned_total_steps,
warmup_steps, train_loss, eval_loss, perplexity, train_runtime_s, wall_time_s` and the
probe-curve descriptor).

## B. Evaluation — `results/eval/set_3_ep12/`

Post-hoc scoring of the **final** trained models.

| File | Producer | What it contains |
|---|---|---|
| `test_indistribution_eval.csv` / `.jsonl` | `evaluation/test_eval.py` (`evaluate_all`) | One row per model: the held-out language-modelling number on `test_indistribution` (15,000 items; each model scored on **its own** grammar's test split, in fp32). Columns: `nats_per_token` = Σ−logq / Σtokens over the whole split, `perplexity` = exp of that, `mean_neg_logq_per_item`, `total_neg_logq`, `n_items`, `n_tokens`, plus provenance (`corpus_set`, `corpus_sha256`, `trained_on_set`, `git_sha`, `recipe_deviations`, `epochs`, `val_loss`). |
| `test_indistribution_items/<run_id>.parquet` | `evaluation/test_eval.py` (`--per-item`) | 15,000 rows per model: per-sentence whole-string −log q (`run_id, grammar, seed, split, id, item_uid, construction, length, n_tokens, neg_logq, neg_logq_per_token`). The model side of the differential-surprisal join Δ(x); join on `item_uid`, never `id`. |
| `probe_scores/<run_id>.parquet` | `evaluation/score_pairs.py` | 8,400 rows per model = 4,200 items × {`structural_probe_IOD`, `structural_probe_OOD`}, the **reported** probes. One row per item (pairing is `aggregate`'s job), including `neg_logq` (whole string) and `neg_logq_crit` (critical span). |
| `baseline_pairs.parquet` | `evaluation/baseline_pairs.py` | Model-independent n-gram margins per pair: `margin_b1` / `margin_b4` and their `_span` variants. `margin_b1` (the pure lexical-frequency reader) is the frequency **and** length control that `aggregate` subtracts to form `margin_corrected`. Depends only on the corpora. |
| `terms.csv` | `evaluation/aggregate.py` | The analysis-ready table: one row per `(run, epoch, grammar, split, term, criterion)`. Pairs the scored items, assigns the whole-string vs critical-span criterion by violation geometry, joins the baselines, and builds `margin`, `margin_corrected`, `margin_over_b4`, `acc`, `n_pairs`, `len_matched`, etc. Filter on the `split` column. |

**Pipeline flow.** `train.py` produces 20 models → on the final models,
`test_eval.py` (held-out LM), `score_pairs.py` (IOD/OOD probes) and
`baseline_pairs.py` (control) run → `aggregate.py` combines the probe scores and
baselines into `terms.csv` → the scripts under `analysis/reports/` turn these into
figures and tables under `results/analysis/`.

## C. Three things not to confuse

1. **Validation vs test.** `eval_final.json` (and the `val_loss` column in the test
   table) is **validation** loss, computed at every epoch as part of watching the run,
   and it is a mean of per-batch means. `test_indistribution_eval.*` is the **held-out
   test** number on unseen data, computed as exact corpus perplexity
   (`exp(Σ−logq / Σtokens)`) with the same scorer as the probes.
2. **`probe_curve` vs `probe_scores`.** `probe_curve.parquet` (in the checkpoint
   bundle) is the **SELECT** probe scored at **every epoch** during training: a
   learning curve, not a reported result. `probe_scores/*.parquet` is the **reported
   IOD + OOD** probes scored on the **final** model.
3. **Join on `item_uid`, never `id`.** `id` (`{grammar}-{split}-{index}`) is a line
   index into a split file and re-points on a corpus rebuild; `item_uid` is stable.

## D. Reproduce

With `THESIS_CORPUS_SET=set_3` and `$DATA` pointing at a large-storage root:

```bash
# 1. Train the 20 models
python -m models.train --device cuda --bf16 --output-dir $DATA/checkpoints/set_3_ep12

# 2. Held-out test LM number (+ per-sentence shards)
python -m evaluation.test_eval \
    --checkpoints $DATA/checkpoints/set_3_ep12 --split test_indistribution \
    --device cuda --per-item --out-dir $DATA/eval/set_3/ep12

# 3. IOD + OOD probe scores, one shard per model
for run in $DATA/checkpoints/set_3_ep12/*__s*; do
    rid=$(basename $run); g=${rid%%__*}
    python -m evaluation.score_pairs --model $run/final_model --grammar $g \
        --split structural_probe_IOD structural_probe_OOD --run-id $rid --device cuda \
        --out $DATA/eval/set_3/ep12/probe_scores/$rid.parquet
done

# 4. N-gram control (model-independent), then the aggregated term table
python -m evaluation.baseline_pairs --grammar all --fit-split train \
    --split structural_probe_IOD structural_probe_OOD \
    --out $DATA/eval/set_3/ep12/baseline_pairs.parquet
python -m evaluation.aggregate "$DATA/eval/set_3/ep12/probe_scores/*.parquet" \
    --half all --baselines $DATA/eval/set_3/ep12/baseline_pairs.parquet \
    --out $DATA/eval/set_3/ep12/terms.csv
```

Evaluation outputs must sit under a directory named after the corpus set (`set_3`):
the scripts refuse an output path that does not name its generation, because every
downstream join key is positional and would silently match another generation's
files. Copy (or symlink) the two output trees to `models/checkpoints/set_3_ep12/` and
`results/eval/set_3_ep12/`, where `analysis/report_common.py` reads them.
