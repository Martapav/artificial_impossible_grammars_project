# Analysis

Everything that turns corpora, model scores and training curves into numbers,
tables and figures. Outputs go to `results/analysis/` (git-ignored); cached
intermediate tables go to `analysis/data/` (git-ignored).

```
analysis/
├── *.py            libraries + corpus-side pipelines (imported by the reports)
├── *.ipynb         corpus report and separability notebooks (+ their builders)
└── reports/        one script per figure/table set, run as python -m
    ├── corpus/     corpus descriptives, separability figures, probe examples
    ├── training/   training curves, reporting-epoch evidence, grid-sweep report
    ├── heldout/    held-out test: surprisal, Δ(x), R, permutation tests
    ├── probes/     structural-probe figures and margin heatmaps
    └── ch2/        evaluation chapter 2 (case-by-case, length/depth, H′/P′, TOST)
```

Methodology lives in `docs/`: [SURPRISAL_REFERENCE_POINTS.md](../docs/SURPRISAL_REFERENCE_POINTS.md)
(the ℓ(x) floor and the n-gram yardstick), [SEPARABILITY.md](../docs/SEPARABILITY.md)
(the separability audit), [RECIPE_SELECTION.md](../docs/RECIPE_SELECTION.md)
(recipe selection) and [results_documents.md](../docs/results_documents.md) (what each
pipeline output contains, and how to regenerate it).

## Corpus-side analyses

Two reports. The **corpus report** is driven by the generation-time metadata
**sidecar** (`corpora/<set>/<split>/<grammar>.meta.jsonl`, TRANSFORM_RULES §8 — no
replay); the **separability audit** replays natural generation to score licensing
on base order.

* **`corpus_analysis.ipynb`** — sentence-level descriptive statistics: lengths,
  length ↔ phenomenon / structural-complexity relationships, phenomenon
  occurrences and co-occurrences with background structure, expansion sites.
* **[`SURPRISAL_REFERENCE_POINTS.md`](../docs/SURPRISAL_REFERENCE_POINTS.md)** — the two reference points for surprisal
  evaluation: the grammar floor `ℓ(x) = −log p(x)` (absolute; "did the model learn
  the grammar") and the weak surface yardstick (relative; "did it get beyond local
  surface statistics"), plus the normalized score `R` that places the model between
  them. Read it before touching the entropy baseline.
* **`separability_analysis.ipynb`** — the pre-training shortcut audit: for each
  phenomenon, how much of the **licensing** decision can a weak, local,
  left-to-right, **bounded-memory** observer recover from surface categories?
  Metric **U** (normalized conditional-entropy reduction) under a decay-weighted
  (memory τ) left-only observer, read beside a per-family neutral determinism
  floor (a context reference with a CI — not a threshold), with a reproducible cue
  taxonomy (bounded-local / memory-graded / not-locally-recoverable) at a
  stipulated 0.6 cutoff. See [`SEPARABILITY.md`](../docs/SEPARABILITY.md).

## Libraries and pipelines (`analysis/*.py`)

| File | Purpose |
|------|---------|
| `extract_features.py` | Per-**sentence** feature table from the sidecar. Caches to `data/<grammar>_<split>_features.parquet`. |
| `stats.py` | Sentence-level descriptive-statistics tables and Matplotlib plots (project data-viz palette). |
| `build_notebook.py` | Regenerates `corpus_analysis.ipynb` from `stats.py`. |
| `corpus_analysis.ipynb` | Sentence-level report: tables + plots. |
| `lexicon_map.py` | Stem → category lookup shared by all four grammars — recovers a sentence's exact category sequence from the surface string alone (no tree/token access). |
| `separability_features.py` | Per-**candidate** base-order feature tables via replay (`build_candidate_frame`) + neutral probe-pair controls (`build_neutral_frame`): decay window, whole-side counts, agreement-match. Emits both band flags: `in_band` (shared precondition, cross-grammar comparable) and `in_rule_band` (the rule's own precondition — `OBJECT_DOMAIN_RULES`, read within a grammar only). |
| `entropy_baseline.py` | Grammar floor `−log p_corpus(x)`: applies the rejection-sampling correction to the sidecar's raw `ℓ(x)`. Rows are keyed on `item_uid`, never on the positional `id`. |
| `differential_surprisal.py` | Joins model scores to `ℓ(x)` on `item_uid` and computes `Δ(x) = neg_logq − ℓ_corpus`; raises on any unmatched score row. Corpus + in-distribution scope only — probe pairs report a within-pair difference instead and need no baseline (`SURPRISAL_REFERENCE_POINTS.md` §6.9). |
| `ngram_baseline.py` | Token-level n-gram baseline `b_tok` (interpolated Witten–Bell, order 4, held-out selected) — the axis-mate for the grammar floor `ℓ(x)`. See `SURPRISAL_REFERENCE_POINTS.md`. |
| `separability_stats.py` | Metric + taxonomy layer: feature specs, **U**/AUC per cell, `neutral_floor` (per **family**, bootstrapped CI), `classify_taxonomy`, and `run_full_analysis` (single entry point). |
| `build_separability_notebook.py` | Regenerates `separability_analysis.ipynb` from the parquet artifacts. |
| `separability_analysis.ipynb` | Separability report: U-vs-τ curves, taxonomy, controls. |
| `binding_credit.py` | Binding forced-draw credit: the per-(grammar, subtype) correction to `ℓ_corpus` without which Δ is spuriously negative on binding. |
| `recipe_selection.py` | Recipe selection on the grid sweep: ranks recipes on the minimal-pair objective `S[r]` behind two admissibility gates ([RECIPE_SELECTION.md](../docs/RECIPE_SELECTION.md) Step 5). Run as `python -m analysis.recipe_selection`. |
| `selection_epoch.py` | Epoch-budget evidence (dev-loss and probe plateaus) and the shared plateau helpers used by the ep12 training report. |
| `report_common.py` | Shared loaders and plot style for every script under `reports/` (paths of the reported `set_3_ep12` models and scores). |
| `category_labels.py` | Display names for lexical categories and inflection parameters (CAT1 → NOM, …), substituted into plots and tables; presentation only, data keys unchanged. |
| `data/` | Cached tables (git-ignored, regenerable): `*_train_features.parquet` (corpus report), `separability_{results,floor,taxonomy}.parquet` (audit). |
| `figures/` | PNG figures written by the corpus report's plot functions. |

## Reports (`analysis/reports/`)

Each script reads existing scores/corpora and writes figures plus a backing CSV
under `results/analysis/`; none of them recomputes a model score. Run from the
repo root, e.g. `python -m analysis.reports.ch2.report_ch2_tost`.

| Folder | Scripts |
|--------|---------|
| `corpus/` | `descriptive_stats` (lexicon + transformation knobs), `report_corpus_length`, `report_separability`, `export_probe_examples` |
| `training/` | `report_training`, `selection_epoch_ep12`, `report_select_control_vs_target`, `analysis_grid_search` (grid-sweep tables/figures) |
| `heldout/` | `report_test`, `report_test_permutation`, `report_per_sentence_surprisal`, `report_differential_surprisal` (run before `report_residual_surprisal` and `report_normalized_score`), `report_binding_credit_correction` |
| `probes/` | `report_probes`, `report_subphenomenon`, `report_probe_margin_heatmap`, `report_margin_correction_heatmap` |
| `ch2/` | `report_ch2_subphenomenon`, `report_ch2_length_depth`, `report_ch2_seed_variance`, `report_ch2_hprime_pprime`, `report_ch2_tost`, `report_ch2_tost_item`, `report_ch2_tost_subcase` |

## Run

```bash
# Sentence-level corpus report
python -m analysis.extract_features --grammar all --split train
jupyter nbconvert --to notebook --execute --inplace analysis/corpus_analysis.ipynb

# Separability audit (replays natural generation; writes the parquet artifacts)
python -c "from analysis.separability_stats import run_full_analysis; run_full_analysis()"
python analysis/build_separability_notebook.py
jupyter nbconvert --to notebook --execute --inplace analysis/separability_analysis.ipynb
```

## Methodology note — the metadata sidecar

The structural signals in the corpus report (embedding depth, clause count,
expansion sites, aux / gap / pronoun presence) are computed **inside** the
generators at emission and persisted line-aligned to each corpus file as
`corpora/<set>/<split>/<grammar>.meta.jsonl` (TRANSFORM_RULES §8, shared stable `id`).
`extract_features.py` reads that sidecar directly — no replay, no staleness. If
the sidecars are missing, regenerate the corpora with
`scripts/build_all_corpora.py`.

The separability audit deliberately does **not** use the sidecar: a binding
target's base-order neighbourhood is unrecoverable from the derived surface, and
the sidecar annotates only the active phenomenon per sentence. It instead replays
natural bases and evaluates the grammar's own licensing site machinery on base
order (see [`SEPARABILITY.md`](../docs/SEPARABILITY.md) §3).

Positional grammars (P, L′) have no constituent tree, so `embedding_depth` is
`NaN` there ([`TRANSFORM_RULES.md`](../docs/TRANSFORM_RULES.md) §8.4); their depth axis is `n_embedded_clauses`, which equals the
positional embedding depth exactly. Cross-family comparison uses `length`,
`n_clauses` and `expansion_sites`.
