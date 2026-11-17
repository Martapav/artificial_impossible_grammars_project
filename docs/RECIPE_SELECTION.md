# Recipe selection — the sweep, the objective, the frozen recipe

How the training recipe is chosen and why: the logic of the sweep, the selection
objective the code implements, the constraints that protect the experiment, and the
frozen result. Every claim points to the function or constant that implements it.

**Frozen recipe:** medium GPT-2 (4 layers / 256 dims / 8 heads), lr 3e-4, batch 64,
`constant_with_warmup` with warmup pinned to 4,689 steps, **12 epochs for every run,
no early stopping**. Every model is reported at its final (epoch-12) checkpoint. The
recipe was chosen inside the seed-noise band by a throughput convention (Step 5.4),
not as a demonstrated best. The concrete values and their comments live in
`models/train.py` (`FROZEN_RECIPE`, `DEFAULT_*`).

The code, end to end:
`models/grid_search.py` (the sweep) → `evaluation/probe_callback.py` (per-epoch
scoring + the stopping rule) → per-run score shards → `evaluation/aggregate.py`
(where the objective `S[r]` is built) → `analysis/recipe_selection.py` (what selects).
`analysis/reports/training/analysis_grid_search.py` produces descriptive tables and
figures and selects nothing. Related:
[`SURPRISAL_REFERENCE_POINTS.md`](SURPRISAL_REFERENCE_POINTS.md) (the `R` reporting
metric).

---

## The sweep

`models/grid_search.py` sweeps a small coarse grid and records, per run, everything
needed to choose a recipe *afterwards*. It does not itself select.

**The grid.** Three model sizes centred on `models.model`'s default point
(`MODEL_SIZES`: small 2/128/4, medium 4/256/8, large 6/256/8) × two learning rates
(`LEARNING_RATES` = 3e-4, 1e-3) × two batch sizes (`BATCH_SIZES` = 32, 64) = **12
configurations**, run over the four grammars (`GRAMMARS` = H, P, Hprime, Lprime) and
two seeds → **96 runs** under `--preset full`. Warmup ratio (0.05) and weight decay
(0.01) are held fixed as low-sensitivity axes. `--preset reduced` narrows to one
grammar per family (H, P), one seed and a 10k train subset (24 runs, ~12× less
compute); runs are keyed by `run_id`, so a later `full` sweep reuses every `reduced`
result, and `load_completed` / `check_budget` refuse to *mix* budgets, since a
`run_id` does not encode its budget. A recipe is the tuple
`(size_name, learning_rate, batch_size)`; seeds are replicates within a cell.

**Production budget, checked before training.** Each run trains on the FULL train
split: `check_corpora` refuses to start unless every grammar's train split has
`EXPECTED_TRAIN_N = 100_000` rows, and `check_corpus_set_binding` refuses a corpus
that is not the `THESIS_CORPUS_SET` that was asked for. Selecting a recipe at a
fraction of the production budget and then training at the full budget is exactly
the confound the sweep exists to remove.

**Trained to a plateau, not a fixed budget.** `--epochs` (default 60) is a *cap*; a
run normally ends earlier when its probe score stops improving (`--patience`,
default 5). `hit_cap` records whether a run reached the cap (a run at the cap has
*not* demonstrated a plateau); `stopped_epoch` records where it ended. Two things make
the plateau meaningful:

- **The schedule is `constant_with_warmup`, not cosine** (`DEFAULT_LR_SCHEDULER`).
  Cosine anneals over `num_train_epochs`, so the learning rate, and therefore the
  plateau, would be a function of the cap, and a cosine run flattens near its end
  merely because the rate is going to zero. Under a constant rate the rate at step
  *t* does not depend on the total, so "no improvement for `--patience` epochs" means
  the same thing in a run that stops at 22 and one that stops at 47.
- **Stopping and scoring read disjoint halves of the probe pairs.** Stopping reads
  the STOP half, selection reads the SCORE half (`evaluation.aggregate.pair_half`,
  split deterministically on `pair_id` with blake2b). Sharing them would let every
  run stop at its own best moment on the very items it is later judged by, and the
  maximum of a noisy curve is biased upward, by an amount that grows with training
  length, so it does not cancel across runs.

**Scored two ways every epoch.** `eval_strategy="epoch"` logs `eval_loss` on the full
validation split into `eval_curve`; `ProbeScoringCallback` scores the minimal-pair
probe. Loss is kept as an adequacy *gate*; the probe is what *selects* (below).

**Per-item scores, sharded per run.** Probe scores are written per ITEM
(`write_scores` → `probe_scores/<run_id>.parquet`, validated by `validate_scores` and
stamped with corpus provenance). `S[r]` z-scores each term *across* the recipes being
compared, so it cannot be computed inside a single run; keeping the raw rows also
means the objective can be re-scoped or re-weighted without retraining.

**Outputs.** `results.jsonl` (source of truth, carries the curves), `results.csv`
(flattened), `summary_by_grammar.csv` (recipes ranked WITHIN each grammar), the score
shards, and `run_manifest.jsonl` (one line per invocation: grid, corpora + row counts +
SHA-256 digests, probe split, device, host, git SHA). `summarize()` deliberately emits
**no** pooled cross-grammar loss table (§1.4).

---

## Why not eval loss

### §1.3 — eval loss weights the phenomena at ~1%

Eval loss is a mean over every next-token prediction in the validation split. On H
that is 237,530 loss targets; the transformation-critical position is exactly one
token span on the 3,000 phenomenon items:

```
3,000 critical positions / 237,530 loss targets  =  1.26%
```

The split is 70% neutral by design, and even inside a phenomenon item only one token
is where the rule bites. So **98.7% of eval loss is lexical choice, inflection and
length**, the part identical across the four grammars by construction. A recipe a
full nat better at *every* critical position would move eval loss by 0.0126 nats,
smaller than the gap between rank 1 and rank 9 of the grid. On grid96 the two
objectives are close to inverted: the loss-ranked winner is 9th of 12 on the probe.

### §1.4 — eval loss also carries a cross-grammar entropy confound

The four grammars are four DIFFERENT corpora with different true entropies
(`analysis/entropy_baseline.py` measures how different), so a grammar whose corpus is
intrinsically higher-entropy contributes a larger loss no matter how well it was
learned. **Averaging loss over grammars ranks recipes partly by which grammars happen
to be easy**, and a recipe can win the pooled average while being the worst choice on
one grammar. This is why `grid_search.summarize()` refuses to write a pooled table.
The per-grammar normalization on the (demoted) loss side
(`analysis/recipe_selection._norm_ratio` and friends) fixes *this* problem; it does
not fix §1.3.

### §1.7 — what should select the recipe, and what must not

The primary learnability metric is grammaticality-judgment accuracy on minimal pairs.
Three constraints, each of which damages the experiment (not just the tuning) if
violated:

1. **Never select on OOD.** The OOD probe measures structural generalisation, which
   *is* the finding. OOD is opened once, at the end, on the frozen recipe.
2. **Never select on the probe set you report.** Argmax over recipes on the reported
   pairs biases that number upward. Selection uses a separate **SELECT** probe at a
   different seed (`+300_000`, `probe_callback.SELECT_SPLIT =
   "structural_probe_SELECT"`), kept out of `models/dataload.SPLITS`.
3. **Never select a per-grammar recipe.** One recipe for all four, chosen on a
   criterion pooled symmetrically over them. A per-grammar recipe would bake a
   grammar-specific optimisation advantage into the H-vs-P contrast the thesis rests
   on. This is why `S[r]` pools symmetrically (Step 3) and why the per-run table from
   `aggregate --select` is explicitly *not* the selection.

`R = D_KL(p‖q)/D_KL(p‖b)` is a **reporting** metric, not a selection metric: it is
diluted the same ~1% way as eval loss. See
[`SURPRISAL_REFERENCE_POINTS.md`](SURPRISAL_REFERENCE_POINTS.md).

---

## Step 2 — The SELECT probe

Three probes: `structural_probe_IOD` / `structural_probe_OOD` (reporting) and
`structural_probe_SELECT` (seed `+300_000`, selection only, never reported).

The binding violation set carries three modes (`BINDING_MODES` in
`corpusgen/structural_probes.py`): `wrong_item` (correct site, wrong pron subtype),
`wrong_site` (correct subtype, unlicensed site), and `no_substitution` (a licensed
configuration left bare: two coreferential full CAT1s where the grammar requires
substitution).

**`no_substitution` is refl-only, and the asymmetry is the reason.** The violation is
ungrammatical only where substitution is obligatory:

| | refl | pron |
|---|---|---|
| **P / P′** | obligatory (`has_licensed_duplicate`) | obligatory (same screen) |
| **H / H′** | obligatory (`has_nucleus_duplicate`) | **optional at the grammar level** |

A no-substitution item for H/H′ pron would label a *grammatical* string
ungrammatical; building it for P-pron but not H-pron would make the two families'
pron terms non-comparable. So the mode is withheld from pron in **both** families and
refl carries it symmetrically. (If H pron is ever forced, it becomes valid in all
eight grammar × subtype cells.)

## Step 3 — The selection objective

Implemented in `evaluation/aggregate.py`; all of the methodology below lives in that
one file. It was fixed before any model was scored.

**The criterion is assigned by the geometry of the violation, not by result.** Which
statistic is well posed depends on how the ungrammatical member was produced, and
that is a fixed, recorded property (`CRITERION_BY_VIOLATION`, keyed on the
`violation` field):

| `violation` | geometry | criterion | why |
|---|---|---|---|
| `unlicensed_site` (aux, wh_i, wh_ii) | relocation | whole-string | members are exact token permutations — a pure word-order judgment |
| `wrong_site` (refl, pron) | relocation | whole-string | span lands at a different index; a span comparison is not minimal |
| `wrong_item` (refl, pron) | in-place substitution | critical-span | strings index-aligned, equal length; the span is minimal and targeted |
| `no_substitution` (refl) | in-place substitution | critical-span | as above — one differing token, span on the divergence |
| `agreement` (neutral) | in-place substitution | critical-span | as above |
| `positional_*` (neutral) | local reorder | whole-string | permutation; either works |

Both statistics are computed on every item (`to_pairs` builds `margin_string` and
`margin_span`); the map only decides which one enters `S[r]`. Fixed before any model
was scored and keyed on a recorded field, this is a **pre-registration**. The map is
exact: `violation` predicts span alignment with zero mismatches across all four
grammars × both probe splits.

**Eight objective terms.** The term set is *derived from the data*
(`aggregate._term_names`: a subcase, or `subcase:violation` where a subcase spans two
geometries), not a hardcoded list, so a probe rebuild that changes the violation mix
cannot leave a stale term list behind. The current probe yields:

```
aux, wh_i, wh_ii                         (whole-string; permutation pairs)
refl:wrong_site, pron:wrong_site         (whole-string)
refl:wrong_item, pron:wrong_item         (critical-span)
refl:no_substitution                     (critical-span; refl-only, see Step 2)
```

`neutral_agreement` and `neutral_positional` are **gates, not terms**
(`GATE_TERMS`): a recipe that has not learned local agreement is not a candidate, but
being excellent at it earns nothing.

**Aggregation.**
```
z[r,g,t] = margin_corrected[r,g,t] standardized across the compared recipes, within (g,t)
S[r]     = mean over g of ( mean over the objective terms of z[r,g,t] )
W[r]     = min over (g,t) of z[r,g,t]                       # worst cell
Select argmax S[r], tie-break (within seed noise) by W[r].
```
- **Margin, not accuracy** for ranking (accuracy saturates and is noisier at
  n=15–100); accuracy is reported alongside as the headline.
- **Z-scored per term** (`selection_table` / `probe_objective`): whole-string margins
  are nats over ~24 tokens, span margins nats over one, so a raw mean would overweight
  the movement terms. A term whose spread across recipes is below `MIN_SPREAD`
  (1e-3 nats) carries no ranking information and contributes 0.
- **Pooled over grammars**: the pooled mean does not optimise the *difference*
  between grammars, so the H-vs-P contrast is not what is tuned. This is the main
  protection for the experiment (§1.7 constraint 3).
- **Gates**: any recipe below 0.90 accuracy on `neutral_agreement` (worst grammar),
  or whose validation loss does not beat the order-1 token baseline `b_tok` (~2.26
  nats/token), is disqualified. Both are floors, not quality bars.

**Frequency/length control (`evaluation/baseline_pairs.py` + `attach_baselines`).**
`margin_b1` (the same margin under the order-1 unigram reader) is **subtracted**:
`S[r]` ranks on `margin_corrected = margin − margin_b1`, which is zero exactly when
the model has captured nothing beyond lexical frequency and token count. That is the
correct null, and it controls token identity and length together. Subtraction (not
per-token normalization) is used because dividing a difference of sums by a length
rescales an *uncorrected* quantity rather than removing the confound. It is applied
**uniformly** across all terms because `margin_b1` is identically 0 on permutation
terms (a unigram is order-blind; `attach_baselines` *asserts* `|margin_b1| < 1e-6` on
`PERMUTATION_VIOLATIONS`), so the correction costs nothing there. For span terms the
members are index-aligned and equal length (asserted), so every non-critical token
cancels and the whole-string `margin_b1` *is* its span-restricted value.
`margin_over_b4 = margin − margin_ref_b4` (order-4 reference, under the term's own
criterion) is reported as **headroom above the local reader and never subtracted**:
order-1 is a nuisance the model should get no credit for, but order-4 is bounded
local structure the model is *supposed* to learn, so subtracting it would dock a
recipe for succeeding.

Two caveats that travel with the numbers:
- **`wrong_item` has zero lexical diversity** (`cat1pron` is exactly `m`/`mm`, with
  identical unigram counts in train), so `margin_b1` is *exactly* 0 there and
  `refl:wrong_item` / `pron:wrong_item` are exact mirrors. Read them as a pair (their
  mean); a blanket `mm` preference scores 100%/0%. `margin_over_b4` is what carries
  information on those two terms. This is *why* `no_substitution` exists: it supplies
  the missing negative and is the term that actually tests whether the binding
  **rule** was learned.
- **`refl:no_substitution` is locally detectable** (a bare repeated stem; large b1
  and b4 margins), so it separates recipes well but must **not** be read as a
  hierarchy result.

Guards in `attach_baselines` / `check_corpus_binding` refuse to proceed when a pair
lacks `margin_b1`, when a span pair is length-mismatched, when a permutation term has
a non-zero `margin_b1`, or when scores and baselines were built on different corpus
generations (the `pair_id` join key is positional and survives a rebuild).

## Step 5 — Select, freeze, set the budget

Selection is `analysis/recipe_selection.py` (`select_on_probe`), run as
`python -m analysis.recipe_selection`.

**Step 5.1 — rank on `S`, not loss.** `recipe_selection` reads the sweep's
`results.jsonl` *and* the terms table from `evaluation.aggregate --out` (`--terms` is
required; ranking on the raw margin lets a length-tracking recipe score on the
`wrong_site` terms). The ranking is `S[r]`; the two Step-3 gates
(`GATE_ACC_MIN = 0.90`, `B_TOK = 2.26`) are the admissibility filter (`step3_gates`).
`probe_objective` standardizes `z` across the **12 recipes** (seeds averaged first),
the recipe-level statistic the selection needs. This differs from
`aggregate.selection_table` / `aggregate --select`, which z-scores per **run**
(grammar × recipe × seed) and is therefore per-grammar; selecting on it would pick a
per-grammar recipe, so `aggregate --select` prints an explicit "this is NOT the
recipe selection" banner.

**Step 5.2 — read the learning curves.** On grid96, validation loss bottoms out
early and the probe keeps improving after it:

- **Dev loss bottoms at ~epoch 8** in every run (mode 8; 86/96 runs between epochs 6
  and 9). Past the minimum, train loss keeps falling while dev loss creeps up: mild,
  genuine overfitting, with a median rise of +0.015 nats (~1% of the ~1.76 floor).
  It is slightly stronger in the P family (median dev−train gap ~0.041 vs ~0.029 for
  H/H′), split along family, not along clause (i).
- **The large loss spikes are optimizer divergence, not overfitting.** Every severe
  run (rise > 0.2 nats, where train loss blows up too) has lr 1e-3 **and** batch 32,
  and within that cell divergence tracks capacity (large 8/8, medium 5/8, small 0/8).
  No run with lr 3e-4 or batch 64 diverges (max rise 0.06). The frozen recipe sits
  in the stable region.
- **The probe peaks later than the loss**: margins keep climbing a median of ~10
  epochs after validation loss bottoms out.

**Step 5.3 — freeze and record.** `--record` writes the selection and its evidence
to JSON (`selection_record`): the chosen recipe, the objective definition,
`S`/`W`/per-grammar `z`, the seed noise, the tied set, the gates, and the sweep's
provenance (git, host, corpus set, input SHA-256s) from `run_manifest.jsonl`.

**Step 5.4 — confirm the lead exceeds seed noise; otherwise a throughput
convention.** `seed_noise` rebuilds `S` one seed at a time and takes the median
per-recipe range; `select_on_probe` treats every admissible recipe within that band
of the best `S` as tied. On grid96 the top recipes form one cluster inside noise
(~0.081), so the choice falls to the secondary criterion `_secondary_key`: **smallest
model, then largest batch**, for throughput on the 20-run production training. A
recipe chosen inside the noise band is fine; one *reported as the best* is not.

**Step 5.5 — the production epoch budget.** One number for all 20 runs, with early
stopping off (`probe_callback` `patience=0`): if each grammar stopped at its own best
moment, the H-vs-P contrast would be confounded by stopping time.
- The budget is **12 epochs**. A longer budget (30 epochs, set from the probe curve)
  turned out to overfit the probe task: past the dev-loss minimum, margin (a
  confidence gap) keeps rising while the harder string-pooling accuracy decays. At a
  cap of 12, dev loss bottoms at ~epoch 7 and flattens without turning up, and the
  probe metrics are still improving at the cap, so the epoch-12 model is reported.
  It is "best probe, still climbing", not a converged point.
- **Warmup is pinned in steps** (4,689 = the grid's 3.0-epoch warmup at 100k
  sentences / batch 64), not as a ratio. A ratio is a fraction of total steps, so it
  would change the warmup length whenever the cap moves. Since the schedule never
  anneals, pinning the warmup makes a run to cap *E* reproduce the first *E* epochs
  of the grid-selected schedule exactly.
- Every epoch's probe scores are recorded (`probe_curve.parquet`), so the curve
  behind this choice is part of each run's output.

**Step 5.6 — open OOD only now**, once, on the frozen recipe.

The loss side of `recipe_selection.py` (`select`, `ConvergenceCriteria`, the
per-grammar normalizers, the leximin ranking) is kept but demoted: reachable with
`--rank-by-loss`, and its convergence check with `--convergence-gate`. The gate is
**off by default**: the sweep stops when the *probe* plateaus, ~10 epochs after
validation loss bottoms out, so at the stopping epoch the loss curve is rising *by
design*, and a curve-shape test rejects nearly every run for something that is not a
property of the recipe. Genuine divergence is caught by the `b_tok` gate. Runs with
no `eval_curve` are inadmissible (`no_eval_curve`).

---

## Descriptive analysis (states no conclusions)

`analysis/reports/training/analysis_grid_search.py` reads `results.csv` and the terms
table and emits numbered tables and figures (`report.txt`, `index.csv`). It selects
nothing and draws no conclusions. It reports every cross-run quantity at two epoch
views: `fixed` (the last epoch at which every run is still alive, so no recipe is
credited for training longer) and `final` (each run at its own stopping epoch). Its
clause-format contrasts encode the grammar layout as **one crossed factor plus
family, not a 2×2**: the binding/aux family `{H,Hprime}` vs `{Lprime,P}` is collinear
with the clause-(ii) format by construction, so only clause (i) is orthogonal to
family (`contrasts` → `clause_i`, `family`, `clause_i_x_family`; the eight non-wh
terms give an empirical placebo band).

---

## Reproduce

`$DATA` is any root with room for the sweep's outputs.

```
# 1. the sweep (96 runs; --preset reduced for the 24-run pilot)
python -m models.grid_search --preset full --device cuda --bf16 \
    --output-dir $DATA/grid96

# 2. build the terms table on the SCORE half, with the n-gram baselines joined
python -m evaluation.aggregate '$DATA/grid96/probe_scores/*.parquet' \
    --half score --baselines results/eval/<set>/baseline_pairs.parquet \
    --out results/eval/<set>/grid96_terms.csv

# 3. select the recipe (S[r]; --terms is required)
python -m analysis.recipe_selection $DATA/grid96/results.jsonl \
    --terms results/eval/<set>/grid96_terms.csv --record <out>/selection.json

# (optional) descriptive tables/figures
python -m analysis.reports.training.analysis_grid_search \
    --results $DATA/grid96/results.csv --terms results/eval/<set>/grid96_terms.csv \
    --outdir results/analysis/grid96/<set>
```

Outputs of different sweeps must **never** be aggregated in one
`evaluation.aggregate` call: `S[r]` z-scores across whatever runs the glob matched.
