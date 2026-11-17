# Ch2 · sub3 — the clause swap (H′ / P′): what each test does and why

Generator: `analysis/reports/ch2/report_ch2_hprime_pprime.py`.
Outputs: `results/analysis/evaluation/structural_probes/ch2/sub3_hprime_pprime/{figures,tables}`.
Inputs: `results/eval/set_3_ep12/probe_scores/*.parquet`, loaded through
`analysis.report_common.load_probe_pairs` → `evaluation.aggregate.to_pairs`.
Nothing in sub3 recomputes a margin; it only groups, averages and tests the pair table.

---

## 1. The design being tested

Each mixed grammar states wh-fronting with two clauses (`grammars/violations.py::wh_sites`):

| grammar | clause (i) = the swap axis | clause (ii) = inherited |
|---|---|---|
| H  | matrix object hosting an embedded clause (structural) — **native** | inside a Type3 under a Type4 (structural) |
| H′ | third CAT1-position (positional) — **foreign** | inside a Type3 under a Type4 (structural) |
| P  | third CAT1-position (positional) — **native** | second CAT1-position after the second CAT6 (positional) |
| P′ (`Lprime`) | structural licensing (`wh_structural`) — **foreign** | second CAT1-position after the second CAT6 (positional) |

So clause (ii) is family-native in all four grammars; clause (i) is native in H and P
and foreign in H′ and P′. In the probe these are the terms `wh_i` (incongruent) and
`wh_ii` (congruent).

The analysis is therefore **within grammar**:

    Δ(g) = acc(clause i) − acc(clause ii)

Each grammar is compared against its own congruent clause, never against another
grammar's base — this removes any cross-grammar difficulty confound from the
quantity of interest. The swap effect proper is the *between-grammar* comparison of
these within-grammar gaps (a difference-in-differences against the homogeneous parent).

## 2. The data and the unit of analysis

- Model set `set_3_ep12`, final-epoch checkpoints; 4 grammars × 5 seeds.
- Two probe splits: `structural_probe_IOD` (length window [2,25], the training range)
  and `structural_probe_OOD` ([26,48], longer-than-trained). They differ only in the
  length window and the RNG offset (`corpusgen/corpus_builder.py`).
- 300 minimal pairs per (grammar, clause, split, seed); 1500 per cell pooled over
  seeds; 24 000 rows total. Verified balanced — every cell is exactly 300.
- Pairs: both members are the same base sentence with the wh-word fronted from a
  *licensed* site (grammatical member) versus a site licensed by **neither** clause
  (ungrammatical member) — `corpusgen/structural_probes.py::_make_wh_pair`. Violation
  type `unlicensed_site`, criterion `string`, so members are exact token permutations
  of each other and the whole-string log-probability is the scored quantity.
- Metric: **accuracy**, `acc = 1[margin > 0]`, margin oriented so positive = the model
  prefers the grammatical member. Accuracy is used throughout deliberately, to avoid
  comparing nats across cells that differ in string content.
- **Sampling unit = the trained model (seed), n = 5.** Every interval and every p-value
  in sub3 is computed on the 5 per-seed cell means; nothing resamples items within a
  seed. An item-level bootstrap that pools the 5 seeds would understate
  model-to-model variation (pseudoreplication).

## 3. The tests

### T1 — Within-grammar clause accuracy, the two mixed grammars
`fig_primes` → `clause_congruence_primes.{png,csv}`

Grouped bars: accuracy on clause (i) vs clause (ii), for H′ and P′, in each split.
Point = mean of the 5 per-seed accuracies (equals the balanced pooled mean); interval =
small-sample t(4) CI over those 5 values (`_acc_across_seed`). The Δ for each grammar is
annotated above its pair of bars.

Logic: this is the descriptive layer — it shows whether the incongruent clause is
*absolutely* harder, before any reference grammar is brought in.

Result: it is not. In both splits and both mixed grammars, clause (i) is scored *higher*
than clause (ii) — e.g. IOD P′ .977 vs .897; OOD H′ .885 vs .785.

### T2 — The within-grammar gap Δ(g), with an across-seed interval
`fig_delta` → `clause_congruence_delta.{png,csv}`, and the formal test in
`report_seed_level` → `seed_within_grammar.csv`

Δ is computed **within seed first** (`_per_seed_delta`: mean acc on clause i minus mean
acc on clause ii, for that one model), giving 5 values per (grammar, split). Reported:
mean, SD, t(4) CI, one-sample t-test of H0: Δ = 0, and — because n = 5 — an **exact
two-sided sign-flip permutation p** over all 2⁵ = 32 relabelings (`_one_sample`).

Logic: the t-test assumes normality on 5 points; the sign-flip permutation makes no
distributional assumption and is exact under the null that a seed's gap is equally
likely to have either sign. Reporting both is a robustness statement, not a choice.

Result (Δ, [t CI], perm p):

| split | H′ | P′ | H (ref) | P (ref) |
|---|---|---|---|---|
| IOD | +0.023 [−0.003, +0.048], p=.067, perm .125 | +0.080 [+0.067, +0.093], p<.001, perm .0625 | +0.051 [+0.032, +0.069] | +0.081 [+0.067, +0.095] |
| OOD | +0.099 [+0.034, +0.165], p=.014, perm .0625 | +0.124 [+0.103, +0.145], p<.001, perm .0625 | +0.059 [+0.032, +0.085] | +0.071 [+0.037, +0.106] |

All 5/5 seeds share the sign of Δ in 7 of the 8 cells (IOD H′ is 4/5).

### T3 — The homogeneous no-swap reference
`fig_homogeneous` → `clause_congruence_homogeneous.{png,csv}`

The identical T1 plot for H and P, whose two clauses are *both* native. These grammars
have no swap, so their Δ is whatever intrinsic difficulty difference exists between the
clause-(i) and clause-(ii) site geometries.

Logic — this is the load-bearing control. Because H and P also show Δ > 0 (+0.051 and
+0.081 IOD), a positive Δ in a mixed grammar **cannot** be read as a congruence effect.
The raw Δ is confounded by the sites themselves; only its excess over the parent's Δ is
attributable to the swap. That is what T4 measures.

### T4 — Difference-in-differences: Δ(prime) − Δ(parent)
Two versions are written, and they do not agree.

**(a) Item-level bootstrap — `clause_congruence_did.csv`** (comparison only).
2000 resamples (rng seed 0) of the four pooled 1500-row accuracy vectors, seeds
pooled. Percentile 95% CI. This has the pseudoreplication problem: the same 300
items recur under every seed, so its interval describes the item sample, not the
population of trained models.

**(b) Seed-level paired test — `seed_did_interaction.csv`.** Δ is formed per seed for
prime and parent, the two are differenced **on the shared seed index** (same seed = same
initial weights, only the training corpus differs, so the contrast is paired), and the 5
differences get the same `_one_sample` treatment as T2 (t(4) + exact sign-flip perm).
This is the intended inference.

| split / contrast | DiD | seed-level CI | t p | perm p | bootstrap CI (a) |
|---|---|---|---|---|---|
| IOD H′ vs H | −0.028 | [−0.069, +0.013] | .130 | .188 | [−0.052, −0.003] |
| IOD P′ vs P | −0.001 | [−0.011, +0.009] | .729 | .875 | [−0.027, +0.025] |
| OOD H′ vs H | +0.041 | [−0.040, +0.122] | .235 | .313 | [+0.005, +0.077] |
| OOD P′ vs P | +0.053 | [+0.020, +0.085] | .011 | .0625 | [+0.015, +0.088] |

Logic of the disagreement: the item bootstrap calls three of four contrasts significant;
the seed-level test calls one. The three it loses are exactly the cells where the
seed-to-seed spread is large relative to the item-to-item spread (H′ OOD: per-seed Δ
ranges +0.017 to +0.153). The seed-level verdict is the one to report.

### T5 — Divergent-items-only robustness variant
Columns `delta_divergent_only`, `delta_div_lo/hi` in `clause_congruence_delta.csv`.

`grammars/metadata.py::licensing_rule_type` tags each clause-(i) item as
`clause_i_divergent` (clause (i) licenses it **and** H′ and P′ disagree about the string —
the discriminating items) or `clause_i_shared` (both grammars agree — a within-rule
control). The variant recomputes Δ using only the divergent subset of clause (i),
against the full clause (ii). Counts: H′ 2410 divergent / 590 shared; P′ 1970 / 1030
(IOD+OOD pooled). H and P are `None` here by design — only mixed grammars have a swap
axis (`RULE_TYPE_GRAMMARS`), which is why those cells are blank.

Logic: if the clause-(i) advantage were an artefact of the non-discriminating items
riding along, sharpening to the divergent subset would move Δ. It does not — e.g.
OOD P′ +0.124 → +0.138, OOD H′ +0.099 → +0.098, IOD H′ +0.023 → +0.019.

### T6 — Clause gap versus embedding depth
`fig_depth` → `incongruent_vs_congruent_depth.{png,csv}`

Within each mixed grammar, accuracy on the two clauses is broken out by
`embedding_depth_g` (the grammatical member's depth: max Type0 nesting − 1 for the
hierarchical family, n_embedded_clauses for the positional family — exact there, not an
approximation). Bins with fewer than 50 items (`FLOOR = 50`) are dropped.

Logic: depth is the obvious alternative explanation — if the incongruent clause simply
selected shallower sentences, the gap would be a depth artefact. It is not a level
effect: both clauses degrade with depth, and the gap persists at each depth (OOD H′ at
depth 3: .730 vs .687; P′ at depth 3: .843 vs .663). Note the depth support differs by
split (IOD reaches depth 2, OOD depth 3) and the deepest OOD bins are thin (95–115 items),
so this is read as a trend check, not a tested interaction.

### T7 — Per-seed reproduction and seed stability
`figures/by_seed/*__s{0..4}.png`, `tables/by_seed/*__s{0..4}.csv`, `seed_stability_acc.csv`

T1 and T3 are re-rendered for each seed alone (raw cell means, no error bars — a single
model has no across-seed spread to show), and `seed_stability_acc.csv` tabulates the
mean / SD / min / max / range of every bar over the 5 seeds.

Logic: between-seed variability is shown by reproduction rather than only by an interval,
so a reference seed can sit in the main text with the other four in the appendix, and a
reader can check that no single model drives a bar. Spread is small IOD (SD ≤ 0.015) and
larger OOD (H′ clause (i) SD 0.030, range 0.073; P clause (ii) range 0.083) — which is
precisely why the OOD DiDs lose significance under the seed-level test.

---

## 4. What the battery establishes, and its limits

Reading T1–T7 together: the incongruent (swapped) clause is **not** harder for the model
than the congruent one — it is consistently easier, in every grammar, both splits. But
so is clause (i) in the homogeneous grammars that never had a swap, so the gap tracks the
clause *geometry*, not family congruence. Once the parent grammar is differenced out
(T4b), only one of four DiD contrasts is distinguishable from zero at the model level
(OOD P′ vs P, +0.053), and the two IOD contrasts are essentially null. The robustness
variant (T5) and the depth breakdown (T6) close the two obvious alternative readings.

Limits that should travel with the numbers:

- **n = 5 models.** The exact sign-flip permutation p has a floor of 2/32 = **0.0625**,
  so no cell in this section can reach p < .05 by permutation; the perm column is a
  consistency check on the t-test, not an independent significance claim.
- **No multiple-comparison correction** is applied to the 8 within-grammar tests or the
  4 DiD tests.
- **Two DiD tables are written.** `seed_did_interaction.csv` (seed-level, paired) is
  the one to cite; `clause_congruence_did.csv` (item bootstrap) is written for
  comparison only, and citing its intervals would overstate three of the four
  contrasts.
- Splits differ in both length window and depth support, so IOD→OOD is a combined
  length-and-depth extrapolation, not a length-only one.
