# Separability audit — methodology

A descriptive, **pre-training**, corpus-only analysis. For each phenomenon it
measures how well a **weak, local, surface-category-only** classifier can predict
whether a candidate token sits in a position **licensed by the phenomenon's rule**.

The audit runs on base order with a comparable band (below) because a naive
candidate-level co-occurrence measure confounds licensing with token category:
reflexives live in different linear neighbourhoods than nouns, so such a "cue"
merely re-derives the word-order template. The numbers come from
`analysis.separability_stats.run_full_analysis()`; the figures and tables from
`python -m analysis.reports.corpus.report_separability` (commands in
[`analysis/README.md`](../analysis/README.md)).

## 1. Goal — how to read a number

- **P / L′ (positional rules):** high local separability is *expected* — a sanity
  check, not a threat.
- **H / H′ (hierarchical rules):** high local separability is a *red flag* — the
  "structural" phenomenon has an accidental n-gram shortcut a model could exploit
  instead of learning hierarchy (`TRANSFORM_RULES.md` §1(b)).

Always report the **same phenomenon side-by-side across the family boundary**
(H vs P, H′ vs L′), annotated with the rule *type*, so a number is read as
expected/alarming correctly and H′ vs L′ never blend.

## 2. What "licensed / candidate" means

Uniform question: *can local category context predict whether a candidate is in a
licensing position?*

- **Candidate** = every eligible token, in **every sentence, all constructions**:
  all CAT1 for wh/binding, all CAT3AUX for aux.
- **Licensed** = the phenomenon rule's own site geometry, read from the probe
  builder's machinery (`violations.wh_sites` / `aux_sites` / `binding_sites`):
  - **wh_i / wh_ii** — per-clause flags (`licensed_i`, `licensed_ii`), which track
    the shipped predicates automatically. Current clauses: `TRANSFORM_RULES.md` §6.
  - **aux** — `licensed` (H-family: matrix aux; P-family: first-in-scan).
  - **refl / pron** — `licensed AND has_antecedent`. Binding **is coreference**: a
    site licenses binding only if there is an antecedent to corefer with. For
    H-pron **both** members of the coreference relation are licensed.

## 3. Two architectural decisions

1. **Base order, via replay — not the derived-string sidecar.** The sidecar stores
   the *derived* surface; a binding sentence's realized target is a `CAT1PRON` whose
   modifiers were stripped, so its base-order neighbourhood is gone, and featurising
   it would leak "pronoun-neighbourhood ⟺ licensed" — the exact confound. The
   sidecar also annotates only the **active** phenomenon per sentence. So we
   **replay** natural generation (`corpusgen.structural_probes._draw_base`, same
   rules/probabilities as training) and **never apply the transform** — we audit the
   *base* configuration.
2. **Consequences of base order.** Every candidate is a plain CAT1 / CAT3AUX with an
   intact neighbourhood; the licensed target is category-identical to distractors
   (centre excluded from features), so any predictive power is genuinely *context*.
   No gap-centering special case.

## 4. Regimes = length window on the base

Same draw machinery, same rules; **only the length window differs**:
- **`train`** — natural lengths [2, 25]; the descriptive baseline / IOD anchor.
- **`ood`** — [26, 48]; length-stability comparison.

## 5. Modules

| File | Role |
|---|---|
| `analysis/separability_features.py` | Data layer. `build_candidate_frame(grammar, regime, n_bases, seed, w)` → one row per (phenomenon, candidate): `is_licensed`, `center`, position-tagged category slots `L1..Lw`/`R1..Rw` (centre excluded, `PAD` at edges), **exact whole-side counts** `nL_<cat>`/`nR_<cat>`, and per-position agreement-match indicators `fmL*/fmR*`. Emits both band flags: `in_band` (shared precondition, cross-grammar comparable) and `in_rule_band` (`OBJECT_DOMAIN_RULES`, within-grammar). `build_neutral_frame` loads neutral probe-pair controls. |
| `analysis/separability_stats.py` | Metric + taxonomy layer. `cell_separability` fits one **feature spec** with grouped-by-base CV (unweighted — U needs true-distribution log-loss) → U + AUC + balanced acc + bootstrap CIs. `neutral_floor(family, ...)` fits the determinism reference once per family. `run_full_analysis()` is the single reproducible entry point → writes `analysis/data/separability_{results,floor,taxonomy}.parquet`. |
| `analysis/build_separability_notebook.py` | Regenerates the notebook from the parquet artifacts. |
| `analysis/separability_analysis.ipynb` | Thin presentation: per-(grammar, phenomenon) panels (H vs P, H′ vs L′ side by side); **never aggregates phenomena within a grammar**. |

## 6. Metric, representation, scope

- **Metric = U** = 1 − H(label|features)/H(label) (normalized conditional-entropy
  reduction, from the CV classifier's log-loss). Robust, fixed-scale. AUC is a
  companion — **read it alongside U above ~0.95**, where U charges for confidence
  (the L2 penalty caps how confident a perfect separator may be), so a perfectly
  separable cell scores ≈0.97 not 1.0 and U differences up there are not information
  differences.
- **Representation = decay-weighted position one-hots**, memory constant **τ**: each
  (offset k × category) one-hot scaled by `e^{−(k−1)/τ}`; τ→0 = immediate
  neighbours, τ→∞ = full window. Distance costs more via L2 — a graded memory, no
  hard cutoff. Companions: `window`, `count`, `winfm` (agreement channel).
- **Scope = LEFT-ONLY (canonical)** — matches the causal model + incremental
  parsing. Verified `left ≈ both` for every rule except a small `wh_i` residual
  (object vs subject: an object's verb precedes it, a subject's follows). Both-sided
  is kept only as that lookahead check.
- **τ-sweep {0.5, 1, 2, 4, ∞}**, headline τ=1 — the "how much memory" diagnostic.

## 7. Bands

- **comparable** (`is_licensed | in_band`) — trims the *unlicensed* pool to
  candidates satisfying the phenomenon's **shared** precondition (binding: a
  preceding antecedent; wh: not the first eligible nominal; aux: no band). This is
  what makes cells **cross-grammar comparable** — use it for cross-grammar claims.
- **rulewise** (`is_licensed | in_rule_band`) — controls for **the rule's own**
  precondition. `OBJECT_DOMAIN_RULES` = {H/H′ `refl`, H/L′ `wh_i`} band on
  objecthood; every other cell inherits `in_band`. **Not comparable across
  grammars** — read within a grammar. It is the population **probe pairs are drawn
  from**, so it is the band that matters for probe design.

## 8. Determinism floor

The same left-only decay estimator predicting the *next category* on neutral text ≈
the **family's** local determinism. Fit per family (H≡H′, P≡L′ share draw machinery
+ lexicon), n=2000 sentences, C=1.0, sentence-level bootstrap CI (±0.01). It is a
**reference line, not subtracted** (it scores a different target — next category
out of 8, H(Y)≈2.7 bits — than the cells, licensed/not, ≤1 bit). Near-flat across τ
(next-category is close to first-order Markov, so unlimited memory buys ≈+0.03).

**The 0.6 recovery cutoff is stipulated**, deliberately *not* derived from the floor.

## 9. Cue taxonomy (`classify_taxonomy`)

Per (grammar, phenomenon), from U at `decay@τ:left`, comparable, train:

- **bounded-local** — U(τ=0.5) ≥ 0.6: a fixed-offset / immediate-neighbour cue.
- **memory-graded** — reaches ≥0.6 only at some larger τ (`tau_recover` = the memory
  cost; the P-family counting rules). For a counting rule this is the *expected*
  outcome, not a defect: an observer with enough left memory essentially *is* the
  rule. What the class asserts is that the cue is **not free** at small τ.
- **not-locally-recoverable** — U < 0.6 even at τ=∞: **counting** (third-position),
  **depth / whole-clause scan** (`wh_ii` Type3-under-Type4, `refl` embedded
  nucleus), and **existential** (H `pron` cross-nucleus) rules.

`tau_recover` distinguishes `inf` (reached only at unbounded memory) from `NaN`
(never reached).

**Abstention (`thin`).** A cell whose minority class has < 30 members yields no
verdict (the label carries `n_pos`/`n_neg`/`thin`). U≈0 in a thin cell means "not
measurable at this sample size", not "no shortcut exists". `aux` is thin in all four
grammars (an aux occurs in ~30% of natural bases, two in <2%) — aux selectivity is
tested by the minimal-pair probes, which construct that contrast deliberately.

## 10. Controls and the key reframing

**Neutral probe-pair controls** (positive control for the method):
`neutral_positional` ≈ 0.90 from categories (a relocated token is locally visible);
`neutral_agreement` ≈ 0.50 from categories (invisible — only the inflection flips)
but ≈0.95 from the agreement-match channel (`winfm`), validating that channel. The
channel works, yet **adds nothing to binding** (Δ≈0): binding licensing is
recoverable via position/structure, not agreement.

**The key reframing** (the whole point):
- **Unbounded count observer** (whole-side category counts): almost everything is
  recoverable, because these grammars **surface-mark hierarchy** (`CAT9` = depth,
  `CAT6` = PP). Under this observer there is no genuinely non-surface rule.
- **Bounded-memory, left-to-right observer** (the cognitively/model-relevant one):
  the diagnostic is **restored** — adjacency rules are shortcuttable at ~zero
  memory, fixed-offset rules at a graded memory cost, and counting/depth/existential
  rules are *not* shortcuttable at any plausible memory.

So the corpus does **not** force hierarchy in the trivial sense (hierarchy is
surface-cued), but under human-like memory constraints a clear class of rules
resists a surface shortcut. Which rules a trained model actually shortcuts, given it
*has* the cues, is the interpretability question, not a corpus question.
