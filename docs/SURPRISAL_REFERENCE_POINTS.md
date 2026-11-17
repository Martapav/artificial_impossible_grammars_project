# Surprisal reference points: the grammar floor and the surface yardstick

The framing for surprisal evaluation: what a model's surprisal is compared
against, why, and the conditions under which each comparison is valid.

---

## 1. Why two reference points

A model's surprisal on a string, `−log q(x)`, means nothing on its own. It becomes
interpretable only against something, and two candidates answer *different*
questions, so we build both.

Let `p` be the corpus distribution the model was trained on, `q` the trained model,
`b` a baseline predictor. For **any** proper distribution over the same strings,
cross-entropy decomposes exactly:

```
E_p[−log q(x)]  =  H(p)  +  D_KL(p ‖ q)
E_p[−log b(x)]  =  H(p)  +  D_KL(p ‖ b)
```

where `H(p) = E_p[ℓ(x)]` is the entropy of the grammar itself.

**`ℓ(x) = −log p(x)` is a floor.** No predictor beats `H(p)` in expectation.
`Δ(x) = −log q(x) − ℓ(x)` has mean `D_KL(p‖q) ≥ 0`, zero exactly when the model has
learned the grammar. It is an *absolute* reference and the only one that licenses
the claim **"the model has (not) learned the grammar."**

**`b` is a yardstick.** It bounds nothing; it is one weak predictor's position on
the same axis. It licenses **"the model has (not) gone beyond a local surface
reader."** Because `b` is the same bounded-memory, left-to-right observer as the
separability audit, the two analyses share one notion of "weak reader": the audit
asks whether such a reader can recover *licensing*, the baseline asks what
*surprisal* it achieves.

## 2. The normalized score

```
R  =  D_KL(p ‖ q) / D_KL(p ‖ b)  =  (E[−log q] − H(p)) / (E[−log b] − H(p))
```

| `R` | reading |
|---|---|
| 0 | model matches the grammar |
| 1 | model is exactly as far from the grammar as the weak surface reader |
| > 1 | model is worse than the weak surface reader |

`1 − R` is the fraction of the surface reader's divergence the model closed. `R`
fixes both ends of the ruler (`D_KL(p‖q)` in nats has no interpretable scale, `b`
alone has no notion of optimal) and divides out the intrinsic entropy rate, which
H, H′, P, L′ need not share — so `R` is the quantity that survives **cross-grammar**
comparison, whereas raw surprisal does not.

## 3. The granularity constraint (this decides the implementation)

The three objects do not natively live at the same grain:

| | predicts | grain |
|---|---|---|
| `p` | strings | whole string only |
| `q` | morpheme tokens (~4.6k vocab) | per token; sums to whole string |
| category observer | 8 categories | per position |

The decomposition requires `b` to be a **proper normalized distribution over the
same strings as `p`**. A category-level observer is not (it predicts a coarsening,
and predicting a category is strictly easier than predicting a token), so its
surprisal is on a different scale and the `≥ H(p)` inequality can fail.

**Consequence: the axis-mate for `ℓ(x)` must be a TOKEN-level baseline** — a smoothed
morpheme n-gram with an explicit `[EOS]`, so `b(x) = Π b(t_i | context)` sums to 1
over variable-length strings. The category observer keeps a separate,
non-substitutable job (§4).

Why `ℓ(x)` is whole-string only: it sums over generative *draws*, which do not
correspond 1:1 to surface positions (structural draws emit no word, a lexical draw
emits 2–3 morpheme tokens, a GAP draw emits zero). The sum is order-invariant (a
product of factors), so derivation-order vs surface-order mismatch is harmless for
the *total* — and that same mismatch is why the total cannot be split across
positions (a transform of a context-free language need not be context-free, so the
surface distribution does not factor per position).

## 4. Two framings, two questions — keep them separate

**String level: `p`, `q`, `b_tok`.** All whole-string, all nats, same vocabulary.
Yields `H(p)`, `D_KL(p‖q)`, `D_KL(p‖b)`, `R`. Answers *how much of the grammar the
model learned* and *how much of that is beyond local surface statistics*. Aggregate
over string subclasses (construction, length, embedding depth) to localize what was
not learned.

**Position level: `q` and `b_cat` only.** Marginalize the model's next-token
distribution onto categories and compare per position against the category observer.
`p` cannot join (no valid per-position true entropy, §3). Answers *where in the
sentence* the model exceeds a local reader — a diagnostic view, not a learnability
view.

The category observer is a positional diagnostic; the token n-gram is the axis-mate
for `ℓ(x)`. Neither substitutes for the other.

## 5. Implementation

| Component | Module |
|---|---|
| Draw tracing (`ℓ(x)` per draw) | `grammars/tracing.py` |
| Rejection-sampling correction → `−log p_corpus(x)` | `analysis/entropy_baseline.py` (`ell_corpus`, `emit`) |
| Token n-gram baseline `b_tok` (interpolated Witten–Bell, order 4, selected on held-out data) | `analysis/ngram_baseline.py` |
| Binding forced-draw credit (below) | `analysis/binding_credit.py` |
| Model per-item surprisal on the test split | `evaluation/test_eval.py --per-item` |
| Differential surprisal `Δ = neg_logq − ℓ_corpus` | `analysis/differential_surprisal.py`, report `analysis/reports/heldout/report_differential_surprisal.py` |
| Normalized score `R` | `analysis/reports/heldout/report_normalized_score.py` |

The corpus sidecars carry each item's derivation trace and a content-derived
`item_uid`, and `<grammar>.accept.json` carries the per-construction acceptance
rates, so `ℓ(x)` is computed for exactly the data the models trained on.

**Binding forced-draw credit.** `ℓ_raw` over-charges binding items relative to
`−log p_corpus` by two per-`(grammar, subtype)` constants: the **target-block
marginalization** (the coindexed nominal's latent head/modifier/number draws, which
coindexation overwrites) and the **geometry-resample log Z** (the within-item
rejection on the subtype's forced geometry, which the acceptance correction does not
see). Uncorrected, mean `Δ` is negative on binding; `analysis/binding_credit.py`
measures both constants on the real generator and the `Δ` report subtracts them.
The credit is a mean correction (the overwritten sub-tree is genuinely latent),
which is the level `Δ` is read at (`E[Δ] = D_KL`; per-item `Δ < 0` is allowed,
§6.2).

## 6. Requirements and honest limits

1. **`ℓ(x)` is the RAW derivation cost, not `−log p_corpus(x)`.** The corpus is
   rejection-sampled (construction quota, length window, `*_skipped` discards), so
   ```
   p_corpus(x) = (quota_c / n) · p_raw(x) / P_raw(accepted, c)
   ℓ_corpus(x) = ℓ(x) + log P_raw(accepted, c) − log(quota_c / n)
   ```
   Corrections are per-`(grammar, construction)` scalars from `<grammar>.accept.json`.
   Skipping this offsets the KL by a per-construction constant: harmless within a
   construction, wrong across them.
2. **`Δ(x) ≥ 0` holds in expectation, not per string.** A model can beat the grammar
   on an individual sentence by being over-confident where the grammar was uncertain.
   Report `Δ` as a mean with an interval; never assert per-item non-negativity.
3. **`b_tok` must be normalized** (proper smoothing, explicit `[EOS]`) or the
   decomposition breaks. It is *estimated*, unlike `p`, so fit on the training
   surface across all constructions and evaluate held out.
4. **Report `D_KL(p‖b)` alongside `R`.** The denominator is small when the grammar
   is nearly deterministic, which makes `R` unstable; the raw denominator must be
   visible.
5. **Units.** `ℓ(x)` is natural log; `F.log_softmax` is nats. Fix nats on both sides
   and assert it.
6. **`ℓ(x) = −log p(x)` requires derivational unambiguity.** Only 6 items across
   the tree family carry two derivation signatures (H train 3, test 1; H′ train 2);
   P and L′ have none. So `ℓ(x)` is exact except those six, where it is an upper
   bound with slack `H(D|X)`.
7. **Grammatical symbols are NOT single tokens**: a word is 1–4 morpheme tokens. Critical-position alignment must go through
   `model_token_span`, never word indices.
8. **Scope.** The baseline applies to grammatical strings only, and specifically to
   the **corpus and in-distribution splits**, not the structural probes. Ungrammatical strings have `p(x)=0`. The grammaticality judgment
   does not use this baseline and is not changed by it.
9. **Probe reporting — no baseline, and why.** Probe results are the **within-pair
   difference of model negative log-probabilities** (ungrammatical − grammatical).
   Both members share one base, so a shared `ℓ(x)` cancels from the difference;
   subtracting it would change nothing and only invite the reader to believe a
   correction had been applied. A shared baseline corrects nothing in which the
   members differ — a `wrong_site` pair whose members differ in length is not made
   comparable by subtracting the same constant. Those asymmetries are handled in
   probe design and in the pair-level analysis (`S[r]` subtracts `margin_b1`), never
   by the entropy baseline. If a per-item corrected value is ever reported for an
   ungrammatical member using its grammatical twin's `ℓ(x)`, label it
   **pair-anchored excess surprisal** — not that item's differential surprisal.

---

Related: [`SEPARABILITY.md`](SEPARABILITY.md) (the same bounded-memory observer),
`grammars/tracing.py`, `docs/TRANSFORM_RULES.md` §8 (sidecar, targets).
