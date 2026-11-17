"""Assemble analysis/separability_analysis.ipynb from the parquet artifacts.

Thin presentation layer over ``analysis/data/separability_{results,floor,taxonomy}``
.parquet (produced by ``separability_stats.run_full_analysis``). Regenerate the
artifacts first, then:

    python analysis/build_separability_notebook.py
    jupyter nbconvert --to notebook --execute --inplace analysis/separability_analysis.ipynb

Every table is preceded by a "how to read this" note; a glossary sits up front and
an interpretation section closes. Never aggregates phenomena within a grammar;
keeps H vs P and H′ vs P′ side by side.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "separability_analysis.ipynb"


def md(t): return nbf.v4.new_markdown_cell(t.strip("\n"))
def code(t): return nbf.v4.new_code_cell(t.strip("\n"))


CELLS = [
    md(r"""
# Separability audit

**The question.** *Before* we train any model, can a **weak, purely surface-level
observer** figure out where each grammatical rule applies — just from the
categories of nearby words? If a "hierarchical" rule can be predicted this way, a
trained model could pass it with a shortcut instead of learning structure, so we
want to know this up front.

**The observer** is deliberately weak and human-like: it reads **left-to-right**,
sees only word **categories** (not the actual words), and has **bounded memory**
— recent words count fully, distant ones fade (a memory constant **τ**: small τ =
only the immediate neighbours; large τ = the whole left context). It is trained
to guess, for each candidate word, whether the rule licenses that position.

**What we measure — U.** For each rule we report **U ∈ [0, 1]**: the fraction of
the licensing decision's uncertainty that the observer's local context resolves.

* **U = 0** → the context is useless: knowing the neighbours tells you nothing
  about licensing beyond the base rate.
* **U = 1** → the context fully determines licensing (a perfect surface shortcut).
* Read U **as a function of τ**: how much *memory* the shortcut needs.

Everything below is **descriptive of the corpus**, not of any model.
"""),
    md(r"""
## Glossary (read once)

| term | meaning |
|---|---|
| **U** | fraction of licensing uncertainty (entropy) the observer resolves; 0 = useless, 1 = perfectly predictable. **Saturates near the top:** U is a log-loss score, so the L2 penalty caps an already-perfect separator at ≈0.97, and the exact ceiling depends on how many features carry the cue. Treat U ≳ 0.95 as "perfect" and compare with `auc`, not with each other. |
| **τ (tau)** | memory span. τ=0.5 ≈ only the 1–2 nearest words; τ=∞ = the entire left context, undimmed. |
| **left-only** | the observer uses only words *to the left* of the candidate — matching a causal, left-to-right reader. |
| **comparable band** | we compare a licensed position only against *unlicensed positions that could plausibly have been licensed* (share the rule's precondition), so U reflects the rule itself, not a trivial giveaway. |
| **floor** | the *family's* generic local determinism: how well the same observer predicts an arbitrary *next word's category*. A rough reference (~0.5) for "how much local structure exists anyway". It targets a different variable (8 categories, H(Y)≈2.7 bits) than the cells (licensed/not, H(Y)≤1 bit), so it is **not** a threshold and is **not** subtracted. Shown with its bootstrap CI (±0.01); read it to 2 dp. Nearly **flat** across τ by construction of the grammar — next-category here is close to first-order Markov, so unlimited memory buys only ≈+0.03. |
| **recovery cutoff 0.6** | we call a rule "recoverable" at a given memory once U ≥ 0.6. A **stipulated** threshold — it is *not* derived from the floor (the floor is not precise enough to support one, and measures a different target). |
| **cue class** | **bounded-local** = recoverable at tiny memory; **memory-graded** = recoverable only with more memory (`tau_recover` = how much); **not-locally-recoverable** = U stays below 0.6 even with unlimited memory. |
| **lookahead gap** | how much a *both-sides* observer beats the left-only one — i.e. how much the rule needs to see words *to the right* (lookahead). |
| **thin** | the cell's minority class has < 30 members, so its U is too weakly estimated to support a verdict. A thin row **abstains**: it is not evidence of a shortcut *or* of its absence. `aux` is thin in every grammar — not because the rule is unselective, but because the discriminating configuration is rare in natural bases (an aux occurs in ~30% of them, two auxes in <2%), so there is almost no negative class. Aux selectivity is tested by the minimal-pair probes, which construct that contrast deliberately instead of waiting for it. |
"""),
    code(r"""
import pandas as pd, numpy as np, matplotlib.pyplot as plt, math
from pathlib import Path
DATA = Path("data")
grid = pd.read_parquet(DATA / "separability_results.parquet")
floor = pd.read_parquet(DATA / "separability_floor.parquet")
tax  = pd.read_parquet(DATA / "separability_taxonomy.parquet")
GORD = ["H","Hprime","P","Lprime"]; PORD = ["wh_i","wh_ii","aux","refl","pron"]
GNAME = {"H":"H","Hprime":"H′","P":"P","Lprime":"P′"}  # display labels (data keys unchanged)
TAUS = [0.5,1.0,2.0,4.0,float("inf")]
def tlab(t): return "inf" if math.isinf(t) else (int(t) if float(t).is_integer() else t)
LOG = grid[grid.model=="logreg"]
def Ucell(regime, band, spec):
    d = LOG[(LOG.regime==regime)&(LOG.band==band)&(LOG.spec==spec)]
    p = d.pivot_table(index="phenomenon", columns="grammar", values="U")
    return p.reindex(index=[x for x in PORD if x in p.index],
                     columns=[g for g in GORD if g in p.columns])
"""),
    md(r"""
## 1. Cue taxonomy — the headline

**One row per rule per grammar.** Each rule is classified by *how much memory a
surface observer needs to shortcut it*.

**How to read the columns:**

* **`taxonomy`** — the verdict: `bounded-local` (shortcuttable with almost no
  memory), `memory-graded` (shortcuttable, but only once memory is long enough),
  `not-locally-recoverable` (no surface shortcut, even with unlimited memory).
* **`tau_recover`** — the *memory cost*: the smallest τ at which U reaches 0.6.
  `inf` = reached **only** with unlimited memory; `NaN` = **never** reached at any
  memory (a genuinely non-local rule). These are different claims — before
  2026-07-31 both printed as `inf`.
* **`U_t0.5 … U_tinf`** — U as memory grows from tiny (0.5) to unlimited (∞).
  Reading a row **left→right** shows how much extra memory buys: flat-high = free
  shortcut; rising = a memory cost; flat-low = no shortcut.
* **`lookahead_gap`** — extra U from also seeing the right side (≈0 for almost
  everything; small positive only where a rule needs lookahead).
* **`n_pos` / `n_neg` / `thin`** — class balance of the headline (τ=1) cell.
  **`thin=True` (minority class < 30) means the row's verdict is not evidence
  either way** — U≈0 there reads "not measurable at this sample size", *not*
  "no shortcut exists". Carried here rather than left in §7 so the caveat cannot
  be read separately from the label. See §7 for the CIs.
* **`floor_t1`** — the generic determinism reference (~0.5) for that grammar's
  *family*. A context number, not a threshold: it scores a different target
  (next category out of 8) and carries a ±0.01 CI.

**What to look for:** rules that stay **below ~0.6 across the whole row** *and* are
not `thin` are the ones a bounded-memory learner cannot shortcut — the
interesting, genuinely structural cases.
"""),
    code(r"""
show = tax.copy()
show["phenomenon"] = pd.Categorical(show.phenomenon, PORD, ordered=True)
show["grammar"] = pd.Categorical(show.grammar.map(GNAME),
                                 [GNAME[g] for g in GORD], ordered=True)
cols = ["grammar","phenomenon","taxonomy","tau_recover",
        "U_t0.5","U_t1.0","U_t2.0","U_t4.0","U_tinf","lookahead_gap",
        "n_pos","n_neg","thin","floor_t1"]
show.sort_values(["phenomenon","grammar"])[cols].round(3).reset_index(drop=True)
"""),
    md(r"""
## 2. The memory-cost curves — U vs τ

Each panel is one phenomenon; each line is one grammar; **x = memory (τ)**,
**y = U**. The **red dotted** line is the 0.6 recovery cutoff; the **grey/black
dotted** lines are the two families' determinism floors, with their bootstrap CI
bands. The floors are near-flat and their bands overlap across finite τ — that is
expected (see the glossary), and they are a *reference*, not a threshold.

**How to read a curve:**

* **flat and high** (well above 0.6 from the start) → *bounded-local*: a free
  surface shortcut, no memory needed. **No rule is in this class in the current
  corpus** — the highest any curve starts is H/H′ `pron` at 0.32. The class is kept
  on the axis because eliminating it is what the rule revisions were for.
* **rising through 0.6** → *memory-graded*: the shortcut exists but costs memory;
  where it crosses 0.6 is the cost (`tau_recover`). P/P′ `pron` and `refl` are the
  two instances.
* **flat and low** (never reaches 0.6) → *not-locally-recoverable*: no surface
  shortcut at any memory. H and P′ `wh_i` are the flattest, sitting near 0.07–0.14
  at every τ — though both are `thin`, so read them with §7.

**Within-family delegation:** binding (refl/pron) and aux delegate H→H′ and P→P′,
and wh_ii's clause is family-shared, so **H≡H′ and P≡P′ exactly** for refl, pron,
aux, wh_ii — the dashed (primed) curve rides on top of its solid base curve. Only
**wh_i** diverges across grammars (the clause-(i) crossover). Coincident curves
are expected, not missing data.
"""),
    code(r"""
specs = [f"decay@{tlab(t)}:left" for t in TAUS]
piv = {s: Ucell("train","comparable",s) for s in specs}          # precompute
xs = [0.5,1,2,4,8]                                                # ∞ at 8
STYLE = {"H":("#1f77b4","-","o"), "Hprime":("#17becf","--","s"),  # H family: blue/cyan
         "P":("#d62728","-","o"), "Lprime":("#ff7f0e","--","s")}  # P family: red/orange
fig, axes = plt.subplots(1, len(PORD), figsize=(17,3.3), sharey=True)
# One floor per FAMILY (H≡H', P≡P' by construction), each with its bootstrap CI
# band — the band is the point: the finite-tau floor is flat WITHIN noise.
FAMOF = {"H":"H", "Hprime":"H", "P":"P", "Lprime":"P"}
fl   = floor.pivot_table(index="grammar", columns="tau", values="floor_U")
fllo = floor.pivot_table(index="grammar", columns="tau", values="floor_U_lo")
flhi = floor.pivot_table(index="grammar", columns="tau", values="floor_U_hi")
for ax, phen in zip(axes, PORD):
    for g in GORD:
        ys = [piv[s].loc[phen, g] if (phen in piv[s].index and g in piv[s].columns)
              else np.nan for s in specs]
        col, ls, mk = STYLE[g]
        ax.plot(xs, ys, color=col, ls=ls, marker=mk, ms=5, lw=1.8, alpha=0.85, label=GNAME[g])
    for rep, fc in (("H", "grey"), ("P", "black")):          # family representatives
        ax.plot(xs, [fl.loc[rep, t] for t in TAUS], ls=":", c=fc, lw=1,
                label=f"floor ({rep} fam)")
        ax.fill_between(xs, [fllo.loc[rep, t] for t in TAUS],
                        [flhi.loc[rep, t] for t in TAUS], color=fc, alpha=0.12, lw=0)
    ax.axhline(0.6, ls=":", c="red", lw=0.8)
    ax.set_title(phen); ax.set_xlabel("memory τ (∞@8)"); ax.set_ylim(-0.05,1.02)
    ax.set_xticks(xs); ax.set_xticklabels(["0.5","1","2","4","∞"])
axes[0].set_ylabel("U (uncertainty resolved)"); axes[-1].legend(fontsize=7)
plt.tight_layout(); plt.show()
"""),
    md(r"""
## 2b. The rulewise band — holding the rule's *own* precondition constant

`comparable` controls for a precondition **shared by all grammars**, so cells stay
comparable across grammars. But where a rule's domain is *narrower* than that
shared precondition, the leftover negatives can still be a freebie — U would then
score the precondition rather than the rule. `rulewise` bands the rule's own
precondition instead (objecthood, for these cells). **Read it within a grammar
only** — it is not cross-grammar comparable. Only the object-domain rules have a
`rulewise` band: H/H′ `refl` and H/P′ `wh_i`.

Both surviving rules stay under 0.6 at every τ once their own precondition is held
constant:

* **H/H′ `refl` — 49 positives against 389 negatives.** Under rule A′ the licensing
  predicate is "the nucleus is embedded", which is independent of objecthood, so
  the rule has real negatives inside its own domain. U is at chance for bounded
  memory (−0.030, AUC 0.46–0.49 at τ≤1) and reaches 0.449 only at τ=∞. The AUC
  there is 0.946: an unbounded ranker orders the cases well, but bounded,
  calibrated recovery fails.
* **H/P′ `wh_i` — 21 and 26 positives, both `thin`.** U sits at or near zero at
  every τ (−0.036 to +0.001) and AUC stays between 0.39 and 0.63, so neither the
  bounded nor the unbounded observer recovers E3 once objecthood is held out. The cells are too small to
  carry a verdict on their own; they agree with the `comparable` band rather than
  adding to it.

Earlier versions of this section reported H/H′ `refl` as `degenerate` (no negatives
inside its own domain). That was a property of the pre-A′ rule, whose licensing
predicate *was* objecthood. It no longer holds.
"""),
    code(r"""
rw = LOG[(LOG.band=="rulewise")&(LOG.regime=="train")&(LOG.kind=="decay")&(LOG.side=="left")]
cp = LOG[(LOG.band=="comparable")&(LOG.regime=="train")&(LOG.kind=="decay")&(LOG.side=="left")]
key = ["grammar","phenomenon","spec"]
cmp_tbl = (rw.merge(cp, on=key, suffixes=("_rulewise","_comparable"))
             [key + ["n_pos_rulewise","n_neg_rulewise","U_rulewise","auc_rulewise",
                     "n_neg_comparable","U_comparable","auc_comparable"]])
cmp_tbl.round(3).assign(grammar=lambda d: d["grammar"].map(GNAME)).reset_index(drop=True)
"""),
    md(r"""
## 3. Does the rule need to look *ahead*?  (both-sides − left-only, τ=1)

The observer is left-to-right, but here we let it also peek right and report the
**gain**: `U(both sides) − U(left only)`. **≈ 0** means the left context already
had everything (the rule is resolvable incrementally). A **positive** value means
the rule genuinely needs to see upcoming words.

**What to look for:** the gains track the *clause*, not the grammar. Clause (i) is
E3 — "matrix object AND it hosts an embedded clause" — and the hosted clause sits
to the target's RIGHT, so its licensor is only visible with lookahead. E3 gains
wherever it is hosted: **P′ +0.210** (AUC 0.720 → 0.914) and **H +0.090** (AUC
0.809 → 0.887). The third-position clause, in the same column of the table, gains
nothing: **H′ −0.011, P −0.009**. E3 is hard for a *left-to-right* observer, which
is what a causal LM is, but not hard in both directions.

Two smaller gains: **H/H′ `refl` +0.126** (AUC 0.808 → 0.906), consistent with
rule A′ requiring an embedded nucleus whose marker can fall to the right; and
**H/H′ `aux` +0.164**, which sits on a `thin` cell (27 negatives) and should not
be read as a result. The whole P/P′ arm is ≈0 or slightly negative, including
`wh_ii` (−0.016) — earlier versions of this section called `wh_ii` a non-trivial
gain, which the current rule does not support.
"""),
    code(r"""
gap = (Ucell("train","comparable","decay@1:both") - Ucell("train","comparable","decay@1:left"))
gap.round(3).rename(columns=GNAME)
"""),
    md(r"""
## 4. Sanity controls (do NOT describe the rules — they validate the method)

These are grammatical-vs-ungrammatical **minimal pairs** from the probe splits.
They check the metric can detect a violation that really *is* local:

* **`neutral_positional`** — one word moved to an illegal slot. Visible from
  **categories** (`window@5:both`), so U should be clearly above zero.
* **`neutral_agreement`** — one inflection flipped so a word disagrees with its
  controller. Invisible to categories (the categories are unchanged), but the
  **agreement-match channel** (`winfm`) should catch it.

**What to look for:** positional above zero under `window` (0.16–0.39); agreement
at exactly 0.000 under `window`, rising under `winfm` in seven of the eight cells
(0.28–0.50). The exception is P at OOD length (0.019). These are 30-pair control
cells, so single values are noisy and the pattern is the point, not the
magnitudes. It confirms the method detects a local violation and that the
agreement channel is live, which is what makes the binding null in §6 meaningful.
"""),
    code(r"""
neu = LOG[LOG.regime.str.startswith("neutral") & LOG.spec.isin(["window@5:both","winfm@5:both"])]
neu.pivot_table(index=["regime","phenomenon"], columns="spec", values="U").round(3)
"""),
    md(r"""
## 5. Does it hold at longer sentences?  (train → OOD, τ=1, left-only)

Same rule, scored on in-distribution lengths (`train`, ≤25 words) vs long
out-of-distribution ones (`ood`, 26–48). **`drop` = train − ood.** A large drop
means the shortcut was propped up by short sentences and weakens with length
(these are the rules whose "local proxy" for a non-local property gets diluted).

**Scope — what a *negative* drop does and does not mean.** `train`/`ood` here are
**length bands of freshly drawn bases**, not corpus splits; this module never
reads `corpora/`. Where U is *higher* at OOD length, that is **not** a shortcut
the model can exploit: models are trained only on ≤25 words, so a regularity that
exists only at 26–48 is never available to learn. The quantity that would bear on
that risk is an observer **fitted at training lengths and evaluated at OOD
lengths**; the grid fits and scores within each band separately, so it is not
computed here. Read this section as a property of the *grammar* across length,
not as a threat model for the trained model.
"""),
    code(r"""
tr = Ucell("train","comparable","decay@1:left").stack().rename("train")
od = Ucell("ood","comparable","decay@1:left").stack().rename("ood")
ls = pd.concat([tr,od],axis=1); ls["drop"]=(ls["train"]-ls["ood"]).round(3)
ls.round(3).rename(index=GNAME, level="grammar")
"""),
    md(r"""
## 6. Companions & the binding agreement-null

**Top table — AUC** (an alternative, ranking-based score in [0.5, 1]) for the
primary decay observer vs the full-window and whole-side-count references, as a
cross-check that U and AUC tell the same story.

**Bottom table — binding via agreement.** Binding U with categories only
(`window`) vs categories + agreement-match (`winfm`). The agreement channel
**works** (proven in §4), yet adds **≈0** here → binding licensing is *not*
recoverable from agreement, only from position/structure.
"""),
    code(r"""
comp = LOG[(LOG.regime=="train")&(LOG.band=="comparable")
           & LOG.spec.isin(["decay@1:left","window@7:left","count:left"])]
display(comp.pivot_table(index=["phenomenon","grammar"], columns="spec", values="auc").round(3)
        .rename(columns=lambda c: "AUC:"+c).rename(index=GNAME, level="grammar"))
bind = LOG[(LOG.regime=="train")&(LOG.band=="comparable")
           & LOG.phenomenon.isin(["refl","pron"])
           & LOG.spec.isin(["window@5:both","winfm@5:both"])]
display(bind.pivot_table(index=["phenomenon","grammar"], columns="spec", values="U").round(3)
        .rename(index=GNAME, level="grammar"))
"""),
    md(r"""
## 7. Reliability — is each number trustworthy?

Per cell: **U** with its **bootstrap 95% CI** (`U_lo`–`U_hi`), plus the class
sizes `n_pos` / `n_neg`. `thin=True` flags a minority class under 30, whose CI is
too wide to support a verdict.

Six cells are thin: `aux` in all four grammars (27 negatives in H/H′, 12 in P/P′)
and `wh_i` in H (21 positives) and P′ (26). Both causes are scarcity of the
discriminating configuration in natural bases, not a property of the rules. Longer
sentences supply it: at OOD length the minority counts rise to 77/60 for `aux` and
78/84 for `wh_i`, and **no cell is thin in the `ood` regime**. That does not
license reading the OOD numbers as the model's threat model — see §5 — but it does
show the thinness is a sampling limit rather than a fact about the grammars.
"""),
    code(r"""
rel = LOG[(LOG.regime=="train")&(LOG.band=="comparable")&(LOG.spec=="decay@1:left")][
    ["phenomenon","grammar","U","U_lo","U_hi","auc","n_pos","n_neg","thin"]]
rel.round(3).sort_values(["phenomenon","grammar"]).assign(
    grammar=lambda d: d["grammar"].map(GNAME)).reset_index(drop=True)
"""),
    md(r"""
## 8. Interpretation & takeaways

**The observer settles the question the raw magnitudes could not.** With an
*unbounded* observer (count every category anywhere), almost every rule looks
recoverable — because these grammars **mark structure on the surface**
(complementizers flag embedding, prepositions flag PPs), so a big-enough counter
can reconstruct even "hierarchical" rules. That is not the human- or model-relevant
question. Under the **bounded-memory, left-to-right** observer the picture
separates cleanly:

* **Bounded-local (free shortcut):** recoverable at τ≈0.5 — an immediate
  neighbour gives it away, so a model needs no structure to pass. **This cell is
  now empty by design.** Every rule that fell in it was replaced for exactly that
  reason: `refl` object (H/H′) → rule A′, the nucleus must be embedded
  (2026-07-29); `wh_i` *matrix object* (H/P′) → E3, the matrix object must also
  host an embedded clause; `wh_ii` *after-CAT5* (P/P′) → the second nominal after
  the second CAT6, since the old clause was a CAT5→nominal bigram (both
  2026-07-30). The rows below are re-derived at each rebuild — read them from the
  tables in this notebook, not from this prose.
* **Memory-graded (shortcut at a cost):** `refl` and `pron` (P/P′). Both are
  **counting** rules, so an observer with enough left memory essentially *is* the
  rule — recovering them at large τ is expected and is not a defect. What matters
  is that neither is free at small memory: `pron` costs `tau_recover`=4 and `refl`
  is reached only at unbounded memory, while both sit at U≈0.11–0.17 at τ=0.5.
* **Not locally recoverable (no surface shortcut):** `wh_i` *third-position*
  (H′/P), `wh_ii` *Type3-under-Type4* (H/H′) and *2nd-nominal-after-2nd-CAT6*
  (P/P′), `refl` *embedded nucleus* (H/H′), and `pron` *cross-nucleus* (H/H′).
  These require **counting**, **scanning the whole clause**, tracking **embedding
  depth**, or a **whole-sentence existential** — none of which a bounded
  left-to-right reader can do. These are where a model genuinely has to compute,
  not memorise. `wh_i` *E3* (H/P′) carries the same label in the table but is
  `thin`, so it is listed under abstaining below rather than counted here.
* **Abstaining (`thin`):** `aux`, in **all four** grammars, and `wh_i` in H (21
  positives) and P′ (26). Their labels above are reported but carry no weight —
  see the `thin` column in §1 and the CIs in §7. The `aux` cells in particular
  have 27 (H/H′) and 12 (P/P′) negatives, and their U CIs straddle zero, so no
  verdict about aux shortcuttability should be drawn from this audit either way.

**Two conclusions for the thesis:**

1. **"Positional" does not mean "easy".** The *positional* third-position `wh_i`
   (H′/P) and the *positional* `wh_ii` (P/P′) are among the hardest to shortcut —
   U reaches only 0.47 and 0.51 at unlimited memory, on well-populated cells —
   because linear counting is non-local. The naive "hierarchical = hard,
   positional = easy" split is wrong; the real axis is *how much reach/computation
   the cue needs*, which cuts across the family label. (The `aux` first-in-scan
   rule was cited here in earlier versions; its cell is thin, so it no longer
   supports the point either way.)

2. **The corpus does not force hierarchy for free, but it does leave a genuine
   non-shortcuttable core.** Because structure is surface-marked, a powerful
   enough model *could* shortcut most rules; but under human-like memory limits a
   clear set (counting / depth / existential rules) resists any surface shortcut.
   Which of these a trained model actually shortcuts — given it *has* the surface
   cues — is then an **interpretability** question about the model, not a property
   of the corpus. This audit's job was to identify that non-shortcuttable core,
   and it does.

*(All numbers reproduce from `analysis/separability_stats.run_full_analysis`;
methods in `docs/SEPARABILITY.md`.)*
"""),
]


def main():
    nb = nbf.v4.new_notebook(); nb["cells"] = CELLS
    nb["metadata"] = {"kernelspec": {"name":"python3","display_name":"Python 3","language":"python"}}
    NB_PATH.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    main()
