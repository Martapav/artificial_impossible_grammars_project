"""Normalized cross-grammar score R on the test split.

Δ = D_KL(p‖q) in nats is not scale-comparable across grammars: a low-entropy
construction (e.g. the positionally-conditioned binding of P/L′, whose floor the
geometry log Z pushes down) inflates D_KL for the same degree of mismatch. The
normalized score divides that scale out (SURPRISAL_REFERENCE_POINTS.md §2):

    R = D_KL(p‖q) / D_KL(p‖b)
      = (E_p[−log q] − H(p)) / (E_p[−log b] − H(p))

with H(p) = E_p[ℓ_corpus] the (credit-corrected) grammar floor and b the token
n-gram baseline (analysis/ngram_baseline.py). R = 0 means the model matches the
grammar; R = 1 means it is exactly as far as the weak surface reader; R > 1 worse.
Because the same floor sits in both numerator and denominator, the intrinsic-
entropy scale cancels — so R, not Δ, is what the binding comparison should use.

Reads the per-item Δ table (analysis/reports/heldout/report_differential_surprisal.py) and the
baseline scores; writes to results/analysis/evaluation/test/:
  normalized_score.csv, normalized_score_by_phenomenon.png.

Run:  python -m analysis.reports.heldout.report_normalized_score
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    CONSTRUCTION_DISPLAY, CONSTRUCTION_ORDER, GRAMMARS, GRAMMAR_COLOR,
    GRAMMAR_DISPLAY, outdir, savefig,
)

OUT = outdir("evaluation", "test")
DIFF = OUT / "differential_surprisal.parquet"
# b_tok fit per grammar on its own train split; the order fixes what counts as
# "local", so we report R against both a 4-gram and a 1-gram (unigram) reader.
BASE = {4: "analysis/data/ngram_baseline_scores.parquet",
        1: "analysis/data/ngram_baseline_scores_order1.parquet"}
SPLIT = "test_indistribution"


def load(order: int) -> pd.DataFrame:
    """Per-item frame with delta_q (model) and delta_b (baseline) over the floor."""
    d = pd.read_parquet(DIFF, columns=["grammar", "seed", "id", "construction",
                                       "neg_logq", "ell_corrected",
                                       "differential_surprisal"])
    d = d.rename(columns={"differential_surprisal": "delta_q"})
    b = pd.read_parquet(BASE[order])
    b = b[b.split == SPLIT][["grammar", "id", "neg_logb"]]
    m = d.merge(b, on=["grammar", "id"], how="left", validate="many_to_one")
    miss = int(m.neg_logb.isna().sum())
    if miss:
        raise ValueError(f"{miss} test items have no b_tok score — refit "
                         "analysis.ngram_baseline on this corpus set (set_3).")
    m["delta_b"] = m.neg_logb - m.ell_corrected
    m["grammar"] = pd.Categorical(m["grammar"], categories=GRAMMARS, ordered=True)
    return m


def _R(sub: pd.DataFrame) -> tuple[float, float, float, float]:
    """(R, R_sem, D_KL(p‖q), D_KL(p‖b)) for one cell.

    D_KL(p‖b) is seed-independent; the numerator varies by seed, so R's spread is
    taken over the per-seed numerators divided by the common denominator."""
    dkl_b = float(sub.delta_b.mean())
    per_seed_q = sub.groupby("seed", observed=True)["delta_q"].mean().to_numpy()
    dkl_q = float(per_seed_q.mean())
    r_seed = per_seed_q / dkl_b
    r = float(r_seed.mean())
    r_sem = float(r_seed.std(ddof=1) / np.sqrt(len(r_seed))) if len(r_seed) > 1 else 0.0
    return r, r_sem, dkl_q, dkl_b


def summarise(m: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GRAMMARS:
        for c in CONSTRUCTION_ORDER + ["__all__"]:
            sub = m[m.grammar == g] if c == "__all__" else \
                m[(m.grammar == g) & (m.construction == c)]
            if sub.empty:
                continue
            r, r_sem, dq, db = _R(sub)
            rows.append(dict(grammar=g, construction=c, R=r, R_sem=r_sem,
                             DKL_q=dq, DKL_b=db, n_items=len(sub)))
    return pd.DataFrame(rows)


def _panel(ax, summary: pd.DataFrame, label: str, legend: bool, ylabel: bool) -> None:
    s = summary[summary.construction != "__all__"]
    x = np.arange(len(CONSTRUCTION_ORDER))
    ng = len(GRAMMARS)
    w = 0.8 / ng
    for i, g in enumerate(GRAMMARS):
        gs = s[s.grammar == g].set_index("construction")
        ys = [gs.loc[c, "R"] if c in gs.index else np.nan for c in CONSTRUCTION_ORDER]
        es = [1.96 * gs.loc[c, "R_sem"] if c in gs.index else 0 for c in CONSTRUCTION_ORDER]
        ax.bar(x + (i - (ng - 1) / 2) * w, ys, width=w, yerr=es, capsize=2,
               color=GRAMMAR_COLOR[g], label=GRAMMAR_DISPLAY[g],
               error_kw=dict(lw=0.8, alpha=0.7))
    ax.set_xticks(x, [CONSTRUCTION_DISPLAY[c] for c in CONSTRUCTION_ORDER])
    ax.set_xlabel("phenomenon")
    if ylabel:
        ax.set_ylabel("normalized divergence  $R$")
    ax.grid(alpha=0.25, axis="y")
    ax.set_title(label, fontsize=9, loc="left")
    if legend:
        ax.legend(frameon=False, fontsize=9)


def plot_two_panel(sum4: pd.DataFrame, sum1: pd.DataFrame, out_png) -> None:
    # Shared y-axis so bar heights are directly comparable across the two
    # baselines (the 1-gram is a weaker reader -> genuinely smaller R).
    fig, (a4, a1) = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    _panel(a4, sum4, "4-gram baseline", legend=True, ylabel=True)
    _panel(a1, sum1, "1-gram baseline", legend=False, ylabel=False)
    gmax = max(pd.concat([sum4, sum1]).query("construction != '__all__'")["R"])
    a4.set_ylim(0, gmax * 1.10)
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    sums = {}
    for order in (4, 1):
        s = summarise(load(order))
        s.insert(0, "baseline_order", order)
        sums[order] = s
    combined = pd.concat(sums.values(), ignore_index=True)
    combined.to_csv(OUT / "normalized_score.csv", index=False)
    plot_two_panel(sums[4], sums[1], OUT / "normalized_score_by_phenomenon.png")

    for order in (4, 1):
        show = sums[order][sums[order].construction != "__all__"].pivot_table(
            index="construction", columns="grammar", values="R", observed=True)
        print(f"R ({order}-gram) by construction × grammar:")
        print(show.reindex(CONSTRUCTION_ORDER).round(3).to_string())
        print()
    print(f"[test] wrote two-panel normalized-score figure + csv -> {OUT}")


if __name__ == "__main__":
    main()
