"""Differential surprisal Δ(x) = neg_logq − ℓ_corpus on the in-distribution test.

Δ is the model's surprisal corrected by the grammar's own entropy floor; its mean
estimates ``D_KL(p‖q) ≥ 0`` — "how far the model is from having learned the
grammar" (``SURPRISAL_REFERENCE_POINTS.md`` §1). The floor ``ℓ_corpus``
(``analysis/entropy_baseline.py``) is additionally corrected on the two binding
constructions by the forced-draw credit (``analysis/binding_credit.py``), without
which Δ is spuriously negative there (the marginalized target block + the
geometry-resample log Z).

Writes, under ``results/analysis/evaluation/test/``:
  * ``differential_surprisal.parquet`` — per-item Δ for every run.
  * ``differential_surprisal.csv``     — mean Δ per grammar × construction, with
    a bootstrap CI over items and the across-seed spread.
  * ``differential_surprisal.png``     — the same, one panel.

Run:  python -m analysis.reports.heldout.report_differential_surprisal
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.binding_credit import apply_credit
from analysis.differential_surprisal import join_differential_surprisal
from analysis.entropy_baseline import ell_corpus
from analysis.report_common import (
    CONSTRUCTION_DISPLAY, CONSTRUCTION_ORDER, GRAMMARS, GRAMMAR_COLOR,
    GRAMMAR_DISPLAY, load_test_items, outdir, savefig,
)

OUT = outdir("evaluation", "test")
SPLIT = "test_indistribution"


def build_differential(items: pd.DataFrame) -> pd.DataFrame:
    """Join per-item model scores to the credit-corrected floor and subtract."""
    frames = []
    for g in GRAMMARS:
        ell = apply_credit(ell_corpus(g, SPLIT), g)          # + ell_corrected
        scores = items[items.grammar == g]
        merged = join_differential_surprisal(
            scores, ell, ell_column="ell_corrected",
            out_column="differential_surprisal")
        frames.append(merged)
    df = pd.concat(frames, ignore_index=True)
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    return df


def _cell(df: pd.DataFrame, g: str, construction: str | None) -> dict | None:
    """One summary row. Uncertainty is across the 5 seeds (model training
    variation) — the meaningful granularity here, and memory-light: the seeds
    share one test set, so item-level resampling would both understate the run
    variation and blow up memory on this box."""
    sub = df[df.grammar == g] if construction is None \
        else df[(df.grammar == g) & (df.construction == construction)]
    if sub.empty:
        return None
    seed_means = sub.groupby("seed", observed=True)[
        "differential_surprisal"].mean().to_numpy()
    point = float(seed_means.mean())
    sem = float(seed_means.std(ddof=1) / np.sqrt(len(seed_means))) \
        if len(seed_means) > 1 else 0.0
    return dict(grammar=g, construction=construction or "__all__",
                mean_delta=point, ci_lo=point - 1.96 * sem,
                ci_hi=point + 1.96 * sem, seed_min=float(seed_means.min()),
                seed_max=float(seed_means.max()), n_seeds=len(seed_means),
                n_items=len(sub))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for g in GRAMMARS:
        for c in CONSTRUCTION_ORDER:
            row = _cell(df, g, c)
            if row is not None:
                rows.append(row)
        rows.append(_cell(df, g, None))
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, out_png) -> None:
    constructions = CONSTRUCTION_ORDER
    x = np.arange(len(constructions))
    width = 0.8 / len(GRAMMARS)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, g in enumerate(GRAMMARS):
        sub = summary[(summary.grammar == g) &
                      (summary.construction != "__all__")].set_index("construction")
        xs, ys, lo, hi = [], [], [], []
        for j, c in enumerate(constructions):
            if c not in sub.index:
                continue
            xs.append(x[j] + (i - (len(GRAMMARS) - 1) / 2) * width)
            ys.append(sub.loc[c, "mean_delta"])
            lo.append(sub.loc[c, "mean_delta"] - sub.loc[c, "ci_lo"])
            hi.append(sub.loc[c, "ci_hi"] - sub.loc[c, "mean_delta"])
        ax.bar(xs, ys, width=width, color=GRAMMAR_COLOR[g], label=GRAMMAR_DISPLAY[g],
               yerr=[lo, hi], capsize=2, error_kw=dict(lw=0.8, alpha=0.7))
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x, [CONSTRUCTION_DISPLAY[c] for c in constructions])
    ax.set_ylabel(r"$\Delta = -\log q(x) - \ell_{\mathrm{corpus}}(x)$  "
                  r"(nats/sentence, $\widehat{D_{KL}}$)")
    ax.set_xlabel("construction")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, ncol=len(GRAMMARS), loc="upper left", framealpha=0.9)
    ax.set_title("Differential surprisal on the held-out test set "
                 "(binding floor corrected)", fontsize=10)
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    items = load_test_items()
    df = build_differential(items)
    df.to_parquet(OUT / "differential_surprisal.parquet", index=False)
    summary = summarise(df)
    summary.to_csv(OUT / "differential_surprisal.csv", index=False)
    plot(summary, OUT / "differential_surprisal.png")

    neg = summary[(summary.construction != "__all__") & (summary.mean_delta < 0)]
    print(summary.round(3).to_string(index=False))
    print(f"\n[test] wrote differential surprisal -> {OUT}")
    if len(neg):
        print(f"WARNING: {len(neg)} grammar×construction cell(s) still have mean "
              f"Δ < 0:\n{neg[['grammar','construction','mean_delta']].to_string(index=False)}")
    else:
        print("All grammar×construction mean Δ ≥ 0 (KL floor respected).")


if __name__ == "__main__":
    main()
