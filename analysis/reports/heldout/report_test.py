"""Generative-test reporting figures -> results/analysis/evaluation/test/.

The held-out generative test is in-distribution only and scored from the final
(ep12) model, so there is no epoch axis. "Trends" are therefore over **sentence
length**: does the model's per-token surprisal grow on longer items, and how do
the four grammars and five phenomena compare.

Outputs (each figure paired with a backing CSV where it holds new numbers):

    perplexity_by_phenomenon_vs_length.{png,csv}
        one macro-panel per phenomenon; mean neg-logq-per-token vs length bin,
        one line per grammar, band = seed min..max.
    loss_vs_length_overlay.png
        one macro plot: the four grammars' neg-logq-per-token vs length bin,
        all phenomena pooled (the "4-grammar loss trend").
    neglogq_boxplot_by_phenomenon.{png,csv}
        per phenomenon, the four grammars' mean + IQR of neg_logq PER ITEM.
    perplexity_summary.csv
        per grammar x phenomenon: n, mean/median neg_logq, mean nats/token,
        perplexity.

Length bins are quantile bins over the pooled test lengths, shared across every
panel so the grammars and phenomena are read on one x-axis.

Run:  python -m analysis.reports.heldout.report_test
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    CONSTRUCTION_DISPLAY, CONSTRUCTION_ORDER, GRAMMARS, GRAMMAR_COLOR,
    GRAMMAR_DISPLAY, grammar_handles, load_test_items, load_test_summary,
    outdir, savefig,
)

OUT = outdir("evaluation", "test")
N_LEN_BINS = 8
MIN_PER_BIN = 50   # drop (grammar, phenomenon) length bins thinner than this


def assign_length_bins(items: pd.DataFrame, n_bins: int = N_LEN_BINS):
    """Quantile length bins shared across all panels; returns (items, centres)."""
    edges = np.unique(np.quantile(items["length"], np.linspace(0, 1, n_bins + 1)))
    items = items.copy()
    items["len_bin"] = pd.cut(items["length"], bins=edges, include_lowest=True)
    centres = (items.groupby("len_bin", observed=True)["length"].median())
    return items, centres


def _grammar_length_curve(sub: pd.DataFrame, centres: pd.Series):
    """Per length-bin: mean-over-items line and seed min..max band, one grammar.

    The line is the mean of the per-seed bin means (so each seed weighs equally);
    the band is the spread of those per-seed means.
    """
    per_seed = (sub.groupby(["seed", "len_bin"], observed=True)["neg_logq_per_token"]
                .mean().reset_index())
    g = per_seed.groupby("len_bin", observed=True)["neg_logq_per_token"]
    stat = g.agg(mean="mean", lo="min", hi="max").reindex(centres.index)
    counts = sub.groupby("len_bin", observed=True).size().reindex(centres.index).fillna(0)
    x = centres.reindex(stat.index).to_numpy()
    ok = (~np.isnan(x) & ~stat["mean"].isna().to_numpy()
          & (counts.to_numpy() >= MIN_PER_BIN))
    return x[ok], stat["mean"].to_numpy()[ok], stat["lo"].to_numpy()[ok], stat["hi"].to_numpy()[ok]


def panels_by_phenomenon(items: pd.DataFrame, centres: pd.Series, out_png, out_csv):
    phen = [c for c in CONSTRUCTION_ORDER if c in set(items["construction"])]
    ncol = 3
    nrow = int(np.ceil(len(phen) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    rows = []
    for k, con in enumerate(phen):
        ax = axes[k // ncol][k % ncol]
        for g in GRAMMARS:
            sub = items[(items.construction == con) & (items.grammar == g)]
            if sub.empty:
                continue
            x, mean, lo, hi = _grammar_length_curve(sub, centres)
            ax.plot(x, mean, color=GRAMMAR_COLOR[g], linewidth=1.6)
            ax.fill_between(x, lo, hi, color=GRAMMAR_COLOR[g], alpha=0.12, linewidth=0)
            for xi, m, l, h in zip(x, mean, lo, hi):
                rows.append(dict(construction=con, grammar=g, length_bin_centre=xi,
                                 mean_neg_logq_per_token=m, seed_min=l, seed_max=h))
        ax.set_title(CONSTRUCTION_DISPLAY[con], fontsize=11)
        ax.grid(alpha=0.25)
    for k in range(len(phen), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    for r in range(nrow):
        axes[r][0].set_ylabel("neg-logq / token (nats)")
    for c in range(ncol):
        axes[nrow - 1][c].set_xlabel("sentence length (tokens)")
    axes[0][ncol - 1].legend(handles=grammar_handles(), title="grammar",
                             fontsize=8, loc="upper left")
    fig.tight_layout()
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def overlay_loss_trend(items: pd.DataFrame, centres: pd.Series, out_png):
    fig, ax = plt.subplots(figsize=(8, 5))
    for g in GRAMMARS:
        sub = items[items.grammar == g]
        x, mean, lo, hi = _grammar_length_curve(sub, centres)
        ax.plot(x, mean, color=GRAMMAR_COLOR[g], linewidth=2, label=GRAMMAR_DISPLAY[g])
        ax.fill_between(x, lo, hi, color=GRAMMAR_COLOR[g], alpha=0.12, linewidth=0)
    ax.set_xlabel("sentence length (tokens)")
    ax.set_ylabel("neg-logq / token (nats)")
    ax.legend(handles=grammar_handles(), title="grammar", fontsize=9)
    ax.grid(alpha=0.25)
    savefig(fig, out_png)


def boxplot_by_phenomenon(items: pd.DataFrame, out_png, out_csv):
    """Per phenomenon, the four grammars' distribution of neg_logq PER ITEM."""
    phen = [c for c in CONSTRUCTION_ORDER if c in set(items["construction"])]
    fig, ax = plt.subplots(figsize=(1.9 * len(phen) + 2, 5.5))
    width = 0.8 / len(GRAMMARS)
    rows = []
    for gi, g in enumerate(GRAMMARS):
        data, positions = [], []
        for pi, con in enumerate(phen):
            vals = items[(items.grammar == g) & (items.construction == con)]["neg_logq"].to_numpy()
            data.append(vals)
            positions.append(pi + (gi - (len(GRAMMARS) - 1) / 2) * width)
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            rows.append(dict(grammar=g, construction=con, n=len(vals),
                             mean=float(vals.mean()), median=float(med),
                             q1=float(q1), q3=float(q3)))
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set(facecolor=GRAMMAR_COLOR[g], alpha=0.75, linewidth=0.6)
    ax.set_xticks(range(len(phen)), [CONSTRUCTION_DISPLAY[c] for c in phen])
    ax.set_ylabel("neg_logq per item (nats)")
    ax.legend(handles=grammar_handles(), title="grammar", fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def summary_table(items: pd.DataFrame, out_csv):
    g = (items.groupby(["grammar", "construction"], observed=True)
         .agg(n_items=("neg_logq", "size"),
              mean_neg_logq=("neg_logq", "mean"),
              median_neg_logq=("neg_logq", "median"),
              mean_nats_per_token=("neg_logq_per_token", "mean"))
         .reset_index())
    g["perplexity"] = np.exp(g["mean_nats_per_token"])
    order = {c: i for i, c in enumerate(CONSTRUCTION_ORDER)}
    g = g.sort_values(["grammar", "construction"],
                      key=lambda s: s.map(order) if s.name == "construction" else s)
    g.to_csv(out_csv, index=False)
    return g


def main() -> None:
    items = load_test_items()
    items, centres = assign_length_bins(items)

    panels_by_phenomenon(items, centres, OUT / "perplexity_by_phenomenon_vs_length.png",
                         OUT / "perplexity_by_phenomenon_vs_length.csv")
    overlay_loss_trend(items, centres, OUT / "loss_vs_length_overlay.png")
    boxplot_by_phenomenon(items, OUT / "neglogq_boxplot_by_phenomenon.png",
                          OUT / "neglogq_boxplot_by_phenomenon.csv")
    summary_table(items, OUT / "perplexity_summary.csv")

    # Corpus-level perplexity per run, straight from the eval summary, for the record.
    load_test_summary().to_csv(OUT / "test_run_summary.csv", index=False)
    print(f"[test] wrote per-phenomenon length panels, overlay, boxplot, summary -> {OUT}")


if __name__ == "__main__":
    main()
