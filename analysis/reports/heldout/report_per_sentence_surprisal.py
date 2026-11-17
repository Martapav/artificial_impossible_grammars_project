"""Per-sentence surprisal distribution, per grammar -> results/analysis/evaluation/test/.

Visualises the quantity ``mean_neg_logq_per_item`` = (sum_x -log q(x)) / N: the
mean per-SENTENCE surprisal, which weights every held-out item equally regardless
of length. The figure shows the full per-item distribution of ``neg_logq`` the
mean is taken over (pooled across seeds), with the mean (the reported metric) and
the median both marked -- so the gap between them makes visible that the mean is
pulled up by long / heavy-phenomenon sentences, which is precisely why it is
reported beside the length-weighted corpus perplexity.

Run:  python -m analysis.reports.heldout.report_per_sentence_surprisal
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY, load_test_items, outdir, savefig,
)

OUT = outdir("evaluation", "test")


def _per_seed_means(sub: pd.DataFrame) -> np.ndarray:
    """The metric computed per seed: one mean-per-sentence value per seed."""
    return (sub.groupby("seed", observed=True)["neg_logq"].mean().to_numpy())


def per_sentence_violin(items: pd.DataFrame, out_png, out_csv) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(GRAMMARS))
    data = [items.loc[items.grammar == g, "neg_logq"].to_numpy() for g in GRAMMARS]

    parts = ax.violinplot(data, positions=positions, widths=0.85,
                          showmeans=False, showmedians=False, showextrema=False)
    for body, g in zip(parts["bodies"], GRAMMARS):
        body.set(facecolor=GRAMMAR_COLOR[g], alpha=0.35, edgecolor=GRAMMAR_COLOR[g],
                 linewidth=1.0)

    rows = []
    for x, g, vals in zip(positions, GRAMMARS, data):
        mean = float(vals.mean())                 # the reported metric
        median = float(np.median(vals))
        seed_means = _per_seed_means(items[items.grammar == g])
        seed_lo, seed_hi = float(seed_means.min()), float(seed_means.max())

        # Median: a wide flat tick across the violin.
        ax.plot([x - 0.28, x + 0.28], [median, median], color="black", lw=1.6,
                zorder=4)
        # Mean (the metric): a diamond, with the across-seed spread as a thin bar.
        ax.plot([x, x], [seed_lo, seed_hi], color=GRAMMAR_COLOR[g], lw=3.0,
                solid_capstyle="round", zorder=4)
        ax.scatter([x], [mean], marker="D", s=70, color=GRAMMAR_COLOR[g],
                   edgecolor="black", linewidth=0.8, zorder=6)
        ax.annotate(f"{mean:.1f}", (x, mean), textcoords="offset points",
                    xytext=(12, 0), va="center", fontsize=9, fontweight="bold")
        ax.annotate(f"med {median:.0f}", (x + 0.28, median),
                    textcoords="offset points", xytext=(4, -2), va="top",
                    fontsize=7.5, color="0.35")
        rows.append(dict(grammar=g, mean_neg_logq_per_item=mean,
                         median_neg_logq=median, seed_min=seed_lo, seed_max=seed_hi,
                         n_items=len(vals)))

    ax.set_xticks(positions, [GRAMMAR_DISPLAY[g] for g in GRAMMARS])
    ax.set_ylabel("per-sentence surprisal  $-\\log q(x)$  (nats/sentence)")
    ax.set_xlabel("grammar")
    ax.grid(alpha=0.25, axis="y")

    handles = [
        plt.Line2D([], [], marker="D", ls="", color="0.5", mec="black",
                   label="mean (reported metric)"),
        plt.Line2D([], [], color="black", lw=1.6, label="median"),
        plt.Line2D([], [], color="0.5", lw=3.0, label="mean, seed min..max"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="upper right", framealpha=0.9)
    fig.tight_layout()
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    items = load_test_items()
    per_sentence_violin(items,
                        OUT / "per_sentence_surprisal_violin.png",
                        OUT / "per_sentence_surprisal_violin.csv")
    print(f"[test] wrote per-sentence surprisal violin -> {OUT}")


if __name__ == "__main__":
    main()
