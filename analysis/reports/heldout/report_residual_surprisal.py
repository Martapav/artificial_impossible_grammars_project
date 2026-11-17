"""Residual (differential) surprisal on the test split: two clean figures.

Δ(x) = −log q(x) − ℓ_corpus(x) is the model's surprisal with each grammar's own
entropy floor removed, so it is comparable across the four grammars. Reads the
per-item table written by ``analysis/reports/heldout/report_differential_surprisal.py``.

  residual_by_phenomenon.png : mean Δ per phenomenon, one bar per grammar.
  residual_by_length.png     : per-token residual (Δ / n_tokens) vs sentence
                               length, one line per grammar.

Uncertainty is across the five seeds. Run:
  python -m analysis.reports.heldout.report_residual_surprisal
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
SRC = OUT / "differential_surprisal.parquet"
MIN_N = 40           # drop (grammar, length) cells thinner than this (noisy tails)


def _seed_stat(df: pd.DataFrame, value: str, by: list[str]):
    """Per-seed cell means -> (mean, sem) over seeds for each `by` group."""
    per_seed = (df.groupby(by + ["seed"], observed=True)[value].mean()
                  .reset_index())
    g = per_seed.groupby(by, observed=True)[value]
    out = g.agg(mean="mean", sd="std", k="size").reset_index()
    out["sem"] = out["sd"] / np.sqrt(out["k"].clip(lower=1))
    return out


def plot_by_phenomenon(df: pd.DataFrame, out_png) -> None:
    stat = _seed_stat(df, "differential_surprisal", ["grammar", "construction"])
    x = np.arange(len(CONSTRUCTION_ORDER))
    ng = len(GRAMMARS)
    w = 0.8 / ng
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for i, g in enumerate(GRAMMARS):
        s = stat[stat.grammar == g].set_index("construction")
        ys = [s.loc[c, "mean"] if c in s.index else np.nan for c in CONSTRUCTION_ORDER]
        es = [1.96 * s.loc[c, "sem"] if c in s.index else 0 for c in CONSTRUCTION_ORDER]
        ax.bar(x + (i - (ng - 1) / 2) * w, ys, width=w, yerr=es, capsize=2,
               color=GRAMMAR_COLOR[g], label=GRAMMAR_DISPLAY[g],
               error_kw=dict(lw=0.8, alpha=0.7))
    ax.set_xticks(x, [CONSTRUCTION_DISPLAY[c] for c in CONSTRUCTION_ORDER])
    ax.set_xlabel("phenomenon")
    ax.set_ylabel("residual surprisal (nats)")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig(fig, out_png)


def plot_by_length(df: pd.DataFrame, out_png) -> None:
    df = df.assign(resid_per_tok=df.differential_surprisal / df.n_tokens)
    counts = df.groupby(["grammar", "length"], observed=True).size()
    stat = _seed_stat(df, "resid_per_tok", ["grammar", "length"])
    stat = stat.merge(counts.rename("n").reset_index(), on=["grammar", "length"])
    stat = stat[stat.n >= MIN_N]
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for g in GRAMMARS:
        s = stat[stat.grammar == g].sort_values("length")
        ax.plot(s.length, s["mean"], color=GRAMMAR_COLOR[g], lw=1.8,
                label=GRAMMAR_DISPLAY[g])
        ax.fill_between(s.length, s["mean"] - 1.96 * s["sem"],
                        s["mean"] + 1.96 * s["sem"], color=GRAMMAR_COLOR[g], alpha=0.15)
    ax.set_xlabel("length (words)")
    ax.set_ylabel("residual surprisal per token (nats)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    df = pd.read_parquet(
        SRC, columns=["grammar", "seed", "construction", "length", "n_tokens",
                      "differential_surprisal"])
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    plot_by_phenomenon(df, OUT / "residual_by_phenomenon.png")
    plot_by_length(df, OUT / "residual_by_length.png")
    print(f"[test] wrote residual-surprisal figures -> {OUT}")


if __name__ == "__main__":
    main()
