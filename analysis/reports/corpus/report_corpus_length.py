"""Corpus length distribution by phenomenon, across train / dev / test.

A descriptive check on the generative splits (``train``, ``validation`` -> dev,
``test_indistribution`` -> test) of ``corpora/set_3``: for each construction
(the generative phenomenon), the sentence-length distribution in each split.
Since the three splits are drawn from one generator, they should overlap -- this
figure makes any split-induced length shift visible per phenomenon.

Grammars are pooled (length is a property of the construction, not the grammar);
the backing table keeps the per (construction, split) quantiles and n.

Figure -> results/analysis/descriptive/figures/ ; table -> .../tables/.

Run:  python -m analysis.reports.corpus.report_corpus_length
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    CONSTRUCTION_DISPLAY, CONSTRUCTION_ORDER, GRAMMARS, REPO, outdir, savefig,
)

CORPUS_ROOT = REPO / "corpora" / "set_3"
SPLITS = {"train": "train", "validation": "dev", "test_indistribution": "test"}
SPLIT_ORDER = ["train", "dev", "test"]
SPLIT_COLOR = {"train": "#4d4d4d", "dev": "#1f77b4", "test": "#d62728"}


def load_corpus_lengths() -> pd.DataFrame:
    """(split, grammar, construction, length) for every generative item."""
    rows = []
    for split_dir, split in SPLITS.items():
        for g in GRAMMARS:
            path = CORPUS_ROOT / split_dir / f"{g}.jsonl"
            with open(path) as f:
                for line in f:
                    o = json.loads(line)
                    rows.append((split, g, o["construction"], o["length"]))
    return pd.DataFrame(rows, columns=["split", "grammar", "construction", "length"])


def figure(df: pd.DataFrame, fig_dir, tab_dir) -> None:
    constr = [c for c in CONSTRUCTION_ORDER if c in set(df["construction"])]
    ncol = 3
    nrow = int(np.ceil(len(constr) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False, sharex=True)
    edges = np.arange(df["length"].min(), df["length"].max() + 2)
    rows = []
    for i, c in enumerate(constr):
        ax = axes[i // ncol][i % ncol]
        for split in SPLIT_ORDER:
            d = df[(df.construction == c) & (df.split == split)]["length"]
            if d.empty:
                continue
            ax.hist(d, bins=edges, density=True, histtype="step", lw=1.5,
                    color=SPLIT_COLOR[split], label=split)
            q = np.percentile(d, [10, 50, 90])
            rows.append(dict(construction=c, split=split, n=int(d.size),
                             mean=float(d.mean()), p10=q[0], median=q[1], p90=q[2]))
        ax.set_title(CONSTRUCTION_DISPLAY[c], fontsize=9)
        ax.grid(alpha=0.2)
        if i % ncol == 0:
            ax.set_ylabel("density")
    for j in range(len(constr), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][ncol - 1].legend(fontsize=8, loc="upper right", title="split")
    fig.supxlabel("length (tokens)", fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir / "corpus_length_by_phenomenon.png")
    pd.DataFrame(rows).to_csv(tab_dir / "corpus_length_by_phenomenon.csv", index=False)


def main() -> None:
    fig_dir = outdir("descriptive", "figures")
    tab_dir = outdir("descriptive", "tables")
    df = load_corpus_lengths()
    figure(df, fig_dir, tab_dir)
    print(f"[corpus] {len(df):,} items -> corpus_length_by_phenomenon "
          f"(fig -> {fig_dir}, table -> {tab_dir})")


if __name__ == "__main__":
    main()
