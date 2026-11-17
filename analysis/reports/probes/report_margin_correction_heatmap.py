"""Triple margin heatmap: raw / 1-gram-corrected / 4-gram-corrected -> results/
analysis/evaluation/structural_probes/margin_heatmap/.

One 2 x 3 grid figure. Rows are the two probe splits (IOD, OOD); columns are the
three readings of the held-out within-pair margin, each term x grammar, in nats:

    raw               margin -- the criterion-native margin (span for in-place
                      substitutions, whole-string for relocations).
    1-gram corrected  margin_corrected = margin - margin_b1: the model's margin
                      minus the order-1 (unigram) n-gram's. Zero = nothing beyond
                      lexical frequency + length. This is what S[r] ranks on.
    4-gram corrected  margin_over_b4 = margin - margin_ref_b4: headroom above the
                      order-4 n-gram (same criterion as the term). >0 = the model
                      beats a 4-gram on that contrast; ~0 = reachable from local
                      statistics alone. A diagnostic, never subtracted for
                      selection.

Reading left-to-right within a row shows how much of the raw margin survives each
baseline. The colour scale is shared across all six panels, diverging about 0
(positive = grammatical member preferred / baseline beaten); every cell is
annotated with its value, so exact numbers are legible regardless of saturation.
Gate (control) terms are flagged with an asterisk.

Backing data: margin_correction_long.csv (term x grammar x split x margin_kind).
Numbers come straight from the ep12 terms.csv (evaluation.aggregate); nothing is
recomputed here. Values are the mean over the 5 seeds.

Run:  python -m analysis.reports.probes.report_margin_correction_heatmap
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.category_labels import relabel
from analysis.report_common import (
    GRAMMARS, GRAMMAR_DISPLAY, SPLIT_DISPLAY, TERM_ORDER, load_terms, outdir, savefig,
)

OUT = outdir("evaluation", "structural_probes", "margin_heatmap")
SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]

# (column key, column title) in left-to-right order.
KINDS = [
    ("margin", "raw margin"),
    ("margin_corrected", "1-gram corrected\n(margin - b1)"),
    ("margin_over_b4", "4-gram corrected\n(margin - b4)"),
]


def build_long(terms: pd.DataFrame) -> pd.DataFrame:
    """term x grammar x split x margin_kind -> mean-over-seeds margin (nats)."""
    value_cols = [k for k, _ in KINDS]
    agg = (terms.groupby(["split", "grammar", "term"], observed=True)[value_cols]
           .mean().reset_index())
    gates = set(terms.loc[terms.is_gate, "term"])
    long = agg.melt(id_vars=["split", "grammar", "term"], value_vars=value_cols,
                    var_name="margin_kind", value_name="margin_nats")
    long["is_gate"] = long["term"].isin(gates)
    return long


def _matrix(agg: pd.DataFrame, split: str, kind: str, present: list[str]) -> np.ndarray:
    idx = agg.set_index(["split", "grammar", "term"])[kind]
    M = np.full((len(present), len(GRAMMARS)), np.nan)
    for ri, term in enumerate(present):
        for ci, g in enumerate(GRAMMARS):
            try:
                M[ri, ci] = idx.loc[(split, g, term)]
            except KeyError:
                pass
    return M


def _draw(ax, M, present, ylabels, vmax, cmap, title, show_y):
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(GRAMMARS)), [GRAMMAR_DISPLAY[g] for g in GRAMMARS],
                  fontsize=9)
    if show_y:
        ax.set_yticks(range(len(present)), ylabels, fontsize=8)
    else:
        ax.set_yticks(range(len(present)), [""] * len(present))
    ax.set_title(title, fontsize=10)
    for ri in range(M.shape[0]):
        for ci in range(M.shape[1]):
            v = M[ri, ci]
            if np.isnan(v):
                continue
            frac = 0.5 + 0.5 * np.clip(v / vmax, -1, 1)
            tc = "white" if abs(frac - 0.5) > 0.32 else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center", color=tc,
                    fontsize=7.5)
    return im


def main() -> None:
    terms = load_terms()
    agg = (terms.groupby(["split", "grammar", "term"], observed=True)
           [[k for k, _ in KINDS]].mean().reset_index())
    build_long(terms).to_csv(OUT / "margin_correction_long.csv", index=False)

    present = [t for t in TERM_ORDER if t in set(terms["term"])]
    gates = set(terms.loc[terms.is_gate, "term"])
    ylabels = [relabel(t) + ("  *" if t in gates else "") for t in present]

    # One diverging scale shared across every panel, so subtracting a baseline is
    # visibly a change in the same colour space.
    vmax = float(np.nanmax(np.abs(agg[[k for k, _ in KINDS]].to_numpy())))
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(2, 3, figsize=(15, 11),
                             gridspec_kw={"wspace": 0.08, "hspace": 0.22})
    im = None
    for ri, split in enumerate(SPLITS):
        for ci, (kind, ktitle) in enumerate(KINDS):
            ax = axes[ri][ci]
            M = _matrix(agg, split, kind, present)
            title = ktitle if ri == 0 else ""
            im = _draw(ax, M, present, ylabels, vmax, cmap, title, show_y=(ci == 0))
        axes[ri][0].annotate(SPLIT_DISPLAY[split], xy=(-0.55, 0.5),
                             xycoords="axes fraction", rotation=90, va="center",
                             ha="center", fontsize=13, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("mean within-pair margin (nats)  ·  >0 = grammatical preferred / "
                   "baseline beaten")
    savefig(fig, OUT / "margin_correction_triple.png")
    print(f"wrote margin_correction_triple.png + margin_correction_long.csv -> {OUT}")


if __name__ == "__main__":
    main()
