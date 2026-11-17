"""Per-term probe-margin heatmaps -> results/analysis/evaluation/
structural_probes/margin_heatmap/.

One figure per probe split (IOD, OOD). Each figure has two panels, probe term
(rows) x grammar (columns), cell = mean within-pair margin in nats:

    left   whole-string margin only, one row per term (margin_string averaged
           over every pair of that term).
    right  whole-string margin for every term PLUS a span-margin row for the
           in-place-substitution terms -- the ones whose native criterion is the
           critical span (refl:wrong_item, refl:no_substitution, pron:wrong_item,
           agreement gate). The span row averages margin_span over that term's
           pairs; span is not a minimal contrast for the relocation terms, so they
           keep only a string row (evaluation.aggregate, report_subphenomenon).

Rows are the fine-grained terms, so the sub-cases of a phenomenon (e.g. pronoun
wrong-item vs wrong-site) are distinguished. Colour scale is shared across both
panels and both probe splits, diverging about 0 (positive = the model preferred
the grammatical member).

Backing data: margin_heatmap/margins_long.csv (term x grammar x split x
margin_kind, mean margin + n pairs).

Run:  python -m analysis.reports.probes.report_probe_margin_heatmap
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    GRAMMARS, GRAMMAR_DISPLAY, SPLIT_DISPLAY, TERM_ORDER,
    load_probe_pairs, outdir, savefig,
)

OUT = outdir("evaluation", "structural_probes", "margin_heatmap")
SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]

# Readable row label per probe term.
TERM_LABEL = {
    "aux": "aux movement",
    "wh_i": "wh (i)",
    "wh_ii": "wh (ii)",
    "refl:wrong_item": "reflexive: wrong item",
    "refl:wrong_site": "reflexive: wrong site",
    "refl:no_substitution": "reflexive: no subst.",
    "pron:wrong_item": "pronoun: wrong item",
    "pron:wrong_site": "pronoun: wrong site",
    "neutral_agreement": "agreement (gate)",
    "neutral_positional": "positional (gate)",
}


def build_margins_long(pairs: pd.DataFrame) -> pd.DataFrame:
    """term x grammar x split x margin_kind -> mean margin, n pairs.

    string : margin_string over every pair of the term.
    span   : margin_span over the term's pairs, ONLY for the span-native
             (in-place-substitution) terms; not emitted for the others.
    """
    # A term is span-native iff its pairs carry criterion == "span".
    span_terms = set(pairs.loc[pairs.criterion == "span", "term"].unique())
    rows = []
    for split in SPLITS:
        s = pairs[pairs.split == split]
        for term in TERM_ORDER:
            tt = s[s.term == term]
            for g in GRAMMARS:
                gg = tt[tt.grammar == g]
                rows.append(dict(split=split, term=term, grammar=g,
                                 margin_kind="string",
                                 margin=gg["margin_string"].mean(),
                                 n=int(gg["margin_string"].notna().sum())))
                if term in span_terms:
                    rows.append(dict(split=split, term=term, grammar=g,
                                     margin_kind="span",
                                     margin=gg["margin_span"].mean(),
                                     n=int(gg["margin_span"].notna().sum())))
    return pd.DataFrame(rows)


def _row_spec(long: pd.DataFrame, with_span: bool) -> list[tuple[str, str]]:
    """(term, margin_kind) rows, top-to-bottom, for a panel."""
    span_terms = set(long.loc[long.margin_kind == "span", "term"].unique())
    spec = []
    for term in TERM_ORDER:
        spec.append((term, "string"))
        if with_span and term in span_terms:
            spec.append((term, "span"))
    return spec


def _matrix(long: pd.DataFrame, split: str, rowspec) -> np.ndarray:
    idx = long.set_index(["split", "term", "grammar", "margin_kind"])["margin"]
    M = np.full((len(rowspec), len(GRAMMARS)), np.nan)
    for ri, (term, kind) in enumerate(rowspec):
        for ci, g in enumerate(GRAMMARS):
            try:
                M[ri, ci] = idx.loc[(split, term, g, kind)]
            except KeyError:
                pass
    return M


def _draw_panel(ax, M, rowspec, vmax, cmap, title):
    im = ax.imshow(M, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(GRAMMARS)))
    ax.set_xticklabels([GRAMMAR_DISPLAY[g] for g in GRAMMARS])
    ax.set_yticks(range(len(rowspec)))
    labels = [f"{TERM_LABEL.get(term, term)}  ·  {kind}"
              for term, kind in rowspec]
    ax.set_yticklabels(labels)
    ax.set_title(title, fontsize=11)
    for ri in range(M.shape[0]):
        for ci in range(M.shape[1]):
            v = M[ri, ci]
            if np.isnan(v):
                continue
            frac = 0.5 + 0.5 * np.clip(v / vmax, -1, 1)
            tc = "white" if abs(frac - 0.5) > 0.32 else "black"
            ax.text(ci, ri, f"{v:.2f}", ha="center", va="center",
                    color=tc, fontsize=8)
    # thin rule above each span row, tying it to its string partner.
    for ri, (_, kind) in enumerate(rowspec):
        if kind == "span":
            ax.axhline(ri - 0.5, color="0.85", lw=0.8)
    return im


def make_figure(long: pd.DataFrame, split: str, vmax: float, out_png):
    left_spec = _row_spec(long, with_span=False)
    right_spec = _row_spec(long, with_span=True)
    ML = _matrix(long, split, left_spec)
    MR = _matrix(long, split, right_spec)
    cmap = plt.get_cmap("RdYlGn")

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7.5),
        gridspec_kw={"width_ratios": [1, 1], "wspace": 0.75})
    _draw_panel(axes[0], ML, left_spec, vmax, cmap,
                "whole-string margin")
    im = _draw_panel(axes[1], MR, right_spec, vmax, cmap,
                     "whole-string + critical-span margin")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("mean margin (nats)  ·  >0 = grammatical preferred")
    savefig(fig, out_png)


def main():
    pairs = load_probe_pairs()
    long = build_margins_long(pairs)
    long.to_csv(OUT / "margins_long.csv", index=False)
    vmax = float(np.nanmax(np.abs(long["margin"].to_numpy())))
    for split in SPLITS:
        make_figure(long, split, vmax,
                    OUT / f"margin_heatmap_{SPLIT_DISPLAY[split]}.png")
    print(f"wrote figures + margins_long.csv to {OUT}")


if __name__ == "__main__":
    main()
