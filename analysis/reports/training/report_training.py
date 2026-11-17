"""Training-side reporting figures for ``set_3_ep12`` -> results/analysis/training/.

Outputs (each figure paired with a backing CSV):

    plateau_table.csv        per grammar x pooling x seed x metric {dev_loss,
                             margin, accuracy}: the plateau/min-onset epoch and
                             context. Descriptive -- no reporting epoch is chosen.
    plateau_summary.csv      per (metric, grammar, pooling): min/median/max
                             plateau epoch across seeds.
    training_panels.png      three stacked panels (dev loss / margin / accuracy)
                             over the full 12 epochs, mean +/- seed band, string
                             solid / span dashed, each metric's own plateau band
                             marked faintly (descriptive; no e=7 line).
    heatmap_term_grammar_final.{png,csv}
                             accuracy and raw margin by term x grammar at the
                             FINAL epoch on the SELECT split.
    grid_panels.png          grid-search context for the selected recipe
                             (medium / lr3e-4 / bs64): the same three curves.

Reuses the plateau machinery and panel layout of
:mod:`analysis.reports.training.selection_epoch_ep12` unchanged; only the output location differs.

Run:  python -m analysis.reports.training.report_training
"""
from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import numpy as np

from analysis.category_labels import relabel
from analysis.report_common import (
    GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY, POOLINGS, POOL_STYLE,
    TERM_ORDER, grammar_handles, load_select_terms_final, outdir, savefig,
)
from analysis import selection_epoch as se
from analysis.reports.training import selection_epoch_ep12 as se12

OUT = outdir("training")


# --- plateau table + panels (reuse the selection machinery) ----------------

def training_curves_and_table():
    paths = sorted(glob.glob(str(se12.CKPT_ROOT / "*" / "probe_curve.parquet")))
    if not paths:
        raise SystemExit(f"no training probe curves under {se12.CKPT_ROOT}")
    curves = se.pooling_curves(se12._terms_from_scores(paths))
    devloss = se12.train_devloss()
    table = se12.plateau_table(curves, devloss)
    summary = se12.plateau_summary(table)
    return curves, devloss, table, summary


# --- final-epoch term x grammar heatmaps -----------------------------------

def heatmap_term_grammar(terms_final, out_png):
    """Two heatmaps (raw margin, accuracy) over term x grammar at the final epoch."""
    present = [t for t in TERM_ORDER if t in set(terms_final["term"])]
    piv_m = (terms_final.pivot_table(index="term", columns="grammar",
                                     values="margin", observed=True)
             .reindex(index=present, columns=GRAMMARS))
    piv_a = (terms_final.pivot_table(index="term", columns="grammar",
                                     values="acc", observed=True)
             .reindex(index=present, columns=GRAMMARS))

    fig, axes = plt.subplots(1, 2, figsize=(11, 6),
                             gridspec_kw={"wspace": 0.35})
    xlabels = [GRAMMAR_DISPLAY[g] for g in GRAMMARS]
    ylabels = [relabel(t) for t in present]

    for ax, piv, title, cmap, fmt, vlims in (
            (axes[0], piv_m, "raw margin (nats)", "viridis", "{:.1f}", (None, None)),
            (axes[1], piv_a, "pair accuracy", "RdYlGn", "{:.2f}", (0.0, 1.0))):
        data = piv.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vlims[0], vmax=vlims[1])
        ax.set_xticks(range(len(GRAMMARS)), xlabels)
        ax.set_yticks(range(len(present)), ylabels, fontsize=8)
        ax.set_title(title, fontsize=11)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if np.isnan(v):
                    continue
                lum = 0.5 if cmap == "viridis" else 0.75
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                        color="white" if (cmap == "viridis" and v < np.nanmedian(data))
                        else "black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    savefig(fig, out_png)


# --- grid-search context panels --------------------------------------------

def grid_panels(out_png):
    grid_paths = sorted(glob.glob(se.GRID_PROBE_GLOB))
    if not grid_paths:
        print(f"[grid] no grid probe scores matched {se.GRID_PROBE_GLOB}; skipping")
        return
    curves = se.pooling_curves(se._terms_from_scores(grid_paths))
    devloss = se.grid_devloss()

    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    xmax = float(max(devloss["epoch"].max(), curves["epoch"].max()))

    ax = axes[0]
    for g in GRAMMARS:
        sub = devloss[devloss.grammar == g]
        if sub.empty:
            continue
        mean, lo, hi = se._mean_band(sub, "dev_loss")
        ax.plot(mean.index, mean.values, color=GRAMMAR_COLOR[g], label=GRAMMAR_DISPLAY[g])
        ax.fill_between(mean.index, lo.values, hi.values, color=GRAMMAR_COLOR[g],
                        alpha=0.12, linewidth=0)
    ax.set_ylabel("dev loss (nats/token)")
    ax.set_title("Validation loss")
    ax.legend(handles=grammar_handles(), title="grammar", fontsize=8, ncol=4,
              loc="upper right")

    for ax, col, lab in ((axes[1], "margin", "raw margin (nats)"),
                         (axes[2], "accuracy", "pair accuracy")):
        for g in GRAMMARS:
            for pool in POOLINGS:
                sub = curves[(curves.grammar == g) & (curves.pooling == pool)]
                if sub.empty:
                    continue
                mean, lo, hi = se._mean_band(sub, col)
                ax.plot(mean.index, mean.values, color=GRAMMAR_COLOR[g],
                        linestyle=POOL_STYLE[pool], linewidth=1.6)
                ax.fill_between(mean.index, lo.values, hi.values,
                                color=GRAMMAR_COLOR[g], alpha=0.08, linewidth=0)
        ax.set_ylabel(lab)
        ax.set_title("Margin (mean over pooling's terms)" if col == "margin"
                     else "Accuracy (mean over pooling's terms)")
    axes[2].legend(handles=[plt.Line2D([], [], color="black", linestyle=POOL_STYLE[p],
                                       label=p) for p in POOLINGS],
                   title="pooling", fontsize=8, loc="lower right")
    for ax in axes:
        ax.set_xlim(1, xmax)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    curves, devloss, table, summary = training_curves_and_table()
    table.to_csv(OUT / "plateau_table.csv", index=False)
    summary.to_csv(OUT / "plateau_summary.csv", index=False)
    curves.to_csv(OUT / "curves_by_seed.csv", index=False)
    se12.make_panels(curves, devloss, table, OUT / "training_panels.png",
                     draw_suptitle=False)

    terms_final = load_select_terms_final()
    terms_final.to_csv(OUT / "heatmap_term_grammar_final.csv", index=False)
    heatmap_term_grammar(terms_final, OUT / "heatmap_term_grammar_final.png")

    grid_panels(OUT / "grid_panels.png")

    print(f"[training] wrote plateau table/summary, training_panels.png, "
          f"heatmap_term_grammar_final.png, grid_panels.png -> {OUT}")
    for metric in ("dev_loss", "margin", "accuracy"):
        lo, med, hi = se12._metric_band(table, metric)
        print(f"  plateau {metric:9s}: ep {lo:.0f}-{hi:.0f} (median {med:.0f})")


if __name__ == "__main__":
    main()
