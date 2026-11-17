"""Reporting-epoch evidence for the 12-epoch-cap retrain (``set_3_ep12``).

The first ``set_3`` models were trained with a hard 30-epoch cap and overfit
(see :mod:`analysis.selection_epoch`). This retrain pins the warmup to the grid
schedule and caps at 12 epochs. This script lays out the evidence to pick the
epoch whose weights to freeze for the held-out test -- but, unlike the 30-cap
script, it does **not** impose a single reporting-epoch rule. Each metric's own
plateau/min is reported and drawn; the choice is left to the reader.

Reads, per run (4 grammars x 5 seeds) under ``models/checkpoints/set_3_ep12/``:

    metrics.csv          per-epoch LM validation loss (dev loss).
    probe_curve.parquet  per-epoch pairs on the ``structural_probe_SELECT``
                         split -> margin / accuracy per term and pooling.

Writes to ``results/analysis/selection/training_ep12/``:

    plateau_table.csv    one row per (grammar, pooling, seed) x metric
                         {dev_loss, margin, accuracy}: the plateau/min-onset
                         epoch and context. dev_loss has pooling '-'.
    plateau_summary.csv  per (metric, grammar, pooling): min / median / max
                         plateau epoch across seeds -- the decision aid.
    panels.png           three stacked panels (dev loss, margin, accuracy) over
                         the full 12 epochs, each marking that metric's own
                         plateau band (min..max across runs, line at median).
    curves_by_seed.csv   the backing per-(grammar, seed, epoch, pooling) curves.

Metric definitions, plateau rule, pooling, and the raw-margin note are all
inherited unchanged from :mod:`analysis.selection_epoch` (SMOOTH=3, HOLD=3,
TOL_FRAC=0.01). ``margin_corrected`` is not applied for the same reason given
there: on the SELECT split ``margin_b1`` is 0 on string terms and an
epoch-independent constant on span terms, so it moves no plateau epoch.

Run:  python -m analysis.reports.training.selection_epoch_ep12
"""
from __future__ import annotations

import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.selection_epoch import (
    GRAMMARS, GRAMMAR_DISPLAY, GRAMMAR_COLOR, POOLINGS, POOL_STYLE,
    _mean_band, _seed_of, _terms_from_scores, plateau_onset, pooling_curves,
)

REPO = Path(__file__).resolve().parents[3]
CKPT_ROOT = REPO / "models" / "checkpoints" / "set_3_ep12"
OUTDIR = REPO / "results" / "analysis" / "selection" / "training_ep12"

MARGIN_NOTE = (
    "Margin panels show RAW margin (margin_corrected unavailable on the SELECT "
    "split locally); the b1 correction is epoch-independent and moves no plateau "
    "epoch -- see analysis/selection_epoch.py.")

# metric -> optimisation direction for the plateau rule.
METRIC_KIND = {"dev_loss": "min", "margin": "max", "accuracy": "max"}


# --- data loading ----------------------------------------------------------

def train_devloss() -> pd.DataFrame:
    """Per (grammar, seed, epoch) dev loss, from each run's metrics.csv."""
    rows = []
    for d in sorted(CKPT_ROOT.glob("*/metrics.csv")):
        run_id = d.parent.name                      # e.g. "P__s0"
        grammar, seed = run_id.split("__")[0], _seed_of(run_id)
        m = pd.read_csv(d)
        ev = m[m["eval_loss"].notna()][["epoch", "eval_loss"]]
        ev = ev.groupby("epoch", as_index=False)["eval_loss"].first()
        for _, rr in ev.iterrows():
            rows.append((grammar, seed, float(rr["epoch"]), float(rr["eval_loss"])))
    return pd.DataFrame(rows, columns=["grammar", "seed", "epoch", "dev_loss"])


# --- table -----------------------------------------------------------------

def plateau_table(curves: pd.DataFrame, devloss: pd.DataFrame) -> pd.DataFrame:
    """One row per (grammar, pooling, seed) x metric with its plateau epoch.

    margin/accuracy use the max-onset (first sustained approach to the running
    max); dev_loss uses the min-onset (first sustained approach to the running
    min). No cross-metric reporting-epoch rule is applied here.
    """
    rows = []
    for (grammar, seed, pool), sub in curves.groupby(
            ["grammar", "seed", "pooling"], observed=True):
        sub = sub.sort_values("epoch")
        ep = sub["epoch"].to_numpy()
        for metric in ("margin", "accuracy"):
            y = sub[metric].to_numpy()
            onset = plateau_onset(ep, y, kind="max")
            i_opt = int(np.nanargmax(y))
            rows.append(dict(
                grammar=grammar, pooling=pool, seed=seed, metric=metric,
                plateau_epoch=onset, last_epoch=float(ep.max()),
                best_epoch=float(ep[i_opt]), best_value=float(y[i_opt]),
                value_at_plateau=float(y[np.argmin(np.abs(ep - onset))]),
            ))
    for (grammar, seed), sub in devloss.groupby(["grammar", "seed"], observed=True):
        sub = sub.sort_values("epoch")
        ep, y = sub["epoch"].to_numpy(), sub["dev_loss"].to_numpy()
        onset = plateau_onset(ep, y, kind="min")
        i_opt = int(np.nanargmin(y))
        rows.append(dict(
            grammar=grammar, pooling="-", seed=seed, metric="dev_loss",
            plateau_epoch=onset, last_epoch=float(ep.max()),
            best_epoch=float(ep[i_opt]), best_value=float(y[i_opt]),
            value_at_plateau=float(y[np.argmin(np.abs(ep - onset))]),
        ))
    df = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(GRAMMARS)}
    metric_order = {"dev_loss": 0, "margin": 1, "accuracy": 2}
    return df.sort_values(
        ["metric", "grammar", "pooling", "seed"],
        key=lambda s: (s.map(metric_order) if s.name == "metric"
                       else s.map(order) if s.name == "grammar" else s),
    ).reset_index(drop=True)


def plateau_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Per (metric, grammar, pooling): min / median / max plateau across seeds."""
    g = (table.groupby(["metric", "grammar", "pooling"], observed=True)["plateau_epoch"]
         .agg(min_ep="min", median_ep="median", max_ep="max", n_seeds="count")
         .reset_index())
    order = {gr: i for i, gr in enumerate(GRAMMARS)}
    metric_order = {"dev_loss": 0, "margin": 1, "accuracy": 2}
    return g.sort_values(
        ["metric", "grammar", "pooling"],
        key=lambda s: (s.map(metric_order) if s.name == "metric"
                       else s.map(order) if s.name == "grammar" else s),
    ).reset_index(drop=True)


def _metric_band(table: pd.DataFrame, metric: str) -> tuple[float, float, float]:
    """(min, median, max) plateau epoch for a metric, over all runs/poolings."""
    vals = table.loc[table["metric"] == metric, "plateau_epoch"]
    return float(vals.min()), float(vals.median()), float(vals.max())


# --- panels ----------------------------------------------------------------

def _mark_plateau(ax, band: tuple[float, float, float], label: str,
                  vpos: str = "top") -> None:
    lo, med, hi = band
    if hi > lo:
        ax.axvspan(lo, hi, color="black", alpha=0.06, linewidth=0)
    ax.axvline(med, color="black", linestyle=":", linewidth=1.4)
    xlo, xhi = ax.get_xlim()
    right = med > 0.5 * (xlo + xhi)          # median in the right half -> text left
    y = ax.get_ylim()[1] if vpos == "top" else ax.get_ylim()[0]
    ax.annotate(f"{label}: ep {lo:.0f}-{hi:.0f} (med {med:.0f})",
                xy=(med, y), xytext=(-4 if right else 4, -10 if vpos == "top" else 10),
                textcoords="offset points", fontsize=8, color="black",
                ha="right" if right else "left",
                va="top" if vpos == "top" else "bottom")


def make_panels(curves: pd.DataFrame, devloss: pd.DataFrame,
                table: pd.DataFrame, out_png: Path,
                draw_suptitle: bool = True) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    xmax = float(max(devloss["epoch"].max(), curves["epoch"].max()))

    # Panel 1: dev loss (one line per grammar, band = seed min..max).
    ax = axes[0]
    for g in GRAMMARS:
        sub = devloss[devloss.grammar == g]
        if sub.empty:
            continue
        mean, lo, hi = _mean_band(sub, "dev_loss")
        ax.plot(mean.index, mean.values, color=GRAMMAR_COLOR[g],
                label=GRAMMAR_DISPLAY[g])
        ax.fill_between(mean.index, lo.values, hi.values,
                        color=GRAMMAR_COLOR[g], alpha=0.12, linewidth=0)
    ax.set_ylabel("dev loss (nats/token)")
    ax.set_title("Validation loss")
    ax.legend(title="grammar", fontsize=8, ncol=4, loc="upper right")
    _mark_plateau(ax, _metric_band(table, "dev_loss"), "dev-loss min", vpos="bottom")

    # Panels 2-3: margin / accuracy, string solid vs span dashed.
    for ax, col, lab, metric in (
            (axes[1], "margin", "raw margin (nats)", "margin"),
            (axes[2], "accuracy", "pair accuracy", "accuracy")):
        for g in GRAMMARS:
            for pool in POOLINGS:
                sub = curves[(curves.grammar == g) & (curves.pooling == pool)]
                if sub.empty:
                    continue
                mean, lo, hi = _mean_band(sub, col)
                ax.plot(mean.index, mean.values, color=GRAMMAR_COLOR[g],
                        linestyle=POOL_STYLE[pool], linewidth=1.6)
                ax.fill_between(mean.index, lo.values, hi.values,
                                color=GRAMMAR_COLOR[g], alpha=0.08, linewidth=0)
        ax.set_ylabel(lab)
        ax.set_title("Margin (mean over pooling's terms)"
                     if col == "margin" else "Accuracy (mean over pooling's terms)")
        _mark_plateau(ax, _metric_band(table, metric), f"{metric} plateau")

    style_handles = [plt.Line2D([], [], color="black", linestyle=POOL_STYLE[p],
                                label=p) for p in POOLINGS]
    axes[2].legend(handles=style_handles, title="pooling", fontsize=8,
                   loc="lower right")

    for ax in axes:
        ax.set_xlim(1, xmax)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("epoch")

    if draw_suptitle:
        fig.tight_layout(rect=(0, 0, 1, 0.98))
    else:
        fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --- driver ----------------------------------------------------------------

def main() -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)

    paths = sorted(glob.glob(str(CKPT_ROOT / "*" / "probe_curve.parquet")))
    if not paths:
        raise SystemExit(f"no probe curves under {CKPT_ROOT}")
    terms = _terms_from_scores(paths)
    curves = pooling_curves(terms)
    devloss = train_devloss()

    table = plateau_table(curves, devloss)
    summary = plateau_summary(table)

    table.to_csv(OUTDIR / "plateau_table.csv", index=False)
    summary.to_csv(OUTDIR / "plateau_summary.csv", index=False)
    curves.to_csv(OUTDIR / "curves_by_seed.csv", index=False)
    make_panels(curves, devloss, table, OUTDIR / "panels.png")

    print(f"[training_ep12] wrote table / summary / panels -> {OUTDIR}")
    print(MARGIN_NOTE)
    print("\nPlateau epoch (min..max across runs, poolings pooled):")
    for metric in ("dev_loss", "margin", "accuracy"):
        lo, med, hi = _metric_band(table, metric)
        print(f"  {metric:9s}: {lo:.0f}-{hi:.0f}  (median {med:.0f})")


if __name__ == "__main__":
    main()
