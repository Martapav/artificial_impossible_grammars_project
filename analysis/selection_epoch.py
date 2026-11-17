"""Choose a principled training epoch cap from grid-search and training curves.

The final `set_3` models were trained with a hard 30-epoch cap and no early
stopping, while the grid search (same recipe) early-stopped on dev loss at
15-23 epochs. This script produces the evidence to pick a correct cap, as two
mirrored sets of outputs under ``results/analysis/selection/``:

    grid/      grid search, restricted to the selected recipe
               (medium / lr 3e-4 / bs 64, seeds s100, s101)
    training/  the set_3 training runs (seeds s0..s4)

Each set contains:

    plateau_table.csv    per (grammar, pooling, seed) x metric {dev_loss,
                         margin, accuracy}: the plateau/onset epoch and context.
    panels.png           three stacked panels -- dev-loss, margin, accuracy --
                         with the recommended reporting epoch marked.
    reporting_epoch.txt  the chosen reporting epoch, the reduction that produced
                         it, the dev-loss overfit onset, and the conflict flag.

Metrics, per epoch:
    dev loss   LM validation loss (grid `eval_curve` / training `metrics.csv`).
    margin     mean pair margin over the terms in each pooling.
    accuracy   mean pair accuracy over the terms in each pooling.

Pooling = the per-term ``criterion`` assigned in :mod:`evaluation.aggregate`
(``string`` vs ``span``); the two are kept separate because their margins are in
different units.

Plateau rule (confirmed with the user): on a lightly smoothed curve (centered
rolling mean, window ``SMOOTH``), the plateau *onset* is the first epoch that
enters a tolerance band of the curve's optimum and stays inside it for ``HOLD``
epochs. ``TOL_FRAC`` sets the band as a fraction of each curve's own range.

Reporting epoch, per set:
    min( max over (grammar, seed, pooling) of plateau(margin, accuracy),
         earliest dev-loss overfit onset ),
flagging the case where the probe metrics are still climbing after the loss has
turned up.

Margin note. ``margin_corrected`` (= margin - margin_b1) is *not* applied. The
only n-gram baseline available locally covers the held-out IOD/OOD splits; these
per-epoch curves live on the ``structural_probe_SELECT`` split, which has no
local baseline. This changes nothing for epoch selection: ``margin_b1`` is
identically 0 on every string-criterion term and an epoch-independent constant
on the span-criterion terms, so the correction shifts the span margin panel's
*level* but moves no plateau epoch. The panels are labelled "raw margin".

Run:  python -m analysis.selection_epoch
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation.aggregate import to_pairs, term_table

# --- configuration ---------------------------------------------------------

REPO = Path(__file__).resolve().parent.parent
OUT_ROOT = REPO / "results" / "analysis" / "selection"

# The recipe the grid search selected, and which set_3 was trained with.
SELECTED_RECIPE = "medium__lr3e-4__bs64"

GRID_PROBE_GLOB = str(
    REPO / "results/grid96/set_2/probe_scores"
    / f"*__{SELECTED_RECIPE}__s*.parquet")
GRID_RESULTS_JSONL = REPO / "results/grid96/set_2/results.jsonl"
TRAIN_CKPT_ROOT = REPO / "models/checkpoints/set_3"

GRAMMARS = ["H", "Hprime", "Lprime", "P"]
# Display: L' is shown as P' (display-only rename); see memory / metadata.
GRAMMAR_DISPLAY = {"H": "H", "Hprime": "H′", "Lprime": "P′", "P": "P"}
GRAMMAR_COLOR = {"H": "#1b9e77", "Hprime": "#d95f02",
                 "Lprime": "#7570b3", "P": "#e7298a"}
POOLINGS = ["string", "span"]
POOL_STYLE = {"string": "-", "span": "--"}

# Plateau-detection parameters (confirmed).
SMOOTH = 3        # centered rolling-mean window
HOLD = 3          # epochs the band must be held (k)
TOL_FRAC = 0.01   # tolerance band = TOL_FRAC * curve range

SEED_RE = re.compile(r"s(\d+)$")


# --- plateau detection -----------------------------------------------------

def _smooth(y: np.ndarray) -> np.ndarray:
    """Centered rolling mean, window SMOOTH, shrinking at the edges."""
    s = pd.Series(y).rolling(SMOOTH, center=True, min_periods=1).mean()
    return s.to_numpy()


def plateau_onset(epochs: np.ndarray, values: np.ndarray, *, kind: str) -> float:
    """First epoch that reaches, and holds, the tolerance band of the optimum.

    ``kind='max'`` for margin/accuracy (optimum = running max), ``kind='min'``
    for dev loss (optimum = running min). Returns the onset epoch; on a curve
    too short or noisy to satisfy the hold, falls back to the arg-optimum epoch.
    """
    order = np.argsort(epochs)
    ep, y = epochs[order], _smooth(values[order])
    n = len(ep)
    if n == 0:
        return float("nan")
    rng = float(np.nanmax(y) - np.nanmin(y))
    tol = TOL_FRAC * rng if rng > 0 else 0.0
    if kind == "max":
        opt = float(np.nanmax(y))
        in_band = y >= opt - tol
    else:
        opt = float(np.nanmin(y))
        in_band = y <= opt + tol
    for i in range(n):
        if not in_band[i]:
            continue
        hold = in_band[i:min(i + HOLD, n)]
        if hold.all():
            return float(ep[i])
    return float(ep[int(np.nanargmax(y) if kind == "max" else np.nanargmin(y))])


def overfit_onset(epochs: np.ndarray, devloss: np.ndarray) -> float:
    """First sustained rise of dev loss above its running minimum.

    Returns the epoch where the smoothed dev loss first exceeds
    ``min + tol`` and stays above it for HOLD epochs. If the loss never rises
    that way within the observed range, returns the last epoch (no overfitting
    signal seen).
    """
    order = np.argsort(epochs)
    ep, y = epochs[order], _smooth(devloss[order])
    n = len(ep)
    rng = float(np.nanmax(y) - np.nanmin(y))
    tol = TOL_FRAC * rng if rng > 0 else 0.0
    i_min = int(np.nanargmin(y))
    floor = float(y[i_min])
    above = y > floor + tol
    for i in range(i_min + 1, n):
        if above[i] and above[i:min(i + HOLD, n)].all():
            return float(ep[i])
    return float(ep[-1])


# --- data loading ----------------------------------------------------------

def _seed_of(run_id: str) -> int:
    m = SEED_RE.search(run_id)
    if not m:
        raise ValueError(f"cannot parse seed from run_id {run_id!r}")
    return int(m.group(1))


def _terms_from_scores(paths: list[str]) -> pd.DataFrame:
    """Per (run, epoch, grammar, term, criterion) margins/acc, over score shards.

    No baselines: the curves live on the SELECT split (see the module note), so
    `margin` here is the raw pair margin. `criterion` is the pooling.
    """
    frames = []
    for p in paths:
        pairs = to_pairs(pd.read_parquet(p))
        frames.append(term_table(pairs))
    terms = pd.concat(frames, ignore_index=True)
    terms["seed"] = terms["run_id"].map(_seed_of)
    return terms


def pooling_curves(terms: pd.DataFrame) -> pd.DataFrame:
    """Reduce the per-term table to one margin/acc per (grammar, seed, epoch, pooling)."""
    g = (terms.groupby(["grammar", "seed", "epoch", "criterion"], observed=True)
         .agg(margin=("margin", "mean"), accuracy=("acc", "mean"))
         .reset_index()
         .rename(columns={"criterion": "pooling"}))
    return g


def grid_devloss() -> pd.DataFrame:
    """Per (grammar, seed, epoch) dev loss for the selected recipe, from the grid."""
    rows = []
    for line in GRID_RESULTS_JSONL.read_text().splitlines():
        r = json.loads(line)
        if r.get("run_id", "").split("__", 1)[-1].rsplit("__s", 1)[0] != SELECTED_RECIPE:
            continue
        seed = _seed_of(r["run_id"])
        for pt in r.get("eval_curve", []):
            rows.append((r["grammar"], seed, float(pt["epoch"]), float(pt["eval_loss"])))
    return pd.DataFrame(rows, columns=["grammar", "seed", "epoch", "dev_loss"])


def train_devloss() -> pd.DataFrame:
    """Per (grammar, seed, epoch) dev loss for set_3, from each run's metrics.csv."""
    rows = []
    for d in sorted(TRAIN_CKPT_ROOT.glob("*/metrics.csv")):
        run_id = d.parent.name          # e.g. "P__s0"
        grammar, seed = run_id.split("__")[0], _seed_of(run_id)
        m = pd.read_csv(d)
        ev = m[m["eval_loss"].notna()][["epoch", "eval_loss"]]
        ev = ev.groupby("epoch", as_index=False)["eval_loss"].first()
        for _, rr in ev.iterrows():
            rows.append((grammar, seed, float(rr["epoch"]), float(rr["eval_loss"])))
    return pd.DataFrame(rows, columns=["grammar", "seed", "epoch", "dev_loss"])


# --- per-set analysis ------------------------------------------------------

def plateau_table(curves: pd.DataFrame, devloss: pd.DataFrame) -> pd.DataFrame:
    """One row per (grammar, pooling, seed) x metric with its plateau epoch."""
    rows = []
    for (grammar, seed, pool), sub in curves.groupby(["grammar", "seed", "pooling"],
                                                      observed=True):
        sub = sub.sort_values("epoch")
        ep = sub["epoch"].to_numpy()
        for metric in ("margin", "accuracy"):
            y = sub[metric].to_numpy()
            i_opt = int(np.nanargmax(y))
            rows.append(dict(
                grammar=grammar, pooling=pool, seed=seed, metric=metric,
                plateau_epoch=plateau_onset(ep, y, kind="max"),
                last_epoch=float(ep.max()), best_epoch=float(ep[i_opt]),
                best_value=float(y[i_opt]),
                value_at_plateau=float(
                    y[np.argmin(np.abs(ep - plateau_onset(ep, y, kind="max")))]),
            ))
    for (grammar, seed), sub in devloss.groupby(["grammar", "seed"], observed=True):
        sub = sub.sort_values("epoch")
        ep, y = sub["epoch"].to_numpy(), sub["dev_loss"].to_numpy()
        i_opt = int(np.nanargmin(y))
        rows.append(dict(
            grammar=grammar, pooling="-", seed=seed, metric="dev_loss",
            plateau_epoch=plateau_onset(ep, y, kind="min"),
            last_epoch=float(ep.max()), best_epoch=float(ep[i_opt]),
            best_value=float(y[i_opt]),
            value_at_plateau=float(
                y[np.argmin(np.abs(ep - plateau_onset(ep, y, kind="min")))]),
        ))
    df = pd.DataFrame(rows)
    order = {g: i for i, g in enumerate(GRAMMARS)}
    return df.sort_values(
        ["metric", "grammar", "pooling", "seed"],
        key=lambda s: s.map(order) if s.name == "grammar" else s
    ).reset_index(drop=True)


def reporting_epoch(table: pd.DataFrame, devloss: pd.DataFrame) -> dict:
    """Latest probe plateau across grammars/seeds/poolings, capped below overfit onset."""
    probe = table[table["metric"].isin(("margin", "accuracy"))]
    probe_plateau_max = float(probe["plateau_epoch"].max())
    onsets = [overfit_onset(sub["epoch"].to_numpy(), sub["dev_loss"].to_numpy())
              for _, sub in devloss.groupby(["grammar", "seed"], observed=True)]
    earliest_overfit = float(min(onsets))
    report = min(probe_plateau_max, earliest_overfit)
    return dict(reporting_epoch=report,
               probe_plateau_max=probe_plateau_max,
               earliest_overfit_onset=earliest_overfit,
               conflict=bool(probe_plateau_max > earliest_overfit))


def _mean_band(sub: pd.DataFrame, col: str):
    g = sub.groupby("epoch")[col]
    return g.mean(), g.min(), g.max()


def make_panels(curves: pd.DataFrame, devloss: pd.DataFrame, report: dict,
                title: str, out_png: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True)
    rep_ep = report["reporting_epoch"]
    overfit = report["earliest_overfit_onset"]
    xmax = float(max(devloss["epoch"].max(), curves["epoch"].max()))

    # Panel 1: dev loss (one line per grammar).
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

    # Panels 2-3: margin / accuracy, string solid vs span dashed.
    for ax, col, lab in ((axes[1], "margin", "raw margin (nats)"),
                         (axes[2], "accuracy", "pair accuracy")):
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

    # Pooling legend on the accuracy panel.
    style_handles = [plt.Line2D([], [], color="black", linestyle=POOL_STYLE[p],
                                label=p) for p in POOLINGS]
    axes[2].legend(handles=style_handles, title="pooling", fontsize=8,
                   loc="lower right")

    for ax in axes:
        ax.axvline(rep_ep, color="black", linestyle=":", linewidth=1.4)
        ax.annotate(f"report @ {rep_ep:.0f}", xy=(rep_ep, ax.get_ylim()[1]),
                    xytext=(3, -10), textcoords="offset points",
                    fontsize=8, color="black", va="top")
        if overfit < xmax:
            ax.axvspan(overfit, xmax, color="grey", alpha=0.08, linewidth=0)
        ax.set_xlim(1, xmax)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("epoch")

    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def write_report(report: dict, title: str, out_txt: Path,
                 extra: str = "") -> None:
    lines = [
        title,
        "=" * len(title),
        "",
        f"reporting_epoch          : {report['reporting_epoch']:.0f}",
        f"  = min( probe_plateau_max , earliest_overfit_onset )",
        f"probe_plateau_max        : {report['probe_plateau_max']:.0f}"
        "   (latest margin/accuracy plateau over grammars x seeds x poolings)",
        f"earliest_overfit_onset   : {report['earliest_overfit_onset']:.0f}"
        "   (first sustained dev-loss rise, earliest grammar/seed)",
        f"conflict                 : {report['conflict']}"
        "   (probe metrics still climbing after the loss turned up)",
        "",
    ]
    if extra:
        lines += [extra, ""]
    out_txt.write_text("\n".join(lines))


def run_set(name: str, title: str, terms: pd.DataFrame,
            devloss: pd.DataFrame, extra: str = "") -> dict:
    outdir = OUT_ROOT / name
    outdir.mkdir(parents=True, exist_ok=True)
    curves = pooling_curves(terms)
    table = plateau_table(curves, devloss)
    report = reporting_epoch(table, devloss)
    table.to_csv(outdir / "plateau_table.csv", index=False)
    make_panels(curves, devloss, report, title, outdir / "panels.png")
    write_report(report, title, outdir / "reporting_epoch.txt", extra=extra)
    curves.to_csv(outdir / "curves_by_seed.csv", index=False)
    print(f"[{name}] reporting epoch = {report['reporting_epoch']:.0f} "
          f"(probe plateau {report['probe_plateau_max']:.0f}, "
          f"overfit onset {report['earliest_overfit_onset']:.0f}, "
          f"conflict={report['conflict']}) -> {outdir}")
    return report


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    margin_note = (
        "Margin panels show RAW margin (margin_corrected unavailable on the "
        "SELECT split locally); the b1 correction is epoch-independent and "
        "moves no plateau epoch -- see the module docstring.")

    # --- grid search (selected recipe only) ---
    grid_paths = sorted(glob.glob(GRID_PROBE_GLOB))
    if not grid_paths:
        raise SystemExit(f"no grid probe scores matched {GRID_PROBE_GLOB}")
    grid_terms = _terms_from_scores(grid_paths)
    grid_dl = grid_devloss()
    grid_rep = run_set(
        "grid", f"Grid search -- {SELECTED_RECIPE} (early-stopped, patience=5)",
        grid_terms, grid_dl, extra=margin_note)

    # --- training (set_3) ---
    train_paths = sorted(glob.glob(str(TRAIN_CKPT_ROOT / "*" / "probe_curve.parquet")))
    if not train_paths:
        raise SystemExit(f"no training probe curves under {TRAIN_CKPT_ROOT}")
    train_terms = _terms_from_scores(train_paths)
    train_dl = train_devloss()
    train_rep = run_set(
        "training", "Training -- set_3 (fixed 30-epoch cap, no early stopping)",
        train_terms, train_dl, extra=margin_note)

    print("\nCross-check: the grid runs for this recipe early-stopped naturally "
          "at 15-23 epochs (patience=5 on dev loss).")
    print(f"Recommended cap (training set) : {train_rep['reporting_epoch']:.0f}")
    print(f"Grid-set reporting epoch       : {grid_rep['reporting_epoch']:.0f}")


if __name__ == "__main__":
    main()
