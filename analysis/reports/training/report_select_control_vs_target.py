"""Controls vs. target phenomena over training, on the SELECT probe.

Learning curves on the held-in SELECT split (the only split scored per epoch,
from each run's ``probe_curve.parquet``), contrasting two bins:

    control  -- the two neutral gate terms (neutral_agreement, neutral_positional)
    target   -- the eight objective terms (aux, wh_i, wh_ii, refl:*, pron:*)

Two panels show how the model's *accuracy* (does it prefer the grammatical
member) and its *margin* (how strongly -- its confidence, in nats) evolve on each
bin across epochs 1..12, pooled over the 20 runs (4 grammars x 5 seeds).

An appendix table (``select_control_vs_target_auc_by_grammar.{csv,tex}``)
summarises each per-grammar accuracy trajectory by the area under it, so the
whole training window is reported by one number per curve rather than only its
endpoint.

Caveat on the margin panel: the target bin mixes span-criterion and whole-string
criterion terms, whose margins are sums over different numbers of tokens, so the
absolute nats gap between the bins is partly a criterion/length artefact. The
informative content is the SHAPE of each curve over epochs; accuracy (unitless)
is the strictly comparable metric.

Run:  python -m analysis.reports.training.report_select_control_vs_target
"""
from __future__ import annotations

import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    CKPT_ROOT, GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY, _seed_of, outdir, savefig,
)
from evaluation.aggregate import GATE_TERMS, to_pairs

OUT = outdir("evaluation", "structural_probes")

BIN_COLOR = {"control": "#7f7f7f", "target": "#1b7837"}
BIN_LABEL = {"control": "controls (neutral gates)", "target": "target phenomena"}
BIN_ORDER = ["control", "target"]


def load_select_pairs() -> pd.DataFrame:
    """Every run's per-epoch SELECT pairs, tagged control/target."""
    frames = []
    for path in sorted(glob.glob(str(CKPT_ROOT / "*" / "probe_curve.parquet"))):
        pairs = to_pairs(pd.read_parquet(path))
        pairs["bin"] = np.where(pairs["term"].isin(GATE_TERMS), "control", "target")
        frames.append(pairs[["run_id", "grammar", "epoch", "bin", "criterion",
                             "acc", "margin"]])
    return pd.concat(frames, ignore_index=True)


def _curve(per_run: pd.DataFrame, value: str, b: str):
    """(epochs, mean, lo, hi) across runs for one bin, one metric."""
    sub = per_run[per_run["bin"] == b]
    g = sub.groupby("epoch")[value].agg(mean="mean", lo="min", hi="max").sort_index()
    return g.index.to_numpy(), g["mean"].to_numpy(), g["lo"].to_numpy(), g["hi"].to_numpy()


def figure(pairs: pd.DataFrame, out_png, out_csv) -> None:
    # Each run weighs equally: reduce to one (run, epoch, bin) mean first.
    per_run = (pairs.groupby(["run_id", "grammar", "epoch", "bin"], observed=True)
               [["acc", "margin"]].mean().reset_index())

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharex=True)
    specs = [("acc", "pair accuracy", axes[0], 0.5, (0, 1.02)),
             ("margin", "within-pair margin (nats)", axes[1], 0.0, None)]
    rows = []
    for value, ylabel, ax, ref, ylim in specs:
        for b in BIN_ORDER:
            x, mean, lo, hi = _curve(per_run, value, b)
            ax.plot(x, mean, color=BIN_COLOR[b], lw=2.2, marker="o", ms=4,
                    label=BIN_LABEL[b])
            ax.fill_between(x, lo, hi, color=BIN_COLOR[b], alpha=0.15, lw=0)
            for xi, m, l, h in zip(x, mean, lo, hi):
                rows.append(dict(metric=value, bin=b, epoch=xi, mean=m,
                                 run_min=l, run_max=h))
        ax.axhline(ref, color="0.4", lw=0.9, ls=":")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("epoch")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# Target split by native criterion, plus the controls as a reference ceiling.
# Colours are fixed across facets so span / string read the same in every panel.
GROUP_STYLE = {
    "control":       dict(color="#7f7f7f", ls="-",  marker="o", label="controls (neutral gates)"),
    "target-string": dict(color="#762a83", ls="-",  marker="s", label="target · string criterion (relocation)"),
    "target-span":   dict(color="#1b7837", ls="--", marker="o", label="target · span criterion (in-place subst.)"),
}
GROUP_ORDER = ["control", "target-string", "target-span"]


def figure_per_grammar_accuracy(pairs: pd.DataFrame, out_png, out_csv) -> None:
    """Accuracy only, one panel per grammar (no pooling), target split by criterion.

    Three curves per facet: the controls (grey), and the target phenomena split
    into their string-criterion terms (relocations: aux, wh, refl/pron wrong-site)
    and their span-criterion terms (in-place substitutions: refl/pron wrong-item,
    refl no-substitution). Accuracy is criterion-native throughout, so each target
    curve uses the statistic that is well posed for its own terms. Band = the
    across-seed (5 seeds) min..max within the grammar.
    """
    p = pairs.copy()
    p["group"] = np.where(p["bin"] == "control", "control",
                          "target-" + p["criterion"].astype(str))
    per_run = (p.groupby(["run_id", "grammar", "epoch", "group"], observed=True)
               ["acc"].mean().reset_index())

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True, sharey=True)
    rows = []
    for gi, g in enumerate(GRAMMARS):
        ax = axes[gi // 2][gi % 2]
        sub = per_run[per_run["grammar"] == g]
        for grp in GROUP_ORDER:
            st = GROUP_STYLE[grp]
            bb = sub[sub["group"] == grp]
            stat = bb.groupby("epoch")["acc"].agg(mean="mean", lo="min", hi="max").sort_index()
            x = stat.index.to_numpy()
            ax.plot(x, stat["mean"].to_numpy(), color=st["color"], lw=2.2,
                    ls=st["ls"], marker=st["marker"], ms=4, label=st["label"])
            ax.fill_between(x, stat["lo"].to_numpy(), stat["hi"].to_numpy(),
                            color=st["color"], alpha=0.13, lw=0)
            for xi, m, l, h in zip(x, stat["mean"], stat["lo"], stat["hi"]):
                rows.append(dict(grammar=g, group=grp, epoch=xi, mean=m,
                                 seed_min=l, seed_max=h))
        ax.axhline(0.5, color="0.4", lw=0.9, ls=":")
        ax.set_title(GRAMMAR_DISPLAY[g], fontsize=12)
        ax.set_ylim(0.45, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="lower right")
    for r in range(2):
        axes[r][0].set_ylabel("pair accuracy")
    for c in range(2):
        axes[1][c].set_xlabel("epoch")
    fig.tight_layout()
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# --- AUC of the per-grammar accuracy trajectories (appendix table) ---------

GRAMMAR_TEX = {"H": "H", "Hprime": "H$'$", "P": "P", "Lprime": "P$'$"}
GROUP_TEX = {
    "control": "controls (neutral gates)",
    "target-string": "target $\\cdot$ string crit.",
    "target-span": "target $\\cdot$ span crit.",
}


def _auc_normalised(epochs: np.ndarray, acc: np.ndarray) -> float:
    """Trapezoidal area under an accuracy curve, divided by the epoch span.

    The result lives on the accuracy scale: it is the time-average of the curve
    over epochs 1..12, so it is directly comparable to a final-epoch accuracy and
    does not depend on the length of the training window.
    """
    order = np.argsort(epochs)
    e, a = epochs[order], acc[order]
    return float(np.trapezoid(a, e) / (e[-1] - e[0]))


def table_auc_by_grammar(pairs: pd.DataFrame, out_csv, out_tex) -> None:
    """Area under each accuracy trajectory of ``figure_per_grammar_accuracy``.

    One row per grammar x curve group (the same three groups the figure draws).
    The AUC is computed **per seed** -- trapezoid over that seed's own epoch
    1..12 curve, normalised by the 11-epoch span -- and only then pooled over the
    5 seeds, so the spread is across-seed (model-level) and not an item pool
    (see the seed-level uncertainty rule). Two readings of the same area:

        auc_mean          time-averaged pair accuracy over training, in [0, 1]
        auc_chance_corr   (auc - 0.5) / 0.5 -- 0 = at chance throughout,
                          1 = perfect from epoch 1; a learning-speed reading

    ``*_min`` / ``*_max`` are the across-seed range, matching the band the figure
    shades.
    """
    p = pairs.copy()
    p["group"] = np.where(p["bin"] == "control", "control",
                          "target-" + p["criterion"].astype(str))
    # Same reduction as the figure: one accuracy per (run, epoch, group).
    per_run = (p.groupby(["run_id", "grammar", "epoch", "group"], observed=True)
               ["acc"].mean().reset_index())

    rows = []
    for g in GRAMMARS:
        for grp in GROUP_ORDER:
            sub = per_run[(per_run["grammar"] == g) & (per_run["group"] == grp)]
            per_seed = {}
            for run_id, r in sub.groupby("run_id", observed=True):
                per_seed[_seed_of(run_id)] = _auc_normalised(
                    r["epoch"].to_numpy(), r["acc"].to_numpy())
            aucs = np.array([per_seed[s] for s in sorted(per_seed)])
            epochs = np.sort(sub["epoch"].unique())
            final = (sub[sub["epoch"] == epochs[-1]]
                     .groupby("run_id", observed=True)["acc"].mean().mean())
            rows.append(dict(
                grammar=g, grammar_display=GRAMMAR_DISPLAY[g], group=grp,
                group_label=GROUP_STYLE[grp]["label"],
                n_seeds=len(aucs),
                epoch_first=epochs[0], epoch_last=epochs[-1],
                auc_mean=aucs.mean(), auc_min=aucs.min(), auc_max=aucs.max(),
                auc_chance_corr=(aucs.mean() - 0.5) / 0.5,
                auc_chance_corr_min=(aucs.min() - 0.5) / 0.5,
                auc_chance_corr_max=(aucs.max() - 0.5) / 0.5,
                acc_final=final,
            ))
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    e0, e1 = int(df.epoch_first.iloc[0]), int(df.epoch_last.iloc[0])
    lines = [
        "% Auto-generated by analysis/reports/training/report_select_control_vs_target.py"
        " -- do not edit.",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        r"  \begin{tabular}{llccc}", r"    \toprule",
        r"    & & \multicolumn{2}{c}{AUC (mean [min, max] over seeds)} & \\",
        r"    \cmidrule(lr){3-4}",
        r"    Gr. & Curve & normalised & chance-corrected & acc.\ at ep.\ "
        + f"${e1}$" + r" \\",
        r"    \midrule",
    ]
    for i, (g, sub) in enumerate(df.groupby("grammar", sort=False)):
        if i:
            lines.append(r"    \addlinespace")
        for j, (_, r) in enumerate(sub.iterrows()):
            first = GRAMMAR_TEX[r.grammar] if j == 0 else ""
            lines.append(
                f"    {first:6} & {GROUP_TEX[r.group]:28} "
                f"& ${r.auc_mean:.3f}$ $[{r.auc_min:.3f}, {r.auc_max:.3f}]$ "
                f"& ${r.auc_chance_corr:.3f}$ "
                f"$[{r.auc_chance_corr_min:.3f}, {r.auc_chance_corr_max:.3f}]$ "
                f"& ${r.acc_final:.3f}$ \\\\")
    lines += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{Area under the SELECT accuracy trajectories of"
        r" Figure~\ref{fig:select-control-vs-target-by-grammar}, one row per"
        r" grammar and curve. The area is the trapezoidal integral over epochs $"
        + f"{e0}$--${e1}$" + r" of a single seed's curve, divided by the epoch"
        r" span, so it is a time-averaged pair accuracy in $[0,1]$: it says how"
        r" much of the training window the model spent already solving the bin,"
        r" not merely where it ended (last column). The chance-corrected column"
        r" rescales it as $(\mathrm{AUC} - 0.5)/0.5$, so $0$ is chance"
        r" throughout and $1$ is perfect from the first epoch. Areas are"
        r" computed per seed and pooled afterwards; brackets give the across-seed"
        r" ($5$ seeds) range, the band shaded in the figure.}",
        r"  \label{tab:select-control-vs-target-auc}", r"\end{table}",
    ]
    out_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    pairs = load_select_pairs()
    figure(pairs, OUT / "select_control_vs_target.png",
           OUT / "select_control_vs_target.csv")
    figure_per_grammar_accuracy(
        pairs, OUT / "select_control_vs_target_accuracy_by_grammar.png",
        OUT / "select_control_vs_target_accuracy_by_grammar.csv")
    table_auc_by_grammar(
        pairs, OUT / "select_control_vs_target_auc_by_grammar.csv",
        OUT / "select_control_vs_target_auc_by_grammar.tex")
    print(f"[probes] wrote control-vs-target SELECT learning curves "
          f"(pooled + per-grammar accuracy) and the per-grammar AUC table "
          f"-> {OUT}")


if __name__ == "__main__":
    main()
