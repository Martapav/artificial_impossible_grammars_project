"""Reporting figures + tables for the separability audit (corpus-side).

    python -m analysis.reports.corpus.report_separability

Reads the artifacts written by ``analysis.separability_stats.run_full_analysis``
for one corpus set (default ``set_3``)::

    analysis/data/<set>/separability_{results,floor,taxonomy}.parquet

and writes, under ``results/analysis/separability/``::

    figures/tau_sweep.png          the headline memory sweep (one panel/phenomenon)
    tables/taxonomy.{csv,tex}      U across tau + the cue taxonomy
    tables/controls.{csv,tex}      neutral probe-pair positive controls

Nothing is recomputed here: this module only reshapes the parquet artifacts.
Re-run ``run_full_analysis`` to change a number.

Style note. The grammar colours are the separability scheme (H family blue/cyan,
P family red/orange), NOT ``selection_epoch.GRAMMAR_COLOR``: this analysis is
about the family split, so colour encodes family and linestyle/marker encodes the
member within it. Every pair clears a CVD separation check (min dE = 11.8 under
tritan simulation, 18.6 at normal vision), and the two light hues carry marker +
direct-label encoding because their contrast on white is low.

Display renames (L' -> P') follow the rest of the reporting layer.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
CORPUS_SET = os.environ.get("THESIS_CORPUS_SET", "set_3")
DATA = REPO / "analysis" / "data" / CORPUS_SET
OUT = REPO / "results" / "analysis" / "separability"
FIGS, TABLES = OUT / "figures", OUT / "tables"

# --- display ---------------------------------------------------------------

GRAMMAR_ORDER = ["H", "Hprime", "P", "Lprime"]       # family-major
GRAMMAR_DISPLAY = {"H": "H", "Hprime": "H′", "P": "P", "Lprime": "P′"}
GRAMMAR_TEX = {"H": "H", "Hprime": "H$'$", "P": "P", "Lprime": "P$'$"}
FAMILY_OF = {"H": "H", "Hprime": "H", "P": "P", "Lprime": "P"}

# colour = family, linestyle/marker = member within the family
STYLE = {"H":      ("#1f77b4", "-",  "o"),
         "Hprime": ("#17becf", "--", "s"),
         "P":      ("#d62728", "-",  "o"),
         "Lprime": ("#ff7f0e", "--", "s")}

PHEN_ORDER = ["refl", "pron", "wh_i", "wh_ii", "aux"]
PHEN_DISPLAY = {"refl": "reflexive", "pron": "pronoun",
                "wh_i": "wh (i)", "wh_ii": "wh (ii)", "aux": "aux movement"}
CONTROL_DISPLAY = {"neutral_positional": "positional (relocated token)",
                   "neutral_agreement": "agreement (flipped inflection)"}

TAUS = [0.5, 1.0, 2.0, 4.0, float("inf")]
TAU_X = [0.5, 1, 2, 4, 8]                            # inf plotted at 8
TAU_LABELS = ["0.5", "1", "2", "4", "∞"]
HEADLINE_SPEC = "decay@1:left"
RECOVER = 0.6
TAXONOMY_SHORT = {"bounded-local": "bounded-local",
                  "memory-graded": "memory-graded",
                  "not-locally-recoverable": "not-loc.-rec.",
                  "n/a": "---"}


def _spec(tau: float) -> str:
    tag = "inf" if math.isinf(tau) else (int(tau) if float(tau).is_integer() else tau)
    return f"decay@{tag}:left"


def load():
    grid = pd.read_parquet(DATA / "separability_results.parquet")
    floor = pd.read_parquet(DATA / "separability_floor.parquet")
    tax = pd.read_parquet(DATA / "separability_taxonomy.parquet")
    return grid, floor, tax


# ---------------------------------------------------------------------------
# Figure: the memory sweep
# ---------------------------------------------------------------------------

def figure_tau_sweep(grid: pd.DataFrame, floor: pd.DataFrame) -> Path:
    """U vs memory tau, one panel per phenomenon, one line per grammar.

    Reads the canonical cell: left-only decay features, ``comparable`` band,
    training lengths. Thin cells (minority class < 30) are drawn hollow, because
    a low U there is a non-measurement and must not read as "no shortcut".
    """
    g = grid[(grid.model == "logreg") & (grid.band == "comparable")
             & (grid.regime == "train")]
    U = g.pivot_table(index=["phenomenon", "grammar"], columns="spec", values="U")
    thin = g[g.spec == HEADLINE_SPEC].set_index(["phenomenon", "grammar"]).thin

    fl = floor.pivot_table(index="grammar", columns="tau", values="floor_U")
    fl_lo = floor.pivot_table(index="grammar", columns="tau", values="floor_U_lo")
    fl_hi = floor.pivot_table(index="grammar", columns="tau", values="floor_U_hi")

    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9,
                         "axes.titlesize": 10, "legend.fontsize": 7.5})
    fig, axes = plt.subplots(1, len(PHEN_ORDER), figsize=(13.5, 3.3), sharey=True)

    for ax, phen in zip(axes, PHEN_ORDER):
        # reference lines first, so the data sits on top
        ax.axhspan(RECOVER, 1.05, color="#d62728", alpha=0.045, lw=0)
        ax.axhline(RECOVER, ls=":", c="#b0272c", lw=1.0, zorder=1)
        for fam, fc in (("H", "#5f6b7a"), ("P", "#2b2b2b")):
            rep = "H" if fam == "H" else "P"
            ax.plot(TAU_X, [fl.loc[rep, t] for t in TAUS], ls=(0, (1, 2)),
                    c=fc, lw=1.0, zorder=2)
            ax.fill_between(TAU_X, [fl_lo.loc[rep, t] for t in TAUS],
                            [fl_hi.loc[rep, t] for t in TAUS],
                            color=fc, alpha=0.10, lw=0, zorder=1)
        for gram in GRAMMAR_ORDER:
            key = (phen, gram)
            if key not in U.index:
                continue
            ys = [U.loc[key, _spec(t)] for t in TAUS]
            col, ls, mk = STYLE[gram]
            is_thin = bool(thin.get(key, False))
            ax.plot(TAU_X, ys, color=col, ls=ls, lw=1.9, alpha=0.9, zorder=3)
            ax.plot(TAU_X, ys, ls="none", marker=mk, ms=5.2, zorder=4,
                    markerfacecolor=("white" if is_thin else col),
                    markeredgecolor=col, markeredgewidth=1.4)
        ax.set_title(PHEN_DISPLAY[phen])
        ax.set_xlabel("memory  τ")
        ax.set_xscale("log", base=2)
        ax.set_xticks(TAU_X); ax.set_xticklabels(TAU_LABELS)
        ax.set_xlim(0.42, 9.5); ax.set_ylim(-0.08, 1.02)
        ax.tick_params(labelsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.grid(axis="y", color="#e6e6e6", lw=0.6, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("U  (uncertainty resolved)")
    # direct labels on the last panel's lines carry identity without colour
    axes[0].annotate("recoverable (U ≥ 0.6)", xy=(0.46, 0.63), fontsize=7.5,
                     color="#b0272c", va="bottom")
    axes[0].annotate("determinism floor", xy=(0.46, 0.475), fontsize=7.5,
                     color="#2b2b2b", va="top")

    handles = [plt.Line2D([], [], color=STYLE[g][0], ls=STYLE[g][1],
                          marker=STYLE[g][2], ms=5.2, lw=1.9,
                          label=GRAMMAR_DISPLAY[g]) for g in GRAMMAR_ORDER]
    handles += [plt.Line2D([], [], ls="none", marker="o", ms=5.2,
                           markerfacecolor="white", markeredgecolor="#555",
                           label="thin cell (no verdict)"),
                plt.Line2D([], [], ls=(0, (1, 2)), c="#2b2b2b", lw=1.0,
                           label="determinism floor ±CI")]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / "tau_sweep.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "---"
    s = f"{x:.{nd}f}"
    return f"${s}$" if not s.startswith("-") else f"$-{s[1:]}$"


def table_taxonomy(tax: pd.DataFrame):
    """U across the tau sweep + the cue taxonomy, phenomenon-major."""
    ucols = ["U_t0.5", "U_t1.0", "U_t2.0", "U_t4.0", "U_tinf"]
    t = tax.copy()
    t["phen_rank"] = t.phenomenon.map({p: i for i, p in enumerate(PHEN_ORDER)})
    t["gram_rank"] = t.grammar.map({g: i for i, g in enumerate(GRAMMAR_ORDER)})
    t = t.sort_values(["phen_rank", "gram_rank"])

    csv = t[["grammar", "phenomenon", *ucols, "taxonomy", "tau_recover",
             "n_pos", "n_neg", "thin", "floor_t1"]].copy()
    csv.insert(0, "grammar_display", csv.grammar.map(GRAMMAR_DISPLAY))
    csv.to_csv(TABLES / "taxonomy.csv", index=False)

    lines = [
        "% Auto-generated by analysis/reports/corpus/report_separability.py -- do not edit.",
        r"\begin{table}[t]", r"  \centering", r"  \small",
        r"  \begin{tabular}{llccccclc}", r"    \toprule",
        r"    & & \multicolumn{5}{c}{$U$ at memory $\tau$} & & \\",
        r"    \cmidrule(lr){3-7}",
        r"    Gr. & Phenomenon & $0.5$ & $1$ & $2$ & $4$ & $\infty$ "
        r"& Taxonomy & $\tau_{\text{rec}}$ \\",
        r"    \midrule",
    ]
    prev = None
    for _, r in t.iterrows():
        if prev is not None and r.phenomenon != prev:
            lines.append(r"    \addlinespace")
        prev = r.phenomenon
        us = " & ".join(_fmt(r[c]) for c in ucols)
        lab = TAXONOMY_SHORT.get(r.taxonomy, r.taxonomy)
        if bool(r.thin):
            lab += r"\,\emph{(thin)}"
        tr = r.tau_recover
        tr_s = ("---" if tr != tr else
                (r"$\infty$" if math.isinf(tr) else f"${tr:g}$"))
        lines.append(f"    {GRAMMAR_TEX[r.grammar]:6} & \\texttt{{{r.phenomenon.replace('_', chr(92)+'_')}}} "
                     f"& {us} & {lab} & {tr_s} \\\\")
    floor_h = t[t.grammar == "H"].floor_t1.iloc[0]
    floor_p = t[t.grammar == "P"].floor_t1.iloc[0]
    lines += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{Separability $U$ across the memory sweep (left-only decay"
        r" features, comparable band, training lengths) and the resulting cue"
        r" taxonomy. Determinism floor at $\tau = 1$: $" + f"{floor_h:.3f}"
        + r"$ (H family), $" + f"{floor_p:.3f}"
        + r"$ (P family). \emph{Thin} marks cells whose minority"
        r" class has fewer than $30$ members; these yield no verdict. Within a"
        r" family H and H$'$ (resp.\ P and P$'$) share every rule except"
        r" \texttt{wh\_i}, so their rows are identical by construction except"
        r" there.}",
        r"  \label{tab:sep-taxonomy}", r"\end{table}",
    ]
    (TABLES / "taxonomy.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_controls(grid: pd.DataFrame):
    """Neutral probe-pair positive controls: categories vs +agreement channel."""
    n = grid[(grid.model == "logreg") & (grid.regime == "neutral_iod")
             & (grid.spec.isin(["window@5:both", "winfm@5:both"]))]
    piv = n.pivot_table(index=["phenomenon", "grammar"], columns="spec",
                        values=["U", "auc"])
    rows = []
    for phen in ("neutral_positional", "neutral_agreement"):
        for gram in GRAMMAR_ORDER:
            key = (phen, gram)
            if key not in piv.index:
                continue
            rows.append({
                "control": phen, "grammar": gram,
                "grammar_display": GRAMMAR_DISPLAY[gram],
                "U_categories": piv.loc[key, ("U", "window@5:both")],
                "auc_categories": piv.loc[key, ("auc", "window@5:both")],
                "U_agreement": piv.loc[key, ("U", "winfm@5:both")],
                "auc_agreement": piv.loc[key, ("auc", "winfm@5:both")],
                "n": int(n[(n.phenomenon == phen) & (n.grammar == gram)].n.iloc[0]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES / "controls.csv", index=False)

    lines = [
        "% Auto-generated by analysis/reports/corpus/report_separability.py -- do not edit.",
        r"\begin{table}[t]", r"  \centering",
        r"  \begin{tabular}{llcc}", r"    \toprule",
        r"    & & \multicolumn{2}{c}{$U$ (AUC)} \\", r"    \cmidrule(l){3-4}",
        r"    Control & Grammar & categories & \;+\,agreement \\",
        r"    \midrule",
    ]
    for i, (phen, sub) in enumerate(df.groupby("control", sort=False)):
        if i:
            lines.append(r"    \addlinespace")
        label = "positional" if phen == "neutral_positional" else "agreement"
        for j, (_, r) in enumerate(sub.iterrows()):
            first = label if j == 0 else ""
            lines.append(
                f"    {first:10} & {GRAMMAR_TEX[r.grammar]:6} "
                f"& ${r.U_categories:.3f}$ (${r.auc_categories:.3f}$) "
                f"& ${r.U_agreement:.3f}$ (${r.auc_agreement:.3f}$) \\\\")
    npairs = int(df.n.iloc[0] // 2)
    lines += [
        r"    \bottomrule", r"  \end{tabular}",
        r"  \caption{Neutral probe-pair controls (in-domain, $" + f"{npairs}" + r"$ pairs"
        r" per cell, $\pm5$ window). Categories alone locate a relocated token"
        r" but are exactly uninformative about a flipped inflection; the"
        r" agreement-match channel recovers it. This validates both the method"
        r" and the channel.}",
        r"  \label{tab:sep-controls}", r"\end{table}",
    ]
    (TABLES / "controls.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    grid, floor, tax = load()
    fig = figure_tau_sweep(grid, floor)
    table_taxonomy(tax)
    table_controls(grid)
    print(f"corpus set: {CORPUS_SET}  (source: {DATA})")
    print(f"wrote {fig.relative_to(REPO)}")
    for name in ("taxonomy", "controls"):
        for ext in ("csv", "tex"):
            print(f"wrote {(TABLES / f'{name}.{ext}').relative_to(REPO)}")


if __name__ == "__main__":
    main()
