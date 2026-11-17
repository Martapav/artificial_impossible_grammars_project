"""Structural-probe reporting figures -> results/analysis/evaluation/structural_probes/.

Held-out probe evaluation of the final (ep12) model on both the in-domain (IOD)
and out-of-domain (OOD) probe splits. Every term-level number comes from the
pre-computed ``terms.csv`` (``evaluation.aggregate``); every per-pair number from
:func:`analysis.report_common.load_probe_pairs`. Nothing is recomputed here.

Core figures (each with a backing CSV):

    margin_over_b4.{png,csv}
        margin above the order-4 n-gram reference, term x grammar, IOD and OOD
        panels (zero = the model only matches a 4-gram on that contrast).
    heatmap_term_grammar.{png,csv}
        accuracy and margin-over-b4 by term x grammar, IOD and OOD (2x2 grid),
        gate terms flagged.
    iod_vs_ood_by_phenomenon.{png,csv}
        per phenomenon, IOD vs OOD paired bars, faceted per grammar (accuracy and
        margin-over-b4); plus a OOD-IOD degradation table.
    graded_surprisal.{png,csv} (+ _accuracy.png)
        E4 graded confidence: within-pair margin for the three ordered bins
        A target(fired)=0 / B licensing-non-fired / C non-licensing, per grammar,
        per phenomenon, IOD and OOD.

Extra figures (proposed and confirmed):

    iod_ood_degradation_heatmap.png     OOD-IOD delta of margin-over-b4 and acc.
    ood_generalization_vs_length.png    accuracy vs length and vs embedding depth,
                                        OOD vs IOD, per grammar.
    graded_gap.{png,csv}                Delta_C - Delta_B with bootstrap CI, where
                                        both bins exist (refl, pron).

Run:  python -m analysis.reports.probes.report_probes
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.category_labels import relabel
from analysis.report_common import (
    BIN_COLOR, BIN_LABEL, BIN_ORDER, GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY,
    SPLIT_DISPLAY, SUBCASE_DISPLAY, SUBCASE_ORDER, TERM_ORDER, bootstrap_ci,
    grammar_handles, load_probe_pairs, load_terms, outdir, savefig,
)

OUT = outdir("evaluation", "structural_probes")
SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]

# term -> coarse phenomenon (for the IOD-vs-OOD comparison).
TERM_TO_PHENOM = {
    "aux": "aux movement", "wh_i": "wh-movement", "wh_ii": "wh-movement",
    "refl:wrong_item": "reflexive", "refl:wrong_site": "reflexive",
    "refl:no_substitution": "reflexive",
    "pron:wrong_item": "pronoun", "pron:wrong_site": "pronoun",
    "neutral_agreement": "agreement", "neutral_positional": "positional",
}
PHENOM_ORDER = ["aux movement", "wh-movement", "reflexive", "pronoun",
                "agreement", "positional"]


# --- 1. margin over b4 -----------------------------------------------------

def fig_margin_over_b4(terms, out_png, out_csv):
    present = [t for t in TERM_ORDER if t in set(terms["term"])]
    y = np.arange(len(present))
    h = 0.8 / len(GRAMMARS)
    # terms.csv is per-seed (5 rows per grammar x term x split): reduce to the
    # mean over seeds, with the across-seed standard error as the bar's error.
    agg = (terms.groupby(["split", "grammar", "term"], observed=True)["margin_over_b4"]
           .agg(mean="mean", sd="std", n="size").reset_index())
    agg["se"] = agg["sd"] / np.sqrt(agg["n"].clip(lower=1))
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, sharex=True)
    rows = []
    for ax, split in zip(axes, SPLITS):
        sub = agg[agg.split == split].set_index(["grammar", "term"])
        for gi, g in enumerate(GRAMMARS):
            vals, errs, ypos = [], [], []
            for ti, term in enumerate(present):
                if (g, term) not in sub.index:
                    continue
                vals.append(float(sub.loc[(g, term), "mean"]))
                errs.append(float(sub.loc[(g, term), "se"]))
                ypos.append(ti + (gi - (len(GRAMMARS) - 1) / 2) * h)
                rows.append(dict(split=SPLIT_DISPLAY[split], grammar=g, term=term,
                                 margin_over_b4=vals[-1], se_across_seeds=errs[-1]))
            ax.barh(ypos, vals, height=h * 0.9, color=GRAMMAR_COLOR[g],
                    xerr=errs, error_kw=dict(lw=0.6, ecolor="0.3"))
        ax.axvline(0, color="black", lw=1)
        ax.set_yticks(y, [relabel(t) for t in present], fontsize=8)
        ax.set_xlabel("margin over b4 (nats)")
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12)
        ax.grid(alpha=0.25, axis="x")
    axes[0].invert_yaxis()
    axes[1].legend(handles=grammar_handles(), title="grammar", fontsize=8,
                   loc="lower right")
    fig.tight_layout()
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# --- 2. heatmaps (margin-over-b4, acc) x (IOD, OOD) -------------------------

def fig_heatmaps(terms, out_png, out_csv):
    present = [t for t in TERM_ORDER if t in set(terms["term"])]
    gates = set(terms.loc[terms.is_gate, "term"])
    ylabels = [relabel(t) + ("  *" if t in gates else "") for t in present]

    # CSV keeps both metrics; the figure shows pair accuracy only (IOD, OOD).
    rows = []
    for col in ("margin_over_b4", "acc"):
        for split in SPLITS:
            sub = terms[terms.split == split]
            piv = (sub.pivot_table(index="term", columns="grammar", values=col,
                                   observed=True)
                   .reindex(index=present, columns=GRAMMARS))
            data = piv.to_numpy(dtype=float)
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    v = data[i, j]
                    if not np.isnan(v):
                        rows.append(dict(metric=col, split=SPLIT_DISPLAY[split],
                                         term=present[i], grammar=GRAMMARS[j], value=v))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"wspace": 0.3})
    fmt, vlims = "{:.2f}", (0.0, 1.0)
    for ci, split in enumerate(SPLITS):
        ax = axes[ci]
        sub = terms[terms.split == split]
        piv = (sub.pivot_table(index="term", columns="grammar", values="acc",
                               observed=True)
               .reindex(index=present, columns=GRAMMARS))
        data = piv.to_numpy(dtype=float)
        im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=vlims[0], vmax=vlims[1])
        ax.set_xticks(range(len(GRAMMARS)), [GRAMMAR_DISPLAY[g] for g in GRAMMARS])
        ax.set_yticks(range(len(present)), ylabels, fontsize=7)
        ax.set_title(f"pair accuracy -- {SPLIT_DISPLAY[split]}", fontsize=10)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, fmt.format(v), ha="center", va="center",
                            fontsize=7, color="black")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, out_png)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


# --- 3. IOD vs OOD per phenomenon ------------------------------------------

def _phenom_table(terms):
    """Per (grammar, split, phenomenon): n_pairs-weighted acc and mean margin_over_b4."""
    t = terms.copy()
    t["phenomenon"] = t["term"].map(TERM_TO_PHENOM)
    def wmean(df, col):
        w = df["n_pairs"].to_numpy(dtype=float)
        return float(np.average(df[col].to_numpy(dtype=float), weights=w))
    rows = []
    for (g, sp, ph), df in t.groupby(["grammar", "split", "phenomenon"], observed=True):
        rows.append(dict(grammar=g, split=SPLIT_DISPLAY[sp], phenomenon=ph,
                         acc=wmean(df, "acc"), margin_over_b4=wmean(df, "margin_over_b4"),
                         n_pairs=int(df["n_pairs"].sum())))
    return pd.DataFrame(rows)


def fig_iod_vs_ood(terms, out_png, out_csv):
    tab = _phenom_table(terms)
    phen = [p for p in PHENOM_ORDER if p in set(tab.phenomenon)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    x = np.arange(len(phen))
    w = 0.38
    for gi, g in enumerate(GRAMMARS):
        ax = axes[gi // 2][gi % 2]
        sub = tab[tab.grammar == g].set_index(["phenomenon", "split"])
        for si, (split, off, hatch) in enumerate((("IOD", -w / 2, None),
                                                  ("OOD", w / 2, "///"))):
            vals = [sub.loc[(p, split), "acc"] if (p, split) in sub.index else np.nan
                    for p in phen]
            ax.bar(x + off, vals, width=w, color=GRAMMAR_COLOR[g],
                   alpha=0.9 if split == "IOD" else 0.55, hatch=hatch,
                   edgecolor="white", label=split)
        ax.set_title(GRAMMAR_DISPLAY[g], fontsize=12)
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25, axis="y")
        ax.set_xticks(x, [p.replace(" ", "\n") for p in phen], fontsize=8)
        if gi == 0:
            ax.legend(title="split", fontsize=8, loc="lower left")
    for r in range(2):
        axes[r][0].set_ylabel("pair accuracy")
    fig.tight_layout()
    savefig(fig, out_png)

    # Degradation table (OOD - IOD), both metrics.
    wide = tab.pivot_table(index=["grammar", "phenomenon"], columns="split",
                           values=["acc", "margin_over_b4"], observed=True)
    deg = pd.DataFrame({
        "acc_IOD": wide[("acc", "IOD")], "acc_OOD": wide[("acc", "OOD")],
        "acc_drop": wide[("acc", "IOD")] - wide[("acc", "OOD")],
        "margin_over_b4_IOD": wide[("margin_over_b4", "IOD")],
        "margin_over_b4_OOD": wide[("margin_over_b4", "OOD")],
        "margin_over_b4_drop": wide[("margin_over_b4", "IOD")] - wide[("margin_over_b4", "OOD")],
    }).reset_index()
    deg.to_csv(out_csv, index=False)


# --- 4. graded surprisal (E4) ----------------------------------------------

def _graded_stats(pairs, value):
    """Per (grammar, split, phenomenon, bin): point + bootstrap CI of `value`."""
    rows = []
    for (g, sp, ph, b), df in pairs.groupby(
            ["grammar", "split", "phenomenon", "bin"], observed=True):
        pt, lo, hi = bootstrap_ci(df[value].to_numpy())
        rows.append(dict(grammar=g, split=SPLIT_DISPLAY[sp], phenomenon=ph, bin=b,
                         n=len(df), point=pt, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def _graded_figure(stats, value_label, out_png, ref_zero):
    phen = [p for p in SUBCASE_ORDER if p in set(stats.phenomenon)]
    xpos = {p: i for i, p in enumerate(phen)}
    fig, axes = plt.subplots(len(GRAMMARS), 2, figsize=(13, 3.0 * len(GRAMMARS)),
                             sharex=True, sharey=True, squeeze=False)
    for gi, g in enumerate(GRAMMARS):
        for ci, split in enumerate(("IOD", "OOD")):
            ax = axes[gi][ci]
            sub = stats[(stats.grammar == g) & (stats.split == split)]
            for bi, b in enumerate(BIN_ORDER):
                bs = sub[sub.bin == b]
                if bs.empty:
                    continue
                off = (bi - 1) * 0.18
                xs = [xpos[p] + off for p in bs.phenomenon]
                ax.errorbar(xs, bs["point"], yerr=[bs["point"] - bs["lo"],
                            bs["hi"] - bs["point"]], fmt="o", ms=5,
                            color=BIN_COLOR[b], capsize=2, lw=1)
            if ref_zero:
                ax.axhline(0, color=BIN_COLOR["A"], lw=1, ls="--")
            ax.set_title(f"{GRAMMAR_DISPLAY[g]} -- {split}", fontsize=10)
            ax.grid(alpha=0.25, axis="y")
    for gi in range(len(GRAMMARS)):
        axes[gi][0].set_ylabel(value_label)
    for ci in range(2):
        axes[-1][ci].set_xticks(range(len(phen)),
                                [SUBCASE_DISPLAY[p].replace(" ", "\n") for p in phen],
                                fontsize=8)
    handles = [plt.Line2D([], [], marker="o", ls="", color=BIN_COLOR[b],
                          label=f"{b}: {BIN_LABEL[b]}") for b in BIN_ORDER]
    axes[0][1].legend(handles=handles, fontsize=8, loc="upper left",
                      title="graded bin")
    fig.tight_layout()
    savefig(fig, out_png)


def fig_graded(pairs, out_png, out_csv):
    m = _graded_stats(pairs, "margin")           # criterion-native within-pair margin
    a = _graded_stats(pairs, "acc")
    m["metric"], a["metric"] = "margin_nats", "accuracy"
    pd.concat([m, a], ignore_index=True).to_csv(out_csv, index=False)
    _graded_figure(m, "within-pair margin (nats)", out_png, ref_zero=True)
    _graded_figure(a, "P(target preferred)",
                   out_png.with_name("graded_surprisal_accuracy.png"), ref_zero=False)


# --- extra 1: IOD->OOD degradation heatmap ---------------------------------

def fig_degradation_heatmap(terms, out_png):
    present = [t for t in TERM_ORDER if t in set(terms["term"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={"wspace": 0.35})
    for ax, col, label in ((axes[0], "margin_over_b4", "delta margin-over-b4 (OOD - IOD)"),
                           (axes[1], "acc", "delta accuracy (OOD - IOD)")):
        piv = terms.pivot_table(index="term", columns=["grammar", "split"],
                                values=col, observed=True)
        delta = pd.DataFrame(index=present, columns=GRAMMARS, dtype=float)
        for g in GRAMMARS:
            for t in present:
                try:
                    delta.loc[t, g] = (piv.loc[t, (g, "structural_probe_OOD")]
                                       - piv.loc[t, (g, "structural_probe_IOD")])
                except KeyError:
                    pass
        data = delta.to_numpy(dtype=float)
        vmax = np.nanmax(np.abs(data))
        im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(GRAMMARS)), [GRAMMAR_DISPLAY[g] for g in GRAMMARS])
        ax.set_yticks(range(len(present)), [relabel(t) for t in present], fontsize=8)
        ax.set_title(label, fontsize=11)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, out_png)


# --- extra 2: OOD generalisation vs length / depth -------------------------

def _acc_vs(pairs, xcol, split, g, bins):
    sub = pairs[(pairs.split == split) & (pairs.grammar == g)].copy()
    sub["bin"] = pd.cut(sub[xcol], bins=bins, include_lowest=True)
    per_seed = sub.groupby(["seed", "bin"], observed=True)["acc"].mean().reset_index()
    stat = per_seed.groupby("bin", observed=True)["acc"].agg(mean="mean", lo="min", hi="max")
    cnt = sub.groupby("bin", observed=True).size()
    centres = sub.groupby("bin", observed=True)[xcol].median()
    keep = cnt.reindex(stat.index).fillna(0) >= 50
    stat, centres = stat[keep.values], centres.reindex(stat.index)[keep.values]
    return centres.to_numpy(), stat["mean"].to_numpy(), stat["lo"].to_numpy(), stat["hi"].to_numpy()


def fig_ood_generalization(pairs, out_png):
    len_bins = np.unique(np.quantile(pairs["length_g"], np.linspace(0, 1, 9)))
    depth_bins = np.arange(pairs["embedding_depth_g"].min() - 0.5,
                           pairs["embedding_depth_g"].max() + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, xcol, bins, xlabel in (
            (axes[0], "length_g", len_bins, "sentence length (tokens)"),
            (axes[1], "embedding_depth_g", depth_bins, "embedding depth")):
        for g in GRAMMARS:
            for split, ls in (("structural_probe_IOD", "--"),
                              ("structural_probe_OOD", "-")):
                x, mean, lo, hi = _acc_vs(pairs, xcol, split, g, bins)
                if len(x) == 0:
                    continue
                ax.plot(x, mean, ls=ls, color=GRAMMAR_COLOR[g], lw=1.6)
                if split.endswith("OOD"):
                    ax.fill_between(x, lo, hi, color=GRAMMAR_COLOR[g], alpha=0.10, lw=0)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("pair accuracy")
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.grid(alpha=0.25)
    style = [plt.Line2D([], [], color="black", ls="-", label="OOD"),
             plt.Line2D([], [], color="black", ls="--", label="IOD")]
    axes[0].legend(handles=grammar_handles(), title="grammar", fontsize=8, loc="lower left")
    axes[1].legend(handles=style, title="split", fontsize=8, loc="lower left")
    fig.tight_layout()
    savefig(fig, out_png)


# --- extra 3: graded gap (Delta_C - Delta_B) -------------------------------

def fig_graded_gap(pairs, out_png, out_csv):
    """Per (grammar, split, phenomenon): mean(margin | C) - mean(margin | B), CI."""
    rows = []
    both = (pairs.groupby(["grammar", "split", "phenomenon"], observed=True)["bin"]
            .agg(lambda s: {"B", "C"}.issubset(set(s))))
    valid = {k for k, v in both.items() if v}
    for (g, sp, ph), df in pairs.groupby(["grammar", "split", "phenomenon"], observed=True):
        if (g, sp, ph) not in valid:
            continue
        b = df[df.bin == "B"]["margin"].to_numpy()
        c = df[df.bin == "C"]["margin"].to_numpy()
        rng = np.random.default_rng(0)
        boot = (rng.choice(c, (2000, len(c))).mean(1) - rng.choice(b, (2000, len(b))).mean(1))
        gap = float(c.mean() - b.mean())
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rows.append(dict(grammar=g, split=SPLIT_DISPLAY[sp], phenomenon=ph,
                         gap=gap, lo=float(lo), hi=float(hi)))
    tab = pd.DataFrame(rows)
    tab.to_csv(out_csv, index=False)
    phen = sorted(set(tab.phenomenon), key=lambda p: SUBCASE_ORDER.index(p))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    x = np.arange(len(phen))
    w = 0.8 / len(GRAMMARS)
    for ax, split in zip(axes, ("IOD", "OOD")):
        sub = tab[tab.split == split].set_index(["phenomenon", "grammar"])
        for gi, g in enumerate(GRAMMARS):
            vals, los, his, xs = [], [], [], []
            for pi, p in enumerate(phen):
                if (p, g) not in sub.index:
                    continue
                r = sub.loc[(p, g)]
                vals.append(r["gap"]); los.append(r["gap"] - r["lo"]); his.append(r["hi"] - r["gap"])
                xs.append(pi + (gi - (len(GRAMMARS) - 1) / 2) * w)
            ax.bar(xs, vals, width=w * 0.9, color=GRAMMAR_COLOR[g],
                   yerr=[los, his], error_kw=dict(lw=0.6, ecolor="0.3"))
        ax.axhline(0, color="black", lw=1)
        ax.set_xticks(x, [SUBCASE_DISPLAY[p] for p in phen], fontsize=9)
        ax.set_title(split, fontsize=12)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("margin(C) - margin(B)  (nats)")
    axes[1].legend(handles=grammar_handles(), title="grammar", fontsize=8)
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    terms = load_terms()
    pairs = load_probe_pairs()

    fig_margin_over_b4(terms, OUT / "margin_over_b4.png", OUT / "margin_over_b4.csv")
    fig_heatmaps(terms, OUT / "heatmap_term_grammar.png", OUT / "heatmap_term_grammar.csv")
    fig_iod_vs_ood(terms, OUT / "iod_vs_ood_by_phenomenon.png",
                   OUT / "iod_vs_ood_degradation.csv")
    fig_graded(pairs, OUT / "graded_surprisal.png", OUT / "graded_surprisal.csv")

    fig_degradation_heatmap(terms, OUT / "iod_ood_degradation_heatmap.png")
    fig_ood_generalization(pairs, OUT / "ood_generalization_vs_length.png")
    fig_graded_gap(pairs, OUT / "graded_gap.png", OUT / "graded_gap.csv")

    print(f"[probes] wrote margin_over_b4, heatmaps, iod_vs_ood, graded_surprisal "
          f"(+accuracy), degradation_heatmap, ood_generalization, graded_gap -> {OUT}")


if __name__ == "__main__":
    main()
