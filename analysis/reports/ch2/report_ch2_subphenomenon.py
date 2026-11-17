"""Chapter-2, subchapter 1: case-by-case probe description (additions).

Complements the existing ``per_subphenomenon`` dot figures (which stay where they
are) with four case-study readouts of the held-out ``set_3_ep12`` probe:

    margin_distribution   per-pair `margin` as an ECDF per grammar x split,
                          faceted per term -- the density read the dot plots lack.
    acc_vs_b4             per term x grammar, the model's pair accuracy beside the
                          accuracy already implied by the order-4 n-gram
                          (`acc_over_b4` from terms.csv): where competence is only
                          4-gram deep the two bars meet.
    calibration           reliability: within a phenomenon, does a larger |margin|
                          (confidence) come with a higher P(correct)?  P(correct)
                          = fraction with margin > 0 in each |margin| bin.
    worked_examples       the most-confident-correct and most-confident-wrong pair
                          per phenomenon x grammar, with both surface strings.

Everything reads ``load_probe_pairs`` / ``load_terms``; nothing recomputes a
margin. Figures -> ch2/sub1_case_by_case/figures ; tables -> .../tables.

Run:  python -m analysis.reports.ch2.report_ch2_subphenomenon
"""
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.category_labels import relabel
from analysis.report_common import (
    GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY, REPO, SPLIT_DISPLAY,
    SUBCASE_DISPLAY, SUBCASE_ORDER, TERM_ORDER, ch2_dirs, grammar_handles,
    load_probe_pairs, load_terms, savefig,
)

CORPUS_ROOT = REPO / "corpora" / "set_3"           # corpus behind set_3_ep12
SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]
SPLIT_LS = {"structural_probe_IOD": "-", "structural_probe_OOD": "--"}


# --- 1. margin distribution (ECDF, faceted per term) -----------------------

def fig_margin_distribution(pairs, fig_dir, tab_dir):
    present = [t for t in TERM_ORDER if t in set(pairs["term"])]
    ncol = 3
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             squeeze=False)
    rows = []
    for i, term in enumerate(present):
        ax = axes[i // ncol][i % ncol]
        sub = pairs[pairs["term"] == term]
        # robust x-range from the pooled 1-99 percentile of this term's margins.
        m_all = sub["margin"].to_numpy(dtype=float)
        lo, hi = np.percentile(m_all, [1, 99]) if m_all.size else (-1, 1)
        for g in GRAMMARS:
            for sp in SPLITS:
                d = sub[(sub.grammar == g) & (sub.split == sp)]["margin"]
                vals = np.sort(d.to_numpy(dtype=float))
                if vals.size == 0:
                    continue
                y = np.arange(1, vals.size + 1) / vals.size
                ax.step(vals, y, where="post", color=GRAMMAR_COLOR[g],
                        ls=SPLIT_LS[sp], lw=1.3, alpha=0.9)
                q = np.percentile(vals, [10, 25, 50, 75, 90])
                rows.append(dict(term=term, grammar=g, split=SPLIT_DISPLAY[sp],
                                 n=int(vals.size), mean=float(vals.mean()),
                                 frac_positive=float((vals > 0).mean()),
                                 p10=q[0], p25=q[1], p50=q[2], p75=q[3], p90=q[4]))
        ax.axvline(0, color="0.4", lw=0.9, ls=":")
        ax.set_xlim(min(lo, 0), max(hi, 0))
        ax.set_ylim(0, 1)
        ax.set_title(relabel(term), fontsize=9)
        ax.grid(alpha=0.2)
        if i % ncol == 0:
            ax.set_ylabel("ECDF")
    for j in range(len(present), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = grammar_handles() + [
        plt.Line2D([], [], color="0.3", ls="-", label="IOD"),
        plt.Line2D([], [], color="0.3", ls="--", label="OOD")]
    axes[0][ncol - 1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.supxlabel("margin (nats)", fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir / "margin_distribution.png")
    pd.DataFrame(rows).to_csv(tab_dir / "margin_distribution.csv", index=False)


# --- 1b. margin distribution (violins, faceted per term) -------------------

def fig_margin_violin(pairs, fig_dir):
    present = [t for t in TERM_ORDER if t in set(pairs["term"])]
    ncol = 3
    nrow = int(np.ceil(len(present) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow),
                             squeeze=False)
    for i, term in enumerate(present):
        ax = axes[i // ncol][i % ncol]
        sub = pairs[pairs["term"] == term]
        m_all = sub["margin"].to_numpy(dtype=float)
        lo, hi = np.percentile(m_all, [1, 99]) if m_all.size else (-1, 1)
        data, pos, colors, alphas = [], [], [], []
        for gi, g in enumerate(GRAMMARS):
            for si, sp in enumerate(SPLITS):
                d = sub[(sub.grammar == g) & (sub.split == sp)]["margin"].to_numpy(dtype=float)
                d = d[(d >= lo) & (d <= hi)]            # clip tails so the KDE reads
                if d.size < 2:
                    continue
                data.append(d)
                pos.append(gi + (si - 0.5) * 0.42)
                colors.append(GRAMMAR_COLOR[g])
                alphas.append(0.85 if sp.endswith("IOD") else 0.38)
        if not data:
            ax.axis("off")
            continue
        parts = ax.violinplot(data, positions=pos, widths=0.38, showmeans=False,
                              showmedians=True, showextrema=False)
        for body, c, a in zip(parts["bodies"], colors, alphas):
            body.set(facecolor=c, edgecolor=c, alpha=a, linewidth=0.6)
        parts["cmedians"].set(color="black", linewidth=1.0)
        ax.axhline(0, color="0.4", lw=0.8, ls=":")
        ax.set_xticks(range(len(GRAMMARS)), [GRAMMAR_DISPLAY[g] for g in GRAMMARS],
                      fontsize=7)
        ax.set_ylim(lo, hi)
        ax.set_title(relabel(term), fontsize=9)
        ax.grid(alpha=0.2, axis="y")
        if i % ncol == 0:
            ax.set_ylabel("margin (nats)")
    for j in range(len(present), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    handles = grammar_handles() + [
        plt.Line2D([], [], marker="s", ls="", mfc="0.3", mec="0.3", alpha=0.85,
                   label="IOD"),
        plt.Line2D([], [], marker="s", ls="", mfc="0.3", mec="0.3", alpha=0.38,
                   label="OOD")]
    axes[0][ncol - 1].legend(handles=handles, fontsize=7, loc="lower right")
    fig.tight_layout()
    savefig(fig, fig_dir / "margin_violin.png")


# --- 2. model accuracy vs the order-4 n-gram accuracy ----------------------

def fig_acc_vs_b4(terms, fig_dir, tab_dir):
    present = [t for t in TERM_ORDER if t in set(terms["term"])]
    # terms.csv is per-seed: reduce to the across-seed mean.
    agg = (terms.groupby(["split", "grammar", "term"], observed=True)
           [["acc", "acc_over_b4"]].mean().reset_index())
    agg["acc_b4"] = agg["acc"] - agg["acc_over_b4"]
    y = np.arange(len(present))
    h = 0.8 / len(GRAMMARS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, sharex=True)
    rows = []
    for ax, split in zip(axes, SPLITS):
        s = agg[agg.split == split].set_index(["grammar", "term"])
        for gi, g in enumerate(GRAMMARS):
            for ti, term in enumerate(present):
                if (g, term) not in s.index:
                    continue
                acc = float(s.loc[(g, term), "acc"])
                acc_b4 = float(s.loc[(g, term), "acc_b4"])
                yc = ti + (gi - (len(GRAMMARS) - 1) / 2) * h
                # model bar; the b4 accuracy drawn as a notch on top.
                ax.barh(yc, acc, height=h * 0.9, color=GRAMMAR_COLOR[g], alpha=0.9)
                ax.plot([acc_b4, acc_b4], [yc - h * 0.45, yc + h * 0.45],
                        color="black", lw=1.3)
                rows.append(dict(split=SPLIT_DISPLAY[split], grammar=g, term=term,
                                 acc=acc, acc_b4=acc_b4,
                                 acc_over_b4=acc - acc_b4))
        ax.axvline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_xlim(0, 1.02)
        ax.set_yticks(y, [relabel(t) for t in present], fontsize=8)
        ax.set_xlabel("accuracy")
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12)
        ax.grid(alpha=0.25, axis="x")
    axes[0].invert_yaxis()
    handles = grammar_handles() + [
        plt.Line2D([], [], color="black", lw=1.3, label="order-4 n-gram acc")]
    axes[1].legend(handles=handles, fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig(fig, fig_dir / "acc_vs_b4.png")
    pd.DataFrame(rows).to_csv(tab_dir / "acc_vs_b4.csv", index=False)


# --- 3. calibration: P(correct) vs |margin| per phenomenon -----------------

def fig_calibration(pairs, fig_dir, tab_dir, n_bins=8, floor=50):
    phen = [p for p in SUBCASE_ORDER if p in set(pairs["phenomenon"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    cmap = plt.get_cmap("tab10")
    rows = []
    for ax, split in zip(axes, SPLITS):
        for pi, ph in enumerate(phen):
            d = pairs[(pairs.phenomenon == ph) & (pairs.split == split)].copy()
            if d.empty:
                continue
            d["absm"] = d["margin"].abs()
            d["correct"] = (d["margin"] > 0).astype(float)
            # per-phenomenon |margin| quantile bins, then keep bins with >= floor.
            edges = np.unique(np.quantile(d["absm"], np.linspace(0, 1, n_bins + 1)))
            if edges.size < 3:
                continue
            d["b"] = pd.cut(d["absm"], bins=edges, include_lowest=True)
            grp = d.groupby("b", observed=True)
            acc = grp["correct"].mean()
            cnt = grp.size()
            xc = grp["absm"].mean()
            keep = cnt >= floor
            if keep.sum() == 0:
                continue
            ax.plot(xc[keep].to_numpy(), acc[keep].to_numpy(), marker="o", ms=4,
                    color=cmap(pi % 10), lw=1.6, label=SUBCASE_DISPLAY[ph])
            for blab in acc.index[keep]:
                rows.append(dict(split=SPLIT_DISPLAY[split], phenomenon=ph,
                                 absm_mean=float(xc[blab]), p_correct=float(acc[blab]),
                                 n=int(cnt[blab])))
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_xlabel("|margin| (nats)")
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12)
        ax.set_ylim(0.4, 1.02)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("P(correct)")
    axes[1].legend(fontsize=8, loc="lower right", title="phenomenon")
    fig.tight_layout()
    savefig(fig, fig_dir / "calibration.png")
    pd.DataFrame(rows).to_csv(tab_dir / "calibration.csv", index=False)


# --- 4. worked examples (extremes, with surface strings) -------------------

def _sentence_map(grammar, split):
    """{item_uid: sentence} for one grammar x probe split from the corpus."""
    path = CORPUS_ROOT / split / f"{grammar}.jsonl"
    out = {}
    with open(path) as f:
        for line in f:
            o = json.loads(line)
            out[o["item_uid"]] = o["sentence"]
    return out


def worked_examples(pairs, tab_dir):
    phen = [p for p in SUBCASE_ORDER if p in set(pairs["phenomenon"])]
    has_uid = "item_uid_g" in pairs.columns and "item_uid_u" in pairs.columns
    rows = []
    for split in SPLITS:
        for g in GRAMMARS:
            smap = _sentence_map(g, split) if has_uid else {}
            for ph in phen:
                d = pairs[(pairs.phenomenon == ph) & (pairs.grammar == g)
                          & (pairs.split == split)]
                if d.empty:
                    continue
                for kind, idx in (("most_confident_correct", d["margin"].idxmax()),
                                  ("most_confident_wrong", d["margin"].idxmin())):
                    r = d.loc[idx]
                    rows.append(dict(
                        phenomenon=ph, grammar=g, split=SPLIT_DISPLAY[split],
                        kind=kind, term=r["term"], violation=r.get("violation"),
                        margin=float(r["margin"]), acc=float(r["acc"]),
                        pair_id=r["pair_id"],
                        sentence_grammatical=smap.get(r.get("item_uid_g"), ""),
                        sentence_ungrammatical=smap.get(r.get("item_uid_u"), "")))
    pd.DataFrame(rows).to_csv(tab_dir / "worked_examples.csv", index=False)
    return len(rows), has_uid


def main() -> None:
    fig_dir, tab_dir = ch2_dirs("sub1_case_by_case")
    pairs = load_probe_pairs()
    terms = load_terms()

    fig_margin_distribution(pairs, fig_dir, tab_dir)
    fig_margin_violin(pairs, fig_dir)
    fig_acc_vs_b4(terms, fig_dir, tab_dir)
    fig_calibration(pairs, fig_dir, tab_dir)
    n_ex, has_text = worked_examples(pairs, tab_dir)

    txt = "with surface strings" if has_text else "item_uid only (no text join)"
    print(f"[ch2/sub1] wrote margin_distribution, margin_violin, acc_vs_b4, "
          f"calibration -> {fig_dir}\n           worked_examples.csv ({n_ex} rows, "
          f"{txt}) -> {tab_dir}")


if __name__ == "__main__":
    main()
