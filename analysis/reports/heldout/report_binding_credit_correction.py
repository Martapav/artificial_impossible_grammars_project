"""Make the entropy-floor correction on binding visible: two figures.

A companion to ``analysis/reports/heldout/report_differential_surprisal.py`` for the test-split
surprisal section. It shows what the binding forced-draw credit
(``analysis/binding_credit.py``) does to the differential surprisal
``Δ = neg_logq − ℓ_corpus``:

* **A — necessity (before/after).** Δ per construction × grammar computed with the
  UNcorrected floor vs the corrected floor, on a broken y-axis. Only the two
  binding constructions move: uncorrected they plunge into the shaded
  "impossible" region (Δ < 0 ⇒ the model beats the grammar's own entropy); the
  credit lifts exactly those back above zero, leaving neutral/wh/aux untouched.

* **B — anatomy (the credit's two parts).** For each binding cell a floating bar
  climbs from the raw (uncorrected) Δ up through the target-block credit and then
  the geometry-resample log Z to the final Δ, so the size of each component and
  the climb past 0 are both visible.

Writes to ``results/analysis/evaluation/test/``:
  binding_correction_before_after.png, binding_credit_anatomy.png,
  binding_correction.csv.

Run:  python -m analysis.reports.heldout.report_binding_credit_correction
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.binding_credit import SUBTYPES
from analysis.differential_surprisal import join_differential_surprisal
from analysis.entropy_baseline import ell_corpus
from analysis.binding_credit import apply_credit
from analysis.report_common import (
    CONSTRUCTION_DISPLAY, CONSTRUCTION_ORDER, GRAMMARS, GRAMMAR_COLOR,
    GRAMMAR_DISPLAY, outdir, savefig,
)

OUT = outdir("evaluation", "test")
SPLIT = "test_indistribution"
CREDIT = pd.read_parquet("analysis/data/binding_credit.parquet")
IMPOSSIBLE = "#f4cccc"          # light red: the Δ < 0 region a KL cannot enter


def _seed_mean(sub: pd.DataFrame, col: str) -> tuple[float, float]:
    """(mean-over-seeds, sem-over-seeds) of the per-seed item means of ``col``."""
    per_seed = sub.groupby("seed", observed=True)[col].mean().to_numpy()
    sem = float(per_seed.std(ddof=1) / np.sqrt(len(per_seed))) if len(per_seed) > 1 else 0.0
    return float(per_seed.mean()), sem


def build_table(items: pd.DataFrame) -> pd.DataFrame:
    """Per grammar × construction: uncorrected Δ, corrected Δ, and the credit
    split into block + log Z (0 for non-binding)."""
    rows = []
    for g in GRAMMARS:
        ell = apply_credit(ell_corpus(g, SPLIT), g)          # ell_corpus + ell_corrected
        scores = items[items.grammar == g]
        m = join_differential_surprisal(scores, ell, ell_column="ell_corpus",
                                        out_column="delta_unc")
        m = join_differential_surprisal(m, ell, ell_column="ell_corrected",
                                        out_column="delta_cor")
        cred = CREDIT[CREDIT.grammar == g].set_index("subtype")
        for c in CONSTRUCTION_ORDER:
            sub = m[m.construction == c]
            if sub.empty:
                continue
            du, du_se = _seed_mean(sub, "delta_unc")
            dc, dc_se = _seed_mean(sub, "delta_cor")
            block = float(cred.loc[c, "block"]) if c in cred.index else 0.0
            logz = float(cred.loc[c, "logz"]) if c in cred.index else 0.0
            rows.append(dict(grammar=g, construction=c, delta_unc=du,
                             delta_unc_sem=du_se, delta_cor=dc, delta_cor_sem=dc_se,
                             block=block, logz=logz, credit=block + logz,
                             is_binding=c in SUBTYPES))
    df = pd.DataFrame(rows)
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# A — before / after, broken y-axis
# ─────────────────────────────────────────────────────────────────────────────

def plot_before_after(tbl: pd.DataFrame, out_png) -> None:
    ng = len(GRAMMARS)
    slot = np.arange(len(CONSTRUCTION_ORDER))
    w = 0.8 / ng
    lo_max = float(tbl.delta_unc.min()) - 0.8            # bottom-panel floor
    fig, (hi, lo) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1.4], hspace=0.06))

    for ax in (hi, lo):
        ax.axhspan(min(lo_max, -20), 0, color=IMPOSSIBLE, zorder=0)   # Δ<0 impossible
    for i, g in enumerate(GRAMMARS):
        gt = tbl[tbl.grammar == g].set_index("construction")
        xs = slot + (i - (ng - 1) / 2) * w
        cor = [gt.loc[c, "delta_cor"] if c in gt.index else np.nan
               for c in CONSTRUCTION_ORDER]
        unc = [gt.loc[c, "delta_unc"] if c in gt.index else np.nan
               for c in CONSTRUCTION_ORDER]
        for ax in (hi, lo):
            ax.bar(xs, cor, width=w, color=GRAMMAR_COLOR[g],
                   label=GRAMMAR_DISPLAY[g] if ax is hi else None, zorder=3)
            # the "before" value: a hollow marker + a thin stem showing the drop
            for x, u, c in zip(xs, unc, cor):
                if np.isnan(u):
                    continue
                ax.plot([x, x], [c, u], color="0.35", lw=0.8, zorder=4)
                ax.plot([x], [u], marker="o", ms=4, mfc="white", mec="0.2",
                        mew=0.9, zorder=5)

    hi.set_ylim(min(-2.6, tbl.delta_cor.min() - 0.4), tbl.delta_cor.max() + 0.5)
    lo.set_ylim(lo_max, lo_max + 4.0)
    hi.axhline(0, color="black", lw=1.0, zorder=2)

    # broken-axis cosmetics
    hi.spines["bottom"].set_visible(False)
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False)
    kw = dict(marker=[(-1, -0.5), (1, 0.5)], ms=8, ls="none", color="k",
              mec="k", mew=1, clip_on=False)
    hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
    lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)

    lo.set_xticks(slot, [CONSTRUCTION_DISPLAY[c] for c in CONSTRUCTION_ORDER])
    lo.set_xlabel("construction")
    fig.text(0.04, 0.5, r"$\Delta = -\log q(x) - \ell_{\mathrm{corpus}}(x)$   "
             r"(nats/sentence,  $\widehat{D_{KL}}$)", va="center",
             rotation="vertical")
    handles, labels = hi.get_legend_handles_labels()
    handles += [plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.2",
                           label="before correction"),
                plt.Rectangle((0, 0), 1, 1, color=IMPOSSIBLE,
                              label=r"impossible ($\Delta<0$)")]
    hi.legend(handles=handles, fontsize=8, ncol=3, loc="upper left", framealpha=0.9)
    hi.set_title("The binding floor correction is necessary and surgical: only the "
                 "binding\nconstructions leave the impossible region", fontsize=10)
    hi.grid(alpha=0.2, axis="y")
    lo.grid(alpha=0.2, axis="y")
    savefig(fig, out_png)


# ─────────────────────────────────────────────────────────────────────────────
# B — anatomy: floating bar from raw Δ, +block, +log Z, to final Δ
# ─────────────────────────────────────────────────────────────────────────────

def plot_anatomy(tbl: pd.DataFrame, out_png) -> None:
    cells = tbl[tbl.is_binding].copy()
    order = {g: i for i, g in enumerate(GRAMMARS)}
    sub_order = {"anaphoric_binding_refl": 0, "anaphoric_binding_pron": 1}
    cells["key"] = (cells.grammar.astype(str).map(order) * 2
                    + cells.construction.astype(str).map(sub_order))
    cells = cells.sort_values("key").reset_index(drop=True)

    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.axhspan(cells.delta_unc.min() - 1, 0, color=IMPOSSIBLE, zorder=0)
    ax.axhline(0, color="black", lw=1.0, zorder=2)

    c_block, c_logz = "#2b8cbe", "#e08214"
    for xi, r in zip(x, cells.itertuples()):
        base = r.delta_unc
        ax.bar(xi, r.block, bottom=base, width=0.62, color=c_block,
               edgecolor="white", lw=0.5, zorder=3)
        ax.bar(xi, r.logz, bottom=base + r.block, width=0.62, color=c_logz,
               edgecolor="white", lw=0.5, zorder=3)
        # raw start marker and final tick
        ax.plot([xi], [base], marker="o", ms=5, mfc="white", mec="0.2", mew=1,
                zorder=5)
        ax.plot([xi - 0.34, xi + 0.34], [r.delta_cor, r.delta_cor], color="black",
                lw=1.6, zorder=5)
        ax.annotate(f"{r.delta_cor:.1f}", (xi, r.delta_cor),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=8, fontweight="bold")
        ax.annotate(f"{base:.1f}", (xi, base), textcoords="offset points",
                    xytext=(0, -12), ha="center", fontsize=7.5, color="0.3")

    labels = [f"{GRAMMAR_DISPLAY[r.grammar]}\n{CONSTRUCTION_DISPLAY[r.construction]}"
              for r in cells.itertuples()]
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel(r"$\Delta$  (nats/sentence)")
    handles = [
        plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.2",
                   label=r"raw $\Delta$ (uncorrected floor)"),
        plt.Rectangle((0, 0), 1, 1, color=c_block, label="+ target-block credit"),
        plt.Rectangle((0, 0), 1, 1, color=c_logz, label="+ geometry log Z"),
        plt.Line2D([], [], color="black", lw=1.6, label=r"final $\Delta$ ($\widehat{D_{KL}}$)"),
        plt.Rectangle((0, 0), 1, 1, color=IMPOSSIBLE, label=r"impossible ($\Delta<0$)"),
    ]
    ax.legend(handles=handles, fontsize=8, loc="lower right", framealpha=0.95)
    ax.set_title("Anatomy of the binding credit: each cell climbs from a negative raw "
                 "Δ\nthrough the two correction terms to a valid D_KL", fontsize=10)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    from analysis.report_common import load_test_items
    tbl = build_table(load_test_items())
    tbl.to_csv(OUT / "binding_correction.csv", index=False)
    plot_before_after(tbl, OUT / "binding_correction_before_after.png")
    plot_anatomy(tbl, OUT / "binding_credit_anatomy.png")
    show = tbl[tbl.is_binding][["grammar", "construction", "delta_unc", "block",
                               "logz", "delta_cor"]]
    print(show.round(3).to_string(index=False))
    print(f"\n[test] wrote binding-correction figures + csv -> {OUT}")


if __name__ == "__main__":
    main()
