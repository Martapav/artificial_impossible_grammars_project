"""Chapter-2, subchapter 2: length and embedding-depth effects.

Separates "the model fails because the sentence is long / deep" from genuine
phenomenon difficulty, on the held-out ``set_3_ep12`` probe. Length is
``length_g`` (grammatical member, tokens); depth is ``embedding_depth_g`` (small
integer). The margin metric is the S[r]-corrected margin
``margin_corrected = margin - margin_b1`` (via
:func:`analysis.report_common.attach_pair_baselines`); cross-phenomenon
comparisons use **accuracy** (unitless), never raw nats.

Note on OOD: OOD is essentially a length shift (IOD length medians 7-22 vs OOD
~29-31), so it doubles as the natural test bed for the length-only counterfactual.

Figures -> ch2/sub2_length_depth/figures ; tables -> .../tables.

    dist_length_by_phenomenon      length distribution per phenomenon, IOD vs OOD
    dist_length_x_depth            2-D length x depth support per phenomenon; the
                                   table also carries the per-phenomenon
                                   length-depth correlation (collinearity)
    slice_controlled               depth-fixed / length-fixed slices, phenomena
                                   compared inside each cell (length-CONTROLLED)
    acc_vs_length_by_phenomenon    per-phenomenon accuracy vs length, IOD vs OOD
    acc_vs_depth_by_phenomenon     per-phenomenon accuracy vs depth, IOD vs OOD
    partial_effects                per-phenomenon standardized partial effects of
                                   length vs depth on the corrected margin (OLS)
    iod_ood_counterfactual         how much of the OOD accuracy drop length alone
                                   (and depth alone) predicts

Run:  python -m analysis.reports.ch2.report_ch2_length_depth
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis.report_common import (
    GRAMMARS, GRAMMAR_COLOR, SPLIT_DISPLAY, SUBCASE_DISPLAY, SUBCASE_ORDER,
    attach_pair_baselines, attach_verb_counts, grammar_handles, load_probe_pairs,
    savefig, ch2_dirs,
)

SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]
SPLIT_COLOR = {"structural_probe_IOD": "#1f77b4", "structural_probe_OOD": "#d62728"}
FLOOR = 50                      # minimum pairs per reported cell
LEN_COL, DEPTH_COL = "length_g", "embedding_depth_g"
VERB_COL = "n_verbs_g"

# Shared depth axis for the cross-grammar depth plots. Tree grammars carry a
# native embedding depth; positional grammars have no hierarchical depth, so they
# are placed on the same axis via verb count (n_verbs ~= 2*depth + 1, verified; at
# matched verb-count H and P carry near-identical mean depth).
TREE_GRAMMARS = ("H", "Hprime")
TOKEN_GRAMMARS = ("P", "Lprime")
DEPTH_LEVELS = (0, 1, 2, 3)
VERB_TOL = 1                     # token-family band is [2d+1 - tol, 2d+1 + tol]


def _verb_for_depth(d: int) -> int:
    return 2 * d + 1


def _depth_cell(df, grammar, d):
    """Row mask selecting grammar ``grammar``'s depth-level-``d`` cell of ``df``.

    Tree family: native ``embedding_depth_g == d``. Token family: verb-count band
    around ``2d+1`` (the depth proxy), since positional depth is not hierarchical.
    """
    base = df.grammar == grammar
    if grammar in TOKEN_GRAMMARS:
        v = _verb_for_depth(d)
        return base & df[VERB_COL].between(v - VERB_TOL, v + VERB_TOL)
    return base & (df[DEPTH_COL] == d)


def _phenomena(pairs):
    return [p for p in SUBCASE_ORDER if p in set(pairs["phenomenon"])]


def _facets(n, ncol=3):
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.0 * nrow),
                             squeeze=False)
    return fig, axes, nrow, ncol


# --- 1. length distribution per phenomenon, IOD vs OOD ---------------------

def fig_dist_length(pairs, fig_dir, tab_dir):
    phen = _phenomena(pairs)
    fig, axes, nrow, ncol = _facets(len(phen))
    edges = np.arange(pairs[LEN_COL].min(), pairs[LEN_COL].max() + 2)
    rows = []
    for i, ph in enumerate(phen):
        ax = axes[i // ncol][i % ncol]
        for sp in SPLITS:
            d = pairs[(pairs.phenomenon == ph) & (pairs.split == sp)][LEN_COL]
            if d.empty:
                continue
            ax.hist(d, bins=edges, density=True, histtype="step", lw=1.5,
                    color=SPLIT_COLOR[sp], label=SPLIT_DISPLAY[sp])
            q = np.percentile(d, [10, 50, 90])
            rows.append(dict(phenomenon=ph, split=SPLIT_DISPLAY[sp], n=int(d.size),
                             p10=q[0], median=q[1], p90=q[2]))
        ax.set_title(SUBCASE_DISPLAY[ph], fontsize=9)
        ax.grid(alpha=0.2)
        if i % ncol == 0:
            ax.set_ylabel("density")
    for j in range(len(phen), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][ncol - 1].legend(fontsize=8, loc="upper right")
    fig.supxlabel("length (tokens)", fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir / "dist_length_by_phenomenon.png")
    pd.DataFrame(rows).to_csv(tab_dir / "dist_length_by_phenomenon.csv", index=False)


# --- 2. 2-D length x depth support + collinearity --------------------------

def fig_length_x_depth(pairs, fig_dir, tab_dir):
    phen = _phenomena(pairs)
    depths = np.arange(int(pairs[DEPTH_COL].min()), int(pairs[DEPTH_COL].max()) + 1)
    len_edges = np.unique(np.quantile(pairs[LEN_COL], np.linspace(0, 1, 11)))
    fig, axes, nrow, ncol = _facets(len(phen))
    rows = []
    for i, ph in enumerate(phen):
        ax = axes[i // ncol][i % ncol]
        d = pairs[pairs.phenomenon == ph]
        H, _, _ = np.histogram2d(d[DEPTH_COL], d[LEN_COL],
                                 bins=[np.append(depths, depths[-1] + 1) - 0.5, len_edges])
        im = ax.imshow(H, aspect="auto", origin="lower", cmap="viridis",
                       extent=(len_edges[0], len_edges[-1], depths[0] - 0.5, depths[-1] + 0.5))
        ax.set_yticks(depths)
        r_p = d[[LEN_COL, DEPTH_COL]].corr().iloc[0, 1]
        r_s = d[[LEN_COL, DEPTH_COL]].corr(method="spearman").iloc[0, 1]
        ax.set_title(f"{SUBCASE_DISPLAY[ph]}  (r={r_p:.2f})", fontsize=9)
        if i % ncol == 0:
            ax.set_ylabel("depth")
        # per (phenomenon, depth) length support + corr, for the table.
        for dep in depths:
            dd = d[d[DEPTH_COL] == dep][LEN_COL]
            if dd.empty:
                continue
            rows.append(dict(phenomenon=ph, depth=int(dep), n=int(dd.size),
                             len_min=int(dd.min()), len_median=float(dd.median()),
                             len_max=int(dd.max()), pearson_len_depth=float(r_p),
                             spearman_len_depth=float(r_s)))
    for j in range(len(phen), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.supxlabel("length (tokens)", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.02, label="count")
    savefig(fig, fig_dir / "dist_length_x_depth.png")
    pd.DataFrame(rows).to_csv(tab_dir / "length_depth_joint.csv", index=False)


# --- 3. controlled slices: depth-fixed / length-fixed ----------------------

def _binned(df, xcol, edges, value, floor=FLOOR):
    """Per bin: center, mean(value), n; keep bins with n >= floor. Pools rows."""
    cats = pd.cut(df[xcol], bins=edges, include_lowest=True)
    g = df.groupby(cats, observed=True)
    out = pd.DataFrame({"mean": g[value].mean(), "n": g.size(),
                        "center": g[xcol].mean()}).dropna()
    return out[out["n"] >= floor]


def fig_slice_controlled(pairs, fig_dir, tab_dir):
    phen = _phenomena(pairs)
    modal_depth = int(pairs[DEPTH_COL].value_counts().idxmax())
    lo, hi = np.percentile(pairs[LEN_COL], [40, 60])
    len_band = (float(lo), float(hi))
    len_edges = np.unique(np.quantile(pairs[LEN_COL], np.linspace(0, 1, 9)))
    depths = np.arange(int(pairs[DEPTH_COL].min()), int(pairs[DEPTH_COL].max()) + 1)
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8), sharex="col")
    rows = []
    for ri, value in enumerate(("acc", "margin_corrected")):
        # col 0: depth fixed at modal_depth, vary length (pool splits+grammars).
        axL = axes[ri][0]
        base = pairs[pairs[DEPTH_COL] == modal_depth]
        for pi, ph in enumerate(phen):
            b = _binned(base[base.phenomenon == ph], LEN_COL, len_edges, value)
            if b.empty:
                continue
            axL.plot(b["center"], b["mean"], marker="o", ms=4, lw=1.5,
                     color=cmap(pi % 10), label=SUBCASE_DISPLAY[ph])
            for _, r in b.iterrows():
                rows.append(dict(control="depth_fixed", fixed=f"depth={modal_depth}",
                                 x=r["center"], phenomenon=ph, metric=value,
                                 value=r["mean"], n=int(r["n"])))
        axL.set_ylabel("accuracy" if value == "acc" else "corrected margin (nats)")
        # col 1: length fixed to central band, vary depth.
        axR = axes[ri][1]
        base = pairs[(pairs[LEN_COL] >= len_band[0]) & (pairs[LEN_COL] <= len_band[1])]
        for pi, ph in enumerate(phen):
            sub = base[base.phenomenon == ph]
            g = sub.groupby(DEPTH_COL, observed=True)
            m = g[value].mean()
            n = g.size()
            keep = n[n >= FLOOR].index
            if len(keep) == 0:
                continue
            axR.plot(list(keep), m.loc[keep].to_numpy(), marker="o", ms=4, lw=1.5,
                     color=cmap(pi % 10))
            for dep in keep:
                rows.append(dict(control="length_fixed",
                                 fixed=f"len in [{len_band[0]:.0f},{len_band[1]:.0f}]",
                                 x=int(dep), phenomenon=ph, metric=value,
                                 value=float(m.loc[dep]), n=int(n.loc[dep])))
        if value == "acc":
            axL.axhline(0.5, color="0.4", lw=0.8, ls=":")
            axR.axhline(0.5, color="0.4", lw=0.8, ls=":")
        for ax in (axL, axR):
            ax.grid(alpha=0.25)
        axR.set_xticks(depths)
    axes[0][0].set_title(f"depth fixed = {modal_depth}", fontsize=10)
    axes[0][1].set_title(f"length fixed in [{len_band[0]:.0f}, {len_band[1]:.0f}]",
                         fontsize=10)
    axes[1][0].set_xlabel("length (tokens)")
    axes[1][1].set_xlabel("depth")
    axes[0][0].legend(fontsize=7, loc="best", ncol=2)
    fig.tight_layout()
    savefig(fig, fig_dir / "slice_controlled.png")
    pd.DataFrame(rows).to_csv(tab_dir / "slices.csv", index=False)


# --- 4. per-phenomenon accuracy vs length / depth, IOD vs OOD --------------

def _acc_curve_across_seed(df, xcol, *, edges=None, integer=False, floor=FLOOR):
    """Per bin: across-seed accuracy with a seed min..max band.

    The line is the mean of the per-seed bin accuracies (each of the 5 seeds weighs
    equally); the band is their seed-to-seed spread -- no within-seed item
    resampling. Bins are the integer x-values (``integer``, for depth) or
    ``edges``-cut intervals (length); a bin is kept when its pooled item count
    clears ``floor``.
    """
    d = df.copy()
    d["_bin"] = d[xcol] if integer else pd.cut(d[xcol], bins=edges, include_lowest=True)
    counts = d.groupby("_bin", observed=True).size()
    per_seed = d.groupby(["seed", "_bin"], observed=True)["acc"].mean().reset_index()
    stat = per_seed.groupby("_bin", observed=True)["acc"].agg(mean="mean", lo="min",
                                                              hi="max")
    centre = d.groupby("_bin", observed=True)[xcol].mean()
    keep = (counts.reindex(stat.index).fillna(0) >= floor).values
    stat, centre = stat[keep], centre.reindex(stat.index)[keep]
    return (centre.to_numpy(dtype=float), stat["mean"].to_numpy(),
            stat["lo"].to_numpy(), stat["hi"].to_numpy())


def fig_acc_vs(pairs, fig_dir, tab_dir, axis):
    phen = _phenomena(pairs)
    fig, axes, nrow, ncol = _facets(len(phen))
    len_edges = np.unique(np.quantile(pairs[LEN_COL], np.linspace(0, 1, 9)))
    depths = np.arange(int(pairs[DEPTH_COL].min()), int(pairs[DEPTH_COL].max()) + 1)
    rows = []
    for i, ph in enumerate(phen):
        ax = axes[i // ncol][i % ncol]
        for sp in SPLITS:
            d = pairs[(pairs.phenomenon == ph) & (pairs.split == sp)]
            if d.empty:
                continue
            if axis == "length":
                x, m, lo, hi = _acc_curve_across_seed(d, LEN_COL, edges=len_edges)
            else:
                x, m, lo, hi = _acc_curve_across_seed(d, DEPTH_COL, integer=True)
            if len(x) == 0:
                continue
            ax.plot(x, m, marker="o", ms=3, lw=1.4, color=SPLIT_COLOR[sp],
                    label=SPLIT_DISPLAY[sp])
            ax.fill_between(x, lo, hi, color=SPLIT_COLOR[sp], alpha=0.15, lw=0)
            for xi, mi, loi, hii in zip(x, m, lo, hi):
                rows.append(dict(phenomenon=ph, split=SPLIT_DISPLAY[sp], axis=axis,
                                 x=float(xi), acc=float(mi), seed_lo=float(loi),
                                 seed_hi=float(hii)))
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_title(SUBCASE_DISPLAY[ph], fontsize=9)
        ax.grid(alpha=0.25)
        if axis == "depth":
            ax.set_xticks(depths)
        if i % ncol == 0:
            ax.set_ylabel("accuracy")
    for j in range(len(phen), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][ncol - 1].legend(fontsize=8, loc="lower left")
    fig.supxlabel("sentence length (tokens)" if axis == "length" else "embedding depth",
                  fontsize=9)
    fig.tight_layout()
    savefig(fig, fig_dir / f"acc_vs_{axis}_by_phenomenon.png")
    pd.DataFrame(rows).to_csv(tab_dir / f"acc_vs_{axis}.csv", index=False)


# --- 5. per-phenomenon partial effects (standardized OLS on the margin) -----

def _zscore(s):
    sd = s.std(ddof=0)
    return (s - s.mean()) / sd if sd > 0 else s * 0.0


def fig_partial_effects(pairs, fig_dir, tab_dir):
    """Standardized partial effect of length vs depth on the corrected margin.

    Accuracy is at ceiling for several phenomena (logistic would separate), so the
    partial effects are read on the S[r]-corrected margin, z-scored WITHIN each
    phenomenon so coefficients are comparable across phenomena. C(grammar) and
    C(split) are controls, so each slope is a within-grammar, within-split effect.
    """
    phen = _phenomena(pairs)
    rows = []
    for ph in phen:
        d = pairs[pairs.phenomenon == ph].copy()
        d["z_margin"] = _zscore(d["margin_corrected"])
        d["z_len"] = _zscore(d[LEN_COL].astype(float))
        d["z_depth"] = _zscore(d[DEPTH_COL].astype(float))
        model = smf.ols("z_margin ~ z_len + z_depth + z_len:z_depth "
                        "+ C(grammar) + C(split)", data=d).fit()
        ci = model.conf_int()
        for term in ("z_len", "z_depth", "z_len:z_depth"):
            rows.append(dict(phenomenon=ph, term=term, coef=float(model.params[term]),
                             ci_lo=float(ci.loc[term, 0]), ci_hi=float(ci.loc[term, 1]),
                             pval=float(model.pvalues[term]), n=int(d.shape[0])))
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "regression_coeffs.csv", index=False)

    main = tab[tab.term.isin(["z_len", "z_depth"])]
    labels = {"z_len": "length", "z_depth": "depth"}
    colors = {"z_len": "#1b7837", "z_depth": "#762a83"}
    y = np.arange(len(phen))
    fig, ax = plt.subplots(figsize=(8, 0.9 * len(phen) + 1.5))
    for term, off in (("z_len", -0.16), ("z_depth", 0.16)):
        sub = main[main.term == term].set_index("phenomenon").reindex(phen)
        ax.errorbar(sub["coef"], y + off,
                    xerr=[sub["coef"] - sub["ci_lo"], sub["ci_hi"] - sub["coef"]],
                    fmt="o", ms=6, color=colors[term], capsize=3, lw=1.2,
                    label=labels[term])
    ax.axvline(0, color="black", lw=1)
    ax.set_yticks(y, [SUBCASE_DISPLAY[p] for p in phen])
    ax.invert_yaxis()
    ax.set_xlabel("standardized partial effect on corrected margin (per SD)")
    ax.grid(alpha=0.25, axis="x")
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    savefig(fig, fig_dir / "partial_effects.png")


# --- 5b. grammar comparison at matched lengths (3 IOD + 3 OOD) --------------

def _length_targets(series):
    """Up to three distinct integer lengths at the 25/50/75th percentiles."""
    q = np.percentile(series, [25, 50, 75])
    return sorted({int(round(v)) for v in q})


def fig_acc_by_grammar_at_lengths(pairs, fig_dir, tab_dir, tol=1, floor=FLOOR,
                                  seed=None, floor_seed=20):
    """Per phenomenon, accuracy of each grammar at three IOD and three OOD lengths.

    Length held (nearly) constant -- each target L is a +/-`tol`-token bin -- so the
    remaining spread across the bars at one L is a pure between-grammar difference.

    Between-seed variability is shown by REPRODUCTION, not error bars: with
    ``seed`` set, the bars are that single model's cell means (a per-seed floor
    ``floor_seed``), written to ``fig_dir/by_seed/..._s<seed>.png``. The column grid
    (target lengths, floor) is defined on the full pooled data so every per-seed
    panel lines up with the pooled reference figure.
    """
    phen = _phenomena(pairs)
    fig, axes, nrow, ncol = _facets(len(phen))
    w = 0.8 / len(GRAMMARS)
    fl = floor if seed is None else floor_seed
    rows = []
    for i, ph in enumerate(phen):
        ax = axes[i // ncol][i % ncol]
        d_all = pairs[pairs.phenomenon == ph]                  # columns: all seeds
        d = d_all if seed is None else d_all[d_all.seed == seed]  # bars: this seed
        # ordered (split, target length) columns: IOD three, then OOD three.
        cols = []
        for sp in SPLITS:
            for L in _length_targets(d_all[d_all.split == sp][LEN_COL]):
                cols.append((sp, L))
        n_iod = sum(1 for sp, _ in cols if sp == "structural_probe_IOD")
        x = np.arange(len(cols))
        for gi, g in enumerate(GRAMMARS):
            xs, ys = [], []
            for ci, (sp, L) in enumerate(cols):
                cell = d[(d.grammar == g) & (d.split == sp)
                         & (d[LEN_COL] >= L - tol) & (d[LEN_COL] <= L + tol)]
                if len(cell) < fl:
                    continue
                acc = float(cell["acc"].mean())
                xs.append(ci + (gi - (len(GRAMMARS) - 1) / 2) * w)
                ys.append(acc)
                rows.append(dict(phenomenon=ph, split=SPLIT_DISPLAY[sp], seed=seed,
                                 target_length=L, grammar=g, acc=acc, n=int(len(cell))))
            ax.bar(xs, ys, width=w * 0.9, color=GRAMMAR_COLOR[g])
        if n_iod < len(cols):                       # shade the OOD block
            ax.axvspan(n_iod - 0.5, len(cols) - 0.5, color="0.85", alpha=0.35, zorder=0)
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(x, [str(L) for _, L in cols], fontsize=8)
        ax.set_title(SUBCASE_DISPLAY[ph], fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        if i % ncol == 0:
            ax.set_ylabel("accuracy")
        # IOD / OOD group labels above the panel.
        if cols:
            ax.text((n_iod - 1) / 2, 1.04, "IOD", ha="center", va="bottom",
                    fontsize=7, color="0.3")
            ax.text((n_iod + len(cols) - 1) / 2, 1.04, "OOD", ha="center",
                    va="bottom", fontsize=7, color="0.3")
    for j in range(len(phen), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][ncol - 1].legend(handles=grammar_handles(), fontsize=7, loc="lower left",
                             ncol=2)
    fig.supxlabel("sentence length (tokens), matched per column", fontsize=9)
    fig.tight_layout()
    stem = "acc_by_grammar_at_lengths"
    if seed is None:
        savefig(fig, fig_dir / f"{stem}.png")
        pd.DataFrame(rows).to_csv(tab_dir / f"{stem}.csv", index=False)
    else:
        savefig(fig, fig_dir / "by_seed" / f"{stem}__s{seed}.png")
        (tab_dir / "by_seed").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(tab_dir / "by_seed" / f"{stem}__s{seed}.csv",
                                  index=False)


# --- 5c. grammar comparison at matched structural depth ---------------------

def fig_acc_by_grammar_at_depths(pairs, fig_dir, tab_dir, floor=FLOOR,
                                 seed=None, floor_seed=20):
    """Depth analogue of :func:`fig_acc_by_grammar_at_lengths`.

    Each column is a structural depth level ``d``; tree grammars are selected by
    native ``embedding_depth``, positional grammars by the verb-count proxy
    (``2d+1``), so the two families sit on one shared depth axis (see
    :func:`_depth_cell`). Bars compare grammars within a column -- a between-grammar
    difference at matched depth.

    Between-seed variability is shown by REPRODUCTION (see
    :func:`fig_acc_by_grammar_at_lengths`): with ``seed`` set, bars are that single
    model's cell means (per-seed floor ``floor_seed``), written to
    ``fig_dir/by_seed/..._s<seed>.png``; the column grid is defined on the full
    pooled data so per-seed panels line up with the pooled reference.
    """
    phen = _phenomena(pairs)
    fig, axes, nrow, ncol = _facets(len(phen))
    w = 0.8 / len(GRAMMARS)
    fl = floor if seed is None else floor_seed
    rows = []
    for i, ph in enumerate(phen):
        ax = axes[i // ncol][i % ncol]
        d_all = pairs[pairs.phenomenon == ph]                  # columns: all seeds
        d = d_all if seed is None else d_all[d_all.seed == seed]  # bars: this seed
        # ordered (split, depth) columns: IOD levels first, then OOD. Keep a level
        # only where some grammar cell clears the (pooled) floor.
        cols = []
        for sp in SPLITS:
            dsp = d_all[d_all.split == sp]
            for lvl in DEPTH_LEVELS:
                if any(int(_depth_cell(dsp, g, lvl).sum()) >= floor for g in GRAMMARS):
                    cols.append((sp, lvl))
        n_iod = sum(1 for sp, _ in cols if sp == "structural_probe_IOD")
        x = np.arange(len(cols))
        for gi, g in enumerate(GRAMMARS):
            xs, ys = [], []
            for ci, (sp, lvl) in enumerate(cols):
                dsp = d[d.split == sp]
                cell = dsp[_depth_cell(dsp, g, lvl)]
                if len(cell) < fl:
                    continue
                acc = float(cell["acc"].mean())
                xs.append(ci + (gi - (len(GRAMMARS) - 1) / 2) * w)
                ys.append(acc)
                if g in TOKEN_GRAMMARS:
                    crit, val = "verbs", f"{_verb_for_depth(lvl)}+/-{VERB_TOL}"
                else:
                    crit, val = "depth", str(lvl)
                rows.append(dict(phenomenon=ph, split=SPLIT_DISPLAY[sp], seed=seed,
                                 depth=lvl, grammar=g, criterion=crit, value=val,
                                 acc=acc, n=int(len(cell))))
            ax.bar(xs, ys, width=w * 0.9, color=GRAMMAR_COLOR[g])
        if n_iod < len(cols):                       # shade the OOD block
            ax.axvspan(n_iod - 0.5, len(cols) - 0.5, color="0.85", alpha=0.35, zorder=0)
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(x, [f"d={lvl}" for _, lvl in cols], fontsize=8)
        ax.set_title(SUBCASE_DISPLAY[ph], fontsize=9)
        ax.grid(alpha=0.25, axis="y")
        if i % ncol == 0:
            ax.set_ylabel("accuracy")
        if cols:
            ax.text((n_iod - 1) / 2, 1.04, "IOD", ha="center", va="bottom",
                    fontsize=7, color="0.3")
            ax.text((n_iod + len(cols) - 1) / 2, 1.04, "OOD", ha="center",
                    va="bottom", fontsize=7, color="0.3")
    for j in range(len(phen), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    axes[0][ncol - 1].legend(handles=grammar_handles(), fontsize=7, loc="lower left",
                             ncol=2)
    fig.supxlabel("structural depth d  (H/H': embedding depth; P/P': verb-count "
                  "proxy 2d+1), matched per column", fontsize=9)
    fig.tight_layout()
    stem = "acc_by_grammar_at_depths"
    if seed is None:
        savefig(fig, fig_dir / f"{stem}.png")
        pd.DataFrame(rows).to_csv(tab_dir / f"{stem}.csv", index=False)
    else:
        savefig(fig, fig_dir / "by_seed" / f"{stem}__s{seed}.png")
        (tab_dir / "by_seed").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(tab_dir / "by_seed" / f"{stem}__s{seed}.csv",
                                  index=False)


# --- 5d. length-vs-depth covariation companion -----------------------------

def _depth_proxy_col(grammar):
    return VERB_COL if grammar in TOKEN_GRAMMARS else DEPTH_COL


def fig_length_by_depth(pairs, fig_dir, tab_dir, floor=FLOOR):
    """How tightly length covaries with the depth axis of the depth plot.

    One line per grammar: mean ``length_g`` of its depth-level-``d`` cell (the same
    selection the depth plot bars on), with an inter-quartile band. The legend
    carries the within-grammar Pearson r between length and the grammar's depth
    proxy (embedding depth for tree, verb count for token) -- so the panel reads as
    the length confound behind every column of ``acc_by_grammar_at_depths``.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5))
    rows, corr_rows = [], []
    for g in GRAMMARS:
        dg = pairs[pairs.grammar == g]
        r = float(dg[LEN_COL].corr(dg[_depth_proxy_col(g)]))
        corr_rows.append(dict(grammar=g, r_len_depthproxy=r,
                              proxy=_depth_proxy_col(g), n=int(len(dg))))
        # per-phenomenon r, for the table (reuses the dist_length_x_depth pattern).
        for ph in _phenomena(dg):
            dph = dg[dg.phenomenon == ph]
            corr_rows.append(dict(grammar=g, phenomenon=ph,
                                  r_len_depthproxy=float(dph[LEN_COL].corr(
                                      dph[_depth_proxy_col(g)])),
                                  proxy=_depth_proxy_col(g), n=int(len(dph))))
        xs, ys, lo, hi = [], [], [], []
        for lvl in DEPTH_LEVELS:
            cell = dg[_depth_cell(dg, g, lvl)]
            if len(cell) < floor:
                continue
            q = np.percentile(cell[LEN_COL], [25, 50, 75])
            xs.append(lvl); ys.append(float(cell[LEN_COL].mean()))
            lo.append(q[0]); hi.append(q[2])
            rows.append(dict(grammar=g, depth=lvl, mean_len=float(cell[LEN_COL].mean()),
                             q25=float(q[0]), median=float(q[1]), q75=float(q[2]),
                             n=int(len(cell))))
        if xs:
            ax.plot(xs, ys, marker="o", ms=5, lw=1.8, color=GRAMMAR_COLOR[g])
            ax.fill_between(xs, lo, hi, color=GRAMMAR_COLOR[g], alpha=0.12, lw=0)
    ax.set_xticks(list(DEPTH_LEVELS), [f"d={d}" for d in DEPTH_LEVELS])
    ax.set_xlabel("structural depth d  (H/H': embedding depth; P/P': verb-count "
                  "proxy 2d+1)")
    ax.set_ylabel("length (tokens)")
    ax.grid(alpha=0.25)
    rmap = {c["grammar"]: c["r_len_depthproxy"] for c in corr_rows
            if "phenomenon" not in c}
    ax.legend(handles=grammar_handles(), fontsize=8, loc="upper left",
              title="grammar")
    # r annotations (kept out of the legend title so both families' proxies show).
    txt = "  ".join(f"r={rmap[g]:.2f}" for g in GRAMMARS if g in rmap)
    ax.set_title(f"length ↔ depth covariation   ({txt})", fontsize=10)
    fig.tight_layout()
    savefig(fig, fig_dir / "length_by_depth.png")
    pd.DataFrame(rows).to_csv(tab_dir / "length_by_depth.csv", index=False)
    pd.DataFrame(corr_rows).to_csv(tab_dir / "length_depth_covariation.csv", index=False)


# --- 6. IOD->OOD counterfactual: what does length (depth) alone predict? ----

def _iod_lookup(d_iod, xcol, integer):
    """Sorted (x_centers, acc) for the IOD accuracy curve; np.interp clamps ends."""
    if integer:
        g = d_iod.groupby(xcol, observed=True)["acc"].agg(["mean", "size"])
        g = g[g["size"] >= FLOOR]
        return g.index.to_numpy(dtype=float), g["mean"].to_numpy()
    edges = np.unique(np.quantile(d_iod[xcol], np.linspace(0, 1, 9)))
    b = _binned(d_iod, xcol, edges, "acc")
    order = np.argsort(b["center"].to_numpy())
    return b["center"].to_numpy()[order], b["mean"].to_numpy()[order]


def fig_counterfactual(pairs, fig_dir, tab_dir):
    phen = _phenomena(pairs)
    rows = []
    for ph in phen:
        d = pairs[pairs.phenomenon == ph]
        iod = d[d.split == "structural_probe_IOD"]
        ood = d[d.split == "structural_probe_OOD"]
        if iod.empty or ood.empty:
            continue
        rec = dict(phenomenon=ph, iod_acc=float(iod["acc"].mean()),
                   ood_acc=float(ood["acc"].mean()),
                   n_iod=int(len(iod)), n_ood=int(len(ood)))
        for axis, xcol, integer in (("length", LEN_COL, False),
                                    ("depth", DEPTH_COL, True)):
            xs, ys = _iod_lookup(iod, xcol, integer)
            if len(xs) >= 2:
                pred = float(np.interp(ood[xcol].to_numpy(dtype=float), xs, ys).mean())
            else:
                pred = float("nan")
            rec[f"pred_ood_from_{axis}"] = pred
        rows.append(rec)
    tab = pd.DataFrame(rows)
    tab.to_csv(tab_dir / "iod_ood_counterfactual.csv", index=False)

    phen_p = list(tab["phenomenon"])
    x = np.arange(len(phen_p))
    w = 0.2
    series = [("iod_acc", "IOD (actual)", "#4d4d4d"),
              ("pred_ood_from_length", "OOD pred. from length", "#1b7837"),
              ("pred_ood_from_depth", "OOD pred. from depth", "#762a83"),
              ("ood_acc", "OOD (actual)", "#d62728")]
    fig, ax = plt.subplots(figsize=(1.4 * len(phen_p) + 3, 5))
    for k, (col, lab, c) in enumerate(series):
        ax.bar(x + (k - 1.5) * w, tab[col], width=w, color=c, label=lab)
    ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
    ax.set_xticks(x, [SUBCASE_DISPLAY[p] for p in phen_p], fontsize=8, rotation=20,
                  ha="right")
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    fig.tight_layout()
    savefig(fig, fig_dir / "iod_ood_counterfactual.png")


def main() -> None:
    fig_dir, tab_dir = ch2_dirs("sub2_length_depth")
    pairs = attach_verb_counts(attach_pair_baselines(load_probe_pairs()))

    fig_dist_length(pairs, fig_dir, tab_dir)
    fig_length_x_depth(pairs, fig_dir, tab_dir)
    fig_slice_controlled(pairs, fig_dir, tab_dir)
    fig_acc_vs(pairs, fig_dir, tab_dir, "length")
    fig_acc_vs(pairs, fig_dir, tab_dir, "depth")
    fig_acc_by_grammar_at_lengths(pairs, fig_dir, tab_dir)      # pooled reference
    fig_acc_by_grammar_at_depths(pairs, fig_dir, tab_dir)       # pooled reference
    # per-seed reproductions (coherence check; reference in main text, rest in
    # appendix) -> figures/by_seed/, tables/by_seed/.
    for s in sorted(pairs["seed"].unique()):
        fig_acc_by_grammar_at_lengths(pairs, fig_dir, tab_dir, seed=int(s))
        fig_acc_by_grammar_at_depths(pairs, fig_dir, tab_dir, seed=int(s))
    fig_length_by_depth(pairs, fig_dir, tab_dir)
    fig_partial_effects(pairs, fig_dir, tab_dir)
    fig_counterfactual(pairs, fig_dir, tab_dir)

    n_seeds = pairs["seed"].nunique()
    print(f"[ch2/sub2] wrote dist_length, dist_length_x_depth, slice_controlled, "
          f"acc_vs_length, acc_vs_depth, acc_by_grammar_at_lengths, "
          f"acc_by_grammar_at_depths, length_by_depth, partial_effects, "
          f"iod_ood_counterfactual -> {fig_dir}  (+ {n_seeds} per-seed "
          f"reproductions of each grammar-bar plot -> {fig_dir}/by_seed)  "
          f"(+ tables -> {tab_dir})")


if __name__ == "__main__":
    main()
