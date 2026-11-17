"""Seed-level TOST equivalence analysis on the probe grammaticality judgements.

The thesis reports a null -- "no general learnability penalty for positional
grammars" -- but backs it with a *failure to reject* the standard null, which is
not evidence of absence. This module supplies the missing inferential support:
for each (subcase, split) it asks, per grammar-family contrast, whether the
cross-grammar accuracy difference is **statistically equivalent** within a
pre-declared bound DELTA, and reports a retrospective minimum detectable effect
(MDE) so the genuinely underpowered cells are labelled rather than counted as
nulls.

Method (two one-sided tests, Lakens' NHST x TOST cross-classification):
  lower  H0: true diff <= -DELTA   t_low = (dbar + DELTA)/se,  p_low = P(T > t_low)
  upper  H0: true diff >= +DELTA   t_upp = (dbar - DELTA)/se,  p_upp = P(T < t_upp)
  p_tost = max(p_low, p_upp);  equivalence at ALPHA iff p_tost < ALPHA, iff the
  90% (= 1 - 2*ALPHA) CI on the difference lies wholly inside (-DELTA, +DELTA).
Combined with the ordinary two-sided test against 0 this gives four verdicts:
equivalent / different / trivial / inconclusive (see `classify`).

The unit of analysis is the **trained model (seed)**, n = 5 per cell -- never the
300 minimal pairs. Pooling pairs across seeds is pseudoreplication: the same item
list recurs under every seed, so a pair-level n answers a question about the item
sample, not about trained models. Same rule as `analysis/reports/heldout/report_test_permutation`.

Seeds are shared across grammars (same seed -> same initial weights; only the
training corpus differs), so the test is **paired**. A Welch two-sample fallback
exists for misaligned seed sets; on the current set_3_ep12 data every cell holds
seeds 0-4, so that branch never fires.

Zero-variance guard: `se == 0` is handled explicitly (exact-equal -> equivalent,
nonzero-but-<DELTA -> trivial, >=DELTA -> different) rather than producing NaN.
Note this is *not* the common case here: the 46/56 zero cells of the seed-variance
section are zero in the *variance component* after subtracting binomial sampling
noise; the raw SD is zero in only 8/56 cells and only 2/28 contrasts degenerate.

No multiple-comparison correction is applied -- each contrast is characterised
descriptively; this is not a single omnibus hypothesis.

Output -> results/analysis/evaluation/structural_probes/ch2/sub4_tost/
    tables/data_long.csv      the 56 x 5 per-seed accuracies the tests consume
    tables/tost_results.csv   one row per contrast (28)
    tost_summary.md           paste-ready appendix prose
    figures/tost_forest_{IOD,OOD}.png

Run:  python -m analysis.reports.ch2.report_ch2_tost
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.report_common import (
    GRAMMAR_DISPLAY, SPLIT_DISPLAY, SUBCASE_DISPLAY, SUBCASE_ORDER,
    ch2_dirs, load_probe_pairs, savefig,
)
from analysis.reports.ch2.report_ch2_seed_variance import seed_components

# --- config -----------------------------------------------------------------

DELTA  = 0.05          # equivalence bound, accuracy points (see `seed_spread_stats`)
ALPHA  = 0.05
POWER  = 0.80
PAIRED = True

# The spec's ("H","P") and ("H_prime","P_prime"). `Lprime` is the DATA KEY of the
# grammar displayed as P'; it is never renamed in the data.
GRAMMAR_PAIRS = [("H", "P"), ("Hprime", "Lprime")]
SPLITS    = ["structural_probe_IOD", "structural_probe_OOD"]
MIN_SEEDS = 3          # below this a contrast is skipped, not tested

VERDICTS = ["equivalent", "trivial", "different", "inconclusive"]
VERDICT_COLOR = {
    "equivalent":   "#1b7837",   # green  -- small and precisely estimated
    "trivial":      "#7fbc41",   # olive  -- nonzero but below DELTA
    "different":    "#d7301f",   # red    -- real, non-negligible
    "inconclusive": "#999999",   # grey   -- underpowered
}

RESULT_COLUMNS = [
    "split", "subcase", "grammar_a", "grammar_b", "paired", "n_seeds",
    "diff", "sd_diff", "se", "df", "ci90_low", "ci90_high", "delta",
    "p_nhst", "p_tost", "verdict", "degenerate", "mde_80",
]


# --- statistical core (authoritative; do not alter the numbers) --------------

def _paired_stats(a, b, delta=DELTA, alpha=ALPHA):
    """Paired TOST + two-sided NHST on d = a - b, with an se == 0 guard."""
    d = np.asarray(a, float) - np.asarray(b, float)
    n = d.size
    df = n - 1
    dbar = float(d.mean())
    sd = float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    if se == 0.0:
        p_nhst = 0.0 if dbar != 0.0 else 1.0
        p_tost = 0.0 if abs(dbar) < delta else 1.0
        return dict(diff=dbar, sd_diff=sd, se=0.0, df=df,
                    p_nhst=p_nhst, p_tost=p_tost,
                    ci90_low=dbar, ci90_high=dbar, degenerate=True)
    p_low = stats.t.sf((dbar + delta) / se, df)    # H0: mu <= -delta
    p_upp = stats.t.cdf((dbar - delta) / se, df)   # H0: mu >= +delta
    p_tost = max(p_low, p_upp)
    p_nhst = 2 * stats.t.sf(abs(dbar / se), df)
    tcrit = stats.t.ppf(1 - alpha, df)             # 90% CI when alpha = 0.05
    return dict(diff=dbar, sd_diff=sd, se=se, df=df,
                p_nhst=p_nhst, p_tost=p_tost,
                ci90_low=dbar - tcrit * se, ci90_high=dbar + tcrit * se,
                degenerate=False)


def _welch_stats(a, b, delta=DELTA, alpha=ALPHA):
    """Unequal-variance two-sample TOST -- the fallback when seeds do not align."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = a.size, b.size
    diff = float(a.mean() - b.mean())
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = float(np.sqrt(va / na + vb / nb))
    if se == 0.0:
        p_nhst = 0.0 if diff != 0.0 else 1.0
        p_tost = 0.0 if abs(diff) < delta else 1.0
        return dict(diff=diff, sd_diff=0.0, se=0.0, df=na + nb - 2,
                    p_nhst=p_nhst, p_tost=p_tost,
                    ci90_low=diff, ci90_high=diff, degenerate=True)
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1))
    p_low = stats.t.sf((diff + delta) / se, df)
    p_upp = stats.t.cdf((diff - delta) / se, df)
    p_tost = max(p_low, p_upp)
    p_nhst = 2 * stats.t.sf(abs(diff / se), df)
    tcrit = stats.t.ppf(1 - alpha, df)
    return dict(diff=diff, sd_diff=float(np.sqrt(0.5 * (va + vb))), se=se, df=df,
                p_nhst=p_nhst, p_tost=p_tost,
                ci90_low=diff - tcrit * se, ci90_high=diff + tcrit * se,
                degenerate=False)


def classify(p_nhst, p_tost, alpha=ALPHA):
    """Lakens' four-way cross-classification of NHST against TOST."""
    sig, equiv = p_nhst < alpha, p_tost < alpha
    if equiv and not sig:
        return "equivalent"      # the clean null: small and precisely estimated
    if sig and not equiv:
        return "different"       # a real, non-negligible effect
    if sig and equiv:
        return "trivial"         # nonzero but smaller than DELTA
    return "inconclusive"        # underpowered / indeterminate


def mde_paired(sd_diff, n, alpha=ALPHA, power=POWER, df=None):
    """Smallest true difference detectable at `power` given the OBSERVED noise.

    A sensitivity statement from the observed SD -- not observed-effect "post-hoc
    power". `df` defaults to the paired n - 1; the Welch branch passes its own.
    """
    if sd_diff == 0.0:
        return 0.0
    if df is None:
        df = n - 1
    dcrit = stats.t.ppf(1 - alpha / 2, df) + stats.t.ppf(power, df)
    return float(dcrit / np.sqrt(n) * sd_diff)


# --- input ------------------------------------------------------------------

def label_map(long: pd.DataFrame) -> dict:
    """Raw data key -> canonical/display label, for each of the three axes.

    Built from what is actually present in the frame, never hardcoded.
    """
    return {
        "grammar": {g: GRAMMAR_DISPLAY.get(g, g) for g in sorted(long["grammar"].unique())},
        "split":   {s: SPLIT_DISPLAY.get(s, s) for s in sorted(long["split"].unique())},
        "subcase": {c: SUBCASE_DISPLAY.get(c, c) for c in sorted(long["subcase"].unique())},
    }


def build_data_long() -> pd.DataFrame:
    """Per-seed pair accuracy: one row per (grammar, subcase, split, seed).

    Delegates to :func:`analysis.reports.ch2.report_ch2_seed_variance.seed_components`, so
    these are by construction the same five dots the seed-variance section plots
    -- the accuracy is never recomputed here.
    """
    _, seed_points = seed_components(load_probe_pairs(splits=tuple(SPLITS)))
    long = (seed_points
            .rename(columns={"phenomenon": "subcase", "acc": "accuracy"})
            .loc[:, ["grammar", "subcase", "split", "seed", "accuracy"]]
            .copy())
    long["grammar"] = long["grammar"].astype(str)
    long["subcase"] = long["subcase"].astype(str)
    long["split"] = long["split"].astype(str)
    long["seed"] = long["seed"].astype(int)
    long["accuracy"] = long["accuracy"].astype(float)
    return long.sort_values(["split", "subcase", "grammar", "seed"]).reset_index(drop=True)


def grid_report(long: pd.DataFrame) -> None:
    """Print the LABEL_MAP and a grid sanity check. Warns; never raises."""
    lm = label_map(long)
    print("[ch2/sub4 tost] LABEL_MAP")
    for axis, mapping in lm.items():
        for raw, canon in mapping.items():
            print(f"    {axis:8s} {raw!r:24s} -> {canon!r}")

    cells = long.groupby(["grammar", "subcase", "split"], observed=True)["seed"].nunique()
    n_g, n_c, n_s = (long[c].nunique() for c in ("grammar", "subcase", "split"))
    print(f"[ch2/sub4 tost] grid: {n_g} grammars x {n_c} subcases x {n_s} splits "
          f"= {n_g * n_c * n_s} expected cells, {len(cells)} found, "
          f"{len(long)} per-seed rows")
    bad = cells[cells != 5]
    if len(cells) != n_g * n_c * n_s or len(bad):
        print(f"    WARNING: {len(bad)} cell(s) do not have 5 seeds; "
              f"contrasts with < {MIN_SEEDS} aligned seeds will be skipped")
        for key, k in bad.items():
            print(f"      {key} -> {k} seed(s)")
    else:
        print(f"    all {len(cells)} cells have 5 seeds")


def seed_spread_stats(long: pd.DataFrame) -> dict:
    """Measured across-seed spread (max - min) per cell -- the empirical noise floor.

    This is what actually justifies DELTA, in place of a quoted approximation.
    """
    spread = (long.groupby(["subcase", "grammar", "split"], observed=True)["accuracy"]
              .agg(lambda s: s.max() - s.min()))
    widest = spread.sort_values(ascending=False).head(3)
    return dict(
        n_cells=int(len(spread)),
        median=float(spread.median()),
        p75=float(spread.quantile(0.75)),
        maximum=float(spread.max()),
        widest=[(sc, g, SPLIT_DISPLAY.get(sp, sp), float(v))
                for (sc, g, sp), v in widest.items()],
        n_zero_sd=int((long.groupby(["subcase", "grammar", "split"], observed=True)["accuracy"]
                       .std(ddof=1) == 0).sum()),
    )


# --- the contrast driver ----------------------------------------------------

def run_contrasts(long: pd.DataFrame) -> pd.DataFrame:
    """One row per (split, subcase, grammar_a, grammar_b). Full precision."""
    subcases = [c for c in SUBCASE_ORDER if c in set(long["subcase"])]
    rows = []
    for split in SPLITS:
        for subcase in subcases:
            cell = long[(long["split"] == split) & (long["subcase"] == subcase)]
            for ga, gb in GRAMMAR_PAIRS:
                a = cell[cell["grammar"] == ga].set_index("seed")["accuracy"].sort_index()
                b = cell[cell["grammar"] == gb].set_index("seed")["accuracy"].sort_index()
                aligned = PAIRED and len(a) == len(b) and set(a.index) == set(b.index)
                if aligned:
                    seeds = sorted(set(a.index))
                    va, vb = a.loc[seeds].to_numpy(), b.loc[seeds].to_numpy()
                    n_seeds = len(seeds)
                else:
                    va, vb = a.to_numpy(), b.to_numpy()
                    n_seeds = min(a.size, b.size)
                if n_seeds < MIN_SEEDS:
                    print(f"    SKIP {SPLIT_DISPLAY.get(split, split)}/{subcase} "
                          f"{ga}-{gb}: only {n_seeds} aligned seed(s)")
                    continue
                if aligned:
                    st = _paired_stats(va, vb)
                    mde = mde_paired(st["sd_diff"], n_seeds)
                else:
                    st = _welch_stats(va, vb)
                    mde = mde_paired(st["sd_diff"], n_seeds, df=st["df"])
                rows.append(dict(
                    split=SPLIT_DISPLAY.get(split, split), subcase=subcase,
                    grammar_a=ga, grammar_b=gb, paired=bool(aligned), n_seeds=n_seeds,
                    diff=st["diff"], sd_diff=st["sd_diff"], se=st["se"], df=st["df"],
                    ci90_low=st["ci90_low"], ci90_high=st["ci90_high"], delta=DELTA,
                    p_nhst=st["p_nhst"], p_tost=st["p_tost"],
                    verdict=classify(st["p_nhst"], st["p_tost"]),
                    degenerate=st["degenerate"], mde_80=mde))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


# --- figure -----------------------------------------------------------------

def _row_label(r) -> str:
    sub = SUBCASE_DISPLAY.get(r["subcase"], r["subcase"])
    return f"{sub}   {GRAMMAR_DISPLAY.get(r['grammar_a'], r['grammar_a'])}" \
           f"-{GRAMMAR_DISPLAY.get(r['grammar_b'], r['grammar_b'])}"


def fig_forest(res: pd.DataFrame, split: str, fig_dir) -> None:
    """Forest of the 90% CIs against the equivalence bound, one panel per split."""
    block = res[res["split"] == split].reset_index(drop=True)
    if block.empty:
        return
    y = np.arange(len(block))[::-1]          # first contrast at the top
    fig, ax = plt.subplots(figsize=(7.2, 0.34 * len(block) + 1.6))

    ax.axvspan(-DELTA, DELTA, color="#f0f0f0", zorder=0)
    for x, ls in ((-DELTA, "--"), (0.0, "-"), (DELTA, "--")):
        ax.axvline(x, color="#666666", ls=ls, lw=0.8, zorder=1)

    for yi, (_, r) in zip(y, block.iterrows()):
        c = VERDICT_COLOR[r["verdict"]]
        ax.plot([r["ci90_low"], r["ci90_high"]], [yi, yi], color=c, lw=1.6,
                solid_capstyle="butt", zorder=2)
        ax.plot([r["diff"]], [yi], "o", color=c, ms=5, zorder=3)

    ax.set_yticks(y)
    ax.set_yticklabels([_row_label(r) for _, r in block.iterrows()], fontsize=7.5)
    ax.set_xlabel("accuracy difference (grammar A - grammar B), 90% CI", fontsize=8)
    ax.set_title(f"Equivalence of cross-grammar probe accuracy — {split}",
                 fontsize=9.5, loc="left")
    ax.tick_params(axis="x", labelsize=7.5)
    ax.grid(alpha=0.2, axis="x")
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # Legend and method note sit in a band reserved at the foot of the figure, so
    # their height does not scale with the number of contrast rows.
    handles = [plt.Line2D([], [], color=VERDICT_COLOR[v], marker="o", ls="-", ms=5,
                          label=v) for v in VERDICTS if (block["verdict"] == v).any()]
    h = fig.get_figheight()                  # positions in inches, then normalised,
    fig.tight_layout(rect=(0, 0.62 / h, 1, 1))   # so the foot stays tight at any
    fig.legend(handles=handles, fontsize=7.5, frameon=False, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, 0.30 / h))   # row count
    fig.text(0.5, 0.10 / h,
             f"shaded band = equivalence region (|diff| < {DELTA:.2f}); "
             f"unit of analysis = seed (n = 5), not pair",
             ha="center", fontsize=7.5)
    savefig(fig, fig_dir / f"tost_forest_{split}.png")


# --- summary ----------------------------------------------------------------

def _fmt(r) -> str:
    return (f"`{SUBCASE_DISPLAY.get(r['subcase'], r['subcase'])}` "
            f"{GRAMMAR_DISPLAY.get(r['grammar_a'], r['grammar_a'])}"
            f"-{GRAMMAR_DISPLAY.get(r['grammar_b'], r['grammar_b'])}: "
            f"diff = {r['diff']:+.4f}, 90% CI [{r['ci90_low']:+.4f}, {r['ci90_high']:+.4f}]")


def write_summary(res: pd.DataFrame, long: pd.DataFrame, out_md) -> None:
    sp = seed_spread_stats(long)
    n_deg = int(res["degenerate"].sum())
    L = []
    A = L.append

    A("# Equivalence (TOST) analysis of cross-grammar probe accuracy\n")
    A("Generated by `python -m analysis.reports.ch2.report_ch2_tost`. "
      "Read-only: computed from the stored ep12 probe scores, no model was retrained.\n")

    A("## Design\n")
    A(f"The unit of analysis is the **trained model (seed)**: n = {int(res['n_seeds'].max())} "
      "per cell. It is *not* the 300 minimal pairs. Every seed scores the identical item "
      "list, so pooling pairs across seeds would be pseudoreplication — it would estimate "
      "the precision of the item sample, not the reproducibility of a trained model, and "
      "would understate the real model-to-model spread.\n")
    A("The five initialization seeds are shared across grammars (same seed -> same initial "
      "weights; only the training corpus differs), so seed *s* corresponds across grammars "
      "and the test is **paired**. A Welch two-sample fallback is implemented for the case "
      "of misaligned seed sets; on this data every cell holds seeds 0-4, so it never fires "
      f"({int((~res['paired']).sum())} of {len(res)} contrasts are unpaired).\n")
    A("Every difference is reported as **grammar A minus grammar B** in the order given, "
      "so a positive value favours the hierarchical member of the pair (H or H′) and "
      "a negative value favours the positional one (P or P′).\n")
    A(f"Equivalence is tested with two one-sided tests at alpha = {ALPHA}, equivalently by "
      f"checking whether the {int((1 - 2 * ALPHA) * 100)}% CI on the difference lies wholly "
      f"inside (-{DELTA}, +{DELTA}). Each contrast is also given the ordinary two-sided test "
      "against zero, and the two are cross-classified (Lakens): **equivalent** (small and "
      "precisely estimated), **different** (real and non-negligible), **trivial** (nonzero "
      "but below the bound), **inconclusive** (underpowered).\n")

    A("## The equivalence bound\n")
    A(f"**Delta = {DELTA}** accuracy points, declared in advance and used unchanged "
      "throughout. It is calibrated on this study's own between-model variability: across "
      f"the {sp['n_cells']} (grammar x subcase x split) cells, an unchanged rule's accuracy "
      f"moves by a median of {sp['median']:.3f} across seeds (75th percentile "
      f"{sp['p75']:.3f}), so a difference below Delta cannot be attributed to the grammar.\n")
    A(f"One caveat, measured rather than assumed: the spread is not uniform. It reaches "
      f"{sp['maximum']:.3f} in the widest cell; the three widest are "
      + "; ".join(f"{SUBCASE_DISPLAY.get(sc, sc)} {GRAMMAR_DISPLAY.get(g, g)} {s} "
                  f"{v:.3f}" for sc, g, s, v in sp["widest"])
      + f" \u2014 all of them OOD. Delta = {DELTA} therefore sits comfortably above the typical "
      "noise floor but **below the OOD worst case**, so OOD verdicts rest on a bound close "
      "to the OOD noise ceiling and should be read as the weaker of the two splits.\n")

    A("## Verdicts\n")
    for split in res["split"].unique():
        block = res[res["split"] == split]
        counts = block["verdict"].value_counts()
        A(f"### {split} ({len(block)} contrasts)\n")
        A(", ".join(f"**{v}** {int(counts.get(v, 0))}" for v in VERDICTS) + "\n")
        diff = block[block["verdict"] == "different"]
        A(f"*Different* ({len(diff)}):"
          + ("\n" + "\n".join(f"- {_fmt(r)}" for _, r in diff.iterrows()) + "\n"
             if len(diff) else " none.\n"))
        inc = block[block["verdict"] == "inconclusive"]
        A(f"*Inconclusive* ({len(inc)}):"
          + ("\n" + "\n".join(f"- {_fmt(r)}" for _, r in inc.iterrows()) + "\n"
             if len(inc) else " none.\n"))

    A("## Sensitivity (minimum detectable effect)\n")
    nz = res.loc[res["mde_80"] > 0, "mde_80"]
    A(f"At 80% power and the observed per-seed noise, the median detectable difference is "
      f"**{res['mde_80'].median():.4f}**, and {nz.min():.4f}-{nz.max():.4f} across the "
      f"{len(nz)} contrasts with nonzero between-seed variance (the remaining "
      f"{len(res) - len(nz)} have an identical difference in every seed, so their MDE is "
      "reported as 0 rather than estimated). This is a sensitivity statement derived from "
      "the observed SD, not observed-effect post-hoc power.\n")
    A("At n = 5 the median cell cannot resolve a difference below about two accuracy points, "
      "and the noisiest OOD cells cannot resolve one below six.\n")
    A("The comparison with a pair-level calculation cuts against the usual intuition, and the "
      "item-level companion analysis (`report_ch2_tost_item`) measures it: at N = 300 items "
      "the item-sampling MDE is 0.056 at the median, roughly **2.1x worse** than the 0.019 "
      "here, and larger in 26 of the 28 contrasts. The reason is that every seed scores the "
      "identical item list, so item-sampling noise cancels out of the per-seed difference "
      "entirely and the seed-level standard error reflects only model-to-model variation, "
      "which is small. The pair count is therefore not just the wrong replication unit for a "
      "cross-grammar claim — on this data it is also the blunter one.\n")

    A("## Reading notes\n")
    A("- **No multiple-comparison correction is applied.** Each contrast is characterised "
      "individually and descriptively; this is not a single omnibus hypothesis, and the "
      "verdicts are not claims about a family-wise error rate.\n")
    A("- With n = 5 the t-based TOST rests on approximate normality of the seed means, so "
      "the per-contrast verdicts are descriptive rather than decisive.\n")
    A(f"- {n_deg} of {len(res)} contrasts have zero between-seed variance in the difference "
      "and take the exact `se == 0` branch (exact-equal -> equivalent; nonzero but below "
      "Delta -> trivial). Note this is much rarer than the zero counts in the seed-variance "
      "section, which report the variance *component* after subtracting binomial sampling "
      f"noise; the raw SD is zero in only {sp['n_zero_sd']} of {sp['n_cells']} cells.\n")

    out_md.write_text("\n".join(L), encoding="utf-8")


# --- entry point ------------------------------------------------------------

def main() -> None:
    fig_dir, tab_dir = ch2_dirs("sub4_tost")
    long = build_data_long()
    grid_report(long)

    res = run_contrasts(long)
    long.to_csv(tab_dir / "data_long.csv", index=False)
    res.round(4).to_csv(tab_dir / "tost_results.csv", index=False)
    write_summary(res, long, tab_dir.parent / "tost_summary.md")
    for split in res["split"].unique():
        fig_forest(res, split, fig_dir)

    counts = res["verdict"].value_counts()
    print(f"[ch2/sub4 tost] {len(res)} contrasts ("
          + ", ".join(f"{v} {int(counts.get(v, 0))}" for v in VERDICTS)
          + f"); MDE median {res['mde_80'].median():.4f} -> {tab_dir.parent}")


if __name__ == "__main__":
    main()
