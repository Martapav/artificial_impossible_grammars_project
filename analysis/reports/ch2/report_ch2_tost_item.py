"""Item-level counterpart to the seed-level TOST (``analysis/reports/ch2/report_ch2_tost.py``).

The seed-level analysis asks whether a cross-grammar gap would survive *retraining*
-- its unit is the trained model, n = 5. This module asks the other question: would
the gap survive a fresh draw of *probe items*, holding the trained models fixed?
Its unit is the minimal pair, N = 300 per grammar per cell.

Effective sample size (resolved empirically, not assumed). The probe corpus is one
file per grammar under ``corpora/set_3/structural_probe_{IOD,OOD}/<grammar>.jsonl``,
built once by the corpus builder before any training; the training seed touches
only model initialisation. Every seed therefore scores a byte-identical item set --
verified directly on the score shards: all five ``<grammar>__s*.parquet`` carry the
same 4200 ``item_uid`` values per split (= 2100 pairs = 7 subcases x 300). Items are
fully CROSSED with seed, not nested, so:

  * the independent unit is **N = 300 items**, not 1500 -- the five seeds re-score
    the same items and contribute no new item information;
  * the cross-seed mean accuracy is the proportion;
  * the two grammars draw from *different* item streams (item_uids are grammar-
    prefixed), so the contrast is **unpaired** -- a two-proportion test, not a
    paired-over-seeds one.

What this does and does not license. ``se_item`` is item-sampling variance alone; it
treats scoring as deterministic given the item, so it excludes the model-to-model
variance the seed-level test measures. An item-level verdict is a claim about the
item sample, never about trained models. The ``ratio_*`` diagnostic quantifies the
gap between the two: between-seed SD divided by the binomial item SD. Near 0 means
the seeds agree almost exactly and item noise dominates; near or above 1 means real
model-to-model spread that the item-level se does not see.

Output -> results/analysis/evaluation/structural_probes/ch2/sub4_tost/
    tables/tost_item_level.csv

Run:  python -m analysis.reports.ch2.report_ch2_tost_item
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from analysis.report_common import SPLIT_DISPLAY, SUBCASE_ORDER, ch2_dirs

DELTA = 0.05
ALPHA = 0.05
N_ITEMS = 300                  # minimal pairs per (grammar, subcase, split)
Z = stats.norm.ppf(1 - ALPHA)  # 1.6449, the 90% CI multiplier
GRAMMAR_PAIRS = [("H", "P"), ("Hprime", "Lprime")]
SPLITS = ["IOD", "OOD"]
RAW_SPLIT = {v: k for k, v in SPLIT_DISPLAY.items()}

RESULT_COLUMNS = [
    "split", "subcase", "pair", "grammar_a", "grammar_b",
    "p_a", "p_b", "diff", "se_item", "ci90_low", "ci90_high",
    "p_tost", "p_nhst", "verdict_item", "verdict_seed", "changed",
    "ratio_a", "ratio_b", "n_req", "certifiable",
]


# --- the item-level core ----------------------------------------------------

def two_proportion_tost(p_a, p_b, n=N_ITEMS, delta=DELTA):
    """Unpaired two-proportion TOST + NHST, normal approximation.

    Guards ``se == 0``, which occurs when both proportions sit exactly at a
    bound (two gate cells are at 1.0 in every seed).
    """
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n + p_b * (1 - p_b) / n)
    if se == 0.0:
        p_nhst = 1.0 if diff == 0.0 else 0.0
        p_tost = 0.0 if abs(diff) < delta else 1.0
        return dict(diff=diff, se_item=0.0, p_tost=p_tost, p_nhst=p_nhst,
                    ci90_low=diff, ci90_high=diff)
    p_low = stats.norm.sf((diff + delta) / se)     # 1 - Phi((diff + delta)/se)
    p_high = stats.norm.cdf((diff - delta) / se)
    p_nhst = 2 * stats.norm.sf(abs(diff) / se)
    return dict(diff=diff, se_item=se, p_tost=max(p_low, p_high), p_nhst=p_nhst,
                ci90_low=diff - Z * se, ci90_high=diff + Z * se)


def classify_item(p_tost, p_nhst, alpha=ALPHA):
    equiv, sig = p_tost < alpha, p_nhst < alpha
    if equiv and not sig:
        return "equivalent"
    if equiv and sig:
        return "trivial"
    if sig and not equiv:
        return "different"
    return "inconclusive"


def required_n(p_a, p_b, diff, delta=DELTA):
    """Probe size needed to certify +/-delta equivalence for an undetermined cell.

    Returns (n_req, certifiable). An observed |diff| at or beyond delta cannot be
    certified equivalent at any N -- more items shrink the interval around a point
    that already sits outside the bound.
    """
    if abs(diff) >= delta:
        return (np.nan, False)
    pbar = 0.5 * (p_a + p_b)
    return (int(math.ceil(2 * pbar * (1 - pbar) * (Z / (delta - abs(diff))) ** 2)), True)


# --- drivers ----------------------------------------------------------------

def cell_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Per (split, subcase, grammar): pooled proportion and the noise diagnostic."""
    rows = []
    for (sp, sc, g), sub in long.groupby(["split", "subcase", "grammar"], observed=True):
        acc = sub["accuracy"].to_numpy(float)
        k = acc * N_ITEMS
        assert np.abs(k - np.round(k)).max() < 1e-6, f"non-integral counts in {(sp, sc, g)}"
        p = float(acc.mean())
        sd_seed = float(acc.std(ddof=1))
        se_bin = math.sqrt(p * (1 - p) / N_ITEMS)
        rows.append(dict(split=SPLIT_DISPLAY.get(sp, sp), subcase=sc, grammar=g,
                         p=p, k_total=int(np.round(k).sum()), sd_seed=sd_seed,
                         se_binom=se_bin,
                         ratio=(sd_seed / se_bin) if se_bin > 0 else np.nan))
    return pd.DataFrame(rows)


def run_item_contrasts(long: pd.DataFrame, seed_res: pd.DataFrame) -> pd.DataFrame:
    cells = cell_stats(long).set_index(["split", "subcase", "grammar"])
    seed_v = seed_res.set_index(["split", "subcase", "grammar_a", "grammar_b"])["verdict"]
    subcases = [c for c in SUBCASE_ORDER if c in set(long["subcase"])]

    rows = []
    for split in SPLITS:
        for sc in subcases:
            for ga, gb in GRAMMAR_PAIRS:
                A, B = cells.loc[(split, sc, ga)], cells.loc[(split, sc, gb)]
                st = two_proportion_tost(A["p"], B["p"])
                verdict = classify_item(st["p_tost"], st["p_nhst"])
                n_req, certifiable = ((np.nan, True) if verdict != "inconclusive"
                                      else required_n(A["p"], B["p"], st["diff"]))
                vs = seed_v.get((split, sc, ga, gb), "n/a")
                rows.append(dict(
                    split=split, subcase=sc, pair=f"{ga}-{gb}",
                    grammar_a=ga, grammar_b=gb, p_a=A["p"], p_b=B["p"],
                    **st, verdict_item=verdict, verdict_seed=vs,
                    changed=bool(vs != verdict),
                    ratio_a=A["ratio"], ratio_b=B["ratio"],
                    n_req=n_req, certifiable=certifiable))
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def main() -> None:
    _, tab_dir = ch2_dirs("sub4_tost")
    long = pd.read_csv(tab_dir / "data_long.csv")
    seed_res = pd.read_csv(tab_dir / "tost_results.csv")

    res = run_item_contrasts(long, seed_res)
    out = tab_dir / "tost_item_level.csv"
    res.round(6).to_csv(out, index=False)

    show = ["split", "subcase", "pair", "diff", "se_item", "ci90_low", "ci90_high",
            "p_tost", "p_nhst", "verdict_item", "verdict_seed", "changed"]
    print("[ch2/sub4 tost-item] N = 300 items per grammar per cell "
          "(probe set FIXED across seeds; unpaired two-proportion)\n")
    print(res[show].to_string(index=False,
                              formatters={c: "{:+.4f}".format
                                          for c in ("diff", "ci90_low", "ci90_high")}))

    print("\n--- verdict_seed x verdict_item ---")
    print(pd.crosstab(res["verdict_seed"], res["verdict_item"],
                      rownames=["seed-level"], colnames=["item-level"]))
    print(f"\nchanged: {int(res['changed'].sum())} of {len(res)} contrasts")

    diag = res[["ratio_a", "ratio_b"]].to_numpy().ravel()
    diag = diag[np.isfinite(diag)]
    print(f"\n--- diagnostic: between_seed_sd / binomial_item_sd over "
          f"{len(diag)} cells ---")
    print(f"median {np.median(diag):.3f} | IQR {np.percentile(diag, 25):.3f}-"
          f"{np.percentile(diag, 75):.3f} | max {diag.max():.3f} | "
          f"share >= 1: {(diag >= 1).mean():.0%}")

    inc = res[res["verdict_item"] == "inconclusive"]
    print(f"\n--- probe size to certify +/-{DELTA} ({len(inc)} inconclusive) ---")
    if inc.empty:
        print("none")
    for _, x in inc.iterrows():
        print(f"  {x.split}/{x.subcase} {x.pair}: diff {x['diff']:+.4f} -> "
              + (f"N_req = {int(x.n_req)} pairs/grammar" if x.certifiable
                 else "UNCERTIFIABLE (|diff| >= delta)"))
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
