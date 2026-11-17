"""Permutation tests on the test-set neg_logq distributions by phenomenon.

Companion to the ``neglogq_boxplot_by_phenomenon`` figure (analysis/reports/heldout/report_test.py):
for every phenomenon bin it compares each pair of grammars' per-item ``neg_logq``
against one another with a two-sided permutation test.

Design (fixed with the user):

* Unit of analysis = the trained model, not the item. Each item's neg_logq is
  first averaged within its run (seed), giving 5 run-level means per
  (grammar, phenomenon). This avoids the pseudoreplication that pooling all
  ~thousands of items would introduce.
* Statistic = |mean_A - mean_B| over those run-level means.
* Test = EXACT permutation over all C(10,5)=252 label assignments of the ten
  run means (two-sided). The exact floor on the raw p-value is therefore
  2/252 = 0.0079.
* Comparisons = all 6 unordered grammar pairs within each of the 5 phenomena
  (30 tests), Holm-corrected as one family. NOTE: the Holm floor is
  30 * 0.0079 = 0.238, so no test can reach alpha=0.05 after correction; the
  raw p and the effect size remain informative.

Output (under the same test/ folder as the plots):

    neglogq_permutation_by_phenomenon.csv  -- one row per (phenomenon, pair):
        run-level means of each grammar, their difference, raw and Holm p.

Run:  python -m analysis.reports.heldout.report_test_permutation
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd

from analysis.report_common import (
    CONSTRUCTION_DISPLAY, CONSTRUCTION_ORDER, GRAMMARS, GRAMMAR_DISPLAY,
    load_test_items, outdir,
)

OUT = outdir("evaluation", "test")


def _run_means(items: pd.DataFrame) -> pd.DataFrame:
    """Mean neg_logq per (grammar, construction, run) -> 5 rows per cell."""
    return (items.groupby(["grammar", "construction", "run_id"], observed=True)
                 ["neg_logq"].mean().reset_index(name="mean_neg_logq"))


def _exact_perm_p(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    """Two-sided exact permutation p for |mean(a) - mean(b)|.

    Enumerates all C(n+m, n) label assignments of the pooled values. Returns
    (signed diff a-b, p, n_perms). Ties counted with a small tolerance.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    pooled = np.concatenate([a, b])
    n = a.size
    obs = abs(a.mean() - b.mean())
    idx = range(pooled.size)
    hits = total = 0
    for comb in combinations(idx, n):
        mask = np.zeros(pooled.size, dtype=bool)
        mask[list(comb)] = True
        stat = abs(pooled[mask].mean() - pooled[~mask].mean())
        total += 1
        if stat >= obs - 1e-12:
            hits += 1
    return float(a.mean() - b.mean()), hits / total, total


def _holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values (family-wise), monotone-enforced."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = min(1.0, (m - rank) * p[i])
        running = max(running, val)          # enforce monotone non-decreasing
        adj[i] = running
    return adj


def main() -> None:
    items = load_test_items()
    rm = _run_means(items)
    phenomena = [c for c in CONSTRUCTION_ORDER if c in set(rm["construction"])]

    rows = []
    for con in phenomena:
        sub = rm[rm.construction == con]
        by_g = {g: sub[sub.grammar == g]["mean_neg_logq"].to_numpy() for g in GRAMMARS}
        for ga, gb in combinations(GRAMMARS, 2):
            a, b = by_g[ga], by_g[gb]
            diff, p, nperm = _exact_perm_p(a, b)
            rows.append(dict(
                phenomenon=CONSTRUCTION_DISPLAY[con],
                comparison=f"{GRAMMAR_DISPLAY[ga]} vs {GRAMMAR_DISPLAY[gb]}",
                mean_a=round(float(a.mean()), 4),
                mean_b=round(float(b.mean()), 4),
                diff=round(diff, 4),
                p_raw=round(p, 5),
                n_perm=nperm,
                n_runs=int(a.size),
            ))

    tab = pd.DataFrame(rows)
    tab["p_holm"] = np.round(_holm(tab["p_raw"].to_numpy()), 5)
    tab = tab[["phenomenon", "comparison", "mean_a", "mean_b", "diff",
               "p_raw", "p_holm", "n_runs", "n_perm"]]

    out_csv = OUT / "neglogq_permutation_by_phenomenon.csv"
    tab.to_csv(out_csv, index=False)
    print(f"[test] wrote {len(tab)} permutation tests "
          f"({len(phenomena)} phenomena x 6 pairs) -> {out_csv}")
    print(f"       exact raw-p floor = 2/{rows[0]['n_perm']} = "
          f"{2 / rows[0]['n_perm']:.4f}; Holm floor over {len(tab)} = "
          f"{len(tab) * 2 / rows[0]['n_perm']:.3f}")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print(tab.to_string(index=False))


if __name__ == "__main__":
    main()
