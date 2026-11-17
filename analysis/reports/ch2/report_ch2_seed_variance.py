"""Chapter-2, subchapter 2 (companion): sampling noise vs between-seed variance.

A single, simple question before the evaluation chapter: of the variation we see
in a grammar's probe accuracy, how much is just measurement noise (a finite set of
minimal pairs) and how much is genuine variation between the five training seeds?

For each cell (phenomenon x grammar x regime) we take the five per-seed accuracies
-- each a proportion over the full N pairs of that cell -- and split their spread
into two parts:

    v_samp      = mean_s p_s(1-p_s)/N_s        within-seed binomial sampling noise
    s2_total    = var_s(p_s)                   observed between-seed variance
    sigma2_seed = max(0, s2_total - v_samp)    genuine between-seed variance

(method of moments / one-way random effects, proportion scale). It is ceiling-safe:
when all five seeds are near-perfect both terms collapse and sigma_seed -> 0, with no
logistic separation. By construction sigma_seed <= SD(dots). IOD and OOD are kept
separate. There is no length/depth binning: every seed scores the identical item set,
so the per-seed accuracy over all pairs is already the clean quantity.

Output -> ch2/sub2_seed_variance/{figures,tables}:
    seed_variance_components.csv   one row per (phenomenon, grammar, regime)
    seed_points.csv                the five per-seed accuracies behind each cell
    seed_vs_sampling_dumbbell.png  dots (seeds) + sampling bar + sigma_seed, per
                                   grammar, with the cross-grammar span in grey

Run:  python -m analysis.reports.ch2.report_ch2_seed_variance
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.report_common import (
    GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY, SUBCASE_DISPLAY, SUBCASE_ORDER,
    SPLIT_DISPLAY, load_probe_pairs, savefig, ch2_dirs,
)

SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]
MIN_SEEDS = 2                    # need >= this many seeds to form a between-seed variance


def _phenomena(pairs):
    return [p for p in SUBCASE_ORDER if p in set(pairs["phenomenon"])]


# --- the decomposition ------------------------------------------------------

def seed_components(pairs):
    """Return (components, seed_points).

    ``seed_points``: one row per (phenomenon, grammar, regime, seed) with that seed's
    accuracy over all its pairs. ``components``: one row per (phenomenon, grammar,
    regime) with the sampling / between-seed split computed directly from those five
    per-seed accuracies, so ``sigma_seed <= SD`` of the seed points holds by
    construction.
    """
    ps = (pairs.groupby(["phenomenon", "grammar", "split", "seed"], observed=True)["acc"]
          .agg(acc="mean", n_pairs="size").reset_index())
    seed_points = ps.copy()

    rows = []
    for (ph, g, sp), sub in ps.groupby(["phenomenon", "grammar", "split"], observed=True):
        P = sub["acc"].to_numpy(dtype=float)
        N = sub["n_pairs"].to_numpy(dtype=float)
        if len(P) < MIN_SEEDS:
            continue
        s2_total = float(np.var(P, ddof=1))
        v_samp = float(np.mean(P * (1.0 - P) / N))
        sigma2 = max(0.0, s2_total - v_samp)
        denom = sigma2 + v_samp
        rows.append(dict(
            phenomenon=ph, grammar=str(g), split=sp, n_pairs=int(N.sum()),
            n_seeds=int(len(P)), p_bar=float((P * N).sum() / N.sum()),
            s2_total=s2_total, v_samp=v_samp, sigma2_seed=sigma2,
            sigma_seed=float(np.sqrt(sigma2)),
            icc=float(sigma2 / denom) if denom > 0 else np.nan))
    return pd.DataFrame(rows), seed_points


# --- the figure -------------------------------------------------------------

def fig_dumbbell(comp, seed_points, fig_dir):
    """Per phenomenon (rows) x regime (cols): each grammar's seed dots, its sampling
    bar, sigma_seed, and the cross-grammar span in grey."""
    phen = _phenomena(comp)
    fig, axes = plt.subplots(len(phen), 2, figsize=(11, 1.5 * len(phen) + 1),
                             squeeze=False, sharex=True)
    for i, ph in enumerate(phen):
        for j, sp in enumerate(SPLITS):
            ax = axes[i][j]
            block = comp[(comp.phenomenon == ph) & (comp.split == sp)]
            gmeans = []
            for gi, g in enumerate(GRAMMARS):
                row = block[block.grammar == g]
                if row.empty:
                    continue
                r = row.iloc[0]
                y = gi
                sc = seed_points[(seed_points.phenomenon == ph) & (seed_points.grammar == g)
                                 & (seed_points.split == sp)]
                seed_acc = sc["acc"].to_numpy(dtype=float)
                ax.scatter(seed_acc, np.full_like(seed_acc, y, dtype=float),
                           s=14, color=GRAMMAR_COLOR[g], alpha=0.55, zorder=3)
                ax.errorbar(r["p_bar"], y, xerr=np.sqrt(r["v_samp"]), fmt="o",
                            ms=6, color=GRAMMAR_COLOR[g], capsize=3, lw=1.4, zorder=4)
                ax.text(1.005, y, f"σs={r['sigma_seed']:.03f}", va="center",
                        fontsize=6, color=GRAMMAR_COLOR[g],
                        transform=ax.get_yaxis_transform())
                gmeans.append(r["p_bar"])
            if len(gmeans) >= 2:                     # cross-grammar span
                ax.axvspan(min(gmeans), max(gmeans), color="0.75", alpha=0.30, zorder=0)
            ax.set_yticks(range(len(GRAMMARS)),
                          [GRAMMAR_DISPLAY[g] for g in GRAMMARS], fontsize=7)
            ax.set_ylim(-0.6, len(GRAMMARS) - 0.4)
            ax.grid(alpha=0.2, axis="x")
            if i == 0:
                ax.set_title(SPLIT_DISPLAY[sp], fontsize=9)
            if j == 0:
                ax.set_ylabel(SUBCASE_DISPLAY[ph], fontsize=8)
    axes[-1][0].set_xlabel("accuracy")
    axes[-1][1].set_xlabel("accuracy")
    fig.suptitle("dots = per-seed accuracy (N pairs each);  bar = ±1 binomial sampling "
                 "SD √(mean p(1−p)/N);  σs = √(SD²(dots) − sampling var) ≤ SD(dots);  "
                 "grey = span of the 4 grammar means", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.97, 0.98))
    savefig(fig, fig_dir / "seed_vs_sampling_dumbbell.png")


# --- driver -----------------------------------------------------------------

def main() -> None:
    fig_dir, tab_dir = ch2_dirs("sub2_seed_variance")
    pairs = load_probe_pairs()

    comp, seed_points = seed_components(pairs)
    comp.to_csv(tab_dir / "seed_variance_components.csv", index=False)
    seed_points.to_csv(tab_dir / "seed_points.csv", index=False)
    fig_dumbbell(comp, seed_points, fig_dir)

    print(f"[ch2/sub2 seed-variance] {len(comp)} cells "
          f"(phenomenon x grammar x regime); figure + tables -> {fig_dir.parent}")


if __name__ == "__main__":
    main()
