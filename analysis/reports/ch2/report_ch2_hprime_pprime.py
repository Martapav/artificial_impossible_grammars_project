"""Chapter-2, subchapter 3: the incongruent clause, H' vs P'.

Each mixed grammar states wh-fronting with two clauses (grammars/metadata.py):

    clause (i)  -- the SWAP axis. Foreign to the family: POSITIONAL inside the
                   hierarchical family (H'), HIERARCHICAL inside the positional
                   family (P').  In the probe this is the ``wh_i`` term.
                   -> the INCONGRUENT clause.
    clause (ii) -- inherited unchanged from the parent, so NATIVE to the family.
                   In the probe this is the ``wh_ii`` term.
                   -> the CONGRUENT clause.

The analysis is WITHIN grammar: for each grammar we measure the accuracy gap
between its own two clauses,

    delta(g) = acc(clause i, incongruent) - acc(clause ii, congruent),

so a grammar is compared against ITSELF (its own congruent clause), never
against a different grammar's base.  The MAIN result is the between-grammar
comparison of these within-grammar gaps -- delta(H') vs delta(P').  The
homogeneous grammars H and P (whose two clauses are both native) give the
no-swap reference gap and are plotted separately.

Metric is accuracy throughout (the user's choice; avoids the cross-cell nats
confound).  The plotted within-grammar gap delta(g) carries an ACROSS-SEED CI --
the seed-to-seed spread of the within-seed gap, with the trained model as the
sampling unit and no within-seed item resampling (see ``_delta_across_seed``);
this replaces the earlier item-level bootstrap, which pooled the 5 seeds and so
understated model-to-model variation.  The between-grammar difference-in-differences
(prime vs its homogeneous parent) is reported in the tables (its inference is still
under review -- a seed-level interaction test is the intended replacement).

Figures -> ch2/sub3_hprime_pprime/figures ; tables -> .../tables.

    clause_congruence_primes       H', P': clause (i) vs clause (ii) accuracy
    clause_congruence_delta        delta(g) for all grammars (primes = main,
                                   H/P = reference), the between-grammar view
    clause_congruence_homogeneous  H, P: clause (i) vs clause (ii) accuracy
    incongruent_vs_congruent_depth within each prime, the two clauses vs depth

Run:  python -m analysis.reports.ch2.report_ch2_hprime_pprime
"""
from __future__ import annotations

from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analysis.report_common import (
    GRAMMAR_COLOR, GRAMMAR_DISPLAY, SPLIT_DISPLAY, ch2_dirs,
    load_probe_pairs, savefig,
)

SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]
DEPTH_COL = "embedding_depth_g"
FLOOR = 50

# clause (i) = incongruent (the swapped clause);  clause (ii) = congruent.
INCONGRUENT = "wh_i"
CONGRUENT = "wh_ii"

PRIMES = ["Hprime", "Lprime"]
HOMOGENEOUS = ["H", "P"]
# each prime inherits clause (ii) from this parent; the parent is the no-swap
# reference for the difference-in-differences.
PARENT = {"Hprime": "H", "Lprime": "P"}


def _acc(w, grammar, term, split, *, divergent_only=False):
    """Accuracy vector for one (grammar, clause, split) cell."""
    d = w[(w.grammar == grammar) & (w.term == term) & (w.split == split)]
    if divergent_only:
        d = d[d.rule_type_g == "clause_i_divergent"]
    return d["acc"].to_numpy(dtype=float)


def _acc_across_seed(w, grammar, term, split):
    """(point, lo, hi) for one (grammar, clause, split) accuracy, ACROSS seeds.

    Point is the mean of the per-seed accuracies (each of the 5 seeds weighs
    equally, == the balanced pooled mean); the interval is a small-sample t CI over
    those per-seed values -- the trained model is the unit, no within-seed item
    resampling.
    """
    per = (w[(w.grammar == grammar) & (w.term == term) & (w.split == split)]
           .groupby("seed")["acc"].mean().to_numpy(dtype=float))
    if per.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(per.mean())
    if per.size == 1:
        return m, m, m
    half = float(stats.t.ppf(0.975, per.size - 1) * per.std(ddof=1) / np.sqrt(per.size))
    return m, m - half, m + half


def _per_seed_delta(w, grammar, split, *, incongruent_divergent_only=False):
    """One delta per seed: mean acc(clause i) - mean acc(clause ii), within seed."""
    inc = w[(w.grammar == grammar) & (w.term == INCONGRUENT) & (w.split == split)]
    if incongruent_divergent_only:
        inc = inc[inc.rule_type_g == "clause_i_divergent"]
    con = w[(w.grammar == grammar) & (w.term == CONGRUENT) & (w.split == split)]
    mi = inc.groupby("seed")["acc"].mean()
    mc = con.groupby("seed")["acc"].mean()
    seeds = mi.index.union(mc.index)
    return (mi.reindex(seeds) - mc.reindex(seeds)).dropna().to_numpy(dtype=float)


def _delta_across_seed(w, grammar, split, *, incongruent_divergent_only=False):
    """(point, lo, hi) for delta(g) with a small-sample t interval ACROSS seeds.

    The uncertainty is the seed-to-seed spread of the within-seed clause gap -- the
    trained model is the sampling unit -- with **no** within-seed (item) resampling.
    The point equals the pooled mean gap (seeds are balanced, 300 items each).
    """
    d = _per_seed_delta(w, grammar, split,
                        incongruent_divergent_only=incongruent_divergent_only)
    if d.size == 0:
        return float("nan"), float("nan"), float("nan")
    m = float(d.mean())
    if d.size == 1:
        return m, m, m
    half = float(stats.t.ppf(0.975, d.size - 1) * d.std(ddof=1) / np.sqrt(d.size))
    return m, m - half, m + half


# --- 1. within-grammar clause accuracy, the two primes ---------------------

def _fig_two_clauses(w, grammars, fig_dir, tab_dir, *, stem, title, seed=None):
    """Grouped clause-(i)/clause-(ii) accuracy bars, one group per grammar.

    Pooled (``seed=None``): bars are the across-seed mean with a t CI (the seed is
    the unit). Between-seed variability is shown by REPRODUCTION -- with ``seed``
    set, ``w`` is restricted to that one model, bars are its raw cell means with no
    error bar, and the figure/table go to ``.../by_seed/{stem}__s<seed>.{png,csv}``
    so a reference seed can sit in the text and the rest in the appendix.
    """
    ws = w if seed is None else w[w.seed == seed]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    rows = []
    width = 0.36
    for ax, split in zip(axes, SPLITS):
        centres = np.arange(len(grammars), dtype=float)
        for j, (term, off, lab, hatch) in enumerate(
                [(INCONGRUENT, -width / 2, "clause (i) — incongruent", "///"),
                 (CONGRUENT, +width / 2, "clause (ii) — congruent", None)]):
            for c, g in zip(centres, grammars):
                pt, lo, hi = _acc_across_seed(ws, g, term, split)
                yerr = None if seed is not None else [[pt - lo], [hi - pt]]
                ax.bar(c + off, pt, width=width, color=GRAMMAR_COLOR[g],
                       hatch=hatch, edgecolor="white", yerr=yerr,
                       error_kw=dict(lw=1.0, ecolor="0.3", capsize=3),
                       label=lab if c == 0 else None)
                rows.append(dict(split=SPLIT_DISPLAY[split], seed=seed, grammar=g,
                                 clause=term, kind=lab.split("—")[1].strip(),
                                 acc=pt, lo=lo, hi=hi))
        # annotate the within-grammar gap
        for c, g in zip(centres, grammars):
            dpt, _, _ = _delta_across_seed(ws, g, split)
            ax.text(c, 1.005, f"Δ={dpt:+.3f}", ha="center", va="bottom",
                    fontsize=8, color="0.15")
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.06)
        ax.set_xticks(centres, [GRAMMAR_DISPLAY[g] for g in grammars], fontsize=11)
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    if seed is None:
        savefig(fig, fig_dir / f"{stem}.png")
        pd.DataFrame(rows).to_csv(tab_dir / f"{stem}.csv", index=False)
    else:
        savefig(fig, fig_dir / "by_seed" / f"{stem}__s{seed}.png")
        (tab_dir / "by_seed").mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(tab_dir / "by_seed" / f"{stem}__s{seed}.csv",
                                  index=False)


def fig_primes(w, fig_dir, tab_dir, seed=None):
    _fig_two_clauses(w, PRIMES, fig_dir, tab_dir, stem="clause_congruence_primes",
                     title="Incongruent (i) vs congruent (ii) clause — mixed grammars",
                     seed=seed)


def fig_homogeneous(w, fig_dir, tab_dir, seed=None):
    _fig_two_clauses(w, HOMOGENEOUS, fig_dir, tab_dir,
                     stem="clause_congruence_homogeneous",
                     title="Clause (i) vs clause (ii) — homogeneous grammars "
                           "(no swap; reference)", seed=seed)


# --- 2. between-grammar comparison of within-grammar gaps -------------------

def fig_delta(w, fig_dir, tab_dir):
    """delta(g) with ACROSS-SEED CIs: primes are the main contrast, H/P the reference.

    The plotted error bars are the seed-to-seed spread of the within-seed clause gap
    (the trained model is the unit; no within-seed item resampling) -- see
    :func:`_delta_across_seed`. Tables also carry the difference-in-differences
    delta(prime)-delta(parent) and the incongruent-divergent-only variant of delta.
    """
    order = PRIMES + HOMOGENEOUS
    xs = [0.0, 1.0, 2.4, 3.4]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    rows = []
    for ax, split in zip(axes, SPLITS):
        for x, g in zip(xs, order):
            dpt, dlo, dhi = _delta_across_seed(w, g, split)
            inc = _acc(w, g, INCONGRUENT, split)
            con = _acc(w, g, CONGRUENT, split)
            main = g in PRIMES
            ax.errorbar(x, dpt, yerr=[[dpt - dlo], [dhi - dpt]], marker="o",
                        ms=9 if main else 7, lw=0, elinewidth=1.8, capsize=4,
                        color=GRAMMAR_COLOR[g],
                        alpha=1.0 if main else 0.5,
                        markeredgecolor="0.2" if main else "none")
            # divergent-only variant of the incongruent clause (sharp subset)
            ddpt, ddlo, ddhi = _delta_across_seed(w, g, split,
                                                  incongruent_divergent_only=True)
            rows.append(dict(
                split=SPLIT_DISPLAY[split], grammar=g, role="main" if main else "reference",
                acc_clause_i=float(np.nanmean(inc)), acc_clause_ii=float(np.nanmean(con)),
                delta=dpt, delta_lo=dlo, delta_hi=dhi,
                n_clause_i=int(inc.size), n_clause_ii=int(con.size),
                delta_divergent_only=ddpt, delta_div_lo=ddlo, delta_div_hi=ddhi))
        ax.axhline(0.0, color="0.4", lw=0.9, ls=":")
        ax.set_xticks(xs, [GRAMMAR_DISPLAY[g] for g in order], fontsize=11)
        ax.set_xlim(-0.5, 3.9)
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12)
        ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("Δ accuracy  (clause i − clause ii)")
    handles = [plt.Line2D([], [], marker="o", ls="", ms=9, mec="0.2",
                          color="0.4", label="mixed grammar (main)"),
               plt.Line2D([], [], marker="o", ls="", ms=7, alpha=0.5,
                          color="0.4", label="homogeneous (reference)")]
    axes[1].legend(handles=handles, fontsize=8, loc="upper right")
    fig.tight_layout()
    savefig(fig, fig_dir / "clause_congruence_delta.png")

    # difference-in-differences: each prime against its homogeneous parent.
    did_rows = []
    for split in SPLITS:
        for prime in PRIMES:
            parent = PARENT[prime]
            inc_p = _acc(w, prime, INCONGRUENT, split)
            con_p = _acc(w, prime, CONGRUENT, split)
            inc_h = _acc(w, parent, INCONGRUENT, split)
            con_h = _acc(w, parent, CONGRUENT, split)
            dp = inc_p.mean() - con_p.mean()
            dh = inc_h.mean() - con_h.mean()
            # 4-sample bootstrap of (delta_prime - delta_parent)
            rng = np.random.default_rng(0)
            def bs(v):
                v = v[~np.isnan(v)]
                return v[rng.integers(0, v.size, size=(2000, v.size))].mean(axis=1)
            boot = (bs(inc_p) - bs(con_p)) - (bs(inc_h) - bs(con_h))
            lo, hi = np.percentile(boot, [2.5, 97.5])
            did_rows.append(dict(split=SPLIT_DISPLAY[split], prime=prime,
                                 parent=parent, delta_prime=float(dp),
                                 delta_parent=float(dh), did=float(dp - dh),
                                 did_lo=float(lo), did_hi=float(hi)))
    pd.DataFrame(rows).to_csv(tab_dir / "clause_congruence_delta.csv", index=False)
    pd.DataFrame(did_rows).to_csv(
        tab_dir / "clause_congruence_did.csv", index=False)


# --- 3. the two clauses vs embedding depth, within each prime ---------------

def fig_depth(w, fig_dir, tab_dir):
    depths = np.arange(int(w[DEPTH_COL].min()), int(w[DEPTH_COL].max()) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    rows = []
    for ax, split in zip(axes, SPLITS):
        for g in PRIMES:
            for term, ls, lab in [(INCONGRUENT, "--", "clause (i) incongruent"),
                                  (CONGRUENT, "-", "clause (ii) congruent")]:
                d = w[(w.grammar == g) & (w.term == term) & (w.split == split)]
                grp = d.groupby(DEPTH_COL, observed=True)["acc"].agg(["mean", "size"])
                grp = grp[grp["size"] >= FLOOR]
                if grp.empty:
                    continue
                ax.plot(grp.index.to_numpy(), grp["mean"].to_numpy(), marker="o",
                        ms=4, lw=1.6, ls=ls, color=GRAMMAR_COLOR[g],
                        label=f"{GRAMMAR_DISPLAY[g]} — {lab}")
                for dep, r in grp.iterrows():
                    rows.append(dict(split=SPLIT_DISPLAY[split], grammar=g,
                                     clause=term, depth=int(dep),
                                     acc=float(r["mean"]), n=int(r["size"])))
        ax.axhline(0.5, color="0.4", lw=0.8, ls=":")
        ax.set_ylim(0.4, 1.02)
        ax.set_xticks(depths)
        ax.set_xlabel("embedding depth")
        ax.set_title(SPLIT_DISPLAY[split], fontsize=12)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("accuracy")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    savefig(fig, fig_dir / "incongruent_vs_congruent_depth.png")
    pd.DataFrame(rows).to_csv(
        tab_dir / "incongruent_vs_congruent_depth.csv", index=False)


# --- 4. seed-level stability + interaction tests ---------------------------

def _per_seed_delta_series(w, grammar, split):
    """delta(g) per seed as a seed-indexed Series (mean acc i - mean acc ii)."""
    inc = (w[(w.grammar == grammar) & (w.term == INCONGRUENT) & (w.split == split)]
           .groupby("seed")["acc"].mean())
    con = (w[(w.grammar == grammar) & (w.term == CONGRUENT) & (w.split == split)]
           .groupby("seed")["acc"].mean())
    return (inc - con).dropna()


def _one_sample(vals):
    """Across-seed summary + one-sample tests of H0: mean == 0.

    Reports the t(n-1) test and, because n is tiny, the EXACT two-sided sign-flip
    permutation p (each seed's value may flip sign; 2**n equally likely relabelings).
    """
    vals = np.asarray(vals, dtype=float)
    n = vals.size
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if n > 1 else 0.0
    out = dict(mean=m, sd=sd, min=float(vals.min()), max=float(vals.max()),
               n=n, n_pos=int((vals > 0).sum()), n_neg=int((vals < 0).sum()),
               values=";".join(f"{v:+.4f}" for v in vals))
    if n > 1 and sd > 0:
        se = sd / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, n - 1))
        out.update(ci_lo=m - tcrit * se, ci_hi=m + tcrit * se,
                   t=float(m / se), p=float(stats.ttest_1samp(vals, 0.0).pvalue))
        means = [abs(float(np.mean(vals * np.array(sg)))) for sg in product([1, -1], repeat=n)]
        out["perm_p"] = float(np.mean(np.array(means) >= abs(m) - 1e-12))
    else:
        out.update(ci_lo=m, ci_hi=m, t=float("nan"), p=float("nan"),
                   perm_p=float("nan"))
    return out


def report_seed_level(w, tab_dir):
    """Seed-level stability descriptives + within-grammar and DiD interaction tests.

    Writes three CSVs and prints a summary. The trained model (seed) is the unit
    throughout; nothing resamples items within a seed.
    """
    order = PRIMES + HOMOGENEOUS

    # (a) per-bar accuracy stability across seeds.
    stab = []
    for split in SPLITS:
        for g in order:
            for term in (INCONGRUENT, CONGRUENT):
                per = (w[(w.grammar == g) & (w.term == term) & (w.split == split)]
                       .groupby("seed")["acc"].mean().to_numpy(dtype=float))
                stab.append(dict(split=SPLIT_DISPLAY[split], grammar=GRAMMAR_DISPLAY[g],
                                 clause=("i (incongruent)" if term == INCONGRUENT
                                         else "ii (congruent)"),
                                 mean=float(per.mean()), sd=float(per.std(ddof=1)),
                                 min=float(per.min()), max=float(per.max()),
                                 range=float(per.max() - per.min()), n_seeds=per.size))
    stab = pd.DataFrame(stab)
    stab.to_csv(tab_dir / "seed_stability_acc.csv", index=False)

    # (b) within-grammar test: H0 delta(g) == 0 across seeds.
    wg = []
    for split in SPLITS:
        for g in order:
            d = _per_seed_delta_series(w, g, split)
            r = _one_sample(d.to_numpy())
            wg.append(dict(split=SPLIT_DISPLAY[split], grammar=GRAMMAR_DISPLAY[g],
                           delta=r["mean"], sd=r["sd"], ci_lo=r["ci_lo"],
                           ci_hi=r["ci_hi"], t=r["t"], p=r["p"], perm_p=r["perm_p"],
                           n_pos=r["n_pos"], per_seed=r["values"]))
    wg = pd.DataFrame(wg)
    wg.to_csv(tab_dir / "seed_within_grammar.csv", index=False)

    # (c) DiD interaction test: H0 delta(prime) - delta(parent) == 0 across seeds.
    did = []
    for split in SPLITS:
        for prime in PRIMES:
            parent = PARENT[prime]
            dp = _per_seed_delta_series(w, prime, split)
            dh = _per_seed_delta_series(w, parent, split)
            diff = (dp - dh).dropna()                     # aligned on seed index
            r = _one_sample(diff.to_numpy())
            did.append(dict(split=SPLIT_DISPLAY[split],
                            prime=GRAMMAR_DISPLAY[prime], parent=GRAMMAR_DISPLAY[parent],
                            delta_prime=float(dp.mean()), delta_parent=float(dh.mean()),
                            did=r["mean"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"],
                            t=r["t"], p=r["p"], perm_p=r["perm_p"],
                            n_pos=r["n_pos"], per_seed=r["values"]))
    did = pd.DataFrame(did)
    did.to_csv(tab_dir / "seed_did_interaction.csv", index=False)

    # console summary
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print("\n[ch2/sub3] SEED-LEVEL STABILITY -- per-bar accuracy across 5 seeds")
        print(stab.round(3).to_string(index=False))
        print("\n[ch2/sub3] WITHIN-GRAMMAR gap test  H0: delta(g)=0 across seeds  (t(4), exact perm p)")
        print(wg.round(4).to_string(index=False))
        print("\n[ch2/sub3] DiD INTERACTION test  H0: delta(prime)-delta(parent)=0 across seeds")
        print(did.round(4).to_string(index=False))
    return stab, wg, did


def main() -> None:
    fig_dir, tab_dir = ch2_dirs("sub3_hprime_pprime")
    w = load_probe_pairs()
    w = w[w.term.isin([INCONGRUENT, CONGRUENT])]

    fig_primes(w, fig_dir, tab_dir)          # main: within-grammar, the primes
    fig_delta(w, fig_dir, tab_dir)           # main: between-grammar Δ comparison
    fig_homogeneous(w, fig_dir, tab_dir)     # reference: H and P
    fig_depth(w, fig_dir, tab_dir)           # within-prime clause gap vs depth
    # per-seed reproductions of the two clause-congruence bar plots (coherence
    # check; reference in main text, rest in appendix) -> figures/by_seed/.
    for s in sorted(w["seed"].unique()):
        fig_primes(w, fig_dir, tab_dir, seed=int(s))
        fig_homogeneous(w, fig_dir, tab_dir, seed=int(s))
    report_seed_level(w, tab_dir)            # stability + within-grammar + DiD tests

    n_seeds = w["seed"].nunique()
    print(f"[ch2/sub3] wrote clause_congruence_primes, clause_congruence_delta, "
          f"clause_congruence_homogeneous, incongruent_vs_congruent_depth "
          f"-> {fig_dir}  (+ {n_seeds} per-seed reproductions of clause_congruence_"
          f"primes/homogeneous -> {fig_dir}/by_seed)  (+ tables -> {tab_dir})")


if __name__ == "__main__":
    main()
