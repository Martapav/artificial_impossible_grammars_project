"""Item-level TOST equivalence on the probe judgements, per *violation subcase*.

Why this module exists alongside :mod:`analysis.reports.ch2.report_ch2_tost_item`. That one
tests the seven reported subcases, and its two anaphoric cells (``pron``,
``refl``) pool their violation types -- ``pron`` averages wrong_item with
wrong_site, ``refl`` averages three. A pooled accuracy cannot be split after the
fact, so the pooled tables (``data_long.csv``, ``tost_item_level.csv``) are *not*
a valid input here: this module goes back to the per-pair probe scores and
decomposes to the ten terms the probe actually contains, each at its own item
count (300 / 150 / 100).

Four contrasts per cell instead of two:

  cross-family     (H, P) and (H', P')  -- the grammar-family effect
  within-family    (H, H') and (P, P')  -- these two grammars differ ONLY in
                   wh clause (i)

so a within-family pair reads two ways depending on the subcase:

  * on ``wh_i`` it is the clean **swap** test -- the rule differs, every other
    rule is held fixed;
  * on the other nine subcases it is a **same-rule control**: both grammars
    implement the identical rule, so the measured difference is corpus-draw plus
    init noise. That is the empirical floor a cross-family verdict on the same
    subcase has to clear, and an empirical check on whether DELTA = 0.05 is a
    defensible bound. A control that comes out "different" is a floor breach,
    never a grammar effect.

Unit of analysis is the minimal pair, and the probe is **fixed**: all five seeds
re-score the identical item list (verified, not assumed -- see
:func:`detect_probe_mode`), so the independent item count is N_sub, not 5*N_sub.
The five seeds only sharpen the estimate of each proportion; ``ratio_a`` /
``ratio_b`` report between-seed SD over the single-probe binomial floor so a cell
whose seed scatter exceeds that floor is visible rather than assumed away.

The two grammars in a contrast are treated as independent samples (unpaired
two-proportion), within-family pairs included: the item lists are
grammar-specific -- ``H-sp-00000`` is not ``P-sp-00000`` -- so there is nothing
to pair on.

Multiplicity: the family is the 40 tests within a split (4 pairs x 10 subcases).
``holm_nhst`` is Holm-adjusted within split. The equivalence side (``p_tost``) is
left uncorrected, which is the conventional and the conservative choice here --
correcting it would make equivalence *harder* to claim.

Output -> results/analysis/evaluation/structural_probes/ch2/sub4_tost/
    tables/tost_subcase_level.csv   one row per (split, subcase, pair) = 80

Run:  python -m analysis.reports.ch2.report_ch2_tost_subcase
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from analysis.report_common import SPLIT_DISPLAY, ch2_dirs, load_probe_pairs

# --- config -----------------------------------------------------------------

DELTA = 0.05
ALPHA = 0.05
Z = stats.norm.ppf(1 - ALPHA)          # 1.6449, the 90% CI multiplier
SEEDS = [0, 1, 2, 3, 4]
GRAMMARS = ["H", "P", "Hprime", "Lprime"]
SPLITS = ["IOD", "OOD"]
RAW_SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]

# The four contrasts. `Lprime` is the DATA KEY of the grammar displayed as P';
# it is never renamed in the data (see memory: lprime-displays-as-pprime).
CONTRASTS = [
    ("H", "P", "cross"),
    ("Hprime", "Lprime", "cross"),
    ("H", "Hprime", "hier"),
    ("P", "Lprime", "pos"),
]
SWAP_SUBCASE = "wh_i"

# The ten terms, in probe order, with their per-cell item counts.
N_SUB = {
    "aux": 300,
    "wh_i": 300,
    "wh_ii": 300,
    "neutral_agreement": 300,
    "neutral_positional": 300,
    "pron_wrong_item": 150,          # the pron 300 splits over two violations
    "pron_wrong_site": 150,
    "refl_wrong_item": 100,          # the refl 300 splits over three
    "refl_wrong_site": 100,
    "refl_no_substitution": 100,
}
SUBCASES = list(N_SUB)

# Minor name variants -> canonical subcase key.
SUBCASE_ALIASES = {
    "auxiliary_movement": "aux", "aux_movement": "aux",
    "wh1": "wh_i", "wh_1": "wh_i", "wh_clause_i": "wh_i",
    "wh2": "wh_ii", "wh_2": "wh_ii", "wh_clause_ii": "wh_ii",
    "neutral_agr": "neutral_agreement", "agreement": "neutral_agreement",
    "neutral_pos": "neutral_positional", "positional": "neutral_positional",
}

RESULT_COLUMNS = [
    "split", "subcase", "pair", "grammar_a", "grammar_b", "family",
    "contrast_role", "swap_flag", "N_sub", "N_eff", "probe_mode",
    "pA", "pB", "diff", "se", "ci90_low", "ci90_high",
    "p_nhst", "holm_nhst", "p_tost", "verdict", "certifiable", "n_req",
    "degenerate", "ratio_a", "ratio_b",
]


class GuardFailure(RuntimeError):
    """A precondition on the input data does not hold; the analysis is invalid."""


# --- input ------------------------------------------------------------------

def _canonical_subcase(subcase: str, violation: str | float) -> str:
    """The ten-term key: the anaphoric bins carry their violation type."""
    sc = SUBCASE_ALIASES.get(subcase, subcase)
    if sc in ("pron", "refl"):
        if not isinstance(violation, str) or not violation:
            raise GuardFailure(
                f"subcase {sc!r} carries no violation type, so it cannot be "
                f"decomposed. Disaggregated per-subcase data is required.")
        return f"{sc}_{violation}"
    return sc


def build_long() -> pd.DataFrame:
    """Per (grammar, split, seed, subcase) correct-count and item-count.

    Reads the per-pair probe scores -- the only source that still holds the
    violation type -- and never the pooled `data_long.csv` / `tost_item_level.csv`.
    """
    pairs = load_probe_pairs()
    pairs = pairs[pairs["split"].isin(RAW_SPLITS)].copy()
    pairs["subcase10"] = [
        _canonical_subcase(sc, v)
        for sc, v in zip(pairs["subcase"].astype(str), pairs["violation"])
    ]
    pairs["split"] = pairs["split"].map(SPLIT_DISPLAY)
    pairs["grammar"] = pairs["grammar"].astype(str)

    g = pairs.groupby(["grammar", "split", "seed", "subcase10"], observed=True)
    long = g.agg(k=("acc", "sum"), n=("acc", "size"),
                 items=("pair_id", "nunique")).reset_index()
    long["k"] = long["k"].round().astype(int)
    long["accuracy"] = long["k"] / long["n"]
    return long, pairs


# --- guards -----------------------------------------------------------------

def check_guards(long: pd.DataFrame) -> None:
    present = set(long["subcase10"])
    missing = [s for s in SUBCASES if s not in present]
    extra = sorted(present - set(SUBCASES))
    if missing or extra:
        pooled = {"pron", "refl"} & present
        if pooled:
            raise GuardFailure(
                f"subcase(s) {sorted(pooled)} arrive as a single undecomposed bin "
                f"({len(present)} subcases found). The pooled accuracy cannot be "
                f"split; disaggregated per-subcase data is required. STOP.")
        raise GuardFailure(
            f"subcase set mismatch -- missing {missing}, unexpected {extra}. STOP.")

    got_g = sorted(set(long["grammar"]))
    if got_g != sorted(GRAMMARS):
        raise GuardFailure(f"grammars are {got_g}, expected {sorted(GRAMMARS)}. STOP.")
    got_s = sorted(set(long["split"]))
    if got_s != sorted(SPLITS):
        raise GuardFailure(f"splits are {got_s}, expected {SPLITS}. STOP.")
    got_seeds = sorted(set(long["seed"]))
    if got_seeds != SEEDS:
        raise GuardFailure(f"seeds are {got_seeds}, expected {SEEDS}. STOP.")

    cells = long.groupby(["subcase10"], observed=True)["n"].nunique()
    ragged = cells[cells > 1]
    if len(ragged):
        raise GuardFailure(
            f"item count varies across (grammar, seed) within subcase(s) "
            f"{list(ragged.index)}. STOP.")
    for sc, n in long.groupby("subcase10", observed=True)["n"].first().items():
        if int(n) != N_SUB[sc]:
            raise GuardFailure(
                f"subcase {sc!r} has {int(n)} items per cell, expected "
                f"{N_SUB[sc]}. STOP.")
    k = long["accuracy"].to_numpy() * long["n"].to_numpy()
    if np.abs(k - np.round(k)).max() > 1e-6:
        raise GuardFailure("accuracy * N_sub is not integral for every row. STOP.")

    expected_rows = len(GRAMMARS) * len(SPLITS) * len(SEEDS) * len(SUBCASES)
    if len(long) != expected_rows:
        raise GuardFailure(
            f"{len(long)} long rows, expected {expected_rows} "
            f"(4 grammars x 2 splits x 5 seeds x 10 subcases). STOP.")


def detect_probe_mode(pairs: pd.DataFrame) -> str:
    """"fixed" if all five seeds score the identical items, "resampled" if disjoint.

    Decided per (grammar, split, subcase) from the ``pair_id`` sets; anything in
    between (partial overlap) is a data bug, not a mode.
    """
    modes = set()
    key = ["grammar", "split", "subcase10"]
    sets = (pairs.groupby(key + ["seed"], observed=True)["pair_id"]
                 .apply(frozenset))
    for _, grp in sets.groupby(level=[0, 1, 2], observed=True):
        s = list(grp.values)
        if all(x == s[0] for x in s):
            modes.add("fixed")
        elif sum(len(x) for x in s) == len(frozenset().union(*s)):
            modes.add("resampled")
        else:
            modes.add("partial")
    if modes == {"fixed"}:
        return "fixed"
    if modes == {"resampled"}:
        return "resampled"
    raise GuardFailure(
        f"probe item sets are inconsistent across seeds (modes seen: "
        f"{sorted(modes)}); N_eff is undefined. STOP.")


# --- statistical core -------------------------------------------------------

def two_proportion_tost(p_a, p_b, n_eff, delta=DELTA):
    """Unpaired two-proportion TOST + two-sided NHST, normal approximation.

    Guards ``se == 0``, which happens when both arms sit exactly on a bound.
    """
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / n_eff + p_b * (1 - p_b) / n_eff)
    if se == 0.0:
        p_nhst = 1.0 if diff == 0.0 else 0.0
        p_tost = 0.0 if abs(diff) < delta else 1.0
        return dict(diff=diff, se=0.0, p_nhst=p_nhst, p_tost=p_tost,
                    ci90_low=diff, ci90_high=diff, degenerate=True)
    p_low = stats.norm.sf((diff + delta) / se)     # H0: diff <= -delta
    p_high = stats.norm.cdf((diff - delta) / se)   # H0: diff >= +delta
    return dict(diff=diff, se=se,
                p_nhst=2 * stats.norm.sf(abs(diff) / se),
                p_tost=max(p_low, p_high),
                ci90_low=diff - Z * se, ci90_high=diff + Z * se,
                degenerate=False)


def classify(p_tost, p_nhst, alpha=ALPHA):
    equiv, sig = p_tost < alpha, p_nhst < alpha
    if equiv and not sig:
        return "equivalent"
    if equiv and sig:
        return "trivial"
    if sig and not equiv:
        return "different"
    return "inconclusive"


def required_n(p_a, p_b, diff, delta=DELTA):
    """Per-arm items for a 90% CI to fall inside +/-delta at the observed diff.

    An observed ``|diff| >= delta`` is not certifiable at any N: more items
    shrink the interval around a point already outside the bound.
    """
    if abs(diff) >= delta:
        return (np.nan, False)
    pbar = 0.5 * (p_a + p_b)
    return (int(math.ceil(2 * pbar * (1 - pbar) * (Z / (delta - abs(diff))) ** 2)),
            True)


def holm(pvals):
    """Holm step-down adjusted p-values, in the input order."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# --- drivers ----------------------------------------------------------------

def cell_stats(long: pd.DataFrame) -> pd.DataFrame:
    """Per (split, subcase, grammar): pooled p over 5*N_sub, and the seed-noise ratio.

    ``ratio`` = between-seed SD of accuracy / single-probe binomial SE. Well below
    1 means the seeds barely move (shared probe, near-zero init variance); at or
    above 1 means genuine init variance the two-proportion SE does not carry.
    """
    rows = []
    for (sp, sc, g), sub in long.groupby(["split", "subcase10", "grammar"],
                                         observed=True):
        n_sub = int(sub["n"].iloc[0])
        acc = sub["accuracy"].to_numpy(float)
        p = float(sub["k"].sum() / (len(acc) * n_sub))     # = acc.mean()
        sd_seed = float(acc.std(ddof=1))
        se_bin = math.sqrt(p * (1 - p) / n_sub)
        rows.append(dict(split=sp, subcase=sc, grammar=g, N_sub=n_sub, p=p,
                         sd_seed=sd_seed, se_binom=se_bin,
                         ratio=(sd_seed / se_bin) if se_bin > 0 else np.nan))
    return pd.DataFrame(rows)


def run_contrasts(long: pd.DataFrame, probe_mode: str,
                  sensitivity: bool) -> pd.DataFrame:
    cells = cell_stats(long).set_index(["split", "subcase", "grammar"])
    rows = []
    for sp in SPLITS:
        for sc in SUBCASES:
            for ga, gb, fam in CONTRASTS:
                ca, cb = cells.loc[(sp, sc, ga)], cells.loc[(sp, sc, gb)]
                n_sub = int(ca["N_sub"])
                n_eff = n_sub if probe_mode == "fixed" else len(SEEDS) * n_sub
                swap = fam in ("hier", "pos") and sc == SWAP_SUBCASE
                role = ("cross_family" if fam == "cross"
                        else "within_family_swap" if swap
                        else "within_family_control")
                st = two_proportion_tost(ca["p"], cb["p"], n_eff)
                verdict = classify(st["p_tost"], st["p_nhst"])
                n_req, certifiable = required_n(ca["p"], cb["p"], st["diff"])
                row = dict(
                    split=sp, subcase=sc, pair=f"{ga}-{gb}",
                    grammar_a=ga, grammar_b=gb, family=fam,
                    contrast_role=role, swap_flag=swap,
                    N_sub=n_sub, N_eff=n_eff, probe_mode=probe_mode,
                    pA=ca["p"], pB=cb["p"], **st,
                    p_tost_=None, verdict=verdict,
                    certifiable=certifiable,
                    n_req=(n_req if verdict == "inconclusive" and certifiable
                           else np.nan),
                    ratio_a=ca["ratio"], ratio_b=cb["ratio"])
                row.pop("p_tost_")
                if sensitivity:
                    st5 = two_proportion_tost(ca["p"], cb["p"], len(SEEDS) * n_sub)
                    row["verdict_5N"] = classify(st5["p_tost"], st5["p_nhst"])
                rows.append(row)
    res = pd.DataFrame(rows)
    res["holm_nhst"] = np.nan
    for sp, idx in res.groupby("split").groups.items():
        res.loc[idx, "holm_nhst"] = holm(res.loc[idx, "p_nhst"].to_numpy())
    cols = RESULT_COLUMNS + (["verdict_5N"] if sensitivity else [])
    return res[cols]


# --- printed report ---------------------------------------------------------

ROLE_ORDER = ["cross_family", "within_family_control", "within_family_swap"]
VERDICT_ORDER = ["equivalent", "trivial", "different", "inconclusive"]
DISPLAY_PAIR = {"H-P": "H-P", "Hprime-Lprime": "H'-P'",
                "H-Hprime": "H-H'", "P-Lprime": "P-P'"}


def _pair_label(pair: str) -> str:
    return DISPLAY_PAIR.get(pair, pair)


def _fmt(x, nd=4):
    return "     --" if pd.isna(x) else f"{x:{nd + 4}.{nd}f}"


def print_header(probe_mode: str, n_eff_note: str) -> None:
    print("=" * 100)
    print("ITEM-LEVEL TOST BY VIOLATION SUBCASE -- 10 subcases x 4 contrasts x 2 splits")
    print("=" * 100)
    print(f"delta = {DELTA}  alpha = {ALPHA}  z = {Z:.4f}  "
          f"(unpaired two-proportion, normal approximation)")
    print("Multiplicity: holm_nhst is Holm-adjusted within each split over its 40 "
          "tests (4 pairs x 10")
    print("subcases). The EQUIVALENCE side (p_tost) is UNCORRECTED -- correcting it "
          "would only make")
    print("equivalence harder to claim, so the uncorrected p_tost is the "
          "conservative reading here.")
    print(f"N_eff basis: {n_eff_note}")
    print()


def print_verdict_counts(res: pd.DataFrame, split: str) -> None:
    sub = res[res["split"] == split]
    print(f"--- {split}: verdict counts by contrast role ---")
    tab = (pd.crosstab(sub["contrast_role"], sub["verdict"])
             .reindex(index=[r for r in ROLE_ORDER if r in set(sub["contrast_role"])],
                      columns=VERDICT_ORDER, fill_value=0))
    print(tab.to_string())
    print()


def print_floor_table(res: pd.DataFrame, split: str) -> None:
    """Each cross-family effect beside the same-rule floor on the same subcase."""
    sub = res[res["split"] == split].set_index(["subcase", "pair"])
    print(f"--- {split}: FLOOR vs EFFECT (cross-family diff | same-rule control diff) ---")
    print("     within-family diffs on subcase != wh_i are SAME-RULE controls: "
          "corpus-draw + init")
    print("     noise on an unchanged rule. Read each cross-family effect against them.")
    print("     on wh_i the within-family columns are the RULE SWAP, not a floor.")
    heads = "".join(f"{_pair_label(f'{ga}-{gb}'):>10} {'verdict':<13}"
                    for ga, gb, _ in CONTRASTS)
    hdr = f"{'subcase':<22}" + heads
    print(hdr)
    print("-" * len(hdr))
    for sc in SUBCASES:
        cells = []
        for ga, gb, _ in CONTRASTS:
            r = sub.loc[(sc, f"{ga}-{gb}")]
            cells.append(f"{r['diff']:>+10.4f} {r['verdict']:<13}")
        tag = "  <= SWAP" if sc == SWAP_SUBCASE else ""
        print(f"{sc:<22}" + "".join(cells) + tag)
    print()


def print_floor_breaches(res: pd.DataFrame, split: str) -> None:
    sub = res[(res["split"] == split)
              & (res["contrast_role"] == "within_family_control")
              & (res["verdict"] == "different")]
    print(f"--- {split}: FLOOR BREACHES (same-rule control came out 'different') ---")
    if sub.empty:
        print("     none: every same-rule control stayed within the bound. delta = "
              "0.05 survives its own check.")
        print()
        return
    print("     Each row below is NOT a grammar effect. Two grammars implementing the")
    print("     IDENTICAL rule on this subcase were separated at alpha = 0.05, so the")
    print("     cause is one of: delta too tight for this probe size / probe too small /")
    print("     cross-rule leakage (the wh clause-(i) change perturbing this subcase).")
    for _, r in sub.iterrows():
        print(f"     FLOOR-BREACH  {r['subcase']:<22} {_pair_label(r['pair']):<6} "
              f"diff={r['diff']:+.4f}  90% CI [{r['ci90_low']:+.4f}, "
              f"{r['ci90_high']:+.4f}]  p_nhst={r['p_nhst']:.2e}  "
              f"holm={r['holm_nhst']:.2e}  N_eff={r['N_eff']}")
    print()


def print_wh_i_block(res: pd.DataFrame, split: str) -> None:
    sub = res[(res["split"] == split) & (res["subcase"] == SWAP_SUBCASE)]
    print(f"--- {split}: wh_i in full -- the swap rows beside the cross-family rows ---")
    hdr = (f"{'role':<22}{'pair':<7}{'pA':>8}{'pB':>8}{'diff':>9}{'se':>8}"
           f"{'90% CI':>21}{'p_nhst':>10}{'holm':>10}{'p_tost':>10}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for role in ["within_family_swap", "cross_family"]:
        for _, r in sub[sub["contrast_role"] == role].iterrows():
            ci = f"[{r['ci90_low']:+.4f},{r['ci90_high']:+.4f}]"
            print(f"{role:<22}{_pair_label(r['pair']):<7}{r['pA']:>8.4f}{r['pB']:>8.4f}"
                  f"{r['diff']:>+9.4f}{r['se']:>8.4f}{ci:>21}"
                  f"{r['p_nhst']:>10.4f}{r['holm_nhst']:>10.4f}{r['p_tost']:>10.4f}"
                  f"  {r['verdict']}")
    print()


def print_inconclusive(res: pd.DataFrame, split: str) -> None:
    sub = res[(res["split"] == split) & (res["verdict"] == "inconclusive")]
    print(f"--- {split}: inconclusive cells (neither difference nor equivalence "
          f"established) ---")
    if sub.empty:
        print("     none.")
        print()
        return
    hdr = (f"{'subcase':<22}{'pair':<7}{'role':<22}{'diff':>9}{'90% CI':>21}"
           f"{'N_eff':>7}{'n_req/arm':>12}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in sub.iterrows():
        ci = f"[{r['ci90_low']:+.4f},{r['ci90_high']:+.4f}]"
        nreq = "not certif." if pd.isna(r["n_req"]) else str(int(r["n_req"]))
        print(f"{r['subcase']:<22}{_pair_label(r['pair']):<7}{r['contrast_role']:<22}"
              f"{r['diff']:>+9.4f}{ci:>21}{int(r['N_eff']):>7}{nreq:>12}")
    print("     n_req is the per-arm item count at which the 90% CI would fall inside")
    print("     +/-delta AT THE OBSERVED diff; 'not certif.' means |diff| >= delta,")
    print("     which no probe size can rescue.")
    print()


def print_diagnostics(res: pd.DataFrame, split: str) -> None:
    sub = res[res["split"] == split]
    hot = sub[(sub["ratio_a"] >= 1.0) | (sub["ratio_b"] >= 1.0)]
    print(f"--- {split}: seed-scatter diagnostic (ratio = between-seed SD / "
          f"single-probe binomial SE) ---")
    print(f"     ratio << 1 -> seeds move less than one probe's sampling noise "
          f"(shared probe, near-zero")
    print(f"     init variance). ratio >= 1 -> genuine init variance the "
          f"two-proportion SE does not carry,")
    print(f"     so that cell's se is optimistic. {len(hot)}/{len(sub)} contrasts "
          f"have an arm at ratio >= 1:")
    if hot.empty:
        print("     none.")
    for _, r in hot.iterrows():
        print(f"       {r['subcase']:<22}{_pair_label(r['pair']):<7}"
              f"ratio_a={_fmt(r['ratio_a'], 2)}  ratio_b={_fmt(r['ratio_b'], 2)}")
    print()


def verify_against_item_level(res: pd.DataFrame, tables) -> None:
    """Cross-check the five non-decomposed subcases against the pooled-file run."""
    path = tables / "tost_item_level.csv"
    print("--- cross-check vs tost_item_level.csv (5 non-decomposed subcases, "
          "2 cross-family pairs) ---")
    if not path.exists():
        print(f"     {path} absent -- skipped.")
        print()
        return
    old = pd.read_csv(path).rename(columns={"p_a": "pA", "p_b": "pB",
                                            "se_item": "se"})
    keep = ["aux", "wh_i", "wh_ii", "neutral_agreement", "neutral_positional"]
    key = ["split", "subcase", "pair"]
    cmp_cols = ["pA", "pB", "diff", "se", "ci90_low", "ci90_high",
                "p_nhst", "p_tost"]
    new = res[res["subcase"].isin(keep) & (res["family"] == "cross")
              & (res["N_eff"] == 300)].set_index(key)[cmp_cols]
    ref = old[old["subcase"].isin(keep)].set_index(key)[cmp_cols]
    common = new.index.intersection(ref.index)
    expected = len(keep) * 2 * len(SPLITS)          # 5 subcases x 2 pairs x 2 splits
    if len(common) != expected:
        print(f"     WARNING: {len(common)} of the expected {expected} rows matched "
              f"on {key}.")
    d = (new.loc[common] - ref.loc[common]).abs()
    worst = d.max().max()
    col = d.max().idxmax()
    print(f"     {len(common)} rows compared. max |discrepancy| = {worst:.3e} "
          f"(column {col!r});")
    print("     the reference file stores 6 decimals, so ~1e-6 is its rounding, "
          "not a disagreement.")
    agree = (res[res["subcase"].isin(keep) & (res["family"] == "cross")]
             .set_index(key)["verdict"].loc[common]
             == old[old["subcase"].isin(keep)].set_index(key)["verdict_item"].loc[common])
    print(f"     verdicts agree on {int(agree.sum())}/{len(common)} rows.")
    print()


# --- main -------------------------------------------------------------------

def main() -> None:
    long, pairs = build_long()
    check_guards(long)
    probe_mode = detect_probe_mode(pairs)
    sensitivity = probe_mode not in ("fixed", "resampled")
    if probe_mode == "fixed":
        n_eff_note = ("FIXED probe -- verified: all 5 seeds re-score the identical "
                      "pair_id set in every (grammar, split, subcase) cell, so "
                      "N_eff = N_sub (300/150/100). The 5 seeds sharpen each "
                      "proportion; they do not multiply the item count. No 5N "
                      "sensitivity column is needed.")
    elif probe_mode == "resampled":
        n_eff_note = ("RESAMPLED probe -- verified: the 5 seeds score disjoint item "
                      "sets, so N_eff = 5 * N_sub.")
    else:
        n_eff_note = ("ASSUMED SINGLE probe -- item identity could not be verified; "
                      "primary N_eff = N_sub, with a labelled 5N sensitivity in "
                      "verdict_5N.")

    res = run_contrasts(long, probe_mode, sensitivity)

    _, tables = ch2_dirs("sub4_tost")
    out = tables / "tost_subcase_level.csv"
    res.to_csv(out, index=False)

    print_header(probe_mode, n_eff_note)
    for sp in SPLITS:
        print("#" * 100)
        print(f"# SPLIT {sp}")
        print("#" * 100)
        print_verdict_counts(res, sp)
        print_floor_table(res, sp)
        print_floor_breaches(res, sp)
        print_wh_i_block(res, sp)
        print_inconclusive(res, sp)
        print_diagnostics(res, sp)
    verify_against_item_level(res, tables)
    print(f"[ch2/sub4 tost-subcase] {len(res)} contrasts -> {out}")


if __name__ == "__main__":
    main()
