"""Metric + taxonomy layer for the separability audit.

Fits a WEAK, LOCAL classifier per cell and reports how separable licensed vs
unlicensed candidates are from surface context alone, under a **cognitively
plausible, graded-memory, left-to-right** observer, then classifies each rule
into a reproducible cue taxonomy. See ``docs/SEPARABILITY.md`` /
``separability_features.py``.

Primary design (confirmed 2026-07-26):
  * **Metric = U** = 1 − H(label|features)/H(label), the normalized
    conditional-entropy reduction, estimated model-based from the CV classifier's
    log-loss (robust where empirical plug-in entropy is not). AUC is a companion.
  * **Representation = decay-weighted position one-hots**, memory constant τ:
    each (offset k × category) one-hot scaled by e^{−(k−1)/τ}. τ→0 = immediate
    neighbours; τ→∞ = full window. Distance costs more (via L2), no hard cutoff.
  * **Scope = LEFT-ONLY** (canonical) — matches the causal model and incremental
    parsing. Both-sided kept only as an object/subject "lookahead" check.
  * **τ-sweep {0.5,1,2,4,∞}** = the "how much memory would it take" diagnostic;
    headline at τ=1.
  * **Neutral floor** (``neutral_floor``): same left-only decay estimator applied
    to next-category prediction on neutral text ≈ the family's local determinism;
    a reference line (targets category, not licensing — NOT subtracted).
    Computed per FAMILY (H≡H′, P≡L′ share draw machinery + lexicon), with a
    sentence-level bootstrap CI. Read it to 2 dp only: precision is ≈±0.01, and
    the whole τ span is only ≈+0.03, so the floor is legitimately near-flat —
    next-category in these grammars is close to first-order Markov. It does NOT
    justify the 0.6 recovery cutoff, which is stipulated independently.

Feature specs: ``("decay", tau, side)`` (primary), plus companions
``("window", w, side)`` / ``("count", None, side)`` / ``("winfm", w, side)``
(the last carries the agreement channel: binding null + neutral control).

Bands (``run_grid``), per phenomenon:
  * **raw** — every enumerated candidate.
  * **comparable** — controls for a precondition SHARED by all grammars
    (``in_band``); uniform by design, so cells are comparable ACROSS grammars.
  * **rulewise** — controls for the RULE'S OWN precondition (``in_rule_band``,
    ``separability_features.OBJECT_DOMAIN_RULES``), so U measures only what the
    rule decides. Emitted only where it differs from comparable. NOT comparable
    across grammars — read it within a grammar. A rulewise cell with no negatives
    (H/H' refl) is reported ``degenerate``: the rule's domain and its extension
    coincide, i.e. the rule has no content beyond its precondition.

Imbalance policy: never rebalance; every cell reports per-class N + bootstrap CI;
CV grouped by base. The primary fit is UNWEIGHTED (U needs true-distribution
log-loss); thin cells surface as wide CIs. The rulewise wh_i cells are strongly
imbalanced (negatives = embedded objects only) — read their CIs, not point U.

Taxonomy (``classify_taxonomy``), per (grammar, phenomenon). The 0.6 cutoff is a
STIPULATED threshold, not one derived from the floor:
  * **bounded-local** — U(τ=0.5, left) ≥ 0.6 (recoverable at minimal memory).
  * **memory-graded** — not bounded-local, but U reaches ≥ 0.6 at some larger τ
    (``tau_recover`` reports the cost; e.g. fixed-offset rules).
  * **not-locally-recoverable** — U stays < 0.6 even at τ=∞ (counting/scan/depth).
"""
from __future__ import annotations

import math
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import (GroupKFold, StratifiedGroupKFold,
                                     cross_val_predict)
from sklearn.tree import DecisionTreeClassifier

from analysis.separability_features import (
    CATEGORY_VOCAB, GRAMMARS, PHENOMENA, PAD, WMAX, _draw_base, _linearise,
    _registry, build_candidate_frame, build_neutral_frame)

THIN_MIN = 30
MAX_FOLDS = 5
DEFAULT_BOOT = 500
LN2 = math.log(2)
MODELS = ("logreg", "tree")

DECAY_TAUS: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, float("inf"))
HEADLINE_TAU = 1.0
RECOVER = 0.6              # U at/above which a rule counts as "recoverable"
                           # STIPULATED, not derived from the floor (see §floor).

# Neutral-floor estimator settings. n must be large: at n=150 the τ=∞ point
# OVERFITS (181 features on ~1.1k rows) and the whole finite-τ span (+0.004) sits
# inside the seed-to-seed noise (sd ≈ 0.009), which made the curve read as
# "flat then falling". At n≥2000 it is monotone rising and saturating.
FLOOR_N_SENT = 2000        # sentences drawn per FAMILY (cells use 800 bases)
FLOOR_C = 1.0              # logreg inverse penalty — the floor's LEVEL is
                           # sensitive to it (U@τ=0.5 ≈ 0.17/0.41/0.50/0.51 at
                           # C = 0.01/0.1/1/10/100), so it is pinned + recorded.
FLOOR_MAX_ITER = 3000      # matches ``_estimator("logreg")``
FLOOR_BOOT = 400           # bootstrap resamples (over SENTENCES) for the CI

# Spec = (kind, param, side). kind ∈ {decay,window,count,winfm};
# param = τ (decay) or w (window/winfm) or None (count); side ∈ {left,right,both}.
Spec = Tuple[str, Optional[float], str]

DEFAULT_SPECS: List[Spec] = (
    [("decay", t, "left") for t in DECAY_TAUS]          # PRIMARY: left τ-sweep
    + [("decay", HEADLINE_TAU, "both")]                 # object/subject check
    + [("window", 7, "left"), ("count", None, "left")]  # companions
    + [("window", 5, "both"), ("winfm", 5, "both")]     # agreement channel
)


# ---------------------------------------------------------------------------
# Feature columns / matrices
# ---------------------------------------------------------------------------

def _window_cols(w: int, side: str = "both") -> List[str]:
    L = [f"L{k}" for k in range(1, w + 1)]
    R = [f"R{k}" for k in range(1, w + 1)]
    return {"left": L, "right": R, "both": L + R}[side]


def _count_cols(side: str = "both") -> List[str]:
    L = [f"nL_{c}" for c in CATEGORY_VOCAB]
    R = [f"nR_{c}" for c in CATEGORY_VOCAB]
    return {"left": L, "right": R, "both": L + R}[side]


def _fmatch_cols(w: int, side: str = "both") -> List[str]:
    L = [f"fmL{k}" for k in range(1, w + 1)]
    R = [f"fmR{k}" for k in range(1, w + 1)]
    return {"left": L, "right": R, "both": L + R}[side]


def spec_label(spec: Spec) -> str:
    kind, p, side = spec
    if kind == "count":
        return f"count:{side}"
    tag = ("inf" if isinstance(p, float) and math.isinf(p) else
           (int(p) if float(p).is_integer() else p))
    return f"{kind}@{tag}:{side}"


def _decay_weight(colname: str, tau: float) -> float:
    k = int(colname.split("_")[0][1:])         # 'L3_CAT1' -> 3
    return 1.0 if math.isinf(tau) else math.exp(-(k - 1) / tau)


def _effective_w(tau: float) -> int:
    """Largest offset whose decay weight is non-negligible (≥1e-3); offsets past
    it contribute ~0, so we drop them (big speedup at small τ)."""
    if math.isinf(tau):
        return WMAX
    return min(WMAX, max(1, int(1 + math.log(1000) * tau)))


def _feature_matrix(df_cell: pd.DataFrame, spec: Spec) -> np.ndarray:
    kind, p, side = spec
    if kind == "count":
        return df_cell[_count_cols(side)].astype(float).to_numpy()
    if kind == "window":
        X = pd.get_dummies(df_cell[_window_cols(int(p), side)].astype(str))
        return X.astype(float).to_numpy()
    if kind == "winfm":
        cols = _window_cols(int(p), side) + _fmatch_cols(int(p), side)
        return pd.get_dummies(df_cell[cols].astype(str)).astype(float).to_numpy()
    if kind == "decay":
        X = pd.get_dummies(df_cell[_window_cols(_effective_w(p), side)].astype(str))
        w = pd.Series({c: _decay_weight(c, p) for c in X.columns})
        return X.mul(w, axis=1).astype(float).to_numpy()
    raise ValueError(f"unknown feature kind {kind!r}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _binent(p: float) -> float:
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def _U(y, proba, base_rate) -> float:
    Hy = _binent(base_rate)
    if Hy <= 0:
        return np.nan
    return 1.0 - (log_loss(y, proba, labels=[0, 1]) / LN2) / Hy


def _bootstrap_ci(y, proba, base_rate, n_boot, seed):
    if n_boot <= 0:
        return (np.nan,) * 4
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    aucs, us = [], []
    for _ in range(n_boot):
        b = rng.choice(idx, len(y), replace=True)
        if np.unique(y[b]).size < 2:
            continue
        aucs.append(roc_auc_score(y[b], proba[b]))
        us.append(_U(y[b], proba[b], y[b].mean()))
    if not aucs:
        return (np.nan,) * 4
    return (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5)),
            float(np.nanpercentile(us, 2.5)), float(np.nanpercentile(us, 97.5)))


def _estimator(model: str):
    # UNWEIGHTED (U needs true-distribution log-loss).
    if model == "logreg":
        return LogisticRegression(penalty="l2", max_iter=3000, solver="lbfgs")
    if model == "tree":
        return DecisionTreeClassifier(max_depth=3, random_state=0)
    raise ValueError(f"unknown model {model!r}")


def cell_separability(df_cell: pd.DataFrame, spec: Spec, model: str, *,
                      n_boot: int = DEFAULT_BOOT, seed: int = 0) -> Dict:
    y = df_cell["is_licensed"].astype(int).to_numpy()
    n = int(y.size); n_pos = int(y.sum()); n_neg = n - n_pos
    minority = min(n_pos, n_neg); base_rate = n_pos / n if n else np.nan
    out = {
        "model": model, "spec": spec_label(spec),
        "kind": spec[0], "param": spec[1], "side": spec[2],
        "n": n, "n_pos": n_pos, "n_neg": n_neg, "base_rate": base_rate,
        "majority_baseline": max(base_rate, 1 - base_rate) if n else np.nan,
        "thin": minority < THIN_MIN, "degenerate": minority < 2,
        "auc": np.nan, "auc_lo": np.nan, "auc_hi": np.nan,
        "U": np.nan, "U_lo": np.nan, "U_hi": np.nan,
        "balanced_acc": np.nan, "n_folds": 0,
    }
    if out["degenerate"]:
        return out
    groups = df_cell["base_id"].to_numpy()
    ng = min(pd.Series(groups[y == 1]).nunique(), pd.Series(groups[y == 0]).nunique())
    k = min(MAX_FOLDS, minority, ng)
    if k < 2:
        out["degenerate"] = True
        return out
    X = _feature_matrix(df_cell, spec)
    cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    try:
        proba = cross_val_predict(_estimator(model), X, y, cv=cv, groups=groups,
                                  method="predict_proba")[:, 1]
    except ValueError:
        # a grouped fold degenerated to a single class (very thin cell)
        out["degenerate"] = True
        return out
    out["auc"] = float(roc_auc_score(y, proba))
    out["U"] = float(_U(y, proba, base_rate))
    out["balanced_acc"] = float(balanced_accuracy_score(y, (proba >= 0.5).astype(int)))
    out["auc_lo"], out["auc_hi"], out["U_lo"], out["U_hi"] = _bootstrap_ci(
        y, proba, base_rate, n_boot, seed)
    out["n_folds"] = k
    return out


def run_grid(frames: Dict[tuple, pd.DataFrame], *, specs: List[Spec] = None,
             models=MODELS, n_boot: int = DEFAULT_BOOT, seed: int = 0
             ) -> pd.DataFrame:
    specs = specs or DEFAULT_SPECS

    def _specs_for(regime, band):
        # neutral: comparable only (in_band all True); just the control specs.
        if regime.startswith("neutral"):
            return ([] if band == "raw"
                    else [s for s in specs
                          if spec_label(s) in ("window@5:both", "winfm@5:both")])
        # raw band on natural regimes: only the headline spec (freebie contrast).
        if band == "raw":
            return [s for s in specs if s == ("decay", HEADLINE_TAU, "left")]
        return specs

    rows: List[Dict] = []
    for (grammar, regime), df in frames.items():
        for phenomenon, dphen in df.groupby("phenomenon"):
            comp = dphen[dphen.is_licensed | dphen.in_band]
            bands = {"raw": dphen, "comparable": comp}
            # rulewise: bands the RULE'S OWN precondition instead of the shared
            # one (separability_features.OBJECT_DOMAIN_RULES). Added only where it
            # actually differs from comparable, so the grid barely grows.
            if not regime.startswith("neutral"):
                rw = dphen[dphen.is_licensed | dphen.in_rule_band]
                if not rw.index.equals(comp.index):
                    bands["rulewise"] = rw
            for band, dband in bands.items():
                for spec in _specs_for(regime, band):
                    for model in models:
                        res = cell_separability(dband, spec, model,
                                                n_boot=n_boot, seed=seed)
                        res.update(grammar=grammar, regime=regime,
                                   phenomenon=phenomenon, band=band)
                        rows.append(res)
    cols = ["grammar", "phenomenon", "regime", "band", "spec", "kind", "param",
            "side", "model", "U", "U_lo", "U_hi", "auc", "auc_lo", "auc_hi",
            "balanced_acc", "base_rate", "majority_baseline", "n", "n_pos",
            "n_neg", "n_folds", "thin", "degenerate"]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Neutral determinism floor (left-only decay, next-category prediction)
# ---------------------------------------------------------------------------

def _floor_U(y: np.ndarray, P: np.ndarray, classes: List[str]) -> float:
    """U for the multiclass next-category target: 1 − H(Y|X)/H(Y), with H(Y) the
    plug-in marginal entropy of THIS sample (so bootstrap resamples are
    self-consistent, as in ``_bootstrap_ci``)."""
    N = len(y)
    if N == 0:
        return np.nan
    Hy = -sum((v / N) * math.log2(v / N) for v in Counter(y).values())
    if Hy <= 0:
        return np.nan
    return 1.0 - (log_loss(y, P, labels=classes) / LN2) / Hy


def _floor_rows(family: str, load_lex, *, n: int, seed: int
                ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Left-context rows for every token position of ``n`` drawn neutral bases.

    NOTE the PAD convention: at token ``t`` exactly ``WMAX − t`` left slots are
    PAD, so the PAD *count* encodes absolute position — an extra channel that
    grows with τ. Kept deliberately: the candidate cells use the same windowing,
    and diverging here would break floor/cell comparability.
    """
    lex = load_lex(); rng = random.Random(seed)
    recs: List[Dict] = []; kept = 0; attempts = 0
    while kept < n and attempts < n * 600:
        attempts += 1
        cats, _infl, _ri = _linearise(_draw_base(family, rng, lex), family)
        if not (2 <= len(cats) <= WMAX):
            continue
        for t in range(1, len(cats)):
            row = {f"L{k}": (cats[t - k] if t - k >= 0 else PAD)
                   for k in range(1, WMAX + 1)}
            row["_y"] = cats[t]; row["_g"] = kept
            recs.append(row)
        kept += 1
    if kept < n:
        raise RuntimeError(f"{family}: only {kept}/{n} bases after {attempts} draws")
    df = pd.DataFrame(recs)
    return df, df["_y"].to_numpy(), df["_g"].to_numpy()


def neutral_floor(family: str, load_lex, taus=DECAY_TAUS, *,
                  n: int = FLOOR_N_SENT, seed: int = 20260726,
                  n_boot: int = FLOOR_BOOT, C: float = FLOOR_C,
                  max_iter: int = FLOOR_MAX_ITER) -> pd.DataFrame:
    """U of predicting a token's category from its left decay-context on natural
    neutral text — the family's local determinism floor, per τ, WITH a CI.

    Computed per **FAMILY**, not per grammar: ``_registry()`` maps H/H′ to the
    same (FAMILY_TREE, grammar_H lexicon) and P/L′ to the same tokens pair, and
    the draw is seeded — so a per-grammar call would fit the identical sentences
    twice and report H≡H′, P≡L′ to the last decimal. ``run_full_analysis`` fans
    the family result back out to its grammars.

    The CI is a bootstrap over **sentences** (not rows — rows within a sentence
    are dependent) on the held-out CV probabilities, matching ``_bootstrap_ci``.
    Precision is ≈±0.01 at n=2000; do not read the floor to more than 2 dp.
    """
    df, y, g = _floor_rows(family, load_lex, n=n, seed=seed)
    classes = sorted(set(y))
    sents = np.unique(g)
    rows_by_sent = {s: np.flatnonzero(g == s) for s in sents}
    rng_b = np.random.default_rng(seed)

    out: List[Dict] = []
    for tau in taus:
        cols = [f"L{k}" for k in range(1, _effective_w(tau) + 1)]
        Xoh = pd.get_dummies(df[cols].astype(str))
        w = pd.Series({c: _decay_weight(c, tau) for c in Xoh.columns})
        X = Xoh.mul(w, axis=1).astype(float).to_numpy()
        P = cross_val_predict(
            LogisticRegression(max_iter=max_iter, C=C), X, y,
            cv=GroupKFold(3), groups=g, method="predict_proba")
        boots = []
        for _ in range(n_boot):
            pick = rng_b.choice(sents, sents.size, replace=True)
            idx = np.concatenate([rows_by_sent[s] for s in pick])
            boots.append(_floor_U(y[idx], P[idx], classes))
        lo, hi = ((float(np.nanpercentile(boots, 2.5)),
                   float(np.nanpercentile(boots, 97.5))) if boots
                  else (np.nan, np.nan))
        out.append({"tau": tau, "floor_U": _floor_U(y, P, classes),
                    "floor_U_lo": lo, "floor_U_hi": hi,
                    "n_sent": int(sents.size), "n_rows": int(len(y)),
                    "n_feat": int(X.shape[1]), "C": C})
    return pd.DataFrame(out)


# ---------------------------------------------------------------------------
# Taxonomy
# ---------------------------------------------------------------------------

def classify_taxonomy(grid: pd.DataFrame, floor: pd.DataFrame = None, *,
                      model: str = "logreg", band: str = "comparable"
                      ) -> pd.DataFrame:
    g = grid[(grid.model == model) & (grid.band == band) & (grid.regime == "train")]

    def U(spec_lbl, phen, gram):
        m = g[(g.spec == spec_lbl) & (g.phenomenon == phen) & (g.grammar == gram)]
        return float(m.U.iloc[0]) if len(m) and not m.U.isna().all() else np.nan

    def reliability(phen, gram):
        """Class balance of the headline cell — a taxonomy label read off a cell
        with a minority class below ``THIN_MIN`` is not evidence either way, so
        it travels WITH the label rather than three sections away."""
        m = g[(g.spec == spec_label(("decay", HEADLINE_TAU, "left")))
              & (g.phenomenon == phen) & (g.grammar == gram)]
        if not len(m):
            return dict(n_pos=np.nan, n_neg=np.nan, thin=True)
        r = m.iloc[0]
        return dict(n_pos=r.n_pos, n_neg=r.n_neg, thin=bool(r.thin))

    rows = []
    for gram in sorted(g.grammar.unique()):
        for phen in sorted(p for p in g.phenomenon.unique() if p in PHENOMENA):
            us = {t: U(spec_label(("decay", t, "left")), phen, gram) for t in DECAY_TAUS}
            u_small = us[DECAY_TAUS[0]]
            recover = [t for t in DECAY_TAUS if us[t] >= RECOVER]
            # NaN = never recovered, at any memory. inf = recovered ONLY at
            # unbounded memory. These are different claims: do not collapse them
            # onto one sentinel (they were both `inf` before 2026-07-31).
            tau_recover = min(recover) if recover else np.nan
            if u_small != u_small:
                tax = "n/a"
            elif u_small >= RECOVER:
                tax = "bounded-local"
            elif recover:
                tax = "memory-graded"
            else:
                tax = "not-locally-recoverable"
            gap = (U(spec_label(("decay", HEADLINE_TAU, "both")), phen, gram)
                   - us[HEADLINE_TAU])          # object/subject lookahead check
            rows.append(dict(
                grammar=gram, phenomenon=phen, taxonomy=tax,
                tau_recover=tau_recover,
                **{f"U_t{('inf' if math.isinf(t) else t)}": us[t] for t in DECAY_TAUS},
                lookahead_gap=gap, **reliability(phen, gram)))
    tax_df = pd.DataFrame(rows)
    if floor is not None:
        f1 = floor[floor.tau == HEADLINE_TAU].set_index("grammar").floor_U
        tax_df["floor_t1"] = tax_df.grammar.map(f1)
    return tax_df


def run_full_analysis(*, n_bases_train: int = 800, n_bases_ood: int = 250,
                      seed: int = 20260726, n_boot: int = 300, models=("logreg",),
                      out_dir=None) -> Dict[str, pd.DataFrame]:
    from pathlib import Path
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames: Dict[tuple, pd.DataFrame] = {}
    for gmr in GRAMMARS:
        frames[(gmr, "train")] = build_candidate_frame(gmr, "train", n_bases_train, seed)
        frames[(gmr, "ood")] = build_candidate_frame(gmr, "ood", n_bases_ood, seed)
    for pr in ("iod", "ood"):
        neu = build_neutral_frame(pr)
        for gmr in GRAMMARS:
            sub = neu[neu.grammar == gmr]
            if len(sub):
                frames[(gmr, f"neutral_{pr}")] = sub

    grid = run_grid(frames, models=models, n_boot=n_boot, seed=0)

    # Floor: one fit per FAMILY, fanned out to the grammars that share it
    # (H≡H′, P≡L′ by construction — see ``neutral_floor``).
    reg = _registry()
    by_family: Dict[str, List[str]] = {}
    for gmr in GRAMMARS:
        by_family.setdefault(reg[gmr][0], []).append(gmr)
    parts = []
    for fam, gmrs in by_family.items():
        fl = neutral_floor(fam, reg[gmrs[0]][1], seed=seed)
        for gmr in gmrs:
            parts.append(fl.assign(grammar=gmr, family=fam))
    floor = pd.concat(parts, ignore_index=True)[
        ["grammar", "family", "tau", "floor_U", "floor_U_lo", "floor_U_hi",
         "n_sent", "n_rows", "n_feat", "C"]]
    tax = classify_taxonomy(grid, floor, model=models[0])
    grid.to_parquet(out_dir / "separability_results.parquet")
    floor.to_parquet(out_dir / "separability_floor.parquet")
    tax.to_parquet(out_dir / "separability_taxonomy.parquet")
    return {"grid": grid, "floor": floor, "taxonomy": tax}


if __name__ == "__main__":  # smoke test (small, train-only, no floor)
    frames = {(gmr, "train"): build_candidate_frame(gmr, "train", 400, 20260726)
              for gmr in GRAMMARS}
    grid = run_grid(frames, models=("logreg",), n_boot=0)
    print(classify_taxonomy(grid).round(3).to_string(index=False))
