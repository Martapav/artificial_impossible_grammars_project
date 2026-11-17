"""Shared loaders and style for the ``set_3_ep12`` reporting figures.

Three sibling scripts consume this module:

    analysis/reports/training/report_training.py  -> results/analysis/training/
    analysis/reports/heldout/report_test.py      -> results/analysis/evaluation/test/
    analysis/reports/probes/report_probes.py    -> results/analysis/evaluation/structural_probes/

Everything is reported at the **final (ep12) model**. The 30-epoch overfitting
that motivated a reporting-epoch rule is gone at cap 12, so no reporting-epoch is
selected and no ``e=7`` marker is drawn anywhere (decision 2026-09-09). The
per-metric plateau *epochs* are still tabulated on the training side, as a
descriptive readout of when each curve settles -- not as a checkpoint choice.

Style, grammar keys/colours, pooling styles and the plateau machinery are
inherited from :mod:`analysis.selection_epoch`; display renames (L' -> P',
CATn -> NOM/...) from :mod:`analysis.category_labels`. Nothing here recomputes a
margin: the held-out term table (``terms.csv``) and the pair builder
(:mod:`evaluation.aggregate`) are the single sources of every probe number.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from analysis.selection_epoch import (  # noqa: E402
    GRAMMARS, GRAMMAR_DISPLAY, GRAMMAR_COLOR, POOLINGS, POOL_STYLE, _seed_of,
)
from evaluation.aggregate import to_pairs, term_table  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
EVAL_ROOT = REPO / "results" / "eval" / "set_3_ep12"
CKPT_ROOT = REPO / "models" / "checkpoints" / "set_3_ep12"
ANALYSIS_ROOT = REPO / "results" / "analysis"
# Reporting corpus the set_3_ep12 models were trained/scored on. Used only to
# recover sidecar structural fields (e.g. n_verbs) absent from the score parquets.
CORPUS_ROOT = REPO / "corpora" / "set_3"

FINAL_EPOCH = 12.0

# --- display names ---------------------------------------------------------

# Test-set generative phenomena (the `construction` column of the test items).
CONSTRUCTION_DISPLAY = {
    "neutral": "neutral",
    "wh_movement": "wh-movement",
    "auxiliary_movement": "aux-movement",
    "anaphoric_binding_refl": "reflexive",
    "anaphoric_binding_pron": "pronoun",
}
CONSTRUCTION_ORDER = list(CONSTRUCTION_DISPLAY)

# Probe phenomena (the `subcase` column of the probe pairs).
SUBCASE_DISPLAY = {
    "refl": "reflexive",
    "pron": "pronoun",
    "wh_i": "wh (i)",
    "wh_ii": "wh (ii)",
    "aux": "aux movement",
    "neutral_agreement": "agreement (gate)",
    "neutral_positional": "positional (gate)",
}
SUBCASE_ORDER = list(SUBCASE_DISPLAY)

# Held-out probe term order (rows of the term x grammar heatmaps).
TERM_ORDER = [
    "aux", "wh_i", "wh_ii",
    "refl:wrong_item", "refl:wrong_site", "refl:no_substitution",
    "pron:wrong_item", "pron:wrong_site",
    "neutral_agreement", "neutral_positional",
]

# --- E4 graded-confidence bins (confirmed 2026-09-09) ----------------------
#
# A three-way ordering of a minimal pair's ungrammatical member, keyed on the
# recorded `violation` (already `positional_*`-collapsed by `to_pairs`):
#
#   A  target (fired)        the grammatical member -- the within-pair reference,
#                            so its margin is 0 by construction.
#   B  licensing, non-fired  same site, wrong/absent realisation: the licensing
#                            environment holds but the correct form did not fire.
#   C  non-licensing         displaced / wrong environment: the form appears where
#                            it is not licensed.
#
# Coverage: refl -> A,B,C ; pron -> A,B,C ; agreement -> A,B ; wh_i/wh_ii/aux ->
# A,C ; positional -> A,C. Expected graded ordering of the model's within-pair
# margin: 0 = Delta_A < Delta_B < Delta_C (a non-licensing error is penalised
# harder than a licensing-but-unfired one).
VIOLATION_TO_BIN = {
    "wrong_item": "B",
    "no_substitution": "B",
    "agreement": "B",
    "wrong_site": "C",
    "unlicensed_site": "C",
    "positional": "C",
}
BIN_ORDER = ["A", "B", "C"]
BIN_LABEL = {
    "A": "target (fired)",
    "B": "licensing, non-fired",
    "C": "non-licensing",
}
BIN_COLOR = {"A": "#4d4d4d", "B": "#2b8cbe", "C": "#d7301f"}
SPLIT_DISPLAY = {"structural_probe_IOD": "IOD", "structural_probe_OOD": "OOD"}


# --- loaders ---------------------------------------------------------------

def _run_id_parts(run_id: str) -> tuple[str, int]:
    return run_id.split("__")[0], _seed_of(run_id)


def load_terms() -> pd.DataFrame:
    """The held-out per-term table (IOD + OOD), one row per grammar x term.

    This is the ep12 model's ``terms.csv`` -- ``margin``, ``margin_over_b4``,
    ``acc``, ``acc_over_b4``, ``is_gate`` already computed by
    :mod:`evaluation.aggregate`. Not recomputed here.
    """
    df = pd.read_csv(EVAL_ROOT / "terms.csv")
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    return df


def load_test_items() -> pd.DataFrame:
    """Per-item generative test scores for every run (in-distribution only).

    Columns: run_id, grammar, seed, construction, length, n_tokens, neg_logq,
    neg_logq_per_token.
    """
    frames = [pd.read_parquet(p)
              for p in sorted((EVAL_ROOT / "test_indistribution_items").glob("*.parquet"))]
    df = pd.concat(frames, ignore_index=True)
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    return df


def load_test_summary() -> pd.DataFrame:
    """Per-run aggregate test metrics (perplexity, nats/token)."""
    df = pd.read_csv(EVAL_ROOT / "test_indistribution_eval.csv")
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    return df


def load_probe_pairs(splits: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Per-pair held-out probe scores for every run, paired and criterion-tagged.

    Runs :func:`evaluation.aggregate.to_pairs` on each ``probe_scores`` shard, so
    every row carries the oriented ``margin`` (criterion-native: span uses
    ``neg_logq_crit``, string the whole string), ``margin_string``, ``acc``,
    the normalised ``violation``, ``subcase``, ``criterion`` and ``term``. The
    graded-confidence bin is attached here from :data:`VIOLATION_TO_BIN`.

    ``splits`` filters to the given probe splits (default: keep IOD and OOD).
    """
    keep = ("structural_probe_IOD", "structural_probe_OOD") if splits is None else splits
    frames = []
    for p in sorted((EVAL_ROOT / "probe_scores").glob("*.parquet")):
        pairs = to_pairs(pd.read_parquet(p))
        pairs = pairs[pairs["split"].isin(keep)]
        frames.append(pairs)
    df = pd.concat(frames, ignore_index=True)
    df["seed"] = df["run_id"].map(_seed_of)
    df["bin"] = df["violation"].map(VIOLATION_TO_BIN)      # B / C on the violation
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMARS, ordered=True)
    df["phenomenon"] = df["subcase_g"] if "subcase_g" in df.columns else df["subcase"]
    return df


def attach_pair_baselines(pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach the n-gram baselines to per-pair rows and derive both corrected
    margins, span-aware.

    Joins ``baseline_pairs.parquet`` on ``(grammar, split, pair_id)`` and returns
    a copy with ``margin_b1``, ``margin_b1_span``, ``margin_b4``,
    ``margin_b4_span`` and:

        margin_corrected = margin - (margin_b1_span if span else margin_b1)
        margin_over_b4   = margin - (margin_b4_span if span else margin_b4)

    where the span reference is used exactly for the in-place-substitution terms
    (``criterion == "span"``) and the whole-string reference otherwise -- the same
    geometry :mod:`evaluation.aggregate` uses. ``margin_corrected`` is the
    objective's own S[r] correction (subtracting the order-1 nuisance); nothing is
    recomputed here beyond the subtraction.

    Raises if any pair is missing a baseline row (a corpus-binding / rebuild bug
    -- the join is positional on ``pair_id``).
    """
    bp = pd.read_parquet(EVAL_ROOT / "baseline_pairs.parquet")
    key = ["grammar", "split", "pair_id"]
    cols = ["margin_b1", "margin_b1_span", "margin_b4", "margin_b4_span"]
    right = bp[key + cols].copy()
    out = pairs.copy()
    for k in key:
        right[k] = right[k].astype(str)
        out[k] = out[k].astype(str)
    m = out.merge(right, on=key, how="left", validate="many_to_one")
    missing = m[cols].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} pair(s) have no n-gram baseline; rebuild "
            f"baseline_pairs (corpus-binding mismatch?)")
    is_span = m["criterion"].eq("span")
    m["margin_corrected"] = m["margin"] - m["margin_b1"].where(~is_span, m["margin_b1_span"])
    m["margin_over_b4"] = m["margin"] - m["margin_b4"].where(~is_span, m["margin_b4_span"])
    return m


def attach_verb_counts(pairs: pd.DataFrame) -> pd.DataFrame:
    """Attach the grammatical member's verb count (``n_verbs_g``) to each pair.

    ``n_verbs`` is not carried in the ``probe_scores`` parquets (only ``length``,
    ``n_morphemes`` and ``embedding_depth`` ride along), so it is recovered from
    the corpus meta sidecar under :data:`CORPUS_ROOT`. Reads the ``grammatical``
    variant -- the member ``length_g`` / ``embedding_depth_g`` already refer to --
    of each ``structural_probe_{IOD,OOD}/<grammar>.meta.jsonl``, keyed on
    ``(grammar, split, pair_id)``.

    Raises if any pair is unmatched (a corpus-binding / rebuild mismatch -- the
    reporting corpus is ``set_3``, and the join is positional on ``pair_id``).
    """
    import json

    recs = []
    for g in GRAMMARS:
        for split in ("structural_probe_IOD", "structural_probe_OOD"):
            meta = CORPUS_ROOT / split / f"{g}.meta.jsonl"
            with open(meta) as fh:
                for line in fh:
                    d = json.loads(line)
                    if d.get("variant") != "grammatical":
                        continue
                    recs.append((str(g), split, str(d["pair_id"]), int(d["n_verbs"])))
    right = pd.DataFrame(recs, columns=["grammar", "split", "pair_id", "n_verbs_g"])

    key = ["grammar", "split", "pair_id"]
    out = pairs.copy()
    for k in key:
        out[k] = out[k].astype(str)
    m = out.merge(right, on=key, how="left", validate="many_to_one")
    missing = m["n_verbs_g"].isna()
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} pair(s) have no sidecar n_verbs; check the "
            f"reporting corpus at {CORPUS_ROOT} (corpus-binding mismatch?)")
    m["n_verbs_g"] = m["n_verbs_g"].astype(int)
    return m


def load_select_terms_final() -> pd.DataFrame:
    """Per (grammar, term) margin/acc at the FINAL epoch on the SELECT split.

    Reads each training ``probe_curve.parquet`` (the per-epoch SELECT split),
    keeps ``epoch == FINAL_EPOCH``, pairs it and reduces to the term table, then
    averages over seeds. No b4 baseline exists on SELECT, so this is raw margin.
    """
    frames = []
    for p in sorted(CKPT_ROOT.glob("*/probe_curve.parquet")):
        curve = pd.read_parquet(p)
        curve = curve[curve["epoch"] == FINAL_EPOCH]
        if curve.empty:
            continue
        frames.append(term_table(to_pairs(curve)))
    terms = pd.concat(frames, ignore_index=True)
    terms["grammar"] = terms["grammar"].astype(str)
    agg = (terms.groupby(["grammar", "term", "criterion"], observed=True)
           .agg(margin=("margin", "mean"), acc=("acc", "mean"),
                n_seeds=("margin", "size"))
           .reset_index())
    agg["grammar"] = pd.Categorical(agg["grammar"], categories=GRAMMARS, ordered=True)
    return agg


# --- small helpers ---------------------------------------------------------

def bootstrap_ci(values: np.ndarray, *, n_boot: int = 2000, alpha: float = 0.05,
                 stat=np.mean, seed: int = 0) -> tuple[float, float, float]:
    """(point, lo, hi) for ``stat`` of ``values`` via the percentile bootstrap."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(stat(values))
    if values.size == 1:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = stat(values[idx], axis=1)
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def grammar_handles():
    """Legend handles mapping each grammar's colour to its display name."""
    return [plt.Line2D([], [], color=GRAMMAR_COLOR[g], label=GRAMMAR_DISPLAY[g])
            for g in GRAMMARS]


def savefig(fig, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def outdir(*parts: str) -> Path:
    d = ANALYSIS_ROOT.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


def ch2_dirs(subname: str) -> tuple[Path, Path]:
    """(figures_dir, tables_dir) for a chapter-2 probe subchapter.

    Keeps plots and tables in separate sibling folders under
    ``results/analysis/evaluation/structural_probes/ch2/<subname>/``.
    """
    base = ("evaluation", "structural_probes", "ch2", subname)
    return outdir(*base, "figures"), outdir(*base, "tables")
