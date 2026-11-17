"""Recipe selection: admissibility by convergence, ranked by the worst grammar.

What this replaces, and why
---------------------------
The previous ``scripts/select_recipe.py`` ranked recipes by ``mean_loss``, the
validation loss averaged over the four grammars, with an adequacy filter of
``eval_loss > 10``. That number is not interpretable. H, P, H′ and L′ are four
DIFFERENT corpora with different true entropies — ``analysis/entropy_baseline.py``
exists precisely to measure how different — so a grammar whose corpus is
intrinsically higher-entropy contributes a larger loss to the average no matter
how well it was learned. Averaging over grammars therefore ranks recipes partly
by which grammars happen to be easy, and a recipe can win the pooled average
while being the worst available choice on one grammar. The pooled column is
deleted rather than annotated: a caveated wrong number still gets read off.

What replaces it
----------------
**The recipe is selected on ``S[r]``, the minimal-pair probe objective — not on
validation loss.** That is `RECIPE_SELECTION.md` Step 5.1, and §1.3 gives
the reason: the transformation-critical positions are ~1.26% of the loss, so no
amount of renormalizing it produces a quantity that resolves rule learning. On
the grid96 sweep the two objectives are close to inverted — the loss-ranked
winner is 9th of 12 on the probe.

1. **Admissibility.** Step 3's two gates, each a floor rather than a quality
   bar: ``neutral_agreement`` accuracy >= 0.90 on the worst grammar, and
   validation loss beating the order-1 token baseline ``b_tok`` (2.26
   nats/token). All four grammars must be present.

2. **Ranking.** ``S[r]``, pooled symmetrically over grammars. See the Step 5.1
   section below for the definition and for why the pooling is load-bearing.

3. **Tie-break (Step 5.4).** This landscape is flat, and the winner usually does
   NOT lead by more than seed noise. When it does not, the choice falls to a
   throughput convention — smallest model, then largest batch — and the report
   says so in those words. A recipe chosen inside the noise band is fine; one
   chosen inside the noise band and *reported as the best* is not.

The loss side is kept below, demoted to diagnostics and an optional gate
(``--convergence-gate``), and the old ranking is reachable with
``--rank-by-loss`` for comparison. Note that the convergence gate is off by
default with reason: this sweep stops when the PROBE plateaus, which happens a
median of 10 epochs after validation loss bottoms out, so at the stopping epoch
the loss curve is rising by design and a curve-shape test rejects nearly every
run for something that is not a property of the recipe.

   Among admissible recipes the loss ranking orders by the WORST grammar. Raw
   losses are not comparable across grammars, so "worst" is defined on a
   per-grammar normalized quantity — by default each grammar's loss expressed as
   a ratio to that grammar's own best observed value anywhere in the grid.
   Selectable with ``--normalize``. Ties break lexicographically down the sorted
   score vector, a leximin order, deliberately NOT by a mean of the four scores.

Reading the curve
-----------------
``models/grid_search.py`` writes ``eval_curve`` (one ``{epoch, eval_loss}`` per
epoch) into ``results.jsonl``. Runs recorded before that field existed carry no
curve, and their convergence is therefore **unknown, not assumed good**: they
come back inadmissible with reason ``no_eval_curve``. That is the honest verdict
— "we never recorded whether it converged" is not evidence that it did — and it
is why the CLI reports the missing-curve count separately from genuine failures.

Usage
-----
    python -m analysis.recipe_selection results/grid_search_full_sweep/results.jsonl
    python -m analysis.recipe_selection <path> --normalize excess --final-k 5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

GRAMMARS: Tuple[str, ...] = ("H", "P", "Hprime", "Lprime")

# A "recipe" is the tuple the grid varies. Seeds are replicates WITHIN a cell,
# not part of the recipe.
RECIPE_FIELDS: Tuple[str, ...] = ("size_name", "learning_rate", "batch_size")

Recipe = Tuple


def recipe_of(run: Dict) -> Recipe:
    """The recipe key of a run record. ``learning_rate``/``batch_size`` are
    normalized to float/int so a CSV row and a JSONL row key identically."""
    return (
        str(run["size_name"]),
        float(run["learning_rate"]),
        int(run["batch_size"]),
    )


def format_recipe(rc: Recipe) -> str:
    return f"{rc[0]}/lr{rc[1]:.0e}/bs{rc[2]}"


# ---------------------------------------------------------------------------
# B2 — convergence stability
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConvergenceCriteria:
    """When a single run counts as having *converged stably*.

    Operationalization (all must hold; every parameter below is configurable so
    the definition can be tightened or loosened without editing this file):

    1. ``finite`` — the reported ``train_loss`` and ``eval_loss`` are finite, and
       no point on the validation curve is NaN or infinite. A NaN anywhere means
       the optimizer left the numerically valid region; nothing downstream of
       that is meaningful.
    2. ``bounded`` — final ``eval_loss <= max_eval_loss``. An absolute sanity
       ceiling that catches a run which stayed finite but never learned. Its
       value is a loose backstop, NOT a quality bar: quality is the ranking's
       job, and this must not become a disguised per-grammar threshold.
    3. ``no_divergence`` — the final loss is not worse than the best loss the run
       ever reached by more than ``divergence_tolerance``. This is what "did not
       diverge" means on a curve: a run that hit 1.78 and then climbed away to
       3.0 has diverged even though every value is finite and bounded.
    4. ``monotone`` — over the final ``final_k`` evaluations the curve is
       non-increasing within ``monotone_tolerance``: each step may rise by at
       most the tolerance. The tolerance is non-zero on purpose — evaluation
       noise makes exact monotonicity a test of luck rather than of stability.
    5. ``plateau`` — the total improvement across those final ``final_k``
       evaluations is at most ``plateau_tolerance``. This is the "converged
       WITHIN the budget" half of the requirement: a run still descending
       steeply when the epoch cap arrived has not converged, it was truncated,
       and its final loss understates what the recipe would reach.

    ``require_curve`` (default True) makes a missing ``eval_curve`` a failure
    rather than a pass. Runs predating the field in ``models/grid_search.py``
    have no curve; treating that as "converged" would silently readmit exactly
    the unverified runs this check exists to exclude.

    ``require_early_stop`` (default False) additionally demands the run stopped
    on patience rather than hitting the epoch cap. It was off by default because
    the sweep then ran to a FIXED epoch budget, where every run hits the cap by
    construction and requiring otherwise would empty the admissible set. That
    stopped being true when the sweep moved to a probe plateau
    (``models/grid_search.py``, "trains until the probe score stops improving");
    on grid96 all 96 runs have ``hit_cap: false``. The flag is therefore now a
    meaningful check rather than a vacuous one, and could reasonably default on.

    **These criteria assume the run stopped when its VALIDATION LOSS stopped
    improving.** Under probe-based stopping they do not hold: the loss minimum
    arrives a median of 10 epochs before the run ends, so the final ``final_k``
    evaluations sit on the rising side and ``monotone``/``no_divergence`` fire on
    nearly every run. At the shipped tolerances only 6 of 96 grid96 runs are
    monotone and 69 "diverge", by 0.002-0.006 nats on a loss of 1.77. Genuine
    divergence is still separable — it runs to 0.5-4.8 nats — but it is caught
    just as well by the ``b_tok`` gate, which is why the probe path does not
    apply these by default.
    """

    final_k: int = 3
    monotone_tolerance: float = 1e-3
    plateau_tolerance: float = 1e-2
    divergence_tolerance: float = 1e-2
    max_eval_loss: float = 10.0
    require_curve: bool = True
    require_early_stop: bool = False


@dataclass
class ConvergenceReport:
    """Per-run convergence verdict plus the numbers behind it."""

    run_id: str
    stable: bool
    reasons: List[str] = field(default_factory=list)
    n_evals: int = 0
    final_loss: float = float("nan")
    best_loss: float = float("nan")
    # Worst upward step among the final k evaluations (<=0 means monotone).
    max_rise: float = float("nan")
    # Total improvement across the final k evaluations (small means plateaued).
    tail_improvement: float = float("nan")

    def summary(self) -> str:
        if self.stable:
            return (f"stable (rise {self.max_rise:+.4f}, "
                    f"tail Δ {self.tail_improvement:.4f})")
        return "; ".join(self.reasons) or "unstable"


def assess_convergence(run: Dict,
                       criteria: ConvergenceCriteria = ConvergenceCriteria(),
                       ) -> ConvergenceReport:
    """Apply :class:`ConvergenceCriteria` to one run record."""
    rep = ConvergenceReport(run_id=str(run.get("run_id", "?")), stable=True)

    eval_loss = run.get("eval_loss")
    train_loss = run.get("train_loss")
    rep.final_loss = float(eval_loss) if eval_loss is not None else float("nan")

    # 1. finite
    for name, val in (("eval_loss", eval_loss), ("train_loss", train_loss)):
        if val is None or not math.isfinite(float(val)):
            rep.stable = False
            rep.reasons.append(f"{name}_not_finite")

    # 2. bounded
    if eval_loss is not None and math.isfinite(float(eval_loss)) \
            and float(eval_loss) > criteria.max_eval_loss:
        rep.stable = False
        rep.reasons.append(
            f"eval_loss {float(eval_loss):.3f} > max {criteria.max_eval_loss}")

    curve = run.get("eval_curve") or []
    losses = [float(p["eval_loss"]) for p in curve if "eval_loss" in p]
    rep.n_evals = len(losses)

    if not losses:
        if criteria.require_curve:
            rep.stable = False
            rep.reasons.append("no_eval_curve")
        return rep

    if any(not math.isfinite(v) for v in losses):
        rep.stable = False
        rep.reasons.append("nan_in_curve")
        return rep

    rep.best_loss = min(losses)
    rep.final_loss = losses[-1]

    # 3. no divergence away from the best point reached
    if losses[-1] > rep.best_loss + criteria.divergence_tolerance:
        rep.stable = False
        rep.reasons.append(
            f"diverged (final {losses[-1]:.4f} vs best {rep.best_loss:.4f})")

    # 4/5. tail behaviour over the final k evaluations
    if len(losses) < criteria.final_k:
        rep.stable = False
        rep.reasons.append(
            f"only {len(losses)} evals, need {criteria.final_k}")
        return rep

    tail = losses[-criteria.final_k:]
    rep.max_rise = max(b - a for a, b in zip(tail, tail[1:])) if len(tail) > 1 else 0.0
    rep.tail_improvement = tail[0] - tail[-1]

    if rep.max_rise > criteria.monotone_tolerance:
        rep.stable = False
        rep.reasons.append(
            f"not monotone over final {criteria.final_k} "
            f"(worst rise {rep.max_rise:+.4f} > {criteria.monotone_tolerance})")

    if rep.tail_improvement > criteria.plateau_tolerance:
        rep.stable = False
        rep.reasons.append(
            f"still improving at budget end (Δ {rep.tail_improvement:.4f} > "
            f"{criteria.plateau_tolerance})")

    # optional: must have stopped on its own
    if criteria.require_early_stop and run.get("hit_cap"):
        rep.stable = False
        rep.reasons.append("hit epoch cap without early stopping")

    return rep


# ---------------------------------------------------------------------------
# B3 — per-grammar normalization
# ---------------------------------------------------------------------------
# Raw validation loss is not comparable across grammars: the four corpora have
# different true entropies, so a loss of 1.78 means something different on H
# than on P. Every normalizer below maps a grammar's losses onto a scale defined
# by THAT grammar's own observed range, so "worst grammar" is a question about
# how far a recipe falls short on each grammar rather than about which grammar
# is intrinsically harder.
#
# `ratio` is the default and the simplest defensible choice: 1.0 == matched the
# best result anyone achieved on this grammar; 1.02 == 2% worse than that. It
# needs no distributional assumption and no second statistic.

Normalizer = Callable[[float, Sequence[float]], float]


def _norm_ratio(loss: float, all_losses: Sequence[float]) -> float:
    """loss / (best loss observed for this grammar across the grid). >= 1.

    GRID-RELATIVE, not absolute. The denominator is the best loss *this grid*
    happened to reach for this grammar, so the value answers "how far off the
    best recipe tried" and NOT "how far off the grammar's true entropy". Three
    consequences worth stating, because the number reads like an absolute one:

    * it is not comparable across grids — adding or removing recipes moves the
      denominator, so a score of 1.02 from one grid and 1.02 from another are
      not the same claim;
    * the best recipe for a grammar always scores exactly 1.0, by construction,
      however far from the entropy floor it actually sits;
    * the whole grid being poor is invisible here. A grid where every recipe
      badly underfits still produces scores near 1.0.

    That is the right property for SELECTION, which is a within-grid ranking
    question, and the wrong one for reporting absolute fit. ℓ(x) is the
    quantity for absolute fit; see ``grammars/scoring.py``.
    """
    best = min(all_losses)
    if best <= 0:
        raise ValueError("non-positive best loss; ratio normalization undefined")
    return loss / best


def _norm_excess(loss: float, all_losses: Sequence[float]) -> float:
    """loss − (best for this grammar), in nats. 0 == matched the grid best."""
    return loss - min(all_losses)


def _norm_range(loss: float, all_losses: Sequence[float]) -> float:
    """Min-max position within this grammar's observed spread. 0 == best."""
    lo, hi = min(all_losses), max(all_losses)
    if hi == lo:
        return 0.0
    return (loss - lo) / (hi - lo)


def _norm_zscore(loss: float, all_losses: Sequence[float]) -> float:
    """Standardized within this grammar. Depends on the grid's composition."""
    n = len(all_losses)
    if n < 2:
        return 0.0
    mu = math.fsum(all_losses) / n
    var = math.fsum((v - mu) ** 2 for v in all_losses) / (n - 1)
    sd = math.sqrt(var)
    return 0.0 if sd == 0 else (loss - mu) / sd


NORMALIZERS: Dict[str, Normalizer] = {
    "ratio": _norm_ratio,
    "excess": _norm_excess,
    "range": _norm_range,
    "zscore": _norm_zscore,
}
DEFAULT_NORMALIZER = "ratio"


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

@dataclass
class Cell:
    """One (grammar, recipe) cell: the seed replicates and their verdicts."""

    grammar: str
    recipe: Recipe
    losses: List[float] = field(default_factory=list)
    reports: List[ConvergenceReport] = field(default_factory=list)

    @property
    def mean_loss(self) -> float:
        """Mean over SEEDS. Never over grammars — that is the deleted quantity."""
        return math.fsum(self.losses) / len(self.losses) if self.losses else float("nan")

    @property
    def stable(self) -> bool:
        return bool(self.reports) and all(r.stable for r in self.reports)

    @property
    def diagnostic(self) -> str:
        if self.stable:
            worst = max(self.reports, key=lambda r: r.max_rise
                        if math.isfinite(r.max_rise) else -math.inf)
            return worst.summary()
        bad = [r for r in self.reports if not r.stable]
        return f"{len(bad)}/{len(self.reports)} seeds: " + \
               "; ".join(sorted({x for r in bad for x in r.reasons}))


@dataclass
class RecipeVerdict:
    recipe: Recipe
    admissible: bool
    reasons: List[str]
    cells: Dict[str, Cell]
    scores: Dict[str, float]                 # per-grammar normalized score
    worst_grammar: Optional[str] = None
    worst_score: float = float("nan")

    def leximin_key(self) -> Tuple[float, ...]:
        """Scores sorted worst-first: compare on the worst, then the next.

        A pure leximin. No mean is taken anywhere, which is deliberate — an
        average over grammars is exactly what B1 removes, and the tie-break
        must not smuggle one back in.
        """
        return tuple(sorted(self.scores.values(), reverse=True))


@dataclass
class SelectionResult:
    verdicts: List[RecipeVerdict]            # admissible, ranked best-first
    inadmissible: List[RecipeVerdict]
    grammars: Tuple[str, ...]
    normalizer: str
    criteria: ConvergenceCriteria
    per_grammar_best: Dict[str, float]
    n_runs: int

    @property
    def selected(self) -> Optional[RecipeVerdict]:
        return self.verdicts[0] if self.verdicts else None


def build_cells(runs: Iterable[Dict],
                criteria: ConvergenceCriteria) -> Dict[Tuple[str, Recipe], Cell]:
    cells: Dict[Tuple[str, Recipe], Cell] = {}
    for run in runs:
        key = (str(run["grammar"]), recipe_of(run))
        cell = cells.setdefault(key, Cell(grammar=key[0], recipe=key[1]))
        loss = run.get("eval_loss")
        if loss is not None and math.isfinite(float(loss)):
            cell.losses.append(float(loss))
        cell.reports.append(assess_convergence(run, criteria))
    return cells


def select(runs: Sequence[Dict],
           criteria: ConvergenceCriteria = ConvergenceCriteria(),
           normalizer: str = DEFAULT_NORMALIZER,
           grammars: Sequence[str] = GRAMMARS) -> SelectionResult:
    """Admissibility + leximin ranking over the grid in ``runs``."""
    if normalizer not in NORMALIZERS:
        raise ValueError(f"unknown normalizer {normalizer!r}; "
                         f"choose from {sorted(NORMALIZERS)}")
    norm = NORMALIZERS[normalizer]
    grammars = tuple(grammars)

    cells = build_cells(runs, criteria)
    recipes = sorted({rc for _, rc in cells})

    # Per-grammar reference set: every mean loss observed for that grammar,
    # across the WHOLE grid (not only admissible recipes), so the reference
    # point does not move when the admissibility parameters are retuned.
    by_grammar: Dict[str, List[float]] = {g: [] for g in grammars}
    for (g, _), cell in cells.items():
        if g in by_grammar and math.isfinite(cell.mean_loss):
            by_grammar[g].append(cell.mean_loss)

    admissible: List[RecipeVerdict] = []
    rejected: List[RecipeVerdict] = []

    for rc in recipes:
        rc_cells = {g: cells[(g, rc)] for g in grammars if (g, rc) in cells}
        reasons: List[str] = []

        missing = [g for g in grammars if g not in rc_cells]
        if missing:
            reasons.append(f"no runs for {', '.join(missing)}")
        for g in grammars:
            cell = rc_cells.get(g)
            if cell is not None and not cell.stable:
                reasons.append(f"{g}: {cell.diagnostic}")

        scores: Dict[str, float] = {}
        for g, cell in rc_cells.items():
            pool = by_grammar.get(g) or []
            if pool and math.isfinite(cell.mean_loss):
                scores[g] = norm(cell.mean_loss, pool)

        verdict = RecipeVerdict(recipe=rc, admissible=not reasons,
                                reasons=reasons, cells=rc_cells, scores=scores)
        if verdict.admissible and scores:
            verdict.worst_grammar = max(scores, key=lambda g: scores[g])
            verdict.worst_score = scores[verdict.worst_grammar]
            admissible.append(verdict)
        else:
            if scores:
                verdict.worst_grammar = max(scores, key=lambda g: scores[g])
                verdict.worst_score = scores[verdict.worst_grammar]
            rejected.append(verdict)

    admissible.sort(key=lambda v: v.leximin_key())

    return SelectionResult(
        verdicts=admissible,
        inadmissible=rejected,
        grammars=grammars,
        normalizer=normalizer,
        criteria=criteria,
        per_grammar_best={g: min(v) for g, v in by_grammar.items() if v},
        n_runs=len(runs),
    )


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_runs(path: Path) -> List[Dict]:
    """Run records from ``results.jsonl`` (preferred) or ``results.csv``.

    JSONL is the source of truth because it is the only format carrying
    ``eval_curve``; a CSV load is accepted for older grids but every run in it
    will fail the convergence check with ``no_eval_curve``, which is correct —
    the evidence is genuinely absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    runs: List[Dict] = []
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    runs.append(json.loads(line))
    else:
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                for k in ("eval_loss", "train_loss", "learning_rate"):
                    if row.get(k) not in (None, ""):
                        row[k] = float(row[k])
                runs.append(row)
    return runs


# ---------------------------------------------------------------------------
# Step 5.1 — the probe objective: what actually selects the recipe
# ---------------------------------------------------------------------------
# `RECIPE_SELECTION.md` Step 5.1: "swap its objective column from
# `eval_loss` to `S` ... only the input changes." Everything above this line
# ranks recipes on validation loss, and §1.3/§1.4 of the plan disqualify that:
# the transformation-critical positions are ~1.26% of the loss, and the four
# corpora have different true entropies. The per-grammar normalization fixes the
# second problem, not the first. On the grid96 sweep the two objectives are
# close to inverted — the loss-ranked winner is 9th of 12 on the probe.
#
# So the loss side is kept, but demoted: it supplies diagnostics and (optionally)
# a convergence gate. The RANKING is `S[r]`, defined in Step 3 of the plan:
#
#     margin[r,g,t] = mean over seeds of margin_corrected, at the chosen epoch
#     z[r,g,t]      = margin[r,g,t] standardized across the compared RECIPES,
#                     within (grammar, term)
#     S[r]          = mean over g of ( mean over the objective terms of z[r,g,t] )
#     W[r]          = min over (g,t) of z[r,g,t]                      # worst cell
#
# Note the unit: `z` is standardized across the twelve RECIPES (seeds averaged
# first), not across the ninety-six runs. `evaluation.aggregate --select`
# computes the per-RUN variant, which is a different statistic — it is
# per-grammar, so it cannot express the "pooled symmetrically over grammars"
# requirement that §1.7 constraint 3 imposes, and selecting on it would pick a
# per-grammar recipe.

# Step 3's two gates. Neither is a quality bar; both are floors.
GATE_ACC_MIN = 0.90        # accuracy on `neutral_agreement`, worst grammar
B_TOK = 2.26               # nats/token: the order-1 token baseline to beat

EPOCH_VIEWS = ("final", "fixed")


def _run_index(runs: Sequence[Dict]) -> Dict[str, Dict]:
    """`run_id` -> the fields the probe objective needs from the run record."""
    index: Dict[str, Dict] = {}
    for run in runs:
        index[str(run["run_id"])] = {
            "recipe": recipe_of(run),
            "grammar": str(run["grammar"]),
            "seed": run.get("seed"),
            "n_params": run.get("n_params"),
            "batch_size": run.get("batch_size"),
            "eval_loss": run.get("eval_loss"),
        }
    return index


def load_terms(path: Path, runs: Sequence[Dict]):
    """The per-run-epoch-term table from `evaluation.aggregate --out`.

    Joined to the run records on `run_id` rather than parsed out of the id
    string, so a change to the id format cannot silently mis-assign a recipe.
    """
    import pandas as pd

    frame = pd.read_csv(path)
    index = _run_index(runs)
    missing = sorted(set(frame.run_id) - set(index))
    if missing:
        raise ValueError(
            f"{len(missing)} run(s) in the terms table are absent from the run "
            f"records, so their recipe is unknown: {missing[:4]}. The two files "
            f"must describe the same sweep.")
    meta = frame.run_id.map(index)
    frame["recipe"] = [m["recipe"] for m in meta]
    frame["seed"] = [m["seed"] for m in meta]
    return frame


def _at_view(frame, view: str):
    """`frame` restricted to one epoch per run.

    ``final`` — each run's own last probe epoch, which is the epoch patience
    chose on the STOP half. Reading it on the SCORE half is what the two-halves
    split is for, so this is the protocol as designed and is the default.

    ``fixed`` — the last epoch at which every run is still alive, so no recipe
    is credited for having trained longer. Reported as a robustness check; on
    grid96 the two views agree on the winner and on the top four.
    """
    if view == "final":
        return frame[frame.epoch == frame.groupby("run_id").epoch.transform("max")]
    if view == "fixed":
        alive = int(frame.groupby("run_id").epoch.max().min())
        return frame[frame.epoch == alive]
    raise ValueError(f"unknown epoch view {view!r}; choose from {EPOCH_VIEWS}")


def probe_objective(terms, view: str = "final", seeds: Optional[Sequence] = None):
    """`S[r]`, `W[r]` and the per-grammar components, one row per recipe.

    `seeds` restricts the seed replicates, which is only used to estimate how
    much of `S` is seed noise (see `seed_noise`).

    The standardization runs over EVERY recipe with data, not only the
    admissible ones — the same reasoning as the loss side's per-grammar
    reference set: the ranking must not shift when the gates are retuned.
    """
    import pandas as pd
    from evaluation.aggregate import GATE_TERMS, MIN_SPREAD

    at = _at_view(terms, view)
    objective = at[~at.term.isin(GATE_TERMS)]
    if seeds is not None:
        objective = objective[objective.seed.isin(seeds)]

    cell = (objective.groupby(["recipe", "grammar", "term"])["margin_corrected"]
            .mean().reset_index())
    group = cell.groupby(["grammar", "term"])["margin_corrected"]
    spread = group.transform(lambda s: s.max() - s.min())
    sd = group.transform(lambda s: s.std(ddof=0))
    cell["z"] = 0.0
    # A term that does not separate the recipes carries no ranking information;
    # z-scoring it would amplify pure noise. Same guard as `aggregate.py`.
    informative = spread >= MIN_SPREAD
    cell.loc[informative, "z"] = (
        (cell.loc[informative, "margin_corrected"] - group.transform("mean")[informative])
        / sd[informative].replace(0.0, float("nan")))
    cell["z"] = cell["z"].fillna(0.0)

    per_grammar = cell.groupby(["recipe", "grammar"])["z"].mean()
    worst = cell.loc[cell.groupby("recipe")["z"].idxmin()].set_index("recipe")
    table = pd.DataFrame({
        "S": per_grammar.groupby("recipe").mean(),
        "W": cell.groupby("recipe")["z"].min(),
        "worst_cell": worst.grammar.astype(str) + "/" + worst.term.astype(str),
    })
    for grammar in sorted(cell.grammar.unique()):
        table["z_" + grammar] = per_grammar.xs(grammar, level="grammar")
    return table.sort_values("S", ascending=False)


def seed_noise(terms, view: str = "final") -> float:
    """How much `S` moves when the objective is rebuilt from one seed at a time.

    Crude by construction — with two seeds it is a range over n=2, per recipe —
    and it is used only to decide whether the winner is separated from the
    runner-up at all (Step 5.4), never as a standard error. It errs large,
    which is the safe direction for that question.
    """
    seeds = sorted(terms.seed.dropna().unique())
    if len(seeds) < 2:
        return 0.0
    per_seed = [probe_objective(terms, view, seeds=[s])["S"] for s in seeds]
    import pandas as pd
    spread = pd.concat(per_seed, axis=1)
    return float((spread.max(axis=1) - spread.min(axis=1)).median())


def step3_gates(terms, runs: Sequence[Dict], view: str = "final"):
    """Step 3's two gates, per recipe. Every grammar must clear both.

    Gates, not terms: a recipe that has not learned local agreement is not a
    candidate, and being excellent at it earns nothing. Likewise `b_tok` — a
    model whose validation loss does not beat the order-1 token baseline has not
    learned the language, whatever its margins say.
    """
    import pandas as pd

    at = _at_view(terms, view)
    agreement = at[at.term == "neutral_agreement"]
    gate_acc = (agreement.groupby(["recipe", "grammar"])["acc"].mean()
                .groupby("recipe").min())
    index = _run_index(runs)
    loss = pd.Series({rid: m["eval_loss"] for rid, m in index.items()})
    recipe = pd.Series({rid: m["recipe"] for rid, m in index.items()})
    worst_loss = loss.groupby(recipe).max()
    return pd.DataFrame({
        "gate_acc": gate_acc,
        "worst_eval_loss": worst_loss,
        "pass_acc": gate_acc >= GATE_ACC_MIN,
        "pass_btok": worst_loss < B_TOK,
    })


@dataclass
class ProbeSelection:
    """The outcome of a probe-driven selection, with its reasoning exposed."""

    table: "object"                 # ranked DataFrame, admissible flag included
    view: str
    noise: float
    tied: List[Recipe]              # recipes within `noise` of the best S
    selected: Optional[Recipe]
    secondary_applied: bool
    other_views: Dict[str, "object"]


def _secondary_key(recipe: Recipe, runs_index: Dict[str, Dict]):
    """Step 5.4's tie-break: smallest model, then largest batch.

    Explicitly a throughput convention, not a quality claim — it applies only
    once `S` has failed to separate the candidates, and the report says so.
    """
    meta = next(m for m in runs_index.values() if m["recipe"] == recipe)
    return (meta["n_params"] or 0, -(meta["batch_size"] or 0))


def _select_one_view(terms, runs: Sequence[Dict], view: str,
                     criteria: Optional[ConvergenceCriteria]):
    """`(table, noise, tied, selected, secondary_applied)` for one epoch view."""
    table = probe_objective(terms, view).join(step3_gates(terms, runs, view))
    table["admissible"] = table.pass_acc & table.pass_btok
    if criteria is not None:
        cells = build_cells(runs, criteria)
        stable: Dict[Recipe, bool] = {}
        for (_, rc), cell in cells.items():
            stable[rc] = stable.get(rc, True) and cell.stable
        table["pass_convergence"] = [bool(stable.get(rc, False)) for rc in table.index]
        table["admissible"] &= table.pass_convergence

    noise = seed_noise(terms, view)
    admissible = table[table.admissible]
    selected: Optional[Recipe] = None
    tied: List[Recipe] = []
    secondary = False
    if len(admissible):
        tied = list(admissible[admissible.S >= admissible.S.max() - noise].index)
        if len(tied) > 1:
            runs_index = _run_index(runs)
            selected = min(tied, key=lambda rc: _secondary_key(rc, runs_index))
            secondary = selected != admissible.S.idxmax()
        else:
            selected = admissible.S.idxmax()
    return table, noise, tied, selected, secondary


def select_on_probe(terms, runs: Sequence[Dict], *, view: str = "final",
                    criteria: Optional[ConvergenceCriteria] = None,
                    ) -> ProbeSelection:
    """Admissibility by the Step-3 gates, ranking by `S[r]`, Step 5.4 tie-break.

    `criteria` is optional and off by default: on a sweep that stops on the
    probe, the validation curve is still rising at the stopping epoch by design,
    so a curve-shape test rejects nearly everything for a reason that is not
    about the recipe. Pass one to apply it anyway.

    The other epoch view gets the SAME procedure, not just its raw argmax —
    comparing a tie-broken selection against a bare argmax would report a
    disagreement whenever the tie-break bit, which is most of the time.
    """
    table, noise, tied, selected, secondary = _select_one_view(
        terms, runs, view, criteria)
    others = {v: _select_one_view(terms, runs, v, criteria)
              for v in EPOCH_VIEWS if v != view}
    return ProbeSelection(table=table, view=view, noise=noise, tied=tied,
                          selected=selected, secondary_applied=secondary,
                          other_views=others)


def selection_record(res: ProbeSelection, runs: Sequence[Dict], *,
                     results_path: Path, terms_path: Path,
                     convergence_gate: bool) -> Dict:
    """Everything needed to say later what was selected, and on what evidence.

    Written because the selection is otherwise only ever printed. Step 5.3 of
    the plan asks for the objective, the sweep and the inputs to be recorded
    beside the frozen recipe; a report on a terminal is not that.
    """
    from models.dataload import sha256_file

    results_path, terms_path = Path(results_path), Path(terms_path)
    row = next(r for rc, r in res.table.iterrows() if rc == res.selected)

    manifest = results_path.parent / "run_manifest.jsonl"
    sweep: Dict = {}
    if manifest.exists():
        lines = [json.loads(l) for l in manifest.read_text().splitlines() if l.strip()]
        if lines:
            # Only the keys the manifest actually carries — a null field here
            # reads as "recorded, and empty" rather than "never recorded".
            wanted = ("git", "hostname", "corpus_set", "corpora_root",
                      "probe_split", "patience", "epoch_cap", "seeds",
                      "started_utc")
            sweep = {k: lines[0][k] for k in wanted if k in lines[0]}
            if len(lines) > 1:
                sweep["n_invocations"] = len(lines)

    return {
        "selected": dict(zip(RECIPE_FIELDS, res.selected)),
        "objective": (
            "S[r] = mean over grammars of ( mean over objective terms of "
            "z[r,g,t] ), where z standardizes margin_corrected across the "
            "compared RECIPES within (grammar, term), seeds averaged first. "
            "RECIPE_SELECTION.md Step 3."),
        "epoch_view": res.view,
        "S": float(row["S"]),
        "W": float(row["W"]),
        "worst_cell": str(row["worst_cell"]),
        "per_grammar_z": {c[2:]: float(row[c]) for c in res.table.columns
                          if c.startswith("z_")},
        "seed_noise": res.noise,
        "tied_within_seed_noise": [format_recipe(r) for r in res.tied],
        "secondary_criterion_applied": res.secondary_applied,
        "secondary_criterion": "smallest model, then largest batch (Step 5.4)",
        "gates": {"neutral_agreement_acc_min": GATE_ACC_MIN, "b_tok": B_TOK},
        "convergence_gate_applied": convergence_gate,
        "inputs": {
            "results": str(results_path),
            "results_sha256": sha256_file(results_path),
            "terms": str(terms_path),
            "terms_sha256": sha256_file(terms_path),
            "n_runs": len(runs),
            "n_recipes": int(len(res.table)),
        },
        "sweep": sweep,
        "written_utc": datetime.now(timezone.utc).isoformat(),
    }


def _named(table):
    """`table` with its Recipe-tuple index rendered as `size/lrX/bsY`."""
    out = table.copy()
    out.index = [format_recipe(rc) for rc in out.index]
    out.index.name = "recipe"
    return out


def format_probe_report(res: ProbeSelection, runs: Sequence[Dict]) -> str:
    """The probe selection, its gates, and why the winner is the winner."""
    out: List[str] = []
    w = out.append
    bar = "=" * 78
    w(bar); w("RECIPE SELECTION — probe objective S[r]"); w(bar)
    at = ("each run's own stopping epoch" if res.view == "final"
          else "the last epoch at which every run is alive")
    w(f"epoch view: {res.view}   ({at})")
    w(f"gates: neutral_agreement acc >= {GATE_ACC_MIN} (worst grammar); "
      f"worst eval_loss < b_tok {B_TOK}")
    w(f"seed noise on S (median range across seed-worlds): {res.noise:.3f}")
    w("")
    w("-" * 78); w("RANKING"); w("-" * 78)
    cols = ["S", "W", "worst_cell", "gate_acc", "worst_eval_loss", "admissible"]
    w(_named(res.table)[cols].round(3).to_string())
    w("")
    if res.selected is None:
        w("No admissible recipe.")
        return "\n".join(out)

    if len(res.tied) > 1:
        names = ", ".join(format_recipe(r) for r in res.tied)
        w(f"Within seed noise of the best S: {names}")
        if res.secondary_applied:
            w("Step 5.4 applies — the winner does not lead by more than seed "
              "noise, so the choice falls to the secondary criterion (smallest "
              "model, then largest batch, for throughput on the production run).")
            w("This is a convention, NOT a claim that the selected recipe is the "
              "best of the tied set. It is not distinguishable from the others.")
    else:
        w(f"S leads the runner-up by more than the seed noise of {res.noise:.3f}.")
    w("")
    w(bar); w(f"SELECTED: {format_recipe(res.selected)}"); w(bar)

    for view, (table, noise, tied, sel, _) in res.other_views.items():
        agree = ("agrees" if sel == res.selected else
                 f"DIFFERS — selects {format_recipe(sel) if sel else 'nothing'}")
        w(f"\nrobustness, epoch view '{view}' (same procedure): {agree}")
        w(f"    seed noise {noise:.3f}; tied set "
          f"{', '.join(format_recipe(r) for r in tied) or 'none'}")
        w(_named(table)[["S", "W", "admissible"]].round(3).to_string())
    return "\n".join(out)


# ---------------------------------------------------------------------------
# B4 — reporting
# ---------------------------------------------------------------------------

def format_report(res: SelectionResult) -> str:
    """Per-grammar admissibility, diagnostic and normalized score; then the
    selection and why. Contains no quantity averaged across grammars."""
    out: List[str] = []
    c = res.criteria
    out.append("=" * 78)
    out.append("RECIPE SELECTION")
    out.append("=" * 78)
    out.append(f"runs: {res.n_runs}   grammars: {', '.join(res.grammars)}")
    out.append(
        f"convergence: final_k={c.final_k}  monotone_tol={c.monotone_tolerance} "
        f"plateau_tol={c.plateau_tolerance}  divergence_tol={c.divergence_tolerance} "
        f"max_eval_loss={c.max_eval_loss}  require_curve={c.require_curve} "
        f"require_early_stop={c.require_early_stop}")
    out.append(f"normalization: {res.normalizer}  "
               "(per-grammar; losses are NOT comparable across grammars)")
    out.append("per-grammar best observed mean loss: " + "  ".join(
        f"{g}={v:.4f}" for g, v in sorted(res.per_grammar_best.items())))

    # Missing-curve tally: distinct from a genuine convergence failure.
    no_curve = sum(
        1 for v in (res.verdicts + res.inadmissible)
        for cell in v.cells.values()
        for r in cell.reports if "no_eval_curve" in r.reasons)
    if no_curve:
        out.append("")
        out.append(f"NOTE: {no_curve} run(s) carry no eval_curve, so their "
                   "convergence is unknown and they are inadmissible on that "
                   "ground alone. Re-run the grid with the current "
                   "models/grid_search.py, which records eval_curve.")

    out.append("")
    out.append("-" * 78)
    out.append("PER-RECIPE, PER-GRAMMAR")
    out.append("-" * 78)
    for v in res.verdicts + res.inadmissible:
        tag = "ADMISSIBLE" if v.admissible else "INADMISSIBLE"
        out.append("")
        out.append(f"{format_recipe(v.recipe):28s} {tag}")
        for g in res.grammars:
            cell = v.cells.get(g)
            if cell is None:
                out.append(f"    {g:8s} —  no runs")
                continue
            score = v.scores.get(g)
            score_s = f"{score:8.4f}" if score is not None else "     n/a"
            ok = "ok  " if cell.stable else "FAIL"
            out.append(f"    {g:8s} {ok}  loss={cell.mean_loss:.4f} "
                       f"({len(cell.losses)} seeds)  score={score_s}  "
                       f"{cell.diagnostic}")

    out.append("")
    out.append("-" * 78)
    out.append("RANKING (admissible only, by worst grammar; leximin)")
    out.append("-" * 78)
    if not res.verdicts:
        out.append("No admissible recipe. Reasons are listed per recipe above.")
    else:
        header = f"{'recipe':28s} {'worst':>9s} {'on':>8s}   per-grammar scores"
        out.append(header)
        for v in res.verdicts:
            cells = "  ".join(f"{g}={v.scores[g]:.4f}"
                              for g in res.grammars if g in v.scores)
            out.append(f"{format_recipe(v.recipe):28s} {v.worst_score:9.4f} "
                       f"{v.worst_grammar or '—':>8s}   {cells}")

        best = res.verdicts[0]
        out.append("")
        out.append("=" * 78)
        out.append(f"SELECTED: {format_recipe(best.recipe)}")
        out.append("=" * 78)
        out.append(
            f"Why: all {len(res.grammars)} grammars converged stably, and its "
            f"worst grammar ({best.worst_grammar}) scores "
            f"{best.worst_score:.4f} under `{res.normalizer}` — the best "
            "worst-case of any admissible recipe.")
        if len(res.verdicts) > 1:
            nxt = res.verdicts[1]
            bk, nk = best.leximin_key(), nxt.leximin_key()
            if math.isclose(nxt.worst_score, best.worst_score, rel_tol=1e-12):
                # Find the coordinate leximin actually separated them on; with a
                # single grammar there is none, and the tie is genuine.
                j = next((i for i in range(min(len(bk), len(nk)))
                          if not math.isclose(bk[i], nk[i], rel_tol=1e-12)), None)
                if j is None:
                    out.append(
                        f"Exact tie with {format_recipe(nxt.recipe)} on every "
                        "grammar; the order between them is arbitrary.")
                else:
                    out.append(
                        f"Tie on the worst grammar with "
                        f"{format_recipe(nxt.recipe)}; broken at rank {j + 1} "
                        f"of the worst-first score vector "
                        f"({bk[j]:.4f} vs {nk[j]:.4f}).")
            else:
                out.append(
                    f"Runner-up {format_recipe(nxt.recipe)} at "
                    f"{nxt.worst_score:.4f} on {nxt.worst_grammar}.")
    return "\n".join(out)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path,
                   help="results.jsonl (preferred) or results.csv from the grid")
    p.add_argument("--normalize", choices=sorted(NORMALIZERS),
                   default=DEFAULT_NORMALIZER,
                   help="per-grammar normalization used to define 'worst' (B3)")
    p.add_argument("--grammars", nargs="+", default=list(GRAMMARS))
    p.add_argument("--final-k", type=int, default=ConvergenceCriteria.final_k)
    p.add_argument("--monotone-tol", type=float,
                   default=ConvergenceCriteria.monotone_tolerance)
    p.add_argument("--plateau-tol", type=float,
                   default=ConvergenceCriteria.plateau_tolerance)
    p.add_argument("--divergence-tol", type=float,
                   default=ConvergenceCriteria.divergence_tolerance)
    p.add_argument("--max-eval-loss", type=float,
                   default=ConvergenceCriteria.max_eval_loss)
    # There is deliberately no --allow-missing-curve. A run with no eval_curve
    # has no recorded convergence, so admitting it means asserting stability
    # that was never measured — which is the one thing this criterion exists to
    # prevent. The escape hatch existed only to make the tool produce a
    # selection on the two grids that predate the field, and a selection
    # produced that way is exactly the unverified answer it looks like. Re-run
    # the grid so the curves exist.
    p.add_argument("--require-early-stop", action="store_true",
                   help="additionally require the run stopped on patience "
                        "rather than hitting the epoch cap")
    p.add_argument("--terms", type=Path, default=None,
                   help="per-run-epoch-term table from `evaluation.aggregate "
                        "--out`. Required: it carries S[r], the objective the "
                        "recipe is selected on (RECIPE_SELECTION Step 5.1)")
    p.add_argument("--epoch-view", choices=EPOCH_VIEWS, default="final",
                   help="which epoch S[r] is read at (default: each run's own "
                        "stopping epoch; the other view is reported beside it)")
    p.add_argument("--convergence-gate", action="store_true",
                   help="additionally gate on the validation-curve shape (B2). "
                        "Off by default: this sweep stops on the probe, so the "
                        "loss curve is still rising at the stopping epoch by "
                        "design and B2 rejects nearly every run for a reason "
                        "that is not about the recipe")
    p.add_argument("--record", type=Path, default=None,
                   help="write the selection and its evidence to this JSON "
                        "(Step 5.3). Without it the selection exists only as "
                        "terminal output")
    p.add_argument("--rank-by-loss", action="store_true",
                   help="the superseded loss-side ranking (B2/B3). Kept for "
                        "comparison only — see the Step 5.1 note in this module")
    args = p.parse_args(argv)

    criteria = ConvergenceCriteria(
        final_k=args.final_k,
        monotone_tolerance=args.monotone_tol,
        plateau_tolerance=args.plateau_tol,
        divergence_tolerance=args.divergence_tol,
        max_eval_loss=args.max_eval_loss,
        require_curve=True,
        require_early_stop=args.require_early_stop,
    )
    runs = load_runs(args.results)

    if args.rank_by_loss:
        res = select(runs, criteria, args.normalize, tuple(args.grammars))
        print(format_report(res))
        return 0 if res.selected is not None else 1

    if args.terms is None:
        p.error(
            "--terms is required. The recipe is selected on S[r], the "
            "minimal-pair objective (RECIPE_SELECTION.md Step 3/5.1), not "
            "on validation loss; build it with `python -m evaluation.aggregate "
            "'<sweep>/probe_scores/*.parquet' --half score --baselines <...> "
            "--out <...>/terms.csv`. Pass --rank-by-loss for the superseded "
            "loss ranking.")

    terms = load_terms(args.terms, runs)
    probe = select_on_probe(terms, runs, view=args.epoch_view,
                            criteria=criteria if args.convergence_gate else None)
    print(format_probe_report(probe, runs))

    if args.record is not None and probe.selected is not None:
        record = selection_record(probe, runs, results_path=args.results,
                                  terms_path=args.terms,
                                  convergence_gate=args.convergence_gate)
        args.record.parent.mkdir(parents=True, exist_ok=True)
        args.record.write_text(json.dumps(record, indent=2) + "\n",
                               encoding="utf-8")
        print(f"\nrecord -> {args.record}")
    return 0 if probe.selected is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
