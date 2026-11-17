"""Recipe selection (Task B): admissibility, normalization, leximin ranking.

Everything here runs on SYNTHETIC grid records — dicts shaped like the ones
``models/grid_search.py`` writes to ``results.jsonl``. That is deliberate: the
two grids on disk predate the ``eval_curve`` field, so testing against them
would only ever exercise the missing-curve path.
"""
from __future__ import annotations

import math

import pytest

from analysis.recipe_selection import (
    ConvergenceCriteria,
    NORMALIZERS,
    RECIPE_FIELDS,
    assess_convergence,
    build_cells,
    format_report,
    recipe_of,
    select,
)

GRAMMARS = ("H", "P", "Hprime", "Lprime")


def _curve(values):
    return [{"epoch": float(i + 1), "eval_loss": v} for i, v in enumerate(values)]


def _run(grammar, size="medium", lr=3e-4, bs=32, seed=100,
         curve=(2.0, 1.9, 1.845, 1.840, 1.838), eval_loss=None,
         train_loss=1.7, **extra):
    """One synthetic run record. Default curve is a clean convergence."""
    losses = list(curve) if curve is not None else []
    rec = {
        "run_id": f"{grammar}__{size}__lr{lr}__bs{bs}__s{seed}",
        "grammar": grammar,
        "size_name": size,
        "learning_rate": lr,
        "batch_size": bs,
        "seed": seed,
        "train_loss": train_loss,
        "eval_loss": losses[-1] if eval_loss is None and losses else eval_loss,
        "eval_curve": _curve(losses) if curve is not None else None,
    }
    rec.update(extra)
    return rec


def _grid(per_grammar_final, **kw):
    """A full 4-grammar grid for ONE recipe, at the given final losses."""
    runs = []
    for g, final in per_grammar_final.items():
        runs.append(_run(g, curve=(final + 0.2, final + 0.05, final + 0.004,
                                   final + 0.001, final), **kw))
    return runs


# ---------------------------------------------------------------------------
# Convergence criterion (B2)
# ---------------------------------------------------------------------------

def test_clean_curve_is_stable():
    rep = assess_convergence(_run("H"))
    assert rep.stable, rep.reasons
    assert rep.n_evals == 5


def test_nan_loss_is_unstable():
    rep = assess_convergence(_run("H", curve=(2.0, 1.9, float("nan"))))
    assert not rep.stable
    assert "nan_in_curve" in rep.reasons


def test_nan_final_eval_loss_is_unstable():
    rep = assess_convergence(_run("H", curve=None, eval_loss=float("nan")))
    assert not rep.stable
    assert any("eval_loss_not_finite" in r for r in rep.reasons)


def test_divergence_after_best_is_unstable():
    """Finite and bounded throughout, but it climbed away from its best."""
    rep = assess_convergence(_run("H", curve=(2.0, 1.8, 1.79, 2.4, 3.0)))
    assert not rep.stable
    assert any("diverged" in r for r in rep.reasons)


def test_non_monotone_tail_is_unstable():
    rep = assess_convergence(_run("H", curve=(2.0, 1.9, 1.80, 1.95, 1.81)))
    assert not rep.stable
    assert any("not monotone" in r for r in rep.reasons)


def test_tiny_wobble_within_tolerance_is_stable():
    """Evaluation noise must not read as instability."""
    rep = assess_convergence(
        _run("H", curve=(2.0, 1.9, 1.8000, 1.8002, 1.7999)),
        ConvergenceCriteria(monotone_tolerance=1e-3))
    assert rep.stable, rep.reasons


def test_still_descending_at_budget_end_is_unstable():
    """Truncated, not converged: the tail is still dropping fast."""
    rep = assess_convergence(_run("H", curve=(4.0, 3.5, 3.0, 2.5, 2.0)))
    assert not rep.stable
    assert any("still improving" in r for r in rep.reasons)


def test_unbounded_loss_is_unstable():
    rep = assess_convergence(
        _run("H", curve=(30.0, 25.0, 20.5, 20.4, 20.4)),
        ConvergenceCriteria(max_eval_loss=10.0))
    assert not rep.stable
    assert any("max" in r for r in rep.reasons)


def test_missing_curve_is_unstable_by_default():
    rep = assess_convergence(_run("H", curve=None, eval_loss=1.78))
    assert not rep.stable
    assert "no_eval_curve" in rep.reasons


def test_missing_curve_can_be_allowed_explicitly():
    rep = assess_convergence(_run("H", curve=None, eval_loss=1.78),
                             ConvergenceCriteria(require_curve=False))
    assert rep.stable, rep.reasons


def test_too_few_evaluations_is_unstable():
    rep = assess_convergence(_run("H", curve=(1.9, 1.85)),
                             ConvergenceCriteria(final_k=3))
    assert not rep.stable
    assert any("need 3" in r for r in rep.reasons)


def test_criteria_parameters_are_configurable_not_hardcoded():
    """The same curve flips verdict when the tolerance moves — i.e. the
    parameters are genuinely wired through, not decorative."""
    run = _run("H", curve=(2.0, 1.9, 1.80, 1.83, 1.80))
    assert not assess_convergence(run, ConvergenceCriteria(monotone_tolerance=1e-3)).stable
    assert assess_convergence(run, ConvergenceCriteria(monotone_tolerance=0.05)).stable


def test_require_early_stop_flag():
    run = _run("H", hit_cap=True)
    assert assess_convergence(run).stable
    assert not assess_convergence(
        run, ConvergenceCriteria(require_early_stop=True)).stable


# ---------------------------------------------------------------------------
# Admissibility (B5: one bad grammar sinks the recipe)
# ---------------------------------------------------------------------------

def test_recipe_failing_one_grammar_is_inadmissible_even_if_best_elsewhere():
    """The headline requirement: a recipe that is the best available on H, P
    and H' but diverges on L' must not be selected."""
    good = _grid({"H": 1.70, "P": 1.70, "Hprime": 1.70}, size="large")
    # L' diverges under the same recipe.
    good.append(_run("Lprime", size="large",
                     curve=(2.0, 1.75, 1.74, 2.6, 3.4)))
    # A mediocre-but-everywhere-stable alternative.
    rival = _grid({g: 1.90 for g in GRAMMARS}, size="medium")

    res = select(good + rival, grammars=GRAMMARS)

    assert res.selected is not None
    assert res.selected.recipe[0] == "medium"
    large = next(v for v in res.inadmissible if v.recipe[0] == "large")
    assert not large.admissible
    assert any("Lprime" in r for r in large.reasons)
    # ...and it really was better on the other three.
    assert large.scores["H"] < res.selected.scores["H"]


def test_missing_grammar_is_inadmissible():
    runs = _grid({"H": 1.8, "P": 1.8, "Hprime": 1.8}, size="small")
    res = select(runs, grammars=GRAMMARS)
    assert res.selected is None
    assert any("Lprime" in r for r in res.inadmissible[0].reasons)


def test_one_bad_seed_sinks_the_cell():
    runs = _grid({g: 1.8 for g in GRAMMARS}, size="medium")
    runs.append(_run("P", size="medium", seed=101,
                     curve=(2.0, 1.9, 1.88, 2.9, 3.5)))
    res = select(runs, grammars=GRAMMARS)
    assert res.selected is None
    assert not res.inadmissible[0].cells["P"].stable


# ---------------------------------------------------------------------------
# Ranking and tie-breaking (B3, B5)
# ---------------------------------------------------------------------------

def test_ranks_by_worst_grammar_not_by_average():
    """`spiky` wins on three grammars by a wide margin and loses badly on one;
    `even` is uniformly mediocre. A pooled mean would pick `spiky`; ranking on
    the worst grammar must pick `even`."""
    spiky = _grid({"H": 1.50, "P": 1.50, "Hprime": 1.50, "Lprime": 2.40},
                  size="large")
    even = _grid({g: 1.80 for g in GRAMMARS}, size="medium")

    res = select(spiky + even, grammars=GRAMMARS)

    assert [v.recipe[0] for v in res.verdicts] == ["medium", "large"]
    # Sanity: the pooled mean really does favour the loser, so the test is
    # discriminating and not passing by accident.
    spiky_mean = (1.50 * 3 + 2.40) / 4
    even_mean = 1.80
    assert spiky_mean < even_mean


def test_tie_on_worst_resolves_to_better_worst_grammar_score():
    """Two recipes tie on their worst grammar; the one with the better
    next-worst grammar wins.

    Each recipe sets the floor on the grammars where it leads, so both come out
    at ratio 1.10 on their own weakest grammar — an exact tie on the quantity
    the ranking is defined by. `alpha` then matches the floor on all three
    remaining grammars while `beta` gives up 1.056 on one, so leximin separates
    them on the second coordinate.
    """
    a = _grid({"H": 1.80, "P": 1.98, "Hprime": 1.80, "Lprime": 1.80},
              size="alpha")
    b = _grid({"H": 1.98, "P": 1.80, "Hprime": 1.90, "Lprime": 1.80},
              size="beta")

    res = select(a + b, grammars=GRAMMARS)

    assert [v.recipe[0] for v in res.verdicts] == ["alpha", "beta"]
    top, second = res.verdicts
    assert math.isclose(top.worst_score, second.worst_score, rel_tol=1e-12)
    assert top.leximin_key()[1] < second.leximin_key()[1]


def test_worst_is_defined_on_normalized_not_raw_loss():
    """P's corpus is intrinsically higher-entropy here: every recipe scores
    ~1.0 nat worse on it. A rule that took the max RAW loss would call P the
    worst grammar for every recipe and rank on P alone; normalizing per grammar
    must not do that."""
    a = _grid({"H": 1.80, "P": 2.80, "Hprime": 1.80, "Lprime": 1.80}, size="alpha")
    b = _grid({"H": 1.95, "P": 2.81, "Hprime": 1.81, "Lprime": 1.81}, size="beta")

    res = select(a + b, grammars=GRAMMARS)

    # b is barely worse on P but much worse on H, so H is what sinks it.
    beta = next(v for v in res.verdicts if v.recipe[0] == "beta")
    assert beta.worst_grammar == "H"
    assert res.selected.recipe[0] == "alpha"


@pytest.mark.parametrize("name", sorted(NORMALIZERS))
def test_every_normalizer_scores_the_grid_best_at_its_floor(name):
    """Whatever the normalization, the best recipe on a grammar must be that
    grammar's floor — otherwise 'worst' is measured against a moving target."""
    runs = (_grid({g: 1.80 for g in GRAMMARS}, size="alpha")
            + _grid({g: 1.95 for g in GRAMMARS}, size="beta"))
    res = select(runs, normalizer=name, grammars=GRAMMARS)
    alpha = next(v for v in res.verdicts if v.recipe[0] == "alpha")
    beta = next(v for v in res.verdicts if v.recipe[0] == "beta")
    for g in GRAMMARS:
        assert alpha.scores[g] <= beta.scores[g]


def test_normalizer_is_selectable_and_changes_the_scale():
    runs = (_grid({g: 1.80 for g in GRAMMARS}, size="alpha")
            + _grid({g: 1.98 for g in GRAMMARS}, size="beta"))
    ratio = select(runs, normalizer="ratio", grammars=GRAMMARS)
    excess = select(runs, normalizer="excess", grammars=GRAMMARS)
    assert math.isclose(ratio.verdicts[0].worst_score, 1.0, rel_tol=1e-12)
    assert math.isclose(excess.verdicts[0].worst_score, 0.0, abs_tol=1e-12)
    assert ratio.verdicts[0].recipe == excess.verdicts[0].recipe


def test_unknown_normalizer_raises():
    with pytest.raises(ValueError, match="unknown normalizer"):
        select(_grid({g: 1.8 for g in GRAMMARS}), normalizer="pooled_mean")


# ---------------------------------------------------------------------------
# Reporting (B4)
# ---------------------------------------------------------------------------

def test_report_shows_admissibility_diagnostic_and_score_per_grammar():
    runs = (_grid({g: 1.80 for g in GRAMMARS}, size="alpha")
            + _grid({g: 1.95 for g in GRAMMARS}, size="beta"))
    text = format_report(select(runs, grammars=GRAMMARS))
    for g in GRAMMARS:
        assert g in text
    assert "ADMISSIBLE" in text
    assert "SELECTED:" in text
    assert "score=" in text
    assert "Why:" in text


def test_report_never_prints_a_cross_grammar_mean():
    """B4: no figure that averages loss across grammars. The pooled mean of the
    synthetic grid is a distinctive number; it must appear nowhere."""
    runs = _grid({"H": 1.50, "P": 1.60, "Hprime": 1.70, "Lprime": 1.80},
                 size="alpha")
    runs += _grid({g: 1.90 for g in GRAMMARS}, size="beta")
    text = format_report(select(runs, grammars=GRAMMARS))
    pooled = (1.50 + 1.60 + 1.70 + 1.80) / 4          # 1.65
    assert f"{pooled:.4f}" not in text
    assert f"{pooled:.3f}" not in text
    assert "mean_loss" not in text
    assert "mean loss across" not in text


def test_report_flags_missing_curves_separately_from_failures():
    runs = [_run(g, curve=None, eval_loss=1.8) for g in GRAMMARS]
    res = select(runs, grammars=GRAMMARS)
    text = format_report(res)
    assert res.selected is None
    assert "no eval_curve" in text
    assert "no_eval_curve" in text


def test_no_admissible_recipe_reports_cleanly():
    runs = [_run(g, curve=(2.0, 1.9, 1.8, 2.9, 3.5)) for g in GRAMMARS]
    res = select(runs, grammars=GRAMMARS)
    assert res.selected is None
    assert "No admissible recipe." in format_report(res)


# ---------------------------------------------------------------------------
# Plumbing
# ---------------------------------------------------------------------------

def test_recipe_key_is_type_normalized():
    """A CSV row (strings) and a JSONL row (numbers) must key identically."""
    a = recipe_of({"size_name": "medium", "learning_rate": "0.0003",
                   "batch_size": "32"})
    b = recipe_of({"size_name": "medium", "learning_rate": 3e-4,
                   "batch_size": 32})
    assert a == b
    assert RECIPE_FIELDS == ("size_name", "learning_rate", "batch_size")


def test_seeds_are_averaged_within_a_cell_only():
    runs = [_run("H", seed=100, curve=(2.0, 1.9, 1.82, 1.81, 1.80)),
            _run("H", seed=101, curve=(2.0, 1.9, 1.92, 1.91, 1.90))]
    cells = build_cells(runs, ConvergenceCriteria())
    cell = cells[("H", recipe_of(runs[0]))]
    assert len(cell.losses) == 2
    assert math.isclose(cell.mean_loss, 1.85, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Step 5.1 — the probe objective S[r], which is what selects the recipe
# ---------------------------------------------------------------------------

def _row(table, recipe):
    """One row of a probe table by Recipe key.

    The index holds Recipe tuples, and `.loc[a_tuple]` is multi-axis indexing to
    pandas, so it has to be looked up positionally.
    """
    position = [i for i, key in enumerate(table.index) if key == recipe]
    assert position, f"{recipe} not in {list(table.index)}"
    return table.iloc[position[0]]


N_PARAMS = {"small": 1_024_256, "medium": 4_414_464, "large": 5_993_984}


def _terms(spec, *, seeds=(100, 101), n_epochs=2, gate_acc=1.0, jitter=None):
    """A terms table shaped like `evaluation.aggregate --out`.

    `spec` maps (size, lr, bs) -> {grammar: {term: margin at the LAST epoch}}.
    Earlier epochs sit 1 nat lower, so the `final` and `fixed` views differ.
    `jitter` maps recipe -> offset applied to the SECOND seed. It has to vary by
    recipe: `z` is shift-invariant, so an offset applied to every recipe alike
    moves no score and `seed_noise` would come back 0.
    """
    jitter = jitter or {}
    import pandas as pd

    rows = []
    for (size, lr, bs), per_grammar in spec.items():
        for grammar, per_term in per_grammar.items():
            for i, seed in enumerate(seeds):
                rid = f"{grammar}__{size}__lr{lr}__bs{bs}__s{seed}"
                for epoch in range(1, n_epochs + 1):
                    behind = float(n_epochs - epoch)
                    for term, margin in per_term.items():
                        rows.append(dict(
                            run_id=rid, epoch=float(epoch), grammar=grammar,
                            split="structural_probe_SELECT", term=term,
                            criterion="string", acc=0.95, is_gate=False,
                            margin_corrected=margin - behind + i * jitter.get((size, lr, bs), 0.0)))
                    rows.append(dict(
                        run_id=rid, epoch=float(epoch), grammar=grammar,
                        split="structural_probe_SELECT", term="neutral_agreement",
                        criterion="span", acc=gate_acc, is_gate=True,
                        margin_corrected=5.0))
    return pd.DataFrame(rows)


def _runs_for(spec, *, seeds=(100, 101), eval_loss=1.80):
    runs = []
    for (size, lr, bs), per_grammar in spec.items():
        for grammar in per_grammar:
            for seed in seeds:
                runs.append(_run(grammar, size=size, lr=lr, bs=bs, seed=seed,
                                 curve=(2.0, 1.9, eval_loss), n_params=N_PARAMS[size]))
    return runs


def _spec(values, terms=("aux", "wh_i"), grammars=GRAMMARS):
    """{recipe: {grammar: {term: margin}}} with one margin per recipe."""
    return {rc: {g: {t: v for t in terms} for g in grammars}
            for rc, v in values.items()}


def test_S_is_the_mean_over_grammars_not_a_pooled_average_of_runs():
    """S[r] pools symmetrically over grammars (plan Step 3, §1.7 constraint 3)."""
    from analysis.recipe_selection import load_terms, probe_objective

    spec = {("small", 3e-4, 64): {g: {"aux": m} for g, m in
                                  zip(GRAMMARS, [10.0, 2.0, 6.0, 6.0])},
            ("medium", 3e-4, 64): {g: {"aux": m} for g, m in
                                   zip(GRAMMARS, [6.0, 6.0, 6.0, 6.0])}}
    terms = load_terms_from(_terms(spec), _runs_for(spec))
    table = probe_objective(terms, "final")
    # Two recipes per (grammar, term) cell: z is +-1 by construction.
    small = _row(table, ("small", 3e-4, 64))
    assert math.isclose(small["z_H"], 1.0, abs_tol=1e-9)     # 10 vs 6
    assert math.isclose(small["z_P"], -1.0, abs_tol=1e-9)    # 2 vs 6
    # Lprime/Hprime tie exactly, so the term is uninformative and contributes 0.
    assert math.isclose(small["S"], 0.0, abs_tol=1e-9)
    assert math.isclose(small["W"], -1.0, abs_tol=1e-9)      # the worst CELL, not the mean


def load_terms_from(frame, runs):
    """`load_terms` without a CSV round trip."""
    import analysis.recipe_selection as rs
    index = rs._run_index(runs)
    meta = frame.run_id.map(index)
    out = frame.copy()
    out["recipe"] = [m["recipe"] for m in meta]
    out["seed"] = [m["seed"] for m in meta]
    return out


def test_z_is_computed_across_recipes_with_seeds_averaged_first():
    """The unit of comparison is the recipe, not the run."""
    from analysis.recipe_selection import probe_objective

    spec = _spec({("small", 3e-4, 64): 4.0, ("medium", 3e-4, 64): 8.0})
    # seed 101 sits 2 nats above seed 100 on small only; the mean is the midpoint,
    # so small's recipe margin is 5.0 against medium's 8.0.
    terms = load_terms_from(_terms(spec, jitter={("small", 3e-4, 64): 2.0}),
                            _runs_for(spec))
    table = probe_objective(terms, "final")
    assert math.isclose(_row(table, ("small", 3e-4, 64))["z_H"], -1.0, abs_tol=1e-9)
    assert math.isclose(_row(table, ("medium", 3e-4, 64))["z_H"], 1.0, abs_tol=1e-9)


def test_low_neutral_agreement_accuracy_is_gated_out():
    from analysis.recipe_selection import select_on_probe

    spec = _spec({("small", 3e-4, 64): 4.0, ("medium", 3e-4, 64): 8.0})
    good = _terms({k: v for k, v in spec.items() if k[0] == "small"}, gate_acc=1.0)
    bad = _terms({k: v for k, v in spec.items() if k[0] == "medium"}, gate_acc=0.5)
    import pandas as pd
    terms = load_terms_from(pd.concat([good, bad], ignore_index=True),
                            _runs_for(spec))
    res = select_on_probe(terms, _runs_for(spec))
    assert not _row(res.table, ("medium", 3e-4, 64))["admissible"]
    assert res.selected == ("small", 3e-4, 64)   # the higher S is gated out


def test_recipe_that_does_not_beat_b_tok_is_gated_out():
    from analysis.recipe_selection import B_TOK, select_on_probe

    spec = _spec({("small", 3e-4, 64): 4.0, ("medium", 3e-4, 64): 8.0})
    runs = (_runs_for({k: v for k, v in spec.items() if k[0] == "small"},
                      eval_loss=1.80)
            + _runs_for({k: v for k, v in spec.items() if k[0] == "medium"},
                        eval_loss=B_TOK + 0.5))
    res = select_on_probe(load_terms_from(_terms(spec), runs), runs)
    assert not _row(res.table, ("medium", 3e-4, 64))["pass_btok"]
    assert res.selected == ("small", 3e-4, 64)


def test_tie_within_seed_noise_falls_to_smallest_model_then_largest_batch():
    """Step 5.4: inside the noise band the choice is a throughput convention."""
    from analysis.recipe_selection import select_on_probe

    spec = _spec({("large", 3e-4, 32): 8.02, ("large", 3e-4, 64): 8.01,
                  ("medium", 3e-4, 64): 8.00, ("small", 3e-4, 64): 1.0})
    jitter = {("large", 3e-4, 32): 0.5, ("large", 3e-4, 64): -0.5,
              ("medium", 3e-4, 64): 0.4}
    terms = load_terms_from(_terms(spec, jitter=jitter), _runs_for(spec))
    res = select_on_probe(terms, _runs_for(spec))
    assert res.noise > 0
    assert set(res.tied) >= {("large", 3e-4, 32), ("medium", 3e-4, 64)}
    assert ("small", 3e-4, 64) not in res.tied      # genuinely separated
    assert res.secondary_applied
    assert res.selected == ("medium", 3e-4, 64)     # smallest of the tied, bs64


def test_a_separated_winner_is_not_tie_broken():
    from analysis.recipe_selection import select_on_probe

    spec = _spec({("large", 3e-4, 32): 20.0, ("medium", 3e-4, 64): 1.0,
                  ("small", 3e-4, 64): 0.5})
    terms = load_terms_from(_terms(spec), _runs_for(spec))
    res = select_on_probe(terms, _runs_for(spec))
    assert res.tied == [("large", 3e-4, 32)]
    assert not res.secondary_applied
    assert res.selected == ("large", 3e-4, 32)


def test_epoch_views_read_different_epochs():
    """`fixed` reads the last epoch every run reaches; `final` each run's own."""
    from analysis.recipe_selection import _at_view

    spec = _spec({("small", 3e-4, 64): 4.0, ("medium", 3e-4, 64): 8.0})
    terms = load_terms_from(_terms(spec, n_epochs=3), _runs_for(spec))
    terms = terms[~(terms.run_id.str.contains("medium") & (terms.epoch > 2))]

    final = _at_view(terms, "final")
    assert set(final[final.run_id.str.contains("small")].epoch) == {3.0}
    assert set(final[final.run_id.str.contains("medium")].epoch) == {2.0}
    # `fixed` is the last epoch EVERY run reaches, so the longer run is cut back.
    assert set(_at_view(terms, "fixed").epoch) == {2.0}


def test_terms_table_from_another_sweep_is_refused():
    from analysis.recipe_selection import load_terms

    spec = _spec({("small", 3e-4, 64): 4.0})
    frame = _terms(spec)
    with pytest.raises(ValueError, match="absent from the run records"):
        load_terms_from_csv(frame, [])


def load_terms_from_csv(frame, runs, tmp=None):
    import tempfile, os
    from analysis.recipe_selection import load_terms
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    frame.to_csv(path, index=False)
    try:
        return load_terms(path, runs)
    finally:
        os.unlink(path)


def test_cli_refuses_to_rank_on_loss_by_default():
    """Selecting without --terms would silently rank on eval_loss again."""
    import tempfile, os, json as _json
    from analysis.recipe_selection import main

    spec = _spec({("small", 3e-4, 64): 4.0})
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    with os.fdopen(fd, "w") as fh:
        for r in _runs_for(spec):
            fh.write(_json.dumps(r) + "\n")
    try:
        with pytest.raises(SystemExit):
            main([path])
    finally:
        os.unlink(path)
