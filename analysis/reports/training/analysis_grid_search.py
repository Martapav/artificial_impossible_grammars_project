#!/usr/bin/env python3
"""
analysis_grid_search.py — descriptive tables and figures for a grid sweep.

Reads `results.csv` (one row per training run) and the terms table written by
`evaluation.aggregate --out` (one row per run x epoch x term) and emits:

  * numbered tables -> <outdir>/tables/T##_<slug>.csv
  * numbered figures -> <outdir>/figures/F##_<slug>.png
  * <outdir>/report.txt   a descriptive index of both
  * <outdir>/index.csv    the same index, machine-readable

Written for the 96-run grid (4 grammars x 2 seeds x 3 sizes x 2 learning rates
x 2 batch sizes = 12 recipes x 8 runs), and degrades to any subset of it: axes
holding a single value are reported as constant and their marginal tables are
skipped.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It states no conclusions. Every headline and caption is limited to:
  (a) the quantity plotted or tabulated and its units,
  (b) how each cell was computed — which rows were aggregated over, at which
      epoch, with what n,
  (c) where the file was written.
It contains no claim about whether a value is high, low, adequate, expected,
resolvable, or evidence for anything. Keep it that way when editing: the reader
supplies the interpretation, and a caption that pre-empts it is harder to
disagree with than a number.

EPOCH VIEWS
-----------
Runs stop at different epochs, so every cross-run quantity is reported twice:

  fixed   all runs at one epoch — the last epoch at which every run is still
          alive (min over runs of max probe epoch), or --fixed-epoch.
  final   each run at its own last probe epoch (= its stopped_epoch).

Table slugs carry the view (`__fixed`, `__final`). Quantities that do not depend
on the epoch (n_pairs, len_matched, criterion, margin_b1, margin_b4) are
reported once.

COLUMNS IN THE TERMS TABLE
--------------------------
  acc      fraction of minimal pairs whose grammatical member scored higher.
  margin   mean log-probability difference between members, in nats.
  *_string / *_span   the same statistic over the whole sentence vs over the
           manipulated token span. `criterion` records which of the two
           `margin` and `acc` were copied from; it is fixed per term.
  margin_b1 / margin_b4        margin of a unigram / 4-gram baseline.
  margin_corrected             margin - margin_b1.
  acc_corrected                sign rate after the same per-item subtraction.
  margin_ref_b4                4-gram margin under the term's own `criterion`.
  margin_over_b4               margin - margin_ref_b4.
  len_matched                  fraction of pairs whose members have equal length.
  is_gate                      neutral_agreement, neutral_positional.

`margin_ref_b4` / `margin_over_b4` are present only when `evaluation.aggregate`
was run with `--baselines`. Without them the b4 sections are skipped and the
index says so; this script does not substitute the whole-string `margin_b4` on
span terms.

GRAMMAR CODING — ONE CROSSED FACTOR, NOT A 2x2
----------------------------------------------
The four grammars are laid out over the format of the two wh-licensing clauses:
H=(structural,structural), Hprime=(positional,structural),
Lprime=(structural,positional), P=(positional,positional). It is tempting to read
that as a 2x2 factorial. It is not one, and this section used to say it was.

The binding/aux rule family is `{H,Hprime}` vs `{Lprime,P}`
(`grammars/metadata.py`: "H'=H, P'=P" — the primes inherit their parent's binding
and aux rules). That partition is *identical* to the clause-(ii) partition. So
the second axis is not a second factor: it is the family, and it is perfectly
collinear with the clause-(ii) format. Only clause (i) varies within family.

Two consequences the tables here are built around:

  * Clause (i) is the one manipulation orthogonal to family, and it is the only
    one that can be attributed to a clause at all:
        contrast_clause_i = [(H - Hprime) + (Lprime - P)] / 2
  * The other axis is reported as `family_contrast`, never as a clause effect:
        family_contrast   = [(H - Lprime) + (Hprime - P)] / 2
    On wh_i/wh_ii it carries the clause-(ii) swap, but confounded with the family
    rules that swap along with it, so it is not clean evidence about clause (ii)
    either. On every other term it is purely the H-family/P-family rule
    difference — those items contain no wh material at all (`wh_site` is unset
    for every aux, refl, pron and neutral probe item in all four grammars), so
    reading 20 nats of aux difference as a "clause (ii) effect" is simply wrong.
  * The remaining contrast is therefore clause_i x family, not an interaction of
    two clause factors:
        clause_i_x_family = [(H - Hprime) - (Lprime - P)] / 2

Because the non-wh items contain no wh material, their clause-(i) contrast is
null by design — the only thing that differs between H and Hprime there is the
training distribution. Those eight values are reported as a *placebo* and give
an empirical null band to read wh_i and wh_ii against, which is a better use of
them than a factorial test the design does not support.

Decoupling clause (ii) from the family would need grammars that swap clause (ii)
while holding the binding/aux rules fixed, i.e. two more grammars.

Usage
-----
    python -m analysis.reports.training.analysis_grid_search \\
        --results results/grid96/set_2/results.csv \\
        --terms   results/eval/set_2/grid96_terms.csv \\
        --outdir  results/analysis/grid96/set_2
"""

import argparse
import csv as csvmod
import json
import os
import sys
import textwrap
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --------------------------------------------------------------------------- #
# presentation constants
# --------------------------------------------------------------------------- #

GRAMMARS = ["H", "Hprime", "Lprime", "P"]
GRAMMAR_LABELS = ["H\n(S,S)", "H′\n(P,S)", "P′\n(S,P)", "P\n(P,P)"]
# Display name per grammar key (data key stays raw; legends/ticks show these).
GNAME = {g: GRAMMAR_LABELS[i].split("\n")[0] for i, g in enumerate(GRAMMARS)}
SIZES = ["small", "medium", "large"]

# Reporting order: gates, then movement, then binding.
TERMS = [
    "neutral_agreement", "neutral_positional",
    "aux", "wh_i", "wh_ii",
    "refl:wrong_item", "refl:wrong_site", "refl:no_substitution",
    "pron:wrong_item", "pron:wrong_site",
]

# The only terms whose items instantiate a wh clause at all. For every other term
# `wh_site` is unset on both members, so neither clause format is present in what
# is being scored and only the training distribution differs. See GRAMMAR CODING.
WH_TERMS = ["wh_i", "wh_ii"]
NON_WH_TERMS = [t for t in TERMS if t not in WH_TERMS]

# Phenomenon families, for the family-level figures. A term's family is the part
# of its name before ':' (aggregate.py builds `term` as subcase or
# subcase:violation), with the two neutral terms grouped as gates.
FAMILIES = {
    "gate": ["neutral_agreement", "neutral_positional"],
    "aux": ["aux"],
    "wh": ["wh_i", "wh_ii"],
    "refl": ["refl:wrong_item", "refl:wrong_site", "refl:no_substitution"],
    "pron": ["pron:wrong_item", "pron:wrong_site"],
}

# The axes the sweep varies. `seed` is a replicate within a cell, not an axis.
RECIPE_FIELDS = ["size_name", "learning_rate", "batch_size"]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
plt.rcParams.update({
    "font.size": 9, "figure.dpi": 160, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})


# --------------------------------------------------------------------------- #
# output plumbing
# --------------------------------------------------------------------------- #

class Report:
    """Writes tables and figures, and indexes them in report.txt / index.csv.

    The index records what each artefact contains and how it was computed. It
    carries no evaluation of the contents.
    """

    def __init__(self, outdir):
        self.tabdir = os.path.join(outdir, "tables")
        self.figdir = os.path.join(outdir, "figures")
        os.makedirs(self.tabdir, exist_ok=True)
        os.makedirs(self.figdir, exist_ok=True)
        self.fh = open(os.path.join(outdir, "report.txt"), "w")
        self.index = []
        self.n_tab = 0
        self.n_fig = 0

    def _w(self, s=""):
        print(s)
        self.fh.write(s + "\n")

    def section(self, title):
        self._w("\n" + "=" * 78)
        self._w(title.upper())
        self._w("=" * 78)

    def note(self, lines):
        """Free-standing factual lines (no table number)."""
        for ln in lines:
            self._w("    " + ln)

    def table(self, headline, computed, df, slug, round_to=3, view=None,
              units=None, extra=None):
        """Emit one numbered table.

        `headline`  what the table contains.
        `computed`  how each cell was produced: rows aggregated, epoch, n.
        `view`      'fixed' / 'final' / None; appended to the slug.
        """
        self.n_tab += 1
        tag = f"T{self.n_tab:02d}"
        slug_full = f"{slug}__{view}" if view else slug
        path = os.path.join(self.tabdir, f"{tag}_{slug_full}.csv")
        head = f"{tag}. {headline}"
        if view:
            head += f"   [epoch view: {view}]"
        self._w("")
        self._w(head)
        self._w("    " + "\n    ".join(textwrap.wrap(computed, 92)))
        if units:
            self._w(f"    units: {units}")
        self._w("")
        body = df.round(round_to) if round_to is not None else df
        self._w(body.to_string())
        if extra:
            self._w("")
            for ln in extra:
                self._w("    " + ln)
        df.to_csv(path)
        self.index.append(dict(tag=tag, kind="table", headline=headline,
                               view=view or "", units=units or "",
                               computed=computed, path=path))
        return df

    def listing(self, headline, lines, slug, frame=None):
        """A numbered block of factual lines; optionally also written as CSV."""
        self.n_tab += 1
        tag = f"T{self.n_tab:02d}"
        path = ""
        self._w("")
        self._w(f"{tag}. {headline}")
        self._w("")
        for ln in lines:
            self._w("    " + ln)
        if frame is not None:
            path = os.path.join(self.tabdir, f"{tag}_{slug}.csv")
            frame.to_csv(path)
        self.index.append(dict(tag=tag, kind="listing", headline=headline,
                               view="", units="", computed="", path=path))

    def figure(self, fig, headline, computed, slug, suptitle=None, y=1.02,
               view=None, units=None):
        """Emit one numbered figure. The F-number is not drawn on the image; it
        survives only in the filename (F##_<slug>.png) and in index.csv, which
        keeps every figure traceable without stamping a number on the plot."""
        self.n_fig += 1
        tag = f"F{self.n_fig:02d}"
        slug_full = f"{slug}__{view}" if view else slug
        path = os.path.join(self.figdir, f"{tag}_{slug_full}.png")
        fig.savefig(path)
        plt.close(fig)
        self._w("")
        head = f"{tag}. {headline}"
        if view:
            head += f"   [epoch view: {view}]"
        self._w(f"{head}   ->  {path}")
        self._w("    " + "\n    ".join(textwrap.wrap(computed, 92)))
        if units:
            self._w(f"    units: {units}")
        self.index.append(dict(tag=tag, kind="figure", headline=headline,
                               view=view or "", units=units or "",
                               computed=computed, path=path))

    def close(self, outdir):
        idx = os.path.join(outdir, "index.csv")
        with open(idx, "w", newline="", encoding="utf-8") as fh:
            w = csvmod.DictWriter(fh, fieldnames=["tag", "kind", "headline",
                                                  "view", "units", "computed",
                                                  "path"])
            w.writeheader()
            w.writerows(self.index)
        self._w("")
        self._w(f"{self.n_tab} tables -> {self.tabdir}")
        self._w(f"{self.n_fig} figures -> {self.figdir}")
        self._w(f"index -> {idx}")
        self.fh.close()


# --------------------------------------------------------------------------- #
# loading, validation, frame construction
# --------------------------------------------------------------------------- #

def load(results_path, terms_path):
    res = pd.read_csv(results_path)
    trm = pd.read_csv(terms_path)
    for col in RECIPE_FIELDS + ["run_id", "grammar", "seed"]:
        if col not in res.columns:
            sys.exit(f"{results_path} has no column '{col}'")
    if "margin_corrected" not in trm.columns:
        sys.exit(f"{terms_path} has no `margin_corrected`. Re-run "
                 f"evaluation.aggregate with --baselines.")
    missing = set(res.grammar.unique()) - set(GRAMMARS)
    if missing:
        sys.exit(f"unknown grammars in results: {sorted(missing)}")
    return res, trm


def has_b4(trm):
    return {"margin_ref_b4", "margin_over_b4"}.issubset(trm.columns)


def lr_str(lr):
    """Learning rate formatted as in `RunSpec.run_id` (models/grid_search.py),
    so a recipe label, a legend entry and a run id all spell it the same way."""
    return f"{float(lr):.0e}".replace("-0", "-")


def recipe_label(size_name, lr, bs):
    return f"{size_name}/lr{lr_str(lr)}/bs{int(bs)}"


def add_recipe(df):
    df = df.copy()
    df["recipe"] = [recipe_label(s, l, b) for s, l, b
                    in zip(df.size_name, df.learning_rate, df.batch_size)]
    return df


def recipe_order(df):
    """Recipe labels in size-major, then lr, then bs order."""
    seen = df[RECIPE_FIELDS].drop_duplicates()
    seen["_k"] = [SIZES.index(s) if s in SIZES else 99 for s in seen.size_name]
    seen = seen.sort_values(["_k", "learning_rate", "batch_size"])
    return [recipe_label(s, l, b) for s, l, b
            in zip(seen.size_name, seen.learning_rate, seen.batch_size)]


def swept_axes(res):
    """Which of the three recipe axes carry more than one value."""
    return [c for c in RECIPE_FIELDS if res[c].nunique() > 1]


def validate(rep, res, trm):
    """Metric identities and structural properties of the two input files.

    Each line is a check name and its pass/fail. No consequence is drawn here.
    """
    checks = []
    s = trm[trm.criterion == "string"]
    p = trm[trm.criterion == "span"]
    checks.append(("margin == margin_string where criterion == 'string'",
                   np.allclose(s.margin, s.margin_string)))
    checks.append(("margin == margin_span where criterion == 'span'",
                   np.allclose(p.margin, p.margin_span)))
    checks.append(("acc likewise tracks `criterion`",
                   np.allclose(s.acc, s.acc_string) and np.allclose(p.acc, p.acc_span)))
    checks.append(("margin_corrected == margin - margin_b1",
                   np.allclose(trm.margin_corrected, trm.margin - trm.margin_b1)))
    if has_b4(trm):
        checks.append(("margin_over_b4 == margin - margin_ref_b4",
                       np.allclose(trm.margin_over_b4, trm.margin - trm.margin_ref_b4)))
        if "margin_b4_span" in trm.columns:
            checks.append(("margin_ref_b4 tracks `criterion`",
                           np.allclose(s.margin_ref_b4, s.margin_b4)
                           and np.allclose(p.margin_ref_b4, p.margin_b4_span)))
    per_term = trm.groupby("term").criterion.nunique()
    checks.append(("`criterion` is constant within each term",
                   bool((per_term == 1).all())))
    mx = trm.groupby("run_id").epoch.max()
    nn = trm.groupby("run_id").epoch.nunique()
    checks.append(("probe epochs are contiguous 1..max for every run",
                   bool((mx == nn).all())))
    stop = res.set_index("run_id").stopped_epoch.reindex(mx.index)
    checks.append(("max probe epoch == stopped_epoch for every run",
                   bool(np.allclose(mx.values, stop.values))))
    checks.append(("every run in the terms table is present in results.csv",
                   bool(set(trm.run_id) <= set(res.run_id))))
    checks.append(("every run in results.csv is present in the terms table",
                   bool(set(res.run_id) <= set(trm.run_id))))

    zero_b1 = trm.groupby("term").margin_b1.apply(
        lambda x: bool(np.allclose(x, 0, atol=1e-12)))
    lines = [f"[{'ok  ' if ok else 'FAIP'}] {name}" for name, ok in checks]
    lines += ["",
              "terms whose margin_b1 is 0 to 1e-12 (the unigram correction is a "
              "no-op there):",
              "    " + (", ".join(sorted(zero_b1[zero_b1].index)) or "(none)")]
    if not has_b4(trm):
        lines += ["",
                  "margin_ref_b4 / margin_over_b4 are absent from the terms "
                  "table: the b4 sections below are skipped. Re-run "
                  "evaluation.aggregate with --baselines to produce them."]
    rep.listing("Metric identities and structural checks on the two input files",
                lines, "input_checks",
                frame=pd.DataFrame(checks, columns=["check", "passed"]).set_index("check"))
    if not all(ok for _, ok in checks):
        print("\nNOTE: one or more identity checks reported FAIL "
              "(see T01).\n")


def build_frames(res, trm, fixed_epoch=None):
    """Merge run metadata onto the terms table and cut the two epoch views."""
    meta_cols = ["run_id", "grammar", "size_name", "learning_rate", "batch_size",
                 "seed", "n_params", "eval_loss", "perplexity", "stopped_epoch",
                 "train_loss", "train_runtime_s"]
    if "hit_cap" in res.columns:
        meta_cols.append("hit_cap")
    meta = res[[c for c in meta_cols if c in res.columns]]

    alive_to = int(trm.groupby("run_id").epoch.max().min())
    epoch = alive_to if fixed_epoch is None else int(fixed_epoch)
    if epoch > alive_to:
        print(f"NOTE: --fixed-epoch {epoch} is beyond the last epoch at which "
              f"every run is alive ({alive_to}); the fixed-view cells are "
              f"computed over the subset of runs that reached it.")

    long = add_recipe(trm.merge(meta, on=["run_id", "grammar"], how="left"))
    fixed = long[long.epoch == epoch].copy()
    final = long.sort_values("epoch").groupby(["run_id", "term"]).tail(1).copy()
    return long, {"fixed": fixed, "final": final}, epoch, alive_to


def n_runs_in(df):
    return int(df.run_id.nunique())


def order(df, index="term", columns="grammar"):
    """Reindex a pivot into reporting order, dropping absent labels."""
    if index == "term":
        idx = [t for t in TERMS if t in df.index]
    elif index == "size":
        idx = [s for s in SIZES if s in df.index]
    elif index == "grammar":
        idx = [g for g in GRAMMARS if g in df.index]
    else:
        idx = list(df.index)
    if columns == "grammar":
        col = [g for g in GRAMMARS if g in df.columns]
    elif columns == "size":
        col = [s for s in SIZES if s in df.columns]
    else:
        col = list(df.columns)
    return df.loc[idx, col]


def contrasts(cell_means):
    """`(clause_i, family, clause_i_x_family)` from grammar-indexed cell means.

    Not a 2x2 decomposition: the second axis is the binding/aux rule family, which
    is collinear with the clause-(ii) format by construction. See GRAMMAR CODING.
    """
    m = cell_means
    ci = ((m["H"] - m["Hprime"]) + (m["Lprime"] - m["P"])) / 2
    fam = ((m["H"] - m["Lprime"]) + (m["Hprime"] - m["P"])) / 2
    inter = ((m["H"] - m["Hprime"]) - (m["Lprime"] - m["P"])) / 2
    return ci, fam, inter


def cell_note(df, over):
    """'mean over N runs (…)' — the standard `computed` clause."""
    return f"cell = mean over {n_runs_in(df)} runs ({over})"


# --------------------------------------------------------------------------- #
# 1. what was run
# --------------------------------------------------------------------------- #

def t_inventory(rep, res, epoch, alive_to, expect_runs):
    res = add_recipe(res)
    inv = res.pivot_table(index="recipe", columns="grammar", values="run_id",
                          aggfunc="count")
    inv = inv.reindex(index=recipe_order(res),
                      columns=[g for g in GRAMMARS if g in inv.columns])
    inv["row total"] = inv.sum(axis=1)
    rep.table(
        "Run count per recipe x grammar",
        f"Counts of rows in results.csv. {len(res)} runs present; the full grid "
        f"is {expect_runs}. A cell equals the number of seeds when that cell is "
        f"complete.",
        inv, "run_inventory", round_to=None)

    arch = (res.groupby("size_name")[["n_layer", "n_embd", "n_head", "n_params"]]
            .first().reindex([s for s in SIZES if s in set(res.size_name)]))
    rep.table(
        "Architecture per size label",
        "Distinct architecture values carried by each size label in results.csv.",
        arch, "architectures", round_to=None)

    swept = swept_axes(res)
    varying, constant = {}, {}
    cols = ["learning_rate", "batch_size", "size_name", "seed", "epochs",
            "patience", "warmup_ratio", "weight_decay", "lr_scheduler",
            "subset_size", "train_split_n", "probe_split", "max_steps"]
    for c in cols:
        if c not in res.columns:
            continue
        u = sorted(map(str, res[c].dropna().unique()))
        (varying if len(u) > 1 else constant)[c] = ", ".join(u) if len(u) > 1 else (u[0] if u else "")
    lines = ["axes taking more than one value:"]
    lines += [f"    {k:>16s} : {v}" for k, v in varying.items()] or ["    (none)"]
    lines += ["", "settings constant across all runs:"]
    lines += [f"    {k:>16s} : {v}" for k, v in constant.items()]
    lines += ["", f"    recipe axes swept : {swept or '(none)'}",
              f"    recipes            : {len(recipe_order(res))}",
              f"    fixed-view epoch   : {epoch} "
              f"(every run is alive through epoch {alive_to})",
              f"    stopped_epoch      : min {int(res.stopped_epoch.min())}, "
              f"median {res.stopped_epoch.median():.1f}, "
              f"max {int(res.stopped_epoch.max())}"]
    if "hit_cap" in res.columns:
        lines += [f"    hit_cap True       : {int(res.hit_cap.sum())} of {len(res)} runs"]
    rep.listing("Swept axes and constant training settings", lines, "grid_settings")

    dur = res.groupby("recipe").agg(
        n_runs=("run_id", "size"),
        stopped_epoch_mean=("stopped_epoch", "mean"),
        stopped_epoch_min=("stopped_epoch", "min"),
        stopped_epoch_max=("stopped_epoch", "max"),
        eval_loss_mean=("eval_loss", "mean"),
        runtime_s_mean=("train_runtime_s", "mean"))
    if "hit_cap" in res.columns:
        dur["n_hit_cap"] = res.groupby("recipe").hit_cap.sum()
    dur["sec_per_epoch"] = dur.runtime_s_mean / dur.stopped_epoch_mean
    rep.table(
        "Training duration and cost per recipe",
        "Rows are the recipes present in results.csv; each aggregates that "
        "recipe's runs over all grammars and seeds. n_hit_cap counts runs whose "
        "stopped_epoch reached the --epochs cap.",
        dur.reindex(recipe_order(res)), "duration_by_recipe", round_to=2,
        units="epochs; nats/token; seconds")


# --------------------------------------------------------------------------- #
# 2. language modelling
# --------------------------------------------------------------------------- #

def t_lm(rep, res):
    res = add_recipe(res)
    lm = res.pivot_table(index="recipe", columns="grammar", values="eval_loss")
    lm = lm.reindex(index=recipe_order(res),
                    columns=[g for g in GRAMMARS if g in lm.columns])
    lm["row mean"] = lm.mean(axis=1)
    rep.table(
        "Dev loss by recipe x grammar, each run at its own checkpoint",
        "cell = mean eval_loss over the seeds in that recipe x grammar cell. "
        "eval_loss is the validation loss recorded at the end of each run.",
        lm, "dev_loss_recipe_grammar", round_to=4, units="nats/token",
        extra=[f"spread across all runs: {res.eval_loss.max() - res.eval_loss.min():.4f} nats; "
               f"sd {res.eval_loss.std():.4f}",
               f"perplexity range: {res.perplexity.min():.2f} to {res.perplexity.max():.2f}"])

    for axis in swept_axes(res):
        marg = res.pivot_table(index=axis, columns="grammar", values="eval_loss")
        if axis == "size_name":
            marg = marg.reindex([s for s in SIZES if s in marg.index])
        rep.table(
            f"Dev loss by {axis} x grammar (marginal)",
            f"cell = mean eval_loss over every run sharing that {axis} value, "
            f"averaged over the other two recipe axes and over seeds.",
            marg, f"dev_loss_marginal_{axis}", round_to=4, units="nats/token")


def t_devloss_probe_corr(rep, views):
    for view, df in views.items():
        rows = []
        for t in TERMS:
            d = df[df.term == t]
            if len(d) < 3:
                continue
            rm = (np.corrcoef(d.eval_loss, d.margin_corrected)[0, 1]
                  if d.margin_corrected.std() else np.nan)
            ra = (np.corrcoef(d.eval_loss, d.acc_corrected)[0, 1]
                  if d.acc_corrected.std() else np.nan)
            rows.append((t, rm, ra, len(d)))
        corr = pd.DataFrame(rows, columns=["term", "r_margin_corrected",
                                           "r_acc_corrected", "n_runs"]).set_index("term")
        rep.table(
            "Pearson r between final dev loss and the probe quantity, across runs",
            f"Each run contributes one point: its eval_loss against its "
            f"{view}-view value for that term. NaN where the probe quantity has "
            f"zero variance across runs.",
            order(corr, index="term", columns=None),
            "devloss_probe_correlation", round_to=3, view=view,
            units="correlation coefficient")


def f_devloss(rep, views, terms=tuple(t for t in TERMS
                                      if t not in FAMILIES["gate"])):
    """Dev loss against corrected margin, one panel per phenomenon subcase.

    The two neutral gates are excluded by default: they are not used for recipe
    selection, so their margin against dev loss is not the relationship of
    interest here. Panels are independently y-scaled — `aux` margins run to tens
    of nats and would flatten every other panel on a shared axis.
    """
    for view, df in views.items():
        keys = [t for t in terms if t in set(df.term)]
        if not keys:
            continue
        ncol = 4
        nrow = int(np.ceil(len(keys) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 3.2 * nrow),
                                squeeze=False)
        flat = axs.ravel()
        for idx, (a, t) in enumerate(zip(flat, keys)):
            d = df[df.term == t]
            for k, s in enumerate(SIZES):
                dd = d[d.size_name == s]
                if len(dd):
                    a.scatter(dd.eval_loss, dd.margin_corrected, s=26,
                              label=s, color=f"C{k}")
            a.set_title(t, fontsize=9)
            if idx // ncol == nrow - 1:            # bottom row only
                a.set_xlabel("eval_loss (nats/token)")
            if idx % ncol == 0:                    # left column only
                a.set_ylabel("margin_corrected (nats)")
        for a in flat[len(keys):]:
            a.axis("off")
        flat[0].legend(frameon=False, fontsize=7, title="size")
        rep.figure(fig,
                   "Relationship between validation loss and selection probe margins",
                   f"x = each run's final eval_loss; y = that run's "
                   f"margin_corrected for the named term at the {view} view. One "
                   f"panel per non-neutral phenomenon subcase (the neutral gates "
                   f"are excluded — they are not used for selection); panels are "
                   f"independently y-scaled. Colour = model size. "
                   f"{n_runs_in(df)} runs.",
                   "devloss_vs_margin", view=view)


# --------------------------------------------------------------------------- #
# 3. instrument diagnostics (epoch-invariant)
# --------------------------------------------------------------------------- #

def t_term_diagnostics(rep, fixed, b4):
    agg = dict(criterion=("criterion", "first"), is_gate=("is_gate", "first"),
               n_pairs=("n_pairs", "mean"), n_pairs_min=("n_pairs", "min"),
               len_matched=("len_matched", "mean"),
               margin_b1=("margin_b1", "mean"), margin_b4=("margin_b4", "mean"),
               margin_se=("margin_corrected_se", "mean"))
    if b4:
        agg["margin_ref_b4"] = ("margin_ref_b4", "mean")
    diag = fixed.groupby("term").agg(**agg)
    diag["binom_se_at_p.9"] = np.sqrt(0.9 * 0.1 / diag.n_pairs)
    rep.table(
        "Per-term instrument properties",
        "Properties of the probe pairs and the n-gram baselines. None of these "
        "columns depends on the epoch or on the trained model, so they are "
        "reported once. n_pairs is per run; len_matched is the fraction of pairs "
        "whose two members have equal word length; binom_se_at_p.9 is the "
        "binomial standard error on an accuracy of 0.9 at that n_pairs.",
        order(diag, index="term", columns=None), "term_diagnostics", round_to=3,
        units="counts; nats; proportions")


# --------------------------------------------------------------------------- #
# 4. main grids
# --------------------------------------------------------------------------- #

def t_main_grid(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        for val, rnd, unit in [("acc_corrected", 3, "proportion"),
                               ("margin_corrected", 2, "nats")]:
            piv = df.pivot_table(index="term", columns="grammar", values=val)
            rep.table(
                f"{val} by term x grammar",
                f"{cell_note(df, 'all recipes and seeds')} at {at}. "
                f"acc_corrected is the sign rate after per-item subtraction of "
                f"the unigram baseline; margin_corrected is margin - margin_b1.",
                order(piv), f"grid_{val}", round_to=rnd, view=view, units=unit)

        piv = df.pivot_table(index="term", columns="recipe",
                             values="margin_corrected")
        cols = [r for r in recipe_order(df) if r in piv.columns]
        rep.table(
            "margin_corrected by term x recipe",
            f"{cell_note(df, 'the 4 grammars and the seeds in each cell')} at "
            f"{at}. Columns are the full recipe (size/lr/bs), so no recipe axis "
            f"is averaged over.",
            order(piv[cols], index="term", columns=None),
            "grid_margin_by_recipe", round_to=2, view=view, units="nats")

        piv = df.pivot_table(index="term", columns=["grammar", "recipe"],
                             values="margin_corrected")
        cols = [(g, r) for g in GRAMMARS for r in recipe_order(df)
                if (g, r) in piv.columns]
        rep.table(
            "margin_corrected by term x grammar x recipe",
            f"{cell_note(df, 'the seeds in each cell only')} at {at}. Nothing is "
            f"averaged except seeds.",
            order(piv[cols], index="term", columns=None),
            "grid_margin_grammar_recipe", round_to=2, view=view, units="nats")


def t_marginals(rep, views, res, epoch):
    axes = swept_axes(res)
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        for axis in axes:
            piv = df.pivot_table(index="term", columns=axis,
                                 values="margin_corrected")
            if axis == "size_name":
                piv = order(piv, index="term", columns="size")
                lv = [c for c in SIZES if c in piv.columns]
            else:
                piv = order(piv, index="term", columns=None)
                lv = list(piv.columns)
            for a, b in zip(lv, lv[1:]):
                piv[f"{a}->{b}"] = piv[b] - piv[a]
            others = [x for x in axes if x != axis]
            rep.table(
                f"margin_corrected by term x {axis} (marginal)",
                f"{cell_note(df, 'grammars, seeds and ' + (', '.join(others) or 'nothing else'))} "
                f"at {at}. Trailing columns are successive differences between "
                f"adjacent levels of {axis}.",
                piv, f"marginal_{axis}", round_to=2, view=view, units="nats")


def f_heatmap(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
        plt.subplots_adjust(wspace=0.35)
        for a, (val, ttl, fmt) in zip(ax, [
                ("acc_corrected", "acc_corrected", "{:.3f}"),
                ("margin_corrected", "margin_corrected (nats)", "{:.1f}")]):
            M = order(df.pivot_table(index="term", columns="grammar", values=val))
            Z = M.values
            if val == "acc_corrected":
                im = a.imshow(Z, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.04,
                             label="acc_corrected")
                tc = "black"
            else:
                im = a.imshow(np.log10(np.clip(Z, 1e-6, None)), cmap="viridis",
                              aspect="auto")
                fig.colorbar(im, ax=a, fraction=0.046, pad=0.04,
                             label="log10 margin_corrected (nats)")
                tc = "white"
            a.set_xticks(range(len(M.columns)))
            a.set_xticklabels([GRAMMAR_LABELS[GRAMMARS.index(g)].split("\n")[0]
                               for g in M.columns])
            a.set_yticks(range(len(M.index)))
            a.set_yticklabels(list(M.index) if val == "acc_corrected" else [])
            for i in range(Z.shape[0]):
                for j in range(Z.shape[1]):
                    a.text(j, i, fmt.format(Z[i, j]), ha="center", va="center",
                           fontsize=7.5, color=tc)
            a.set_title(ttl)
            a.set_xlabel("grammar")
        rep.figure(fig, "All terms by grammar",
                   f"Left: acc_corrected on a 0.5-1.0 colour scale. Right: "
                   f"margin_corrected on a log10 colour scale. "
                   f"{cell_note(df, 'all recipes and seeds')} at {at}. "
                   f"Printed values are the cell means.",
                   "overview_heatmap", view=view,
                   suptitle=f"acc_corrected and margin_corrected by term x grammar ({view} view)",
                   y=1.03)


def f_recipe_grid(rep, views, res, epoch):
    """margin_corrected per recipe, one panel per term family."""
    axes = swept_axes(res)
    if "size_name" not in axes:
        return
    lrs = sorted(res.learning_rate.unique())
    bss = sorted(res.batch_size.unique())
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        keys = [t for t in TERMS if t in set(df.term)]
        ncol, nrow = 5, int(np.ceil(len(keys) / 5))
        fig, axs = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.5 * nrow),
                                sharex=True)
        axs = np.atleast_2d(axs)
        for a, t in zip(axs.ravel(), keys):
            d = df[df.term == t]
            for i, lr in enumerate(lrs):
                for j, bs in enumerate(bss):
                    dd = d[(d.learning_rate == lr) & (d.batch_size == bs)]
                    means = [dd[dd.size_name == s].margin_corrected.mean()
                             for s in SIZES]
                    a.plot(range(len(SIZES)), means, marker="os"[j % 2], ms=3.5,
                           ls=["-", "--"][j % 2], color=f"C{i}",
                           label=f"lr{lr_str(lr)}/bs{bs}")
            a.set_title(t, fontsize=8)
            a.set_xticks(range(len(SIZES)))
            a.set_xticklabels(SIZES, fontsize=7)
        for a in axs.ravel()[len(keys):]:
            a.axis("off")
        axs[0, 0].legend(frameon=False, fontsize=6)
        fig.supylabel("margin_corrected (nats)", fontsize=9, x=0.06)
        fig.supxlabel("model size", fontsize=9)
        rep.figure(fig, "margin_corrected across the three recipe axes, per term",
                   f"x = model size; colour = learning rate; line style and "
                   f"marker = batch size. Each point is the mean over the 4 "
                   f"grammars and the seeds in that cell, at {at}. Panels are "
                   f"independently scaled.",
                   "recipe_grid", view=view)


# --------------------------------------------------------------------------- #
# 5. clause-format contrasts (one crossed factor, plus family)
# --------------------------------------------------------------------------- #

def _contrast_rows(df, terms, quantities):
    """One row per (term, quantity): the four cell means and the contrasts."""
    rows = []
    for t in terms:
        d = df[df.term == t]
        for val, lab in quantities:
            m = d.groupby("grammar")[val].mean()
            if not set(GRAMMARS).issubset(m.index):
                continue
            ci, fam, inter = contrasts(m)
            sd = d.groupby("grammar")[val].std().mean()
            rows.append(dict(term=t, quantity=lab, H=m["H"], Hprime=m["Hprime"],
                             Lprime=m["Lprime"], P=m["P"], clause_i=ci,
                             family=fam, clause_i_x_family=inter,
                             within_cell_sd=sd,
                             clause_i_over_sd=ci / sd if sd else np.nan))
    return rows


def _quantities(b4):
    q = [("margin_corrected", "margin - b1"), ("acc_corrected", "accuracy")]
    if b4:
        q.insert(1, ("margin_over_b4", "margin - b4"))
    return q


def t_wh_contrasts(rep, views, b4, epoch):
    """The wh terms, where a clause format is actually present in the items."""
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        rows = _contrast_rows(df, WH_TERMS, _quantities(b4))
        if not rows:
            continue
        tab = pd.DataFrame(rows).set_index(["term", "quantity"])
        rep.table(
            "clause-format contrasts on the wh terms",
            f"H/Hprime/Lprime/P are the grammar cell means at {at}, each "
            f"{cell_note(df, 'recipes and seeds').lower()}. clause_i = "
            f"[(H-Hprime)+(Lprime-P)]/2, the one manipulation orthogonal to the "
            f"binding/aux rule family. family = [(H-Lprime)+(Hprime-P)]/2: on "
            f"these two terms it carries the clause-(ii) swap, but the family "
            f"partition {{H,Hprime}} vs {{Lprime,P}} is collinear with the "
            f"clause-(ii) format by construction, so a value here cannot be "
            f"attributed to clause (ii) rather than to the binding/aux rules "
            f"that swap with it. clause_i_x_family = [(H-Hprime)-(Lprime-P)]/2 "
            f"is how much the clause-(i) effect differs between the two "
            f"families — not an interaction of two clause factors. "
            f"within_cell_sd is the mean over grammars of the sd across runs "
            f"within a grammar.",
            tab, "wh_clause_contrasts", round_to=3, view=view,
            units="nats (margin rows); proportion (accuracy rows)",
            extra=["read clause_i against the placebo band in the next table, "
                   "not against zero"])


def t_family_and_placebo(rep, views, b4, epoch):
    """The eight terms with no wh material: a family contrast and a placebo."""
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        rows = _contrast_rows(df, NON_WH_TERMS, _quantities(b4))
        if not rows:
            continue
        tab = pd.DataFrame(rows).set_index(["term", "quantity"])
        band = tab.loc[(slice(None), "margin - b1"), "clause_i"]
        note = (f"placebo band on margin - b1: {band.min():+.2f} to "
                f"{band.max():+.2f} nats over {len(band)} terms "
                f"(max |value| {band.abs().max():.2f})")
        rep.table(
            "family contrast and clause-(i) placebo, terms with no wh material",
            f"H/Hprime/Lprime/P are the grammar cell means at {at}, each "
            f"{cell_note(df, 'recipes and seeds').lower()}. No item scored here "
            f"contains a wh clause (`wh_site` is unset on both members in all "
            f"four grammars), so neither clause format is present in what is "
            f"being compared. family = [(H-Lprime)+(Hprime-P)]/2 is therefore "
            f"the whole H-family/P-family binding and aux rule difference and "
            f"nothing to do with clause (ii). clause_i = "
            f"[(H-Hprime)+(Lprime-P)]/2 is null by design — the primes differ "
            f"from their parents only in a wh clause these items do not use — "
            f"so its spread is an empirical null band for the wh table's "
            f"clause_i, driven only by what the change did to the training "
            f"distribution. clause_i_x_family is a difference of two "
            f"null-by-design quantities and is reported for completeness.",
            tab, "family_contrast_and_placebo", round_to=3, view=view,
            units="nats (margin rows); proportion (accuracy rows)",
            extra=[note])


def f_wh_contrasts(rep, views, epoch):
    """Clause-(i) format on x, one line per family. Restricted to the wh terms.

    The two lines are the two rule families, which is also the clause-(ii) split —
    the same thing, which is why the legend names both and the caption says they
    cannot be separated. Only the movement *between* the two points on a line is
    a clean clause effect.
    """
    cells = {"S": {"S": "H", "P": "Lprime"}, "P": {"S": "Hprime", "P": "P"}}
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        keys = [t for t in WH_TERMS if t in set(df.term)]
        if not keys:
            continue
        fig, ax = plt.subplots(1, len(keys), figsize=(4.5 * len(keys), 3.6))
        ax = np.atleast_1d(ax)
        for a, term in zip(ax, keys):
            d = df[df.term == term]
            for ii, mk, lab in [("S", "o", "H family = clause (ii) structural"),
                                ("P", "s", "P family = clause (ii) positional")]:
                gs = [cells["P"][ii], cells["S"][ii]]
                ys = [d[d.grammar == g].margin_corrected.mean() for g in gs]
                es = [d[d.grammar == g].margin_corrected.std() for g in gs]
                a.errorbar([0, 1], ys, yerr=es, marker=mk, capsize=3, label=lab)
                for x, g in zip([0, 1], gs):
                    v = d[d.grammar == g].margin_corrected
                    a.scatter([x] * len(v), v, s=8, alpha=0.35, color="gray", zorder=0)
            a.set_xticks([0, 1])
            a.set_xticklabels(["positional", "structural"])
            a.set_xlim(-0.3, 1.3)
            a.set_xlabel("format of clause (i)")
            a.set_ylabel("margin_corrected (nats)")
            a.set_title(term)
        ax[0].legend(frameon=False, fontsize=7.5)
        rep.figure(fig, "margin_corrected by clause-(i) format, within family",
                   f"x = format of clause (i), the only manipulation orthogonal "
                   f"to the rule family, so the slope of each line is the clean "
                   f"contrast. The two lines are the two binding/aux rule "
                   f"families, which is the same partition as the clause-(ii) "
                   f"format — the vertical gap between them mixes both and is "
                   f"not a clause-(ii) effect. Points are grammar cell means at "
                   f"{at}, error bars the sd across the runs in the cell, grey "
                   f"dots the individual runs ({n_runs_in(df)} total).",
                   "wh_clause_contrasts", view=view,
                   suptitle=f"wh terms: clause (i) within family ({view} view)")


# --------------------------------------------------------------------------- #
# 6. scoring window and baselines
# --------------------------------------------------------------------------- #

def t_string_span(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        for stat in ["acc", "margin"]:
            a = df.pivot_table(index="term", columns="grammar", values=f"{stat}_string")
            b = df.pivot_table(index="term", columns="grammar", values=f"{stat}_span")
            out = pd.concat({"whole_string": order(a), "target_span": order(b),
                             "span_minus_string": order(b) - order(a)}, axis=1)
            rep.table(
                f"{stat} under both scoring windows, by term x grammar",
                f"whole_string sums log-probability over every token; "
                f"target_span sums only over the manipulated span. "
                f"{cell_note(df, 'recipes and seeds')} at {at}. `criterion` in "
                f"the diagnostics table records which window the reported "
                f"`{stat}` was copied from for each term.",
                out, f"string_vs_span_{stat}", round_to=3, view=view,
                units="proportion" if stat == "acc" else "nats")


def f_string_span(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        terms = [t for t in TERMS if t in set(df.term)]
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        w = 0.8 / max(len(GRAMMARS), 1)
        for k, g in enumerate(GRAMMARS):
            d = df[df.grammar == g].groupby("term")[["acc_string", "acc_span"]].mean()
            d = d.reindex(terms)
            x = np.arange(len(terms)) + (k - (len(GRAMMARS) - 1) / 2) * w
            ax.vlines(x, d.acc_string, d.acc_span, color=f"C{k}", alpha=0.5, lw=1)
            ax.scatter(x, d.acc_string, marker="v", s=34, color=f"C{k}")
            ax.scatter(x, d.acc_span, marker="^", s=34, color=f"C{k}", facecolors="none")
        ax.axhline(0.5, ls=":", c="k", lw=0.8)
        ax.text(len(terms) - 0.6, 0.51, "0.5", fontsize=7)
        ax.set_xticks(range(len(terms)))
        ax.set_xticklabels(terms, rotation=35, ha="right")
        ax.set_ylabel("accuracy")
        h = [plt.Line2D([], [], marker="v", ls="", color="k", label="whole string"),
             plt.Line2D([], [], marker="^", ls="", mfc="none", color="k", label="target span")]
        h += [plt.Line2D([], [], marker="s", ls="", color=f"C{k}", label=GNAME[g])
              for k, g in enumerate(GRAMMARS)]
        ax.legend(handles=h, frameon=False, fontsize=7.5, ncol=2, loc="lower left")
        rep.figure(fig, "Whole-string and target-span accuracy, per term and grammar",
                   f"Each pair of markers is one term x grammar cell: filled "
                   f"triangle = acc_string, open triangle = acc_span, joined by a "
                   f"line. {cell_note(df, 'recipes and seeds')} at {at}. The "
                   f"dotted line marks 0.5.",
                   "string_vs_span", view=view,
                   suptitle=f"acc_string vs acc_span ({view} view)", y=0.99)


def t_ngram_screen(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        m = df.pivot_table(index="term", columns="grammar", values="margin")
        b = df.pivot_table(index="term", columns="grammar", values="margin_ref_b4")
        d = df.pivot_table(index="term", columns="grammar", values="margin_over_b4")
        out = pd.concat({"model": order(m), "b4_reference": order(b),
                         "model_minus_b4": order(d)}, axis=1)
        rep.table(
            "Model margin against the 4-gram reference, by term x grammar",
            f"model = mean `margin` under the term's own criterion; "
            f"b4_reference = `margin_ref_b4`, the 4-gram margin under that same "
            f"criterion; model_minus_b4 = `margin_over_b4`. "
            f"{cell_note(df, 'recipes and seeds')} at {at}. The baseline columns "
            f"do not depend on the model or the epoch.",
            out, "ngram_screen", round_to=2, view=view, units="nats")


def f_ngram(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        terms = [t for t in TERMS if t in set(df.term)]
        fig, ax = plt.subplots(figsize=(8.8, 4.4))
        x = np.arange(len(terms))
        w = 0.8 / max(len(GRAMMARS), 1)
        for k, g in enumerate(GRAMMARS):
            d = df[df.grammar == g].groupby("term")[["margin", "margin_ref_b4"]].mean()
            d = d.reindex(terms)
            xi = x + (k - (len(GRAMMARS) - 1) / 2) * w
            ax.bar(xi, d.margin, width=w * 0.88, color=f"C{k}", alpha=0.45, label=GNAME[g])
            ax.scatter(xi, d.margin_ref_b4, marker="_", s=110, color="k", lw=1.6, zorder=5)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(terms, rotation=35, ha="right")
        ax.set_ylabel("margin (nats)")
        h = [plt.Line2D([], [], marker="s", ls="", color=f"C{k}", alpha=0.45, label=GNAME[g])
             for k, g in enumerate(GRAMMARS)]
        h += [plt.Line2D([], [], marker="_", ls="", color="k", label="b4 reference")]
        ax.legend(handles=h, frameon=False, fontsize=7.5, ncol=5, loc="upper right")
        rep.figure(fig, "Model margin (bars) and the 4-gram reference (dashes)",
                   f"Bars are the mean `margin` per term x grammar; black dashes "
                   f"are `margin_ref_b4` for the same cell. "
                   f"{cell_note(df, 'recipes and seeds')} at {at}.",
                   "ngram_screen", view=view,
                   suptitle=f"margin against the b4 reference ({view} view)", y=0.99)


# --------------------------------------------------------------------------- #
# 7. trajectories and seed variability
# --------------------------------------------------------------------------- #

def t_trajectory(rep, long, res):
    warnings.filterwarnings("ignore", message="An input array is constant")
    rows = []
    for t in TERMS:
        rs, ra = [], []
        for _, d in long[long.term == t].groupby("run_id"):
            d = d.sort_values("epoch")
            rs.append(d.epoch.corr(d.margin_corrected, method="spearman"))
            ra.append(d.epoch.corr(d.acc_corrected, method="spearman"))
        if rs:
            rows.append((t, np.nanmedian(rs), np.nanmin(rs), np.nanmax(rs),
                         np.nanmedian(ra)))
    mono = pd.DataFrame(rows, columns=["term", "rho_margin_median",
                                       "rho_margin_min", "rho_margin_max",
                                       "rho_acc_median"]).set_index("term")

    stop = res.set_index("run_id").stopped_epoch
    last = long.sort_values("epoch").groupby(["run_id", "term"]).tail(1)
    early = long[long.epoch == long.run_id.map(stop) - 5]
    j = last.merge(early, on=["run_id", "term"], suffixes=("_last", "_pat"))
    if len(j):
        j["gain"] = ((j.margin_corrected_last - j.margin_corrected_pat) /
                     j.margin_corrected_pat.replace(0, np.nan))
        win = j.groupby("term").gain.agg(["mean", "max"])
        win.columns = ["last5_relative_gain_mean", "last5_relative_gain_max"]
        mono = mono.join(win)

    e5 = long[long.epoch == 5]
    if len(e5):
        ratio = (last.set_index(["run_id", "term"]).margin_corrected /
                 e5.set_index(["run_id", "term"]).margin_corrected)
        ratio = ratio.groupby(level="term").mean()
        ratio.name = "margin_last_over_epoch5"
        mono = mono.join(ratio)

    rep.table(
        "Within-run epoch trends of the probe quantities",
        "rho is the Spearman correlation between epoch and the quantity within "
        "one run, aggregated over runs. last5_relative_gain is the relative "
        "change in margin_corrected over each run's final five epochs. "
        "margin_last_over_epoch5 is the ratio of the last-epoch value to the "
        "epoch-5 value, averaged over runs.",
        order(mono, index="term", columns=None), "trajectory_trends", round_to=3,
        units="correlation; ratios")


def _eval_curves(results_path, exclude=None):
    """Per-grammar ``{epoch: [eval_loss, ...]}`` from the per-run ``eval_curve``
    in the results ``.jsonl`` sibling of ``results_path``. Runs whose id is in
    ``exclude`` are skipped. Empty if the file is absent."""
    exclude = exclude or set()
    jsonl = os.path.splitext(results_path)[0] + ".jsonl"
    if not os.path.exists(jsonl):
        return {}
    per = {}
    with open(jsonl) as fh:
        for line in fh:
            r = json.loads(line)
            if r["run_id"] in exclude:
                continue
            d = per.setdefault(r["grammar"], {})
            for e in r.get("eval_curve", []):
                d.setdefault(int(e["epoch"]), []).append(float(e["eval_loss"]))
    return per


def _degenerate_runs(results_path, train_rise_thresh=0.1):
    """Run ids that DIVERGED: train loss (not just dev) ends more than
    ``train_rise_thresh`` nats above its own minimum. These are the lr=1e-3/bs=32
    blow-ups where both losses skyrocket — as opposed to genuine overfitting,
    where train loss keeps falling. Empty if the .jsonl is absent."""
    jsonl = os.path.splitext(results_path)[0] + ".jsonl"
    if not os.path.exists(jsonl):
        return set()
    deg = set()
    with open(jsonl) as fh:
        for line in fh:
            r = json.loads(line)
            tl = [e["loss"] for e in r.get("train_curve", [])]
            if tl and tl[-1] - min(tl) > train_rise_thresh:
                deg.add(r["run_id"])
    return deg


def _probe_acc_curves(long):
    """Per grammar ``{epoch: [per-run mean acc_corrected over non-gate terms]}``."""
    ng = long[~long["is_gate"]]
    per_run = ng.groupby(["grammar", "run_id", "epoch"])["acc_corrected"].mean().reset_index()
    out = {}
    for g, sub in per_run.groupby("grammar"):
        out[str(g)] = {int(ep): list(s["acc_corrected"].values)
                       for ep, s in sub.groupby("epoch")}
    return out


def _draw_loss_probe(rep, loss, probe, headline, caption, slug):
    """Shared stacked twin-panel plot: dev loss (top) and probe accuracy (bottom)
    vs epoch, per grammar, median + IQR over the ≥75%-alive region."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 6.8), sharex=True)
    _pergram_epoch_curve(ax1, loss)
    _pergram_epoch_curve(ax2, probe)
    ax1.set_ylabel("dev loss (nats/token)")
    ax2.set_ylabel("probe accuracy")
    ax2.set_xlabel("epoch")
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.legend(frameon=False, ncol=4, fontsize=8, loc="upper right")
    rep.figure(fig, headline, caption, slug)


def _alive_mask(epochs, n_alive):
    """Same rule the trajectory figure uses: keep epochs where at least 75% of a
    grammar's peak run count is still alive."""
    return n_alive >= max(1, int(n_alive.max() * 0.75))


def dev_loss_plateaus(results_path, frac=0.01):
    """Per-grammar dev-loss plateau epoch, for the trajectory figure.

    For each grammar it takes the mean eval_loss over runs alive at each epoch,
    applies the ≥75%-alive mask, and returns the first masked epoch at which the
    mean comes within ``frac`` of its minimum. {grammar: epoch}; empty if absent.
    """
    out = {}
    for g, d in _eval_curves(results_path).items():
        epochs = np.array(sorted(d))
        n_alive = np.array([len(d[e]) for e in epochs])
        mean = np.array([float(np.mean(d[e])) for e in epochs])
        keep = _alive_mask(epochs, n_alive)
        if not keep.any():
            continue
        ep, mn = epochs[keep], mean[keep]
        within = np.where(mn <= mn.min() * (1.0 + frac))[0]
        if len(within):
            out[g] = int(ep[within[0]])
    return out


def f_loss_curve(rep, results_path):
    """Dev-loss curve per grammar: median line + interquartile band.

    Held-out eval_loss against epoch, runs pooled within each grammar. The line
    is the median and the band the 25-75th percentile over that grammar's runs at
    each epoch (robust: a few runs that overfit hard late do not drag the line
    out of its band). Shown only while at least 75% of the grammar's runs are
    still alive (they stop at different epochs). Colours match the trajectory
    figure.
    """
    curves = _eval_curves(results_path)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    for k, g in enumerate(GRAMMARS):
        d = curves.get(g)
        if not d:
            continue
        epochs = np.array(sorted(d))
        n_alive = np.array([len(d[e]) for e in epochs])
        ep = epochs[_alive_mask(epochs, n_alive)]
        med = np.array([float(np.percentile(d[e], 50)) for e in ep])
        q1 = np.array([float(np.percentile(d[e], 25)) for e in ep])
        q3 = np.array([float(np.percentile(d[e], 75)) for e in ep])
        ax.fill_between(ep, q1, q3, color=f"C{k}", alpha=0.18, lw=0)
        ax.plot(ep, med, color=f"C{k}", lw=1.9, label=GNAME[g])
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev loss (nats/token)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, title="grammar")
    rep.figure(fig, "Dev-loss curve by grammar (median + IQR band)",
               "Held-out eval_loss against epoch. Solid line = median over the "
               "grammar's runs alive at that epoch; shaded band = interquartile "
               "range (25-75th percentile) across those runs. Shown only while at "
               "least 75% of the grammar's runs are still alive. Median/IQR are "
               "used so a few runs that overfit hard at late epochs do not distort "
               "the curve.",
               "loss_curve")


def f_loss_curve_full(rep, results_path):
    """Dev-loss curve per grammar over the FULL epoch range.

    Companion to :func:`f_loss_curve`. The region where at least 75% of a
    grammar's runs are alive is drawn solid with the IQR band; beyond it the
    median continues as a faded dashed line with no band (the survivorship tail:
    few runs, count annotated at the end). The median suppresses the diverging
    runs throughout, and those stop by ~epoch 16, so the far tail tracks the
    longest-lived, stable runs still gently overfitting.
    """
    curves = _eval_curves(results_path)
    if not curves:
        return
    fig, ax = plt.subplots(figsize=(9.4, 5.2))
    for k, g in enumerate(GRAMMARS):
        d = curves.get(g)
        if not d:
            continue
        epochs = np.array(sorted(d))
        n_alive = np.array([len(d[e]) for e in epochs])
        med = np.array([float(np.percentile(d[e], 50)) for e in epochs])
        q1 = np.array([float(np.percentile(d[e], 25)) for e in epochs])
        q3 = np.array([float(np.percentile(d[e], 75)) for e in epochs])
        cut = int(np.where(_alive_mask(epochs, n_alive))[0].max())   # last dense epoch
        ax.fill_between(epochs[:cut + 1], q1[:cut + 1], q3[:cut + 1],
                        color=f"C{k}", alpha=0.18, lw=0)
        ax.plot(epochs[:cut + 1], med[:cut + 1], color=f"C{k}", lw=1.9, label=GNAME[g])
        ax.plot(epochs[cut:], med[cut:], color=f"C{k}", lw=1.3,
                ls=(0, (2, 2)), alpha=0.45)                          # faded tail
        ax.annotate(f"n={n_alive[-1]}", xy=(epochs[-1], med[-1]), xytext=(3, 0),
                    textcoords="offset points", color=f"C{k}", fontsize=7,
                    va="center", alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("dev loss (nats/token)")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, title="grammar", loc="upper left")
    rep.figure(fig, "Dev-loss curve by grammar, full epoch range",
               "As the median+IQR loss curve, extended to every grammar's last "
               "epoch. Solid line + band = while at least 75% of the grammar's "
               "runs are alive; faded dashed = the survivorship tail (few runs, "
               "count n annotated at the end). The median suppresses the diverging "
               "runs (which stop by ~epoch 16), so the tail tracks the "
               "longest-lived, stable runs still gently overfitting.",
               "loss_curve_full")


def _pergram_epoch_curve(ax, per):
    """Plot median + IQR over the ≥75%-alive region for {grammar: {epoch: [v]}},
    in the canonical grammar order so legend and colours line up."""
    for k, g in enumerate(GRAMMARS):
        d = per.get(g)
        if not d:
            continue
        epochs = np.array(sorted(d))
        n = np.array([len(d[e]) for e in epochs])
        ep = epochs[_alive_mask(epochs, n)]
        med = np.array([float(np.percentile(d[e], 50)) for e in ep])
        q1 = np.array([float(np.percentile(d[e], 25)) for e in ep])
        q3 = np.array([float(np.percentile(d[e], 75)) for e in ep])
        ax.fill_between(ep, q1, q3, color=f"C{k}", alpha=0.16, lw=0)
        ax.plot(ep, med, color=f"C{k}", lw=1.9, label=GNAME[g])


def f_loss_vs_probe(rep, long, results_path):
    """Dev loss and probe accuracy vs epoch, stacked and sharing the epoch axis.

    Top: median held-out dev loss (+IQR) per grammar (the F44 curve). Bottom:
    median probe accuracy (+IQR) per grammar, where a run's value at an epoch is
    the mean ``acc_corrected`` over the 8 non-gate terms. Both are shown while at
    least 75% of a grammar's runs are alive. Reading straight down shows the
    decoupling: dev loss bottoms ~epoch 8 and then rises, while probe accuracy
    climbs and then holds — which is why stopping is on the probe, not the loss.
    """
    _draw_loss_probe(
        rep, _eval_curves(results_path), _probe_acc_curves(long),
        "Dev loss and probe accuracy vs epoch, by grammar",
        "Stacked, shared epoch axis. Top: median dev loss (+IQR) per grammar. "
        "Bottom: median probe accuracy (+IQR), a run's value being the mean "
        "acc_corrected over the 8 non-gate terms. Both shown while at least 75% of "
        "the grammar's runs are alive. Dev loss bottoms ~epoch 8 then rises while "
        "probe accuracy climbs then plateaus — the decoupling that motivates "
        "probe-based stopping.",
        "loss_vs_probe")


def f_loss_vs_probe_clean(rep, long, results_path):
    """Same as :func:`f_loss_vs_probe`, with the degenerate (diverged) runs removed.

    The runs whose *train* loss also skyrockets (all lr=1e-3/bs=32) are dropped
    from both panels, so the curves show only training-stable runs — isolating
    genuine overfitting and the grammar contrast from optimizer failure.
    """
    deg = _degenerate_runs(results_path)
    _draw_loss_probe(
        rep, _eval_curves(results_path, exclude=deg),
        _probe_acc_curves(long[~long["run_id"].isin(deg)]),
        "Dev loss and probe accuracy vs epoch, degenerate runs excluded",
        f"As the loss-vs-probe figure, but the {len(deg)} diverged runs (where "
        "train loss ALSO skyrockets — all lr=1e-3/bs=32) are dropped from both "
        "panels, leaving only training-stable runs. This isolates genuine "
        "overfitting and the grammar contrast from optimizer failure.",
        "loss_vs_probe_clean")


def f_trajectory(rep, long, plateaus=None, keys=tuple(TERMS)):
    plateaus = plateaus or {}
    keys = [k for k in keys if k in set(long.term)]
    ncol = 5
    nrow = int(np.ceil(len(keys) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.7 * nrow),
                            sharex=True, squeeze=False)
    flat = axs.ravel()
    for a, t in zip(flat, keys):
        for k, g in enumerate(GRAMMARS):
            d = long[(long.term == t) & (long.grammar == g)]
            if not len(d):
                continue
            n_alive = d.groupby("epoch").run_id.nunique()
            mean = d.groupby("epoch").margin_corrected.mean()
            keep = n_alive >= max(1, int(n_alive.max() * 0.75))
            a.plot(mean[keep].index, mean[keep].values, color=f"C{k}", label=GNAME[g])
        for k, g in enumerate(GRAMMARS):        # dev-loss plateau epoch per grammar
            if g in plateaus:
                a.axvline(plateaus[g], color=f"C{k}", ls=":", lw=1.0,
                          alpha=0.7, zorder=0)
        a.set_title(t, fontsize=8)
    for a in flat[len(keys):]:
        a.axis("off")
    for c in range(ncol):                       # xlabel on each column's last panel
        col_idx = [i for i in range(len(keys)) if i % ncol == c]
        if col_idx:
            flat[max(col_idx)].set_xlabel("epoch")
    flat[0].legend(frameon=False, fontsize=7)
    flat[0].xaxis.set_major_locator(MaxNLocator(integer=True))   # epochs are integers
    fig.supylabel("margin_corrected (nats)", fontsize=9, x=0.06)
    plateau_note = (
        " Dotted vertical lines mark each grammar's dev-loss plateau epoch — the "
        "first epoch at which its mean eval_loss (pooled over runs alive, same "
        "≥75%-alive rule) comes within 1% of its minimum."
        if plateaus else "")
    rep.figure(fig, "margin_corrected against epoch, by grammar",
               "Each line is the mean over all runs of that grammar alive at "
               "that epoch, plotted only while at least 75% of the grammar's "
               "runs are still alive. Runs stop at different epochs, so the "
               "number of runs behind each point falls with epoch." + plateau_note,
               "trajectories")


def t_variability(rep, views, epoch):
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        out = {}
        for val, lab in [("acc_corrected", "accuracy"), ("margin_corrected", "margin")]:
            piv = df.pivot_table(index=["grammar", "recipe", "term"],
                                 columns="seed", values=val)
            seeds = list(piv.columns)
            if len(seeds) < 2:
                continue
            piv["abs_diff"] = (piv[seeds[0]] - piv[seeds[-1]]).abs()
            out[f"{lab}_seed_absdiff"] = piv.groupby(level="term").abs_diff.agg(
                ["mean", "median", "max"])
        if not out:
            continue
        tab = pd.concat({k: order(v, index="term", columns=None)
                         for k, v in out.items()}, axis=1)
        rep.table(
            "Seed-to-seed absolute difference within each grammar x recipe cell",
            f"For each grammar x recipe x term cell the absolute difference "
            f"between the two seeds, then summarized over cells per term, at "
            f"{at}. Cells with fewer than two seeds are excluded.",
            tab, "seed_variability", round_to=3, view=view,
            units="proportion (accuracy); nats (margin)")


# --------------------------------------------------------------------------- #
# 8. per-family and per-term figures
# --------------------------------------------------------------------------- #

def f_family(rep, views, epoch):
    """One figure per phenomenon family: its terms by grammar x size."""
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        for fam, terms in FAMILIES.items():
            keys = [t for t in terms if t in set(df.term)]
            if not keys:
                continue
            fig, axs = plt.subplots(1, len(keys), figsize=(3.5 * len(keys), 3.3),
                                    squeeze=False)
            for a, t in zip(axs[0], keys):
                d = df[df.term == t]
                for k, g in enumerate(GRAMMARS):
                    dd = d[d.grammar == g]
                    means = [dd[dd.size_name == s].margin_corrected.mean()
                             for s in SIZES]
                    a.plot(range(len(SIZES)), means, marker="o", ms=4,
                           color=f"C{k}", label=GNAME[g])
                    a.scatter([SIZES.index(s) for s in dd.size_name],
                              dd.margin_corrected, s=7, color=f"C{k}", alpha=0.35)
                a.set_title(t, fontsize=9)
                a.set_xticks(range(len(SIZES)))
                a.set_xticklabels(SIZES, fontsize=7.5)
                a.margins(y=0.18)
            axs[0][0].set_ylabel("margin_corrected (nats)")
            axs[0][0].legend(frameon=False, fontsize=7, loc="upper left")
            rep.figure(fig, f"Family '{fam}': margin_corrected by grammar and size",
                       f"One panel per term in the family. Lines are grammar "
                       f"means over lr, bs and seeds; small dots are the "
                       f"individual runs behind them, at {at}. Panels are "
                       f"independently scaled.",
                       f"family_{fam}", view=view,
                       suptitle=f"Family {fam} ({view} view)")


def f_per_term(rep, long, views, res, epoch, b4):
    """One four-panel figure per term."""
    lrs = sorted(res.learning_rate.unique())
    bss = sorted(res.batch_size.unique())
    for view, df in views.items():
        at = f"epoch {epoch}" if view == "fixed" else "each run's last epoch"
        for t in [x for x in TERMS if x in set(df.term)]:
            d = df[df.term == t]
            fig, axs = plt.subplots(1, 4, figsize=(14.5, 3.3))

            # P1 — grammar x size
            a = axs[0]
            for k, g in enumerate(GRAMMARS):
                dd = d[d.grammar == g]
                means = [dd[dd.size_name == s].margin_corrected.mean() for s in SIZES]
                a.plot(range(len(SIZES)), means, marker="o", ms=4, color=f"C{k}", label=GNAME[g])
                a.scatter([SIZES.index(s) for s in dd.size_name],
                          dd.margin_corrected, s=7, color=f"C{k}", alpha=0.3)
            a.set_xticks(range(len(SIZES)))
            a.set_xticklabels(SIZES, fontsize=7.5)
            a.set_ylabel("margin_corrected (nats)")
            a.set_title("by grammar and size", fontsize=8.5)
            a.margins(y=0.18)      # headroom so the legend clears the top series
            a.legend(frameon=False, fontsize=6.5, ncol=2, loc="upper left")

            # P2 — lr x bs, pooled over grammars and sizes
            a = axs[1]
            for i, lr in enumerate(lrs):
                ys = [d[(d.learning_rate == lr) & (d.batch_size == bs)]
                      .margin_corrected.mean() for bs in bss]
                a.plot(range(len(bss)), ys, marker="o", ms=4, color=f"C{i}",
                       label=f"lr {lr_str(lr)}")
                for j, bs in enumerate(bss):
                    v = d[(d.learning_rate == lr) & (d.batch_size == bs)].margin_corrected
                    a.scatter([j] * len(v), v, s=7, color=f"C{i}", alpha=0.3)
            a.set_xticks(range(len(bss)))
            a.set_xticklabels([str(b) for b in bss], fontsize=7.5)
            a.set_xlabel("batch size")
            a.set_title("by learning rate and batch size", fontsize=8.5)
            if len(lrs) > 1:
                a.legend(frameon=False, fontsize=6.5)

            # P3 — epoch trajectory by grammar
            a = axs[2]
            dl = long[long.term == t]
            for k, g in enumerate(GRAMMARS):
                dd = dl[dl.grammar == g]
                if not len(dd):
                    continue
                n_alive = dd.groupby("epoch").run_id.nunique()
                mean = dd.groupby("epoch").margin_corrected.mean()
                keep = n_alive >= max(1, int(n_alive.max() * 0.75))
                a.plot(mean[keep].index, mean[keep].values, color=f"C{k}", label=GNAME[g])
            a.set_xlabel("epoch")
            a.set_title("by epoch (≥75% of runs alive)", fontsize=8.5)

            # P4 — string vs span accuracy, or the b4 reference
            a = axs[3]
            gs = [g for g in GRAMMARS if g in set(d.grammar)]
            xs = np.arange(len(gs))
            if b4:
                mm = [d[d.grammar == g].margin.mean() for g in gs]
                bb = [d[d.grammar == g].margin_ref_b4.mean() for g in gs]
                a.bar(xs, mm, width=0.6, color="C0", alpha=0.45, label="model")
                a.scatter(xs, bb, marker="_", s=140, color="k", lw=1.8,
                          label="b4 reference", zorder=5)
                a.axhline(0, color="k", lw=0.8)
                a.set_ylabel("margin (nats)")
                a.set_title("margin and b4 reference", fontsize=8.5)
            else:
                a.scatter(xs, [d[d.grammar == g].acc_string.mean() for g in gs],
                          marker="v", s=34, color="C0", label="whole string")
                a.scatter(xs, [d[d.grammar == g].acc_span.mean() for g in gs],
                          marker="^", s=34, facecolors="none", color="C0",
                          label="target span")
                a.axhline(0.5, ls=":", c="k", lw=0.8)
                a.set_ylabel("accuracy")
                a.set_title("acc_string vs acc_span", fontsize=8.5)
            a.set_xticks(xs)
            a.set_xticklabels([GNAME[g] for g in gs], fontsize=7.5)
            a.legend(frameon=False, fontsize=6.5)

            p4 = ("model margin with the b4 reference" if b4
                  else "acc_string against acc_span")
            slug = t.replace(":", "_")
            rep.figure(
                fig, f"Term '{t}': four views",
                f"Panel 1: margin_corrected by grammar and size, dots = "
                f"individual runs. Panel 2: the same quantity by learning rate "
                f"and batch size, pooled over grammars and sizes. Panel 3: "
                f"margin_corrected against epoch, one line per grammar, plotted "
                f"while at least 75% of that grammar's runs are alive. Panel 4: "
                f"{p4}, per grammar. Panels 1, 2 and 4 are at {at}; "
                f"{n_runs_in(d)} runs.",
                f"term_{slug}", view=view,
                suptitle=f"{t} ({view} view)", y=1.04)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True,
                    help="results.csv from the sweep output directory")
    ap.add_argument("--terms", required=True,
                    help="terms table written by `evaluation.aggregate --out`")
    ap.add_argument("--outdir", default="grid_search_analysis")
    ap.add_argument("--fixed-epoch", type=int, default=None,
                    help="epoch for the 'fixed' view; default is the last epoch "
                         "at which every run is alive")
    ap.add_argument("--views", nargs="+", default=["fixed", "final"],
                    choices=["fixed", "final"],
                    help="which epoch views to emit (default: both)")
    ap.add_argument("--expect-runs", type=int, default=96,
                    help="size of the full grid, reported in the inventory")
    ap.add_argument("--no-per-term", action="store_true",
                    help="skip the per-term figures (10 per epoch view)")
    args = ap.parse_args()

    res, trm_raw = load(args.results, args.terms)
    os.makedirs(args.outdir, exist_ok=True)
    rep = Report(args.outdir)
    b4 = has_b4(trm_raw)

    rep.section("0. inputs")
    rep.note([
        f"results: {args.results}  ({len(res)} runs, "
        f"{res.grammar.nunique()} grammars, {len(recipe_order(add_recipe(res)))} recipes)",
        f"terms:   {args.terms}  ({len(trm_raw)} run-epoch-term rows, "
        f"{trm_raw.term.nunique()} terms, splits: {sorted(trm_raw.split.unique())})",
        f"b4 reference columns present: {b4}",
    ])
    validate(rep, res, trm_raw)

    long, all_views, epoch, alive_to = build_frames(res, trm_raw, args.fixed_epoch)
    views = {k: v for k, v in all_views.items() if k in args.views}

    rep.section("1. what was run")
    t_inventory(rep, res, epoch, alive_to, args.expect_runs)

    rep.section("2. language modelling")
    t_lm(rep, res)
    t_devloss_probe_corr(rep, views)
    f_devloss(rep, views)

    rep.section("3. instrument properties")
    t_term_diagnostics(rep, all_views["fixed"], b4)

    rep.section("4. main grids")
    t_main_grid(rep, views, epoch)
    t_marginals(rep, views, res, epoch)
    f_heatmap(rep, views, epoch)
    f_recipe_grid(rep, views, res, epoch)

    rep.section("5. clause-format contrasts (one crossed factor, plus family)")
    t_wh_contrasts(rep, views, b4, epoch)
    t_family_and_placebo(rep, views, b4, epoch)
    f_wh_contrasts(rep, views, epoch)

    rep.section("6. scoring window and n-gram reference")
    t_string_span(rep, views, epoch)
    f_string_span(rep, views, epoch)
    if b4:
        t_ngram_screen(rep, views, epoch)
        f_ngram(rep, views, epoch)

    rep.section("7. epoch trends and seed variability")
    t_trajectory(rep, long, res)
    f_trajectory(rep, long, dev_loss_plateaus(args.results))
    t_variability(rep, views, epoch)

    rep.section("8. per-family figures")
    f_family(rep, views, epoch)

    if not args.no_per_term:
        rep.section("9. per-term figures")
        f_per_term(rep, long, views, res, epoch, b4)

    rep.section("10. dev-loss curve")
    f_loss_curve(rep, args.results)
    f_loss_curve_full(rep, args.results)
    f_loss_vs_probe(rep, long, args.results)
    f_loss_vs_probe_clean(rep, long, args.results)

    rep.close(args.outdir)


if __name__ == "__main__":
    main()
