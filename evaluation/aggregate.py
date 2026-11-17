"""Pairs scored items, applies the per-violation criterion, builds `S[r]`.

Consumes the long per-item table from `score_pairs.py` and produces the recipe
selection statistic. All of the methodology decided in
`docs/RECIPE_SELECTION.md` lives here, in one place.

The criterion is assigned by geometry, not by result
----------------------------------------------------
The two candidate statistics are well posed under different manipulations, and
which manipulation an item underwent is a fixed, recorded property:

* **Relocation** (``unlicensed_site``, ``wrong_site``) — the ungrammatical member
  moves material, so the critical span lands at a different index on a different
  category. A span comparison is not minimal there. **Whole-string margin.**
  For the three movement terms the two members are exact token *permutations*,
  so this margin is a pure word-order judgment: no frequency term, no length
  term.
* **In-place substitution** (``wrong_item``, ``agreement``) — the strings are
  index aligned and equal length, differing at one token. The span comparison is
  minimal and targeted. **Critical-span margin.**

Both statistics are computed for every item regardless; the assignment only
decides which one enters `S[r]`. Because the map is keyed on the recorded
``violation`` field and fixed before any model was scored, this is a
pre-registration rather than a choice among results. The mapping is exact:
checked over all four grammars × all three probe splits, ``violation`` predicts
span alignment with zero mismatches.

Terms
-----
``refl`` and ``pron`` are internally mixed — their pairs are allocated in equal
blocks over ``BINDING_MODES`` (``pron``: wrong_item / wrong_site; ``refl``: those
two plus ``no_substitution``) — so they are *split into terms* rather than forced
onto one criterion. The block sizes follow ``n_per_subcase`` (300 by default, so
150 per pron mode and 100 per refl mode), not a fixed count. The term set is
derived from the data (`_term_names`), never hardcoded, so a probe rebuild that
changes the violation mix cannot leave a stale list behind. With the current
probe that is eight objective terms:

    aux, wh_i, wh_ii                         (whole-string; permutation pairs)
    refl:wrong_item, pron:wrong_item         (critical-span)
    refl:wrong_site, pron:wrong_site         (whole-string)
    refl:no_substitution                     (critical-span; refl-only)

``neutral_agreement`` and ``neutral_positional`` are **gates, not terms**: a
recipe that has not learned local agreement is not a candidate, but being
excellent at it earns nothing.

Aggregation
-----------
The eight terms are not in the same units — whole-string margins are nats over
~24 tokens, span margins are nats over one — so a raw mean would silently
overweight the movement terms. Each term is therefore **z-scored across the
recipes being compared**, within ``(grammar, term)``, before averaging. `S[r]`
is unit free and answers "how does this recipe rank on this rule", which is the
only question selection needs. Raw nats are reported alongside and never
discarded.

    S[r] = mean over grammars of ( mean over the 8 terms of z[r, grammar, term] )
    W[r] = min over (grammar, term) of z[r, grammar, term]        # worst cell

Guard: a term whose spread across recipes is below `MIN_SPREAD` nats carries no
information about the ranking, and z-scoring it would amplify pure noise. Such
terms contribute 0 and are listed in the report.

The frequency and length correction
-----------------------------------
``margin_b1`` — the same margin computed under the **order-1** n-gram, the pure
lexical-frequency reader — is **subtracted**. The objective is

    margin_corrected = margin − margin_b1

which is the model's log-probability difference minus the unigram model's, and
is therefore zero exactly when the model has captured nothing beyond lexical
frequency and token count. That is the correct null, and it controls token
identity and length together.

*Why subtraction and not length normalization.* A per-token margin divides a
difference of sums by a length, which yields neither the margin nor a corrected
margin — it rescales an uncorrected quantity. Subtraction removes the confounded
component in the units the margin is already in.

*Why uniformly, across all eight terms.* ``margin_b1`` is identically zero
wherever the members are token permutations — ``aux``, ``wh_i``, ``wh_ii``,
``neutral_positional`` — because a unigram model is order blind
(`baseline_pairs.check_permutation_terms` asserts it). So uniform subtraction
costs exactly nothing on those terms and removes the alternative, which was to
compute the two ``wrong_site`` terms differently from the other six.

*Units.* The correction is well posed under both criteria. For a whole-string
term both quantities sum over the same tokens. For a **span** term the members
are index aligned, equal length and differ at one token, so every other token
cancels out of ``−log b1(x)`` and the whole-string ``margin_b1`` *is* its own
span-restricted value. ``attach_baselines`` asserts that equal-length premise
rather than trusting it.

*What this replaced.* ``margin_b1`` used to be reported beside each term and
never subtracted, with `S[r]` ranking on the raw margin. That left the two
``wrong_site`` terms — whose members genuinely differ in length, since a
relocated substitution site carries a different number of CAT4/CAT2 modifiers —
exposed to a direct mechanical confound, with nothing between a length-tracking
recipe and the ranking except a human comparing two reported columns. It also
made the length exposure invisible in the objective while being real in it.

*Known limitation.* A unigram baseline controls **frequency and count**. It does
not control any correlation between block length and structural position: if the
licensed and unlicensed sites differ systematically in block length, then length
is not merely noise but is confounded with the licensing contrast itself, and
subtracting a unigram term does not remove that. Measuring that correlation is
outstanding — the standing figure, "11–12 of 15 pairs differ in length", rests on
n=15 and is not a usable estimate.

``margin`` (raw) and ``margin_b1`` both stay in the reported table beside the
corrected term, as diagnostics: a term whose raw ``margin`` sits close to its
``margin_b1`` has not been learned, and that comparison stays visible. What was
removed is ``margin_b1``'s role as the *only* safeguard, not its role as a
reference point.

The order-4 reference (headroom, not a correction)
--------------------------------------------------
``margin_over_b4 = margin − margin_ref_b4`` is reported alongside, and is the
more informative of the two comparisons on most terms. ``margin_ref_b4`` is the
order-4 n-gram's margin taken **under the same criterion as the term's own
margin** — ``margin_b4_span`` on span terms, ``margin_b4`` on whole-string terms.
That criterion matching is required rather than tidy: the cancellation identity
that lets the *whole-string* ``margin_b1`` stand in for its span-restricted value
does not survive above order 1, because changing the token under the span also
changes the context of the next ``order − 1`` positions. Mixing the two would
subtract a sum over ~25 tokens from a sum over one.

It is **not** subtracted for selection, and `S[r]` still ranks on
``margin_corrected``. The two baselines are not the same kind of object: order 1
is a nuisance the model should get no credit for, so it is removed; order 4 is
bounded *local structure*, much of which the model is supposed to learn, so
subtracting it would dock a recipe for succeeding. Read ``margin_over_b4`` as
headroom above the local reader — positive means the model beats a 4-gram on that
contrast, near zero means the margin it earns is reachable from local statistics
alone and the term is not evidence of structure.

Where this bites hardest is ``refl:wrong_item`` and ``pron:wrong_item``.
``margin_b1`` is **exactly** 0 on both, and legitimately so: ``cat1pron`` has two
members, so those violations always swap the same pair of morphemes (``mm`` ↔
``m``), and the phenomenon allocation puts exactly 5000 of each in every
grammar's train split — identical unigram counts, hence a bit-identical
frequency term. The b1 correction is therefore a no-op there and says nothing.
``margin_over_b4`` is what carries information on those two terms.

The two halves
--------------
The pilot stops each run when its probe score stops improving, and then selects
the recipe on the same probe. Doing both on the same items would let every run
stop at its own best moment on the items it is judged by, and the maximum of a
noisy curve is biased upward — a bias that grows with the number of epochs a run
was allowed, so it does not even cancel across runs. `pair_half` therefore
splits the pairs into a disjoint stopping half and scoring half. Selection reads
`--half score`.

Usage
-----
    python -m evaluation.aggregate scores.parquet --out results/eval/terms.csv
    python -m evaluation.aggregate 'results/eval/grid_*.parquet' --select --half score
"""
from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Geometry of each violation -> which statistic is well posed for it.
CRITERION_BY_VIOLATION = {
    "unlicensed_site": "string",   # relocation: aux, wh_i, wh_ii
    "wrong_site": "string",        # relocation: refl, pron
    "wrong_item": "span",          # in-place substitution: refl, pron
    "no_substitution": "span",     # in-place substitution: refl only
    "agreement": "span",           # in-place substitution: neutral control
    "positional": "string",        # local reorder: neutral control (permutation)
}

# Violations whose two members are exact token PERMUTATIONS of each other. A
# unigram model is order blind, so `margin_b1` must be identically 0 on these and
# the correction is a no-op — which is what makes applying it uniformly free.
# Keyed on `violation` rather than on term names, so a probe rebuild that renames
# or splits a term cannot quietly drop a member of this set.
PERMUTATION_VIOLATIONS = ("unlicensed_site", "positional")

# Controls, not objective terms.
GATE_TERMS = ("neutral_agreement", "neutral_positional")

# A term must move by at least this much (nats, across the compared recipes)
# before its z-score is trusted to carry ranking information.
MIN_SPREAD = 1e-3


# --- the stopping / scoring split ------------------------------------------

STOP_HALF = 0
SCORE_HALF = 1


def pair_half(pair_id: object) -> int:
    """`STOP_HALF` or `SCORE_HALF` for one `pair_id`, deterministically.

    Hashed with blake2b rather than the builtin `hash()`: Python salts string
    hashing per process, so `hash()` would assign a pair to the stopping half
    during training and to the scoring half during analysis — silently undoing
    the separation the split exists to create.
    """
    digest = hashlib.blake2b(str(pair_id).encode("utf-8"), digest_size=8).digest()
    return digest[0] & 1


def select_half(pairs: "pd.DataFrame", half: str) -> "pd.DataFrame":
    """`pairs` restricted to one half. `half="all"` is a no-op.

    Applied AFTER `to_pairs`, never before: `_term_names` decides which subcases
    are geometry-mixed from the data it is given, so naming the terms on half the
    pairs could name them differently in the two halves.
    """
    if half == "all":
        return pairs
    if half not in ("stop", "score"):
        raise ValueError(f"half must be 'all', 'stop' or 'score'; got {half!r}")
    want = STOP_HALF if half == "stop" else SCORE_HALF
    return pairs[pairs["pair_id"].map(pair_half) == want]


def normalize_violation(violation: object) -> object:
    """Collapses the four `positional_*` variants into one geometry class."""
    if isinstance(violation, str) and violation.startswith("positional_"):
        return "positional"
    return violation


# --- pairing ---------------------------------------------------------------

def to_pairs(items: "pd.DataFrame") -> "pd.DataFrame":
    """One row per minimal pair, with both margins and both accuracies.

    Pairs on ``pair_id`` *within* ``(run_id, grammar, split)`` — `pair_id` is not
    unique across splits (it repeats between IOD and OOD), so it is never a join
    key on its own.
    """
    import pandas as pd

    df = items.copy()
    if "run_id" not in df.columns:
        df["run_id"] = "single"
    if "epoch" not in df.columns:
        df["epoch"] = float("nan")

    gram = df[df.variant == "grammatical"]
    ungram = df[df.variant == "ungrammatical"]
    keys = ["run_id", "epoch", "grammar", "split", "pair_id"]

    merged = gram.merge(
        ungram, on=keys, suffixes=("_g", "_u"), validate="one_to_one",
    )
    if len(merged) != len(gram):
        raise ValueError(
            f"pairing lost items: {len(gram)} grammatical members produced "
            f"{len(merged)} pairs; a pair_id is missing a partner."
        )

    merged["violation"] = merged["violation_u"].map(normalize_violation)
    merged["subcase"] = merged["subcase_g"]
    merged["criterion"] = merged["violation"].map(CRITERION_BY_VIOLATION)
    unknown = merged.loc[merged.criterion.isna(), "violation"].unique()
    if len(unknown):
        raise ValueError(
            f"no criterion assigned for violation(s) {list(unknown)}; add them "
            f"to CRITERION_BY_VIOLATION with an explicit geometry."
        )

    # Margins are oriented so that POSITIVE = the model prefers the grammatical
    # member, for both statistics.
    merged["margin_string"] = merged["neg_logq_u"] - merged["neg_logq_g"]
    merged["margin_span"] = merged["neg_logq_crit_u"] - merged["neg_logq_crit_g"]
    merged["acc_string"] = (merged["margin_string"] > 0).astype(float)
    merged["acc_span"] = (merged["margin_span"] > 0).astype(float)

    is_span = merged["criterion"] == "span"
    merged["margin"] = merged["margin_string"].where(~is_span, merged["margin_span"])
    merged["acc"] = merged["acc_string"].where(~is_span, merged["acc_span"])

    # A span-criterion pair whose member carries no target scores NaN, which
    # would drop that term from `S[r]` without a word — the failure mode that
    # `no_substitution` hit on its first build (the sidecar's target-finder looks
    # for a CAT1PRON, and the unsubstituted member has none). Fail loudly.
    missing = merged[is_span & merged["margin"].isna()]
    if len(missing):
        offenders = sorted(missing["violation"].unique())
        raise ValueError(
            f"{len(missing)} span-criterion pair(s) have no critical span "
            f"(violations: {offenders}). Their term would silently vanish from "
            f"S[r]. The probe builder must stamp a target on BOTH members — see "
            f"`_divergence_targets` in corpusgen/structural_probes.py."
        )

    merged["len_matched"] = merged["n_morphemes_g"] == merged["n_morphemes_u"]
    merged["term"] = _term_names(merged)
    return merged


def check_corpus_binding(score_patterns: Sequence[str], baselines: Path, *,
                         allow_unstamped: bool = False) -> None:
    """Refuse to join scores and baselines built on different corpus generations.

    The join in `attach_baselines` is on `(grammar, split, pair_id)`, and
    `pair_id` is positional — it survives a corpus rebuild and re-points onto a
    different sentence. So a mismatch here produces a table whose `margin`
    describes one corpus and whose `margin_b1` describes another, with every
    other guard in this module passing. `evaluation/provenance.py` records what
    happened when that went unnoticed.

    Checked per score shard, because a sweep's shards can be written by
    different invocations.
    """
    from models.dataload import compare_provenance
    from evaluation.provenance import describe, read_provenance

    base_prov = read_provenance(baselines)
    shards: list[str] = []
    for pattern in score_patterns:
        shards.extend(sorted(glob.glob(pattern)) or [pattern])

    unstamped = [str(baselines)] if base_prov is None else []
    problems: list[str] = []
    for shard in shards:
        prov = read_provenance(Path(shard))
        if prov is None:
            unstamped.append(shard)
            continue
        if base_prov is None:
            continue
        for line in compare_provenance(prov, base_prov,
                                       label_a="scores", label_b="baselines"):
            problems.append(f"{Path(shard).name}: {line}")

    if problems:
        shown = "\n  ".join(sorted(set(problems))[:12])
        raise ValueError(
            f"scores and baselines were computed on DIFFERENT corpora:\n"
            f"  {shown}\n"
            f"baselines: {describe(base_prov)}\n"
            f"The join key `pair_id` is positional, so this would have produced "
            f"a table mixing two generations rather than an error. Rebuild the "
            f"baselines against the corpus the scores were produced on:\n"
            f"  THESIS_CORPUS_SET=<set> python -m evaluation.baseline_pairs ...")

    if unstamped and not allow_unstamped:
        shown = "\n  ".join(Path(p).name for p in sorted(set(unstamped))[:8])
        raise ValueError(
            f"these inputs carry no corpus provenance, so the join cannot be "
            f"verified:\n  {shown}\n"
            f"They were written before the stamp existed (2026-08-07). Either "
            f"regenerate them, or pass --allow-unstamped-corpus to proceed with "
            f"an UNVERIFIED join — after confirming the corpus by hand (compare "
            f"`corpus_sha256` in the sweep's run_manifest.jsonl against the "
            f"set's CHECKSUMS.sha256; `python -m scripts.verify_corpus_binding` "
            f"does it for you).")


def attach_baselines(pairs: "pd.DataFrame", baselines: "pd.DataFrame") -> "pd.DataFrame":
    """Join the n-gram margins and build the corrected objective.

    ``margin_corrected = margin − margin_b1`` (see the module docstring). Three
    things are checked rather than assumed, because each would silently corrupt
    the ranking:

    1. **Coverage.** Every pair must receive a ``margin_b1``. A missing one would
       propagate NaN into `S[r]`, and a term that vanishes from the objective is
       indistinguishable from a term the recipe scored zero on.
    2. **Span units.** Span-criterion pairs must be equal length, which is what
       makes the whole-string ``margin_b1`` equal to its own span-restricted
       value. If a span-criterion pair is ever length mismatched the subtraction
       is comparing different numbers of tokens and the term is meaningless.
    3. **Permutation terms.** Where the members are token permutations
       ``margin_b1`` must be ~0, so the correction is a no-op there. A non-zero
       value means the pairing or the tokenization disagree with each other, and
       the subtraction would then inject an artifact into terms that had none.
    """
    import pandas as pd

    keep = ["grammar", "split", "pair_id", "margin_b1", "margin_b4",
            "margin_b1_span", "margin_b4_span"]
    on = ["grammar", "split", "pair_id"]
    baselines = baselines[[c for c in keep if c in baselines.columns]]

    # Merging a dictionary-encoded key against a plain one decodes both sides for
    # the whole frame — several hundred MB on a sweep, at the point where the
    # pairs are already live. Giving the baselines the same dtype avoids it. A
    # baseline row whose pair_id is not among the pairs' categories becomes null
    # and fails to match, which is what an unmatched right-hand row does anyway;
    # the reverse direction, a pair with no baseline, is caught just below.
    for key in on:
        if isinstance(pairs[key].dtype, pd.CategoricalDtype):
            recoded = baselines[key].astype(pairs[key].dtype)
            # A baseline row for a grammar or pair the scores do not cover recodes
            # to null. It could never match a left key (the pairs carry none), but
            # left as null it makes the right-hand keys non-unique and trips the
            # many_to_one validation below, so drop it rather than weaken that.
            baselines = baselines.assign(**{key: recoded})[recoded.notna().values]

    out = pairs.merge(baselines, on=on, how="left", validate="many_to_one")

    if "margin_b1" not in out.columns:
        raise ValueError(
            "baselines carry no `margin_b1`; it is the objective's correction "
            "term, not an optional annotation. Build it with "
            "`python -m evaluation.baseline_pairs --split <split>`.")

    missing = out["margin_b1"].isna()
    if missing.any():
        offenders = sorted(out.loc[missing, "term"].unique())
        raise ValueError(
            f"{int(missing.sum())} pair(s) have no margin_b1 (terms: "
            f"{offenders}). S[r] now ranks on margin − margin_b1, so an "
            f"unmatched pair would silently drop out of its term rather than "
            f"score. Rebuild the baselines for every scored split.")

    is_span = out["criterion"] == "span"
    bad = out[is_span & ~out["len_matched"]]
    if len(bad):
        raise ValueError(
            f"{len(bad)} span-criterion pair(s) are length mismatched (terms: "
            f"{sorted(bad['term'].unique())}). The whole-string margin_b1 only "
            f"equals its span-restricted value when every other token cancels, "
            f"which needs equal length. Subtracting it here would compare "
            f"different numbers of tokens.")

    perm = out[out["violation"].isin(PERMUTATION_VIOLATIONS)]
    if len(perm):
        worst = float(perm["margin_b1"].abs().max())
        if worst > 1e-6:
            raise ValueError(
                f"margin_b1 reaches {worst:.2e} on permutation violations, "
                f"where a unigram model must score both members identically. "
                f"The pairing or the tokenization is wrong; correcting with "
                f"this would inject an artifact into terms that had none.")

    out["margin_corrected"] = out["margin"] - out["margin_b1"]
    out["acc_corrected"] = (out["margin_corrected"] > 0).astype(float)
    _attach_b4_reference(out, is_span)
    return out


def _attach_b4_reference(out: "pd.DataFrame", is_span: "pd.Series") -> None:
    """Add the order-4 reference margin and the headroom over it, in place.

    ``margin_ref_b4`` is the b4 margin taken under the *same* criterion as the
    term's own margin — span-restricted where the term is scored on its critical
    span, whole-string where it is scored on the whole string. The b1 correction
    can get away with using the whole-string value everywhere because a unigram's
    out-of-span terms cancel on an in-place substitution (`baseline_pairs.
    check_span_equals_string` verifies exactly that). Order 4 has no such
    identity: changing the token under the span also changes the context of the
    next three positions, so a whole-string ``margin_b4`` includes surprisal from
    outside the span and subtracting it from a span margin would compare sums
    over different numbers of tokens.

    ``margin_over_b4 = margin − margin_ref_b4`` is a **diagnostic, not the
    objective.** `S[r]` continues to rank on ``margin_corrected`` (the b1
    correction). The difference is what each baseline is: b1 is a nuisance the
    model should not get credit for, so it is subtracted; b4 is *bounded local
    structure*, a large part of which the model is supposed to learn, so
    subtracting it would penalise a recipe for succeeding. Read ``margin_over_b4``
    as headroom above the local reader — positive means the model beats a 4-gram
    on that contrast, near zero means what it scores is reachable from local
    statistics alone.
    """
    if "margin_b4" not in out.columns:
        return
    ref = out["margin_b4"]
    if "margin_b4_span" in out.columns:
        ref = ref.where(~is_span, out["margin_b4_span"])
        missing = is_span & out["margin_b4_span"].isna()
        if missing.any():
            offenders = sorted(out.loc[missing, "term"].unique())
            raise ValueError(
                f"{int(missing.sum())} span-criterion pair(s) have no "
                f"margin_b4_span (terms: {offenders}). The b4 reference would "
                f"silently fall back to the whole-string value on some pairs of "
                f"a term and not others. Rebuild the baselines with "
                f"`python -m evaluation.baseline_pairs`.")
    out["margin_ref_b4"] = ref
    out["margin_over_b4"] = out["margin"] - ref
    out["acc_over_b4"] = (out["margin_over_b4"] > 0).astype(float)


def _term_names(pairs: "pd.DataFrame") -> "pd.Series":
    """`subcase`, or `subcase:violation` where a subcase spans two geometries.

    Derived from the data rather than hardcoded, so a probe rebuild that changes
    the violation mix cannot leave a stale term list behind.

    Computed over the distinct ``(subcase, violation)`` combinations — there are
    ten of them — and joined back, rather than row-wise: a sweep frame is
    millions of pairs and a `.apply(axis=1)` over it is a Python loop of that
    length. Neither column can be null here: `to_pairs` calls this only after
    the `criterion` lookup has raised on any unrecognised violation.
    """
    import pandas as pd

    keys = ["subcase", "violation"]
    frame = pairs[keys]
    combos = frame.drop_duplicates().reset_index(drop=True)
    per_subcase = combos.groupby("subcase", observed=True)["violation"].nunique()
    mixed = set(per_subcase[per_subcase > 1].index)
    combos["term"] = [f"{s}:{v}" if s in mixed else s
                      for s, v in zip(combos["subcase"], combos["violation"])]
    named = frame.merge(combos, on=keys, how="left")["term"]
    return pd.Series(named.to_numpy(), index=pairs.index, name="term")


# --- per-term table --------------------------------------------------------

def term_table(pairs: "pd.DataFrame") -> "pd.DataFrame":
    """Mean margin / accuracy per (run, grammar, term), both criteria retained."""
    import pandas as pd

    group = ["run_id", "epoch", "grammar", "split", "term", "criterion"]
    # `observed=True` throughout: the group keys arrive dictionary-encoded from
    # `load_pairs`, and the default would materialise every unobserved
    # combination of run x grammar x term as an all-NaN row.
    out = pairs.groupby(group, dropna=False, observed=True).agg(
        n_pairs=("margin", "size"),
        margin=("margin", "mean"),
        margin_se=("margin", lambda s: s.std(ddof=1) / max(len(s), 1) ** 0.5),
        acc=("acc", "mean"),
        margin_string=("margin_string", "mean"),
        margin_span=("margin_span", "mean"),
        acc_string=("acc_string", "mean"),
        acc_span=("acc_span", "mean"),
        len_matched=("len_matched", "mean"),
    ).reset_index()
    extra = [c for c in ("margin_b1", "margin_b4", "margin_b4_span",
                         "margin_corrected", "acc_corrected",
                         "margin_ref_b4", "margin_over_b4", "acc_over_b4")
             if c in pairs.columns]
    if extra:
        more = pairs.groupby(group, dropna=False, observed=True)[extra].mean().reset_index()
        out = out.merge(more, on=group, how="left")
    for column in ("margin_corrected", "margin_over_b4"):
        if column not in pairs.columns:
            continue
        se = pairs.groupby(group, dropna=False, observed=True)[column].agg(
            lambda s: s.std(ddof=1) / max(len(s), 1) ** 0.5)
        out = out.merge(se.rename(f"{column}_se").reset_index(),
                        on=group, how="left")
    out["is_gate"] = out["term"].isin(GATE_TERMS)
    return out.sort_values(group).reset_index(drop=True)


# --- selection -------------------------------------------------------------

def selection_table(terms: "pd.DataFrame", *, split: str = "structural_probe_SELECT",
                    min_spread: float = MIN_SPREAD) -> "pd.DataFrame":
    """`S[r]` and `W[r]` per run, from the z-scored objective terms.

    Objective terms only: the two neutral terms are gates and are reported as
    separate columns rather than folded into the score.
    """
    import pandas as pd

    scoped = terms[terms.split == split] if split in set(terms.split) else terms
    objective = scoped[~scoped.is_gate].copy()
    if objective.empty:
        raise ValueError("no objective terms found; only gates were scored")

    # `S[r]` ranks on the frequency/length-corrected margin (module docstring).
    # Falling back to the raw margin when the correction is absent would rank on
    # a quantity the two `wrong_site` terms are mechanically confounded in, so
    # its absence is an error rather than a default.
    if "margin_corrected" not in objective.columns:
        raise ValueError(
            "selection needs `margin_corrected` (= margin − margin_b1). Pass "
            "--baselines so `attach_baselines` can build it; ranking on the raw "
            "margin lets a length-tracking recipe score on the wrong_site terms.")
    objective["margin"] = objective["margin_corrected"]

    runs = sorted(objective.run_id.unique())
    if len(runs) < 2:
        raise ValueError(
            f"S[r] z-scores each term across the compared recipes; "
            f"{len(runs)} run(s) present, need >= 2. Use --terms for a "
            f"single-model report."
        )

    cell = objective.groupby(["grammar", "term"], observed=True)["margin"]
    objective["_mean"] = cell.transform("mean")
    objective["_sd"] = cell.transform(lambda s: s.std(ddof=0))
    objective["_spread"] = cell.transform(lambda s: s.max() - s.min())

    # Terms that do not separate the recipes contribute nothing rather than
    # contributing amplified noise.
    informative = objective["_spread"] >= min_spread
    objective["z"] = 0.0
    objective.loc[informative, "z"] = (
        (objective.loc[informative, "margin"] - objective.loc[informative, "_mean"])
        / objective.loc[informative, "_sd"].replace(0.0, float("nan"))
    )
    objective["z"] = objective["z"].fillna(0.0)

    per_grammar = objective.groupby(["run_id", "grammar"], observed=True)["z"].mean()
    rows = []
    for run in runs:
        z = objective[objective.run_id == run]
        rows.append({
            "run_id": run,
            "S": per_grammar.loc[run].mean(),
            "W": z["z"].min(),
            "worst_cell": (z.loc[z["z"].idxmin(), "grammar"] + "/"
                           + z.loc[z["z"].idxmin(), "term"]),
            "mean_margin_nats": z["margin"].mean(),
            "mean_acc": z["acc"].mean(),
            "n_terms": len(z),
        })
    table = pd.DataFrame(rows).sort_values(["S", "W"], ascending=False)

    gates = scoped[scoped.is_gate]
    if not gates.empty:
        gate_acc = gates.pivot_table(index="run_id", columns="term",
                                     values="acc", aggfunc="mean", observed=True)
        table = table.merge(gate_acc.add_prefix("gate_acc_"),
                            left_on="run_id", right_index=True, how="left")

    uninformative = objective.loc[~informative, ["grammar", "term"]].drop_duplicates()
    if len(uninformative):
        table.attrs["uninformative_terms"] = [
            f"{r.grammar}/{r.term}" for r in uninformative.itertuples()
        ]
    return table.reset_index(drop=True)


# --- loading ---------------------------------------------------------------
#
# A sweep's score shards do not fit in memory read naively, and the failure is a
# kernel OOM kill rather than an exception. The grid96 sweep is 96 shards x
# 4200 items x 10-33 epochs = 7.2M rows; at 22 columns, 14 of them repeated short
# strings held as Python objects, that is ~52 MB per shard in pandas (against
# 2 MB on disk, where parquet dictionary-encodes them) — 6.9 GB for the table and
# twice that across a `pd.concat` of a fully materialised list.
#
# Three things keep it inside a couple of GB, none of which change a number:
#
#   1. read only the columns this module uses (11 of 22),
#   2. dictionary-encode the string columns, and keep them encoded — hence
#      `observed=True` on every groupby over them,
#   3. pair each shard as it is read. Pairing is within
#      `(run_id, epoch, grammar, split, pair_id)` and a shard is one run and one
#      grammar, so no pair straddles two files; doing it per shard halves the
#      rows and drops the unpaired string columns before anything accumulates.
#
# Measured on grid96: 52 MB -> 2.4 MB per shard, ~1.2 GB peak for the run.

# The columns `to_pairs` and everything downstream actually read.
PAIR_COLUMNS = ("run_id", "epoch", "grammar", "split", "pair_id", "variant",
                "violation", "subcase", "neg_logq", "neg_logq_crit",
                "n_morphemes")

# Low-cardinality strings, before and after pairing. `pair_id` is the largest of
# them and repeats across every run and epoch in the sweep.
DICT_COLUMNS = ("run_id", "grammar", "split", "variant", "violation",
                "subcase", "pair_id", "criterion", "term")


def _shard_paths(patterns: Sequence[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matched = sorted(glob.glob(pattern))
        if not matched:
            raise FileNotFoundError(f"no files match {pattern!r}")
        paths.extend(matched)
    return paths


def _encode(df: "pd.DataFrame") -> "pd.DataFrame":
    """Dictionary-encode the string columns that are known to be repetitive."""
    for column in DICT_COLUMNS:
        if column in df.columns and df[column].dtype == object:
            df[column] = df[column].astype("category")
    return df


def _read_shard(path: str, columns: Sequence[str] | None = PAIR_COLUMNS):
    """One score shard, pruned and dictionary-encoded.

    Columns absent from the shard are skipped rather than requested: single-model
    score files predate `run_id`/`epoch`, and `to_pairs` supplies both.
    """
    import pandas as pd
    import pyarrow.parquet as pq

    if columns is None:
        return _encode(pd.read_parquet(path))
    present = set(pq.read_schema(path).names)
    wanted = [c for c in columns if c in present]
    return _encode(pd.read_parquet(path, columns=wanted))


def _concat(frames: list) -> "pd.DataFrame":
    """`pd.concat`, keeping dictionary-encoded columns encoded.

    Plain `concat` falls back to object dtype whenever two frames carry the same
    categorical column with different categories, which is exactly the case here
    (each shard is one run) and would undo the encoding at the worst moment —
    with all 96 shards live. Unifying the categories first keeps it encoded.
    """
    import pandas as pd
    from pandas.api.types import union_categoricals

    if len(frames) == 1:
        return frames[0].reset_index(drop=True)
    encoded = [c for c in frames[0].columns
               if isinstance(frames[0][c].dtype, pd.CategoricalDtype)]
    for column in encoded:
        dtype = union_categoricals([f[column] for f in frames]).dtype
        for frame in frames:
            frame[column] = frame[column].astype(dtype)
    return pd.concat(frames, ignore_index=True)


def load_scores(patterns: Sequence[str],
                columns: Sequence[str] | None = PAIR_COLUMNS) -> "pd.DataFrame":
    """The per-item table for every matching shard, pruned and encoded.

    Pass `columns=None` for every column in the files — needed only by a caller
    that wants a field this module does not use, and not affordable for a sweep.
    """
    return _concat([_read_shard(p, columns) for p in _shard_paths(patterns)])


def load_pairs(patterns: Sequence[str]) -> "pd.DataFrame":
    """Pairs for every matching shard, without holding the item table whole.

    Equivalent to `to_pairs(load_scores(...))`, and verified against it, with one
    ordering difference that matters: `term` is re-derived over the concatenated
    frame at the end. `_term_names` decides which subcases are geometry-mixed
    from the data it is handed, and a per-shard call sees one run and one
    grammar, so the per-shard names it assigns are not authoritative.
    """
    frames = []
    for path in _shard_paths(patterns):
        shard = to_pairs(_read_shard(path))
        # The `_g`/`_u` source columns have done their work by the time
        # `to_pairs` returns — every quantity derived from them (both margins,
        # both accuracies, `len_matched`) is already a column. Carrying them
        # across 96 shards is ~170 MB of numerics plus the paired string columns.
        keep = [c for c in shard.columns if not c.endswith(("_g", "_u"))]
        frames.append(_encode(shard[keep].copy()))
    pairs = _encode(_concat(frames))
    del frames
    pairs["term"] = _term_names(pairs).astype("category")
    return pairs


# --- CLI -------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scores", nargs="+", help="per-item parquet(s); globs ok")
    parser.add_argument("--out", type=Path, default=None, help="write the term table")
    parser.add_argument("--allow-unscoped-out", action="store_true",
                        help="permit an --out path that does not name the corpus set")
    parser.add_argument("--select", action="store_true",
                        help="also compute S[r]/W[r] across the runs present")
    parser.add_argument("--split", default="structural_probe_SELECT",
                        help="split the selection statistic is computed on")
    parser.add_argument("--half", choices=("all", "stop", "score"), default="all",
                        help="restrict to one half of the pairs (split on "
                             "pair_id). Pass 'score' when the runs were early "
                             "stopped: they consumed the 'stop' half.")
    parser.add_argument("--baselines", type=Path, default=None,
                        help="n-gram pair margins to join (evaluation.baseline_pairs)")
    parser.add_argument("--allow-unstamped-corpus", action="store_true",
                        help="proceed when an input carries no corpus "
                             "provenance (files written before 2026-08-07). The "
                             "join is then UNVERIFIED — see evaluation/"
                             "provenance.py. Never use it to silence a real "
                             "mismatch.")
    args = parser.parse_args()

    from models.dataload import require_set_scoped
    if args.out is not None:
        args.out = require_set_scoped(args.out,
                                      override=args.allow_unscoped_out)

    import pandas as pd
    pd.set_option("display.width", 200)

    # Before the expensive load: reading a million rows only to discover they
    # describe a different corpus than the baselines wastes minutes and, worse,
    # invites overriding the check to "just see the numbers".
    if args.baselines is not None:
        check_corpus_binding(args.scores, args.baselines,
                             allow_unstamped=args.allow_unstamped_corpus)

    pairs = load_pairs(args.scores)
    if args.half != "all":
        before = len(pairs)
        pairs = select_half(pairs, args.half)
        print(f"half={args.half}: {len(pairs)} of {before} pairs")

    if args.baselines is not None:
        pairs = attach_baselines(pairs, pd.read_parquet(args.baselines))

    terms = term_table(pairs)
    show = ["grammar", "split", "term", "criterion", "n_pairs",
            "margin", "margin_se", "acc", "len_matched"]
    if "margin_corrected" in terms.columns:
        # Objective first, then the two diagnostics it was built from.
        for col in ("margin_b1", "margin_corrected"):
            show.insert(show.index("acc"), col)
    if "margin_over_b4" in terms.columns:
        # Headroom over the local reader — reported, never ranked on.
        for col in ("margin_ref_b4", "margin_over_b4"):
            show.insert(show.index("acc"), col)
    print(terms[show].to_string(index=False))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        if args.out.suffix == ".parquet":
            terms.to_parquet(args.out, index=False)
        else:
            terms.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    if args.select:
        table = selection_table(terms, split=args.split)
        print("\nPer-RUN probe score (diagnostic; NOT the recipe selection):\n")
        print(table.to_string(index=False))
        stale = table.attrs.get("uninformative_terms")
        if stale:
            print(f"\nterms below MIN_SPREAD (contributed 0): {', '.join(stale)}")
        # Deliberately no "SELECTED" line. This table is keyed on `run_id` =
        # grammar x recipe x seed, so its top row is a RUN and its z-scores are
        # standardized within one grammar. The recipe is selected on the
        # recipe-level `S[r]`, pooled symmetrically over the four grammars
        # (RECIPE_SELECTION Step 3; §1.7 constraint 3 forbids a per-grammar
        # recipe). Printing a winner here invited reading this as that.
        print("\nThe recipe is NOT selected from this table — it is per-grammar "
              "and per-seed.\nRun: python -m analysis.recipe_selection "
              "<sweep>/results.jsonl --terms <this --out>")


if __name__ == "__main__":
    main()
