"""A4 — differential surprisal: model score minus the grammar's own entropy.

    Δ(x)  =  neg_logq(x)  −  ℓ_corpus(x)

``neg_logq`` is the model's negative log-probability of the string;
``ℓ_corpus`` is the corpus-side derivation cost after the rejection-sampling
correction (:mod:`analysis.entropy_baseline`). Subtracting it is what makes
cross-grammar comparisons legitimate: the four grammars have different
entropies, so a raw ``neg_logq`` difference between H and P confounds "the model
learned this grammar less well" with "this grammar is intrinsically less
predictable".

Scope
-----
**Corpus and in-distribution splits only** — the same scope ``ℓ(x)`` itself has.
It is *not* an evaluation metric for the structural probes. Probe results are
reported as the within-pair difference of model negative log-probabilities
between the ungrammatical and grammatical member; both members share one base,
so a shared ``ℓ(x)`` cancels from that difference and no baseline is needed
(``docs/SURPRISAL_REFERENCE_POINTS.md`` §6.9).

Because every item on this scope is grammatical and was produced by
``π_corpus``, ``ℓ_corpus`` can never be undefined here. There is therefore no
null-by-reason accounting in this module and no fallback value: an item either
has an ``ℓ(x)`` record or the join is wrong, and a wrong join raises.
(``grammars/probe_scoring.py`` keeps its nullable ``ProbeScore.neg_logp`` — that
is the *probe* path, a different scope.)

The join key
------------
``item_uid`` and nothing else. Never row position, never the surface string:

* ``id`` is ``{grammar}-{split}-{index}``, a dense index into the split file as
  it was written. It identifies a LINE, not an item. Rebuild the corpus with a
  different item order and every previously written score joins onto the wrong
  record, silently, with no error anywhere.
* the surface string is not unique — 100k H train draws yielded 97,870 distinct
  strings — so joining on it merges records that are different items.

A score row with no matching ``ℓ(x)`` record raises :class:`UnmatchedScoreRows`.
There is no override flag and no silent drop, because both failure modes here
are silent by nature: the arithmetic succeeds on whatever rows did match and
produces a plausible number for a biased subset.

Ordering limitation (read this before running it on real data)
--------------------------------------------------------------
**The end-to-end path cannot be exercised yet.** Corpus meta records do not
currently carry a position-independent identifier — ``CorpusBuilder._write``
stamps ``item_uid = None`` on every corpus item, and only the structural-probe
builder assigns a real one. The corpus-side identifier is written by task A3
(persist ``ℓ(x)``, ``n_choices`` and the stable identifier into every split at
generation time, then rebuild all six splits per grammar — §10 defines six;
earlier text said five, omitting ``structural_probe_SELECT``).

So this module and its tests are written against the *contract*, on synthetic
records. Until A3 lands, :func:`analysis.entropy_baseline.load_derivations`
will raise on every on-disk sidecar, which is the intended behaviour: the
corpora are stale and their ``ℓ(x)`` rows have no stable key to join on.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

# The one admissible join key. Assigned by the generator, stable across rebuilds.
STABLE_KEY = "item_uid"

# Keys that identify a LINE rather than an item. Never join on these.
POSITIONAL_KEYS = ("id", "index", "row", "line")

SCORE_COLUMN = "neg_logq"
ELL_COLUMN = "ell_corpus"
OUT_COLUMN = "differential_surprisal"


class JoinKeyError(ValueError):
    """The stable identifier is absent, null, or not unique."""


class UnmatchedScoreRows(ValueError):
    """Score rows that no ℓ(x) record covers. Never dropped, never defaulted."""


# ─────────────────────────────────────────────────────────────────────────────
# Key validation
# ─────────────────────────────────────────────────────────────────────────────

def require_stable_key(frame: pd.DataFrame, what: str,
                       *, unique: bool = False) -> None:
    """Raise unless ``frame`` carries a usable ``item_uid`` column.

    ``unique=True`` additionally requires the key to identify at most one row,
    which the ℓ(x) side must satisfy for the join to be well defined.
    """
    if STABLE_KEY not in frame.columns:
        present = [k for k in POSITIONAL_KEYS if k in frame.columns]
        hint = (f" It carries {present!r}, which index a line in a written file, "
                "not an item; joining on those re-points every score the next "
                "time the corpus is rebuilt.") if present else ""
        raise JoinKeyError(
            f"{what} has no {STABLE_KEY!r} column.{hint} The corpus-side "
            f"identifier is written by task A3; until it runs, corpus records "
            f"carry {STABLE_KEY} = None and cannot be joined."
        )
    col = frame[STABLE_KEY]
    n_null = int(col.isna().sum())
    if n_null:
        raise JoinKeyError(
            f"{what}: {n_null}/{len(frame)} rows have a null {STABLE_KEY!r}. "
            "A null key is exactly the case task A3 fills in; it is not a "
            "value to fall back from."
        )
    if unique:
        dupes = col[col.duplicated()].unique().tolist()
        if dupes:
            raise JoinKeyError(
                f"{what}: {STABLE_KEY!r} is not unique "
                f"({len(dupes)} repeated, e.g. {dupes[:5]!r}). "
                "The join would fan out and double-count items."
            )


# ─────────────────────────────────────────────────────────────────────────────
# Duplicate items on the ℓ(x) side
# ─────────────────────────────────────────────────────────────────────────────

def collapse_duplicate_ell(ell: pd.DataFrame, ell_column: str) -> pd.DataFrame:
    """Collapse repeated ``item_uid`` rows on the ℓ(x) side. Lossless.

    The corpus contains genuine duplicate items — ~2% of train, concentrated in
    the short tail (at length 2, over half the items are repeats), because the
    space of short strings is finite and the corpus builder has no seen-set
    screen. Accepted as a property of the length distribution (decision
    2026-08-06); TRANSFORM_RULES §10 records the rates.

    A repeated ``item_uid`` means the two records agree on grammar, split,
    surface string AND derivation signature. Identical derivation signature
    implies identical ``ell_raw``; identical construction implies identical
    correction; so ``ell_corpus`` is necessarily identical too, and keeping one
    row per key discards nothing. That is what makes this collapse admissible
    where a general "drop duplicates" would not be — it is not a policy choice
    about which row wins, because the rows are the same row.

    The invariant is CHECKED, not trusted: if duplicate keys disagree on the
    value, the premise is false somewhere upstream and this raises rather than
    silently picking the first.
    """
    if STABLE_KEY not in ell.columns or ell_column not in ell.columns:
        return ell                      # let require_stable_key report it
    dup_mask = ell[STABLE_KEY].duplicated(keep=False)
    if not dup_mask.any():
        return ell
    spread = (ell[dup_mask].groupby(STABLE_KEY)[ell_column]
              .agg(lambda s: s.max() - s.min()))
    inconsistent = spread[spread > 1e-9]
    if len(inconsistent):
        raise JoinKeyError(
            f"{len(inconsistent)} repeated {STABLE_KEY!r} value(s) disagree on "
            f"{ell_column!r} (max spread {inconsistent.max():.6g}), e.g. "
            f"{inconsistent.index[:3].tolist()!r}. Identical item_uid implies an "
            "identical derivation signature and therefore an identical ℓ(x), so "
            "this cannot happen unless the uid or the correction is wrong. "
            "Refusing to pick a row."
        )
    return ell.drop_duplicates(subset=[STABLE_KEY], keep="first")


# ─────────────────────────────────────────────────────────────────────────────
# The join
# ─────────────────────────────────────────────────────────────────────────────

def join_differential_surprisal(
    scores: pd.DataFrame,
    ell: pd.DataFrame,
    *,
    score_column: str = SCORE_COLUMN,
    ell_column: str = ELL_COLUMN,
    out_column: str = OUT_COLUMN,
) -> pd.DataFrame:
    """Join model scores to ℓ(x) records on ``item_uid`` and subtract.

    Parameters
    ----------
    scores:
        One row per scored item; must carry ``item_uid`` and ``score_column``.
    ell:
        One row per corpus item; must carry ``item_uid`` and ``ell_column``,
        with ``item_uid`` unique.

    Returns
    -------
    The score frame with ``ell_column`` and ``out_column`` added, in the score
    frame's own row order.

    Raises
    ------
    JoinKeyError
        A frame is missing the stable key, or the ℓ(x) side repeats it.
    UnmatchedScoreRows
        A score row has no ℓ(x) record. Every item on this scope is grammatical
        and was produced by π_corpus, so this is a join defect — a stale corpus,
        a wrong split, a rebuild between scoring and this call — never a
        legitimately undefined value.
    """
    require_stable_key(scores, "the score frame")
    ell = collapse_duplicate_ell(ell, ell_column)
    require_stable_key(ell, "the ℓ(x) frame", unique=True)
    for frame, column, what in ((scores, score_column, "score frame"),
                                (ell, ell_column, "ℓ(x) frame")):
        if column not in frame.columns:
            raise KeyError(f"the {what} has no {column!r} column "
                           f"(has {list(frame.columns)!r})")

    merged = scores.merge(ell[[STABLE_KEY, ell_column]], on=STABLE_KEY,
                          how="left", validate="many_to_one")
    missing = merged.loc[merged[ell_column].isna(), STABLE_KEY]
    if len(missing):
        raise UnmatchedScoreRows(
            f"{len(missing)} score row(s) have no {ell_column} record, e.g. "
            f"{missing.head(5).tolist()!r}. On the corpus / in-distribution "
            "scope every item is grammatical and was produced by π_corpus, so "
            "ℓ(x) is defined for all of them: an unmatched row means the two "
            "sides describe different builds, not a missing value. Rebuild or "
            "re-score — there is no fallback."
        )
    merged[out_column] = merged[score_column] - merged[ell_column]
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Derivational unambiguity
# ─────────────────────────────────────────────────────────────────────────────
#
# ℓ(x) = −log p(x) only if each surface string has ONE derivation. If a string
# has two, ℓ(x) is the cost of the derivation that happened to be taken and is
# an upper bound on −log p(x), with slack H(D|X) (SURPRISAL_REFERENCE_POINTS
# §6.6). The sidecar records derivation *summaries*, not full derivations, so
# two derivations that differ only in an unrecorded choice share a signature —
# this detects collisions, it does not prove their absence.

def derivation_signature(derivation: Mapping[str, Any]) -> Tuple:
    """A comparable fingerprint of one ``derivation`` record."""
    by_label = derivation.get("by_label") or {}
    return (
        round(float(derivation["neg_logp"]), 9),
        int(derivation["n_draws"]),
        tuple(sorted((str(k), round(float(v), 9)) for k, v in by_label.items())),
    )


def find_derivational_collisions(
    records: Iterable[Mapping[str, Any]],
    *,
    sentence_key: str = "sentence",
    derivation_key: str = "derivation",
) -> Dict[str, Sequence[Tuple]]:
    """Surface strings carrying more than one distinct derivation signature.

    Returns ``{sentence: [signature, ...]}`` for the offending strings only, so
    an empty dict is the passing result.
    """
    seen: Dict[str, set] = {}
    for record in records:
        derivation = record.get(derivation_key)
        if derivation is None:
            raise ValueError(
                f"record {record.get(STABLE_KEY) or record.get('id')!r} has no "
                f"{derivation_key!r}; the corpora predate the tracing layer."
            )
        seen.setdefault(record[sentence_key], set()).add(
            derivation_signature(derivation))
    return {s: sorted(sigs) for s, sigs in seen.items() if len(sigs) > 1}


def assert_derivationally_unambiguous(
    records: Iterable[Mapping[str, Any]], *, what: str = "the split",
    **kwargs: Any,
) -> None:
    """Raise if any surface string carries two distinct derivation signatures."""
    collisions = find_derivational_collisions(records, **kwargs)
    if collisions:
        example = next(iter(collisions))
        raise ValueError(
            f"{what}: {len(collisions)} surface string(s) carry more than one "
            f"derivation signature, e.g. {example!r} with "
            f"{collisions[example]!r}. ℓ(x) = −log p(x) requires derivational "
            "unambiguity; where it fails, ℓ(x) is only an upper bound on "
            "−log p(x) and the slack is H(D|X)."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: ℓ(x) straight from the sidecar
# ─────────────────────────────────────────────────────────────────────────────

def load_ell(grammar: str, split: str,
             corpora_dir: Optional[Path] = None) -> pd.DataFrame:
    """``item_uid``, ``construction``, ``ell_raw``, ``correction``, ``ell_corpus``.

    Thin wrapper over :func:`analysis.entropy_baseline.ell_corpus`; raises the
    same way when the sidecar has no stable identifier (i.e. until A3 runs).
    """
    from analysis.entropy_baseline import CORPORA_DIR, ell_corpus
    return ell_corpus(grammar, split, corpora_dir or CORPORA_DIR)
