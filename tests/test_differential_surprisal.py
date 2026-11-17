"""A5 — tests for the ℓ(x) join (A4) and the identifier contract it rests on.

Two of A5's five clauses are already pinned elsewhere, on the real generators
rather than on synthetic frames, and are not duplicated here:

  * ``ℓ(x) > 0`` and ``n_choices > 0`` on every grammatical corpus item —
    ``tests/test_scoring.py::test_grammatical_corpus_items_have_positive_ell_and_choices``
  * round-trip consistency, scoring a corpus-generated derivation reproduces the
    traced value — ``tests/test_scoring.py::test_roundtrip_reproduces_traced_ell_exactly``
    (and ``..._per_label_attribution``, which pins the same thing site by site).

What is new here: the join raises on an unmatched score row, the
derivational-unambiguity check triggers on a constructed collision, and
``entropy_baseline`` refuses a positional-key join.

**Synthetic records throughout, deliberately.** Corpus meta records do not yet
carry ``item_uid`` — it is written by task A3, which has not run — so the
end-to-end path on real corpus data cannot be exercised. These tests are written
against the contract A3 must satisfy.

There are no corpus-side null tests. ℓ(x)'s scope is the corpus and
in-distribution splits; every item there is grammatical and was produced by
π_corpus, so ``ell_corpus`` can never be null on that scope and there is nothing
to null. (The nullable path lives in ``grammars/probe_scoring.py``, a different
scope, and is covered by ``tests/test_probe_scoring.py``.)
"""
from __future__ import annotations

import json

import pytest

pd = pytest.importorskip("pandas")

from analysis.differential_surprisal import (  # noqa: E402
    JoinKeyError,
    STABLE_KEY,
    UnmatchedScoreRows,
    assert_derivationally_unambiguous,
    find_derivational_collisions,
    join_differential_surprisal,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic records shaped like the real ones
# ---------------------------------------------------------------------------

def _scores(uids, neg_logq=None):
    neg_logq = neg_logq or [10.0 + i for i in range(len(uids))]
    return pd.DataFrame({STABLE_KEY: list(uids), "neg_logq": neg_logq})


def _ell(uids, ell=None):
    ell = ell or [1.0 + i for i in range(len(uids))]
    return pd.DataFrame({STABLE_KEY: list(uids), "ell_corpus": ell,
                         "construction": ["neutral"] * len(uids)})


def _derivation(neg_logp, n_draws, by_label=None):
    return {"neg_logp": neg_logp, "n_draws": n_draws,
            "by_label": by_label or {"phenomenon": neg_logp}}


# ---------------------------------------------------------------------------
# The join
# ---------------------------------------------------------------------------

def test_join_subtracts_ell_from_neg_logq():
    out = join_differential_surprisal(
        _scores(["a", "b"], [10.0, 20.0]), _ell(["a", "b"], [1.0, 4.0]))
    assert out["differential_surprisal"].tolist() == [9.0, 16.0]


def test_join_is_by_identifier_not_by_row_position():
    """The whole point of A4: reorder the ℓ(x) side and nothing moves."""
    scores = _scores(["a", "b", "c"], [10.0, 20.0, 30.0])
    ell = _ell(["a", "b", "c"], [1.0, 2.0, 3.0])
    straight = join_differential_surprisal(scores, ell)
    shuffled = join_differential_surprisal(scores, ell.iloc[::-1].reset_index(drop=True))
    assert straight["differential_surprisal"].tolist() == [9.0, 18.0, 27.0]
    assert (straight["differential_surprisal"].tolist()
            == shuffled["differential_surprisal"].tolist())


def test_join_raises_on_an_unmatched_score_row():
    """A5: no silent drop, no fallback value, no override flag."""
    with pytest.raises(UnmatchedScoreRows, match="no ell_corpus record"):
        join_differential_surprisal(_scores(["a", "b", "ghost"]), _ell(["a", "b"]))


def test_unmatched_error_names_the_offending_items():
    with pytest.raises(UnmatchedScoreRows, match="ghost"):
        join_differential_surprisal(_scores(["a", "ghost"]), _ell(["a"]))


def test_join_has_no_override_flag():
    """A dropped row is the failure mode this raises to prevent."""
    import inspect
    params = inspect.signature(join_differential_surprisal).parameters
    assert not any(p in params for p in ("force", "how", "drop_unmatched",
                                         "allow_missing", "fillna", "default"))


def test_join_rejects_a_frame_with_only_a_positional_key():
    scores = pd.DataFrame({"id": ["H-train-00000"], "neg_logq": [10.0]})
    with pytest.raises(JoinKeyError, match=STABLE_KEY):
        join_differential_surprisal(scores, _ell(["a"]))


def test_join_rejects_a_null_identifier():
    scores = _scores(["a", "b"])
    scores.loc[1, STABLE_KEY] = None
    with pytest.raises(JoinKeyError, match="null"):
        join_differential_surprisal(scores, _ell(["a", "b"]))


def test_duplicate_ell_rows_collapse_losslessly():
    """The corpus has ~2% genuine duplicate items; identical uid => identical ℓ.

    Collapsing is admissible precisely because the rows are the same row — same
    surface string, same derivation signature — so there is no choice being made
    about which one wins.
    """
    ell = _ell(["a", "a", "b"], [1.0, 1.0, 2.0])
    out = join_differential_surprisal(_scores(["a", "b"], [10.0, 20.0]), ell)
    assert out["differential_surprisal"].tolist() == [9.0, 18.0]
    assert len(out) == 2                    # no fan-out


def test_duplicate_ell_rows_that_disagree_still_raise():
    """If the premise fails, refuse to pick a row rather than silently guess."""
    ell = _ell(["a", "a"], [1.0, 7.5])
    with pytest.raises(JoinKeyError, match="disagree"):
        join_differential_surprisal(_scores(["a"]), ell)


def test_join_preserves_the_score_frames_own_row_order():
    scores = _scores(["c", "a", "b"], [30.0, 10.0, 20.0])
    out = join_differential_surprisal(scores, _ell(["a", "b", "c"], [1.0, 2.0, 3.0]))
    assert out[STABLE_KEY].tolist() == ["c", "a", "b"]
    assert out["differential_surprisal"].tolist() == [27.0, 9.0, 18.0]


def test_join_raises_when_the_value_column_is_absent():
    with pytest.raises(KeyError, match="neg_logq"):
        join_differential_surprisal(
            pd.DataFrame({STABLE_KEY: ["a"], "logq": [1.0]}), _ell(["a"]))


# ---------------------------------------------------------------------------
# Derivational unambiguity
# ---------------------------------------------------------------------------

def test_unambiguity_check_triggers_on_a_constructed_collision():
    """A5: one surface string, two distinct derivation signatures."""
    records = [
        {STABLE_KEY: "u1", "sentence": "a b c", "derivation": _derivation(5.0, 7)},
        {STABLE_KEY: "u2", "sentence": "a b c", "derivation": _derivation(6.5, 9)},
        {STABLE_KEY: "u3", "sentence": "d e f", "derivation": _derivation(5.0, 7)},
    ]
    collisions = find_derivational_collisions(records)
    assert list(collisions) == ["a b c"]
    assert len(collisions["a b c"]) == 2
    with pytest.raises(ValueError, match="more than one"):
        assert_derivationally_unambiguous(records)


def test_identical_derivations_of_one_string_are_not_a_collision():
    """Two items may share a string; only differing DERIVATIONS are ambiguity."""
    records = [
        {STABLE_KEY: "u1", "sentence": "a b c", "derivation": _derivation(5.0, 7)},
        {STABLE_KEY: "u2", "sentence": "a b c", "derivation": _derivation(5.0, 7)},
    ]
    assert find_derivational_collisions(records) == {}
    assert_derivationally_unambiguous(records)


def test_collision_visible_only_in_the_label_attribution_is_caught():
    """Same total and draw count, different sites — still two derivations."""
    records = [
        {STABLE_KEY: "u1", "sentence": "a b c",
         "derivation": _derivation(5.0, 7, {"phenomenon": 5.0})},
        {STABLE_KEY: "u2", "sentence": "a b c",
         "derivation": _derivation(5.0, 7, {"insitu_aux": 5.0})},
    ]
    assert list(find_derivational_collisions(records)) == ["a b c"]


def test_unambiguity_check_rejects_a_record_with_no_derivation():
    records = [{STABLE_KEY: "u1", "sentence": "a b c", "derivation": None}]
    with pytest.raises(ValueError, match="tracing layer"):
        find_derivational_collisions(records)


# ---------------------------------------------------------------------------
# entropy_baseline refuses a positional-key join
# ---------------------------------------------------------------------------

def _write_sidecar(tmp_path, split, grammar, records):
    d = tmp_path / split
    d.mkdir(parents=True, exist_ok=True)
    with (d / f"{grammar}.meta.jsonl").open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    return tmp_path


def _meta(i, uid="present"):
    rec = {"id": f"H-validation-{i:05d}", "grammar_type": "H",
           "construction": "neutral", "sentence": f"s{i}",
           "derivation": _derivation(40.0 + i, 30)}
    if uid == "present":
        rec[STABLE_KEY] = f"H-validation/{i}"
    elif uid == "null":
        rec[STABLE_KEY] = None
    return rec


def test_entropy_baseline_fails_when_handed_a_positional_key_join(tmp_path):
    """A5: 'id' indexes a line, not an item — it is not an acceptable key."""
    from analysis.entropy_baseline import load_derivations
    root = _write_sidecar(tmp_path, "validation", "H",
                          [_meta(0, uid="absent"), _meta(1, uid="absent")])
    with pytest.raises(ValueError, match=STABLE_KEY):
        load_derivations("H", "validation", root)


def test_entropy_baseline_fails_on_a_null_stable_identifier(tmp_path):
    """The current on-disk state: CorpusBuilder stamps item_uid = None."""
    from analysis.entropy_baseline import load_derivations
    root = _write_sidecar(tmp_path, "validation", "H",
                          [_meta(0, uid="null")])
    with pytest.raises(ValueError, match="A3"):
        load_derivations("H", "validation", root)


def test_entropy_baseline_keys_rows_on_the_stable_identifier(tmp_path):
    """What A3 must produce for the join to work at all."""
    from analysis.entropy_baseline import load_derivations
    root = _write_sidecar(tmp_path, "validation", "H", [_meta(0), _meta(1)])
    df = load_derivations("H", "validation", root)
    assert STABLE_KEY in df.columns
    assert df[STABLE_KEY].tolist() == ["H-validation/0", "H-validation/1"]
    assert df[STABLE_KEY].is_unique


def test_ell_corpus_output_joins_against_scores(tmp_path):
    """End-to-end on synthetic records: the shape A3 has to deliver."""
    from analysis.entropy_baseline import ell_corpus
    root = _write_sidecar(tmp_path, "validation", "H", [_meta(0), _meta(1)])
    (root / "validation" / "H.accept.json").write_text(json.dumps({
        "p_eligible_by_construction": {"neutral": 0.6},
        "corpus_proportion": {"neutral": 1.0},
    }), encoding="utf-8")
    ell = ell_corpus("H", "validation", root)
    scores = _scores(ell[STABLE_KEY].tolist(), [100.0, 200.0])
    out = join_differential_surprisal(scores, ell)
    assert out["differential_surprisal"].tolist() == [
        100.0 - ell.ell_corpus.iloc[0], 200.0 - ell.ell_corpus.iloc[1]]
