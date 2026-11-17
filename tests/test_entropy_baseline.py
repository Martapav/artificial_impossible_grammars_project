"""The rejection-sampling correction from ``ℓ(x)`` to ``−log p_corpus(x)``.

The failure this guards against is silent: raw ``ℓ(x)`` looks like a perfectly
reasonable number, and a KL computed from it is wrong only by a per-construction
constant. Within a construction it cancels; across constructions it does not, and
the quota moves some labels by more than 2×. See
``docs/SURPRISAL_REFERENCE_POINTS.md`` §6.1.
"""
from __future__ import annotations

import json
import math

import pytest

from analysis.entropy_baseline import corrections, ell_corpus, load_acceptance
from grammars.base_generator import CONSTRUCTION_MIX
from grammars.tracing import AcceptanceCounter

pytest.importorskip("pandas")


# ---------------------------------------------------------------------------
# The correction itself
# ---------------------------------------------------------------------------

def test_correction_formula():
    accept = {
        "p_eligible_by_construction": {"a": 0.05, "b": 0.60},
        "corpus_proportion": {"a": 0.10, "b": 0.90},
    }
    c = corrections(accept)
    assert math.isclose(c["a"], math.log(0.05) - math.log(0.10))
    assert math.isclose(c["b"], math.log(0.60) - math.log(0.90))


def test_correction_is_zero_when_corpus_matches_nature():
    """No reweighting, no correction — the quota is the only thing moving mass."""
    accept = {"p_eligible_by_construction": {"a": 0.3, "b": 0.7},
              "corpus_proportion": {"a": 0.3, "b": 0.7}}
    assert all(abs(v) < 1e-12 for v in corrections(accept).values())


def test_upweighted_label_gets_negative_correction():
    """A label the quota over-represents is LESS surprising under p_corpus."""
    accept = {"p_eligible_by_construction": {"rare": 0.02},
              "corpus_proportion": {"rare": 0.10}}
    assert corrections(accept)["rare"] < 0


def test_old_schema_acceptance_record_is_rejected():
    """Quota-capped accept counts cannot estimate P_raw(c, len ok)."""
    stale = {"p_accept_by_construction": {"a": 0.1},
             "corpus_proportion": {"a": 0.1}}
    with pytest.raises(ValueError, match="p_eligible_by_construction"):
        corrections(stale)


def test_degenerate_record_is_rejected():
    accept = {"p_eligible_by_construction": {"a": 0.0},
              "corpus_proportion": {"a": 0.1}}
    with pytest.raises(ValueError, match="degenerate"):
        corrections(accept)


# ---------------------------------------------------------------------------
# The counter must measure the grammar, not the quota
# ---------------------------------------------------------------------------

def test_eligible_is_not_the_quota_capped_accept_count():
    """The bug this replaced: accepted[c] is pinned to quota_c by construction."""
    c = AcceptanceCounter()
    for _ in range(100):                      # eligible, but quota allows only 10
        c.note_attempt("wh", eligible=True)
    for _ in range(10):
        c.note_accept("wh")
    for _ in range(900):
        c.note_attempt("neutral", eligible=True)
    for _ in range(90):
        c.note_accept("neutral")
    s = c.summary()
    assert math.isclose(s["p_eligible_by_construction"]["wh"], 0.10)
    assert math.isclose(s["corpus_proportion"]["wh"], 0.10)
    # accepted/attempts would have said 0.01 — off by 10x
    assert s["accepted_by_construction"]["wh"] / s["attempts"] == pytest.approx(0.01)


def test_ineligible_draws_count_as_attempts_only():
    c = AcceptanceCounter()
    c.note_attempt("wh", eligible=True)
    c.note_attempt("wh", eligible=False)      # too long, or skipped
    s = c.summary()
    assert s["drawn_by_construction"]["wh"] == 2
    assert s["eligible_by_construction"]["wh"] == 1
    assert math.isclose(s["p_eligible_by_construction"]["wh"], 0.5)


# ---------------------------------------------------------------------------
# End to end on a freshly generated corpus
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A real build. ``CorpusBuilder`` now stamps ``item_uid`` itself (A3).

    This fixture used to post-process the sidecar and write a POSITIONAL uid
    (``f"{grammar}-{split}/{i}"``) as a stand-in for A3. That stand-in was
    removed on 2026-08-06: it was a re-spelling of ``id``, so it quietly became
    the de facto spec for a field whose entire purpose is to not be positional.
    The builder's content-derived digest is the spec; nothing here fills in for
    it, and if the builder stops stamping the uid these tests fail rather than
    papering over it.
    """
    from corpusgen.corpus_builder import CorpusBuilder
    from grammars.grammar_H.generate import GrammarHGenerator
    root = tmp_path_factory.mktemp("a2")
    CorpusBuilder(GrammarHGenerator(seed=42), root).build_train(
        1500, constructions=CONSTRUCTION_MIX)
    return root


def test_builder_stamps_a_content_derived_item_uid(built):
    """Not positional, not a build counter: a digest of the item's content."""
    from grammars.tracing import item_uid
    path = built / "train" / "H.meta.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    for rec in records:
        assert rec["item_uid"] == item_uid(
            grammar="H", split="train", sentence=rec["sentence"],
            derivation=rec["derivation"])
        # 32 hex chars, and no trace of the line it happened to land on.
        assert len(rec["item_uid"]) == 32
        assert rec["id"] not in rec["item_uid"]


def test_item_uid_survives_a_rebuild_in_a_different_order(built):
    """The property the positional and build-counter forms both failed."""
    from grammars.tracing import item_uid
    path = built / "train" / "H.meta.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    for rec in reversed(records):
        # Re-derived from content alone — different index, different
        # neighbouring context — the key is unchanged.
        assert item_uid(grammar="H", split="train", sentence=rec["sentence"],
                        derivation=rec["derivation"]) == rec["item_uid"]


def test_repeated_uids_are_genuine_duplicates_and_are_flagged(built):
    """A repeat is a duplicate ITEM, not a digest accident — so it is flagged.

    The digest covers grammar, split, surface string AND derivation signature,
    so two records sharing one agree on all four. At 128 bits over corpus-scale
    n a hash accident is not the explanation. Policy (A3): flag per item,
    continue, report the rate — never merge, never drop.
    """
    from collections import Counter
    from grammars.tracing import derivation_signature
    path = built / "train" / "H.meta.jsonl"
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    counts = Counter(r["item_uid"] for r in records)
    repeated = [u for u, n in counts.items() if n > 1]
    for uid in repeated:
        group = [r for r in records if r["item_uid"] == uid]
        assert len({g["sentence"] for g in group}) == 1
        assert len({derivation_signature(g["derivation"]) for g in group}) == 1
        # Every record after the first carries the back-reference.
        assert sum("uid_duplicate_of" in g for g in group) == len(group) - 1
    # No item is dropped to make the keys unique.
    assert len(records) == 1500


def test_corrected_means_shift_by_exactly_the_correction(built):
    """The plan's acceptance criterion for A2."""
    df = ell_corpus("H", "train", built)
    g = df.groupby("construction").agg(
        raw=("ell_raw", "mean"), corrected=("ell_corpus", "mean"),
        correction=("correction", "first"))
    shift = g["corrected"] - g["raw"]
    assert ((shift - g["correction"]).abs() < 1e-9).all()


def test_correction_cancels_within_a_construction(built):
    """Within-construction contrasts are identical raw or corrected."""
    df = ell_corpus("H", "train", built)
    for _, sub in df.groupby("construction"):
        if len(sub) < 2:
            continue
        raw_spread = sub.ell_raw.max() - sub.ell_raw.min()
        cor_spread = sub.ell_corpus.max() - sub.ell_corpus.min()
        assert math.isclose(raw_spread, cor_spread, rel_tol=1e-9)


def test_correction_does_not_cancel_across_constructions(built):
    """If it did, A2 would be unnecessary; the quota really does move mass."""
    df = ell_corpus("H", "train", built)
    corr = df.groupby("construction").correction.first()
    assert corr.max() - corr.min() > 0.1        # nats


def test_realized_corpus_proportions_match_the_quota(built):
    accept = load_acceptance("H", "train", built)
    prop = accept["corpus_proportion"]
    assert math.isclose(sum(prop.values()), 1.0, rel_tol=1e-9)
    for c, want in CONSTRUCTION_MIX.items():
        assert math.isclose(prop[c], want, abs_tol=1e-6), c


def test_ell_is_finite_and_positive(built):
    df = ell_corpus("H", "train", built)
    assert df.ell_raw.gt(0).all()
    assert df.ell_corpus.notna().all()
    assert (df.ell_corpus.abs() < math.inf).all()


def test_missing_derivation_field_is_a_clear_error(built, tmp_path):
    """Corpora predating the tracing layer must fail loudly, not silently."""
    src = built / "train" / "H.meta.jsonl"
    dst = tmp_path / "train"
    dst.mkdir(parents=True)
    with (dst / "H.meta.jsonl").open("w") as out:
        for line in src.open():
            rec = json.loads(line)
            rec.pop("derivation", None)
            out.write(json.dumps(rec) + "\n")
    from analysis.entropy_baseline import load_derivations
    with pytest.raises(ValueError, match="no 'derivation' field"):
        load_derivations("H", "train", tmp_path)
