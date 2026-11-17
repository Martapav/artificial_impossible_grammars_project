"""Guards that keep one corpus generation from being mistaken for another.

Every join key in this project is positional, so artifacts from two generations
join cleanly and produce silently wrong numbers. These are the four places that
now make that impossible; each test names the failure it prevents.

Context: `evaluation/provenance.py` and `models/dataload.require_set_scoped`.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dataload import (compare_provenance, require_set_scoped,  # noqa: E402
                             resolve_corpora_root, set_scoped)
from scripts.build_all_corpora import guard_seed_unique, seeds_in_use  # noqa: E402


def make_set(base: Path, name: str, *, seed: int | None = None) -> Path:
    """A directory that `is_split_root` accepts, optionally with a manifest."""
    root = base / name
    (root / "train").mkdir(parents=True)
    if seed is not None:
        (root / "manifest.json").write_text(json.dumps({
            "corpus_set": name, "builds": [{"seed": seed}], "files": {}}))
    return root


# --- the shadowing failure (2026-08-05) ------------------------------------

def test_named_set_wins_over_a_flat_corpus_at_the_parent(tmp_path):
    """A flat `base/train/` must not shadow an explicitly named `base/set_1/`.

    This is the failure that cost a pilot: the sweep asked for set_1, a leftover
    flat corpus sat at the parent, and the old rule 1 returned it without ever
    reading THESIS_CORPUS_SET.
    """
    (tmp_path / "train").mkdir()               # the stale flat corpus
    make_set(tmp_path, "set_1")                # the set that was shipped
    assert resolve_corpora_root(tmp_path, "set_1") == tmp_path / "set_1"


def test_named_but_missing_set_refuses_to_fall_back_to_the_flat_corpus(tmp_path):
    """Asking for a set that is absent is an error, never "use that other one"."""
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError, match="named explicitly"):
        resolve_corpora_root(tmp_path, "set_9")


def test_root_pointed_at_the_set_itself_with_the_set_also_named(tmp_path):
    """`THESIS_CORPORA_ROOT=<...>/set_2` + `THESIS_CORPUS_SET=set_2`.

    A supported pattern, and a regression: the first version of the
    explicit-set rule looked only for `base/<set>` and resolved this to
    `<...>/set_2/set_2` — a path that does not exist — while a guard comparing
    the final component still called it OK.
    """
    root = make_set(tmp_path, "set_2")
    assert resolve_corpora_root(root, "set_2") == root


def test_root_pointed_at_a_DIFFERENT_set_than_the_one_named(tmp_path):
    """Naming set_2 while the root IS set_1 must raise, not silently use set_1."""
    root = make_set(tmp_path, "set_1")
    with pytest.raises(FileNotFoundError, match="named explicitly"):
        resolve_corpora_root(root, "set_2")


def test_flat_root_still_works_when_no_set_is_named(tmp_path, monkeypatch):
    """Pre-set-layout corpora and per-split symlinks keep resolving."""
    monkeypatch.delenv("THESIS_CORPUS_SET", raising=False)
    monkeypatch.setattr("models.dataload.CORPUS_SET_IS_EXPLICIT", False)
    (tmp_path / "train").mkdir()
    assert resolve_corpora_root(tmp_path) == tmp_path


# --- the duplicate-set failure ---------------------------------------------

def test_seed_reuse_across_sets_is_refused(tmp_path):
    """`--set set_2` at set_1's seed reproduces set_1 byte for byte."""
    make_set(tmp_path, "set_1", seed=42)
    assert seeds_in_use(tmp_path) == {42: "set_1"}
    with pytest.raises(SystemExit, match="already used to build 'set_1'"):
        guard_seed_unique(tmp_path / "set_2", 42, base=tmp_path)


def test_a_set_may_reuse_its_own_seed_across_invocations(tmp_path):
    """Sets are filled one grammar at a time; only a SIBLING clash is fatal."""
    out = make_set(tmp_path, "set_1", seed=42)
    guard_seed_unique(out, 42, base=tmp_path)          # must not raise


def test_a_fresh_seed_is_accepted(tmp_path):
    make_set(tmp_path, "set_1", seed=42)
    guard_seed_unique(tmp_path / "set_2", 43, base=tmp_path)


# --- derived artifacts must name their generation --------------------------

def test_unscoped_output_path_is_refused(tmp_path):
    root = make_set(tmp_path, "set_1")
    with pytest.raises(SystemExit, match="not addressed by corpus set"):
        require_set_scoped(Path("results/eval/ngram_margins.parquet"), root)


def test_scoped_output_path_passes_and_set_scoped_builds_one(tmp_path):
    root = make_set(tmp_path, "set_1")
    scoped = set_scoped(Path("results/eval"), root) / "ngram_margins.parquet"
    assert scoped == Path("results/eval/set_1/ngram_margins.parquet")
    assert require_set_scoped(scoped, root) == scoped


# --- provenance comparison --------------------------------------------------

def test_provenance_agreement_and_disagreement():
    a = {"corpus_set": "set_1", "files": {"train/H.jsonl": "aa"}}
    assert compare_provenance(a, a) == []

    b = {"corpus_set": "set_1", "files": {"train/H.jsonl": "bb"}}
    assert any("train/H.jsonl" in p for p in compare_provenance(a, b))

    c = {"corpus_set": "set_2", "files": {"train/H.jsonl": "aa"}}
    assert any("corpus set differs" in p for p in compare_provenance(a, c))


def test_provenance_with_no_shared_file_is_not_treated_as_agreement():
    """Two records that overlap in nothing prove nothing, and must say so."""
    a = {"corpus_set": "set_1", "files": {"train/H.jsonl": "aa"}}
    b = {"corpus_set": "set_1", "files": {"train/P.jsonl": "aa"}}
    assert any("no corpus file in common" in p for p in compare_provenance(a, b))


def test_provenance_survives_a_parquet_round_trip(tmp_path):
    pd = pytest.importorskip("pandas")
    from evaluation.provenance import read_provenance, write_parquet

    frame = pd.DataFrame({"pair_id": ["p0", "p1"], "margin_b1": [0.0, 1.5]})
    record = {"corpus_set": "set_1", "corpus_root": "/x",
              "files": {"train/H.jsonl": "aa"}}
    path = tmp_path / "margins.parquet"
    write_parquet(frame, path, record)

    assert read_provenance(path) == record
    pd.testing.assert_frame_equal(pd.read_parquet(path), frame)


def test_unstamped_parquet_reports_none_rather_than_agreement(tmp_path):
    pd = pytest.importorskip("pandas")
    from evaluation.provenance import read_provenance

    path = tmp_path / "plain.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(path, index=False)
    assert read_provenance(path) is None
