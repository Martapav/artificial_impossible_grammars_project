"""Tests for the probe disjointness guard (corpusgen/probe_checks.py).

Pins: the guard passes on a real three-probe build; it fires on a pair
duplicated within one probe, on a pair shared across probes, and on a merely
shared grammatical member; missing probe files are skipped, not failed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from corpusgen.corpus_builder import CorpusBuilder
from corpusgen.probe_checks import (PROBE_SPLITS, ProbeOverlapError,
                                  check_probe_disjointness)
from grammars.grammar_H.generate import GrammarHGenerator
from grammars.grammar_P.generate import GrammarPGenerator

GEN_CLASSES = {"H": GrammarHGenerator, "P": GrammarPGenerator}
N = 3          # pairs per subcase in the reporting probes
N_SELECT = 5   # pairs per subcase in SELECT (larger, as in the real build)


@pytest.fixture(scope="module")
def built_root(tmp_path_factory) -> Path:
    """A real build of all three probes for two grammars."""
    root = tmp_path_factory.mktemp("probe_check_corpora")
    for g, cls in GEN_CLASSES.items():
        b = CorpusBuilder(cls(seed=42), root)
        b.build_structural_probe_iod(n_per_subcase=N)
        b.build_structural_probe_ood(n_per_subcase=N)
        b.build_structural_probe_select(n_per_subcase=N_SELECT)
    return root


@pytest.mark.parametrize("grammar", sorted(GEN_CLASSES))
def test_real_build_is_disjoint(built_root, grammar):
    counts = check_probe_disjointness(built_root, grammar)
    assert set(counts) == set(PROBE_SPLITS)
    assert counts["structural_probe_IOD"] > 0
    assert counts["structural_probe_SELECT"] > counts["structural_probe_IOD"]


def _read(path: Path):
    return [json.loads(l) for l in path.open(encoding="utf-8")]


def _write(path: Path, records):
    with path.open("w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def _clone(built_root: Path, tmp_path: Path, grammar: str) -> Path:
    """Copy one grammar's three probe files into a scratch root."""
    for probe in PROBE_SPLITS:
        d = tmp_path / probe
        d.mkdir(parents=True, exist_ok=True)
        src = built_root / probe / f"{grammar}.jsonl"
        _write(d / f"{grammar}.jsonl", _read(src))
    return tmp_path


def test_duplicate_pair_within_a_probe_raises(built_root, tmp_path):
    root = _clone(built_root, tmp_path, "H")
    path = root / "structural_probe_SELECT" / "H.jsonl"
    recs = _read(path)
    # Re-emit the first pair's two members under a fresh pair_id.
    first = [r for r in recs if r["pair_id"] == recs[0]["pair_id"]]
    for r in first:
        dup = dict(r, pair_id="H-sp-99999", id="H-structural_probe_SELECT-99999")
        recs.append(dup)
    _write(path, recs)

    with pytest.raises(ProbeOverlapError, match="more than once"):
        check_probe_disjointness(root, "H")


def test_pair_shared_across_probes_raises(built_root, tmp_path):
    root = _clone(built_root, tmp_path, "H")
    iod = _read(root / "structural_probe_IOD" / "H.jsonl")
    sel_path = root / "structural_probe_SELECT" / "H.jsonl"
    recs = _read(sel_path)
    leaked = [r for r in iod if r["pair_id"] == iod[0]["pair_id"]]
    for r in leaked:
        recs.append(dict(r, pair_id="H-sp-88888",
                         id="H-structural_probe_SELECT-88888"))
    _write(sel_path, recs)

    with pytest.raises(ProbeOverlapError, match="identical pair"):
        check_probe_disjointness(root, "H")


def test_shared_grammatical_member_alone_raises(built_root, tmp_path):
    """Half a shared item still couples selection to reporting."""
    root = _clone(built_root, tmp_path, "H")
    iod = _read(root / "structural_probe_IOD" / "H.jsonl")
    iod_gram = next(r for r in iod if r["variant"] == "grammatical")
    sel_path = root / "structural_probe_SELECT" / "H.jsonl"
    recs = _read(sel_path)
    # Overwrite one SELECT grammatical member; its partner stays untouched, so
    # the full pair differs and only the grammatical-member check can catch it.
    target = next(r for r in recs if r["variant"] == "grammatical")
    target["sentence"] = iod_gram["sentence"]
    _write(sel_path, recs)

    with pytest.raises(ProbeOverlapError, match="shared grammatical member"):
        check_probe_disjointness(root, "H")


def test_missing_probe_file_is_skipped(built_root, tmp_path):
    root = _clone(built_root, tmp_path, "H")
    (root / "structural_probe_OOD" / "H.jsonl").unlink()
    counts = check_probe_disjointness(root, "H")
    assert "structural_probe_OOD" not in counts
    assert "structural_probe_SELECT" in counts


# --- builder-side dedup (what keeps the guard above green) ------------------

def test_pairs_within_a_probe_are_unique():
    """The short tail of the length distribution is a small finite space, so
    the builder must reject repeats rather than trust the seed."""
    from corpusgen.structural_probes import build_structural_probe

    items = build_structural_probe("H", seed=42 + 200_000, min_length=2,
                                   max_length=25, n_per_subcase=20)
    pairs = [(items[i]["sentence"], items[i + 1]["sentence"])
             for i in range(0, len(items), 2)]
    assert len(set(pairs)) == len(pairs)


def test_exclude_grammatical_is_honoured():
    """Same seed, so without the exclusion the two builds would be identical —
    which makes any shared member proof the exclusion did nothing."""
    from corpusgen.structural_probes import build_structural_probe

    kw = dict(grammar="H", seed=42 + 200_000, min_length=2, max_length=25,
              n_per_subcase=10)
    first = build_structural_probe(**kw)
    members = {i["sentence"] for i in first if i["variant"] == "grammatical"}

    second = build_structural_probe(**kw, exclude_grammatical=members)
    assert len(second) == len(first)          # quota still met by resampling
    assert not members & {i["sentence"] for i in second
                          if i["variant"] == "grammatical"}


def test_select_excludes_reporting_probes_on_disk(tmp_path):
    """CorpusBuilder must wire the exclusion from the written IOD/OOD files."""
    root = tmp_path / "corpora"
    b = CorpusBuilder(GEN_CLASSES["H"](seed=42), root)
    b.build_structural_probe_iod(n_per_subcase=10)
    b.build_structural_probe_select(n_per_subcase=10)
    assert check_probe_disjointness(root, "H")


def test_half_a_pair_raises(built_root, tmp_path):
    root = _clone(built_root, tmp_path, "H")
    path = root / "structural_probe_IOD" / "H.jsonl"
    recs = _read(path)
    dropped = recs[0]["pair_id"]
    _write(path, [r for r in recs
                  if not (r["pair_id"] == dropped and r["variant"] == "ungrammatical")])

    with pytest.raises(ProbeOverlapError, match="missing its"):
        check_probe_disjointness(root, "H")
