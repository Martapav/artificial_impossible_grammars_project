"""Structural tests for the in-distribution 4-bucket probe builder.

Pins the bucket balance, the wh clause-(i) verdict/divergence structure,
the neutral sub-bucket quotas, and tokenizer-cleanliness of every probe sentence.
"""

from collections import Counter

import pytest

from corpora.probes import build_probe_buckets
from models.dataload import check_alignment

SEED = 42 + 300_000
MIN, MAX = 2, 25

_CACHE = {}


def _rows(grammar):
    if grammar not in _CACHE:
        _CACHE[grammar] = build_probe_buckets(grammar, "test_indistribution", SEED, MIN, MAX)
    return _CACHE[grammar]


@pytest.mark.parametrize("grammar", ["H", "P", "Hprime", "Lprime"])
def test_bucket_balance_and_tokenization(grammar):
    rows = _rows(grammar)
    assert len(rows) == 400
    assert Counter(r["bucket"] for r in rows) == {
        "binding": 100, "aux": 100, "wh": 100, "neutral": 100,
    }
    # Neutral sub-buckets 33/33/34.
    assert Counter(r["blocked_phenomenon"] for r in rows if r["bucket"] == "neutral") == {
        "binding": 33, "aux": 33, "wh": 34,
    }
    # Length window respected; every sentence tokenizes cleanly.
    assert all(MIN <= r["length"] <= MAX for r in rows)
    assert check_alignment([r["sentence"] for r in rows]) == []


@pytest.mark.parametrize("grammar", ["H", "P"])
def test_pure_grammars_wh_has_no_verdicts(grammar):
    # Verdicts (H'/L') are a mixed-grammar concept; the pure grammars carry none.
    wh = [r for r in _rows(grammar) if r["bucket"] == "wh"]
    assert len(wh) == 100
    assert all(r["construction"] == "wh_movement" for r in wh)
    assert all(r["divergence"] is None and r["hprime_verdict"] is None
               and r["lprime_verdict"] is None for r in wh)


def test_hprime_wh_divergence_structure():
    wh = [r for r in _rows("Hprime") if r["bucket"] == "wh"]
    div = [r for r in wh if r["divergence"]]
    coh = [r for r in wh if not r["divergence"]]
    assert len(div) == 50 and len(coh) == 50
    # An H'-fronted target was licensed by H''s own set (3rd position or
    # Type3-under-Type4), so the only reachable divergence direction within
    # H' items is H'-licenses / L'-not (a matrix object is never inside a
    # Type3, and an unlicensed-by-both target is never fronted).
    assert all((r["hprime_verdict"], r["lprime_verdict"]) == ("license", "not")
               for r in div)
    assert all(r["hprime_verdict"] == r["lprime_verdict"] for r in coh)


def test_lprime_wh_divergence_structure():
    wh = [r for r in _rows("Lprime") if r["bucket"] == "wh"]
    div = [r for r in wh if r["divergence"]]
    coh = [r for r in wh if not r["divergence"]]
    assert len(div) == 50 and len(coh) == 50
    # L' items reach both divergence directions: the bare-transitive matrix
    # object (L'-licenses / H'-not) and a post-CAT5 non-object nominal at
    # ordinal 3 (H'-licenses / L'-not).
    assert all(r["hprime_verdict"] != r["lprime_verdict"] for r in div)
    assert all(r["hprime_verdict"] == r["lprime_verdict"] for r in coh)
    directions = Counter((r["hprime_verdict"], r["lprime_verdict"]) for r in div)
    assert ("not", "license") in directions
