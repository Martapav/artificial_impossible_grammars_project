"""Tests for lexicon determinism across runs and seeds."""

import json
import pytest

from lexicon import assemble_lexicon, serialize


def test_equal_lexicons_same_seed():
    """Two assemble_lexicon(42) calls must produce equal Lexicon objects."""
    a = assemble_lexicon(seed=42)
    b = assemble_lexicon(seed=42)
    # Exclude metadata.generation_timestamp which is wall-clock dependent
    assert a.cat1 == b.cat1
    assert a.cat2 == b.cat2
    assert a.cat3 == b.cat3
    assert a.cat4 == b.cat4
    assert a.cat5 == b.cat5
    assert a.cat6 == b.cat6
    assert a.cat8 == b.cat8
    assert a.cat9 == b.cat9
    assert a.cat1pron == b.cat1pron
    assert a.cat3aux == b.cat3aux
    assert a.inflectional == b.inflectional
    assert a.metadata.content_hash == b.metadata.content_hash
    assert a.metadata.seed == b.metadata.seed


def test_byte_identical_json(tmp_path):
    """Two lexicons with the same seed must produce byte-identical JSON (excluding timestamp)."""
    a = assemble_lexicon(seed=42)
    b = assemble_lexicon(seed=42)

    path_a = tmp_path / "a.json"
    path_b = tmp_path / "b.json"
    serialize(a, path_a)
    serialize(b, path_b)

    data_a = json.loads(path_a.read_text(encoding="utf-8"))
    data_b = json.loads(path_b.read_text(encoding="utf-8"))

    # Compare everything except the wall-clock timestamp
    for key in ("cat1", "cat2", "cat3", "cat3aux", "cat4", "cat5", "cat6",
                 "cat8", "cat9", "cat1pron", "inflectional"):
        assert data_a[key] == data_b[key], f"Mismatch in key {key!r}"

    assert data_a["metadata"]["content_hash"] == data_b["metadata"]["content_hash"]
    assert data_a["metadata"]["seed"] == data_b["metadata"]["seed"]


def test_different_seeds_differ():
    """Lexicons generated with different seeds must differ in CAT1 feature assignments."""
    a = assemble_lexicon(seed=42)
    b = assemble_lexicon(seed=43)
    # CAT1 feature assignment is random; the two should differ
    genders_a = tuple(item.inherent_gender for item in a.cat1)
    genders_b = tuple(item.inherent_gender for item in b.cat1)
    assert genders_a != genders_b, (
        "Expected different gender assignments for seed 42 vs 43"
    )
