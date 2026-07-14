"""Tests for lexicon/serialization.py."""

import json
import pytest

from lexicon import assemble_lexicon, serialize


@pytest.fixture(scope="module")
def lexicon():
    return assemble_lexicon(seed=42)




def test_serialized_json_has_sorted_keys(lexicon, tmp_path):
    """The top-level keys in the output JSON must be in sorted order."""
    path = tmp_path / "lexicon.json"
    serialize(lexicon, path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    keys = list(data.keys())
    assert keys == sorted(keys)
