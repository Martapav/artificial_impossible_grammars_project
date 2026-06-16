"""Tests for lexicon/serialization.py."""

import json
import pytest

from lexicon import assemble_lexicon, deserialize, serialize
from lexicon.validation import LexiconIntegrityError


@pytest.fixture(scope="module")
def lexicon():
    return assemble_lexicon(seed=42)


def test_round_trip(lexicon, tmp_path):
    """serialize then deserialize must produce an equal Lexicon object."""
    path = tmp_path / "lexicon.json"
    serialize(lexicon, path)
    loaded = deserialize(path)
    assert loaded == lexicon


def test_tamper_detection(lexicon, tmp_path):
    """Modifying the serialized JSON must cause deserialization to raise LexiconIntegrityError."""
    path = tmp_path / "lexicon_tampered.json"
    serialize(lexicon, path)

    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)

    # Tamper: change the first CAT1 stem form
    original_form = data["cat1"][0]["form"]
    data["cat1"][0]["form"] = original_form + "TAMPERED"

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, sort_keys=True, indent=2)

    with pytest.raises(LexiconIntegrityError):
        deserialize(path)


def test_serialized_json_has_sorted_keys(lexicon, tmp_path):
    """The top-level keys in the output JSON must be in sorted order."""
    path = tmp_path / "lexicon.json"
    serialize(lexicon, path)
    raw = path.read_text(encoding="utf-8")
    data = json.loads(raw)
    keys = list(data.keys())
    assert keys == sorted(keys)
