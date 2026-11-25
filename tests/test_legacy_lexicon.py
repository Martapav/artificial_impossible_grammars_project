"""The frozen v0.2.0 generator (lexicon/legacy_v020.py) must keep reproducing
lexicon.json — the lexicon every corpus up to set_3 was built with."""
import json
from pathlib import Path

from lexicon.legacy_v020 import assemble_lexicon
from lexicon.serialization import _dataclass_to_dict

REF = Path(__file__).resolve().parent.parent / "lexicon.json"


def test_legacy_generator_reproduces_lexicon_json():
    ref = json.loads(REF.read_text(encoding="utf-8"))
    new = _dataclass_to_dict(assemble_lexicon(seed=ref["metadata"]["seed"]))
    assert set(new) == set(ref)
    for key in ref:
        if key == "metadata":
            continue
        assert new[key] == ref[key], key
    for key in ref["metadata"]:
        if key != "generation_timestamp":
            assert new["metadata"][key] == ref["metadata"][key], key


def test_legacy_is_independent_of_the_live_config():
    import lexicon.legacy_v020 as legacy
    src = Path(legacy.__file__).read_text(encoding="utf-8")
    assert "from .config" not in src and "import config" not in src
    assert legacy.VERSION == "0.2.0"
