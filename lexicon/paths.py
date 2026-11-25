"""Which lexicon file is in force — one resolver for the whole codebase.

A corpus set is only meaningful with the lexicon that built it (vocabulary,
stems, affixes). Resolution order:

1. ``THESIS_LEXICON`` in the environment (absolute, or relative to the repo root);
2. the ``lexicon_path`` recorded in ``corpora/<THESIS_CORPUS_SET>/manifest.json``,
   when ``THESIS_CORPUS_SET`` is set and the manifest records one (set_4 onward);
3. the repo-root ``lexicon.json`` (lexicon v0.2.0, which built set_3 and earlier).

``scripts/build_all_corpora.py --lexicon PATH`` sets (1) for the build and records
the path + sha256 in the new set's manifest, so training/scoring on that set later
resolves it through (2) without any flag.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
DEFAULT_LEXICON_PATH: Path = REPO_ROOT / "lexicon.json"
ENV_LEXICON = "THESIS_LEXICON"
ENV_CORPUS_SET = "THESIS_CORPUS_SET"


def _abs(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def manifest_lexicon_path(corpus_set: str) -> Path | None:
    """The lexicon a corpus set records in its manifest, or None (pre-set_4)."""
    manifest = REPO_ROOT / "corpora" / corpus_set / "manifest.json"
    if not manifest.is_file():
        return None
    recorded = json.loads(manifest.read_text(encoding="utf-8")).get("lexicon_path")
    return _abs(recorded) if recorded else None


def resolve_lexicon_path() -> Path:
    """The lexicon file in force (see module docstring for the order)."""
    env = os.environ.get(ENV_LEXICON)
    if env:
        path = _abs(env)
    else:
        corpus_set = os.environ.get(ENV_CORPUS_SET)
        path = (manifest_lexicon_path(corpus_set) if corpus_set else None) or DEFAULT_LEXICON_PATH
    if not path.is_file():
        raise FileNotFoundError(f"Resolved lexicon file does not exist: {path}")
    return path
