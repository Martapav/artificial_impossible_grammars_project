"""Lexicon loader and lookup helpers for Grammar P."""

from __future__ import annotations

import json
from pathlib import Path

from lexicon.paths import resolve_lexicon_path

# Default (v0.2.0) path, kept for reference; the file actually loaded is the one
# lexicon.paths.resolve_lexicon_path() selects at call time (THESIS_LEXICON /
# corpus-set manifest / this default).
LEXICON_PATH: Path = Path(__file__).resolve().parents[2] / "lexicon.json"

_NUM_TO_CAT4_NUM: dict = {"3": "singular", "4": "plural"}


def load(path: Path | None = None) -> dict:
    """Load the lexicon in force (``resolve_lexicon_path``) as the raw JSON dict."""
    with open(path or resolve_lexicon_path(), encoding="utf-8") as fh:
        return json.load(fh)


def cat1_items(lex: dict) -> list:
    return lex["cat1"]


_CAT3_BY_TR: dict = {}


def cat3_items(lex: dict, transitivity: str | None = None) -> list:
    """CAT3 entries, optionally filtered by transitivity. The filtered list is
    memoized per (lexicon list, transitivity) so ``pick`` sees a stable list
    object and can reuse its cached sampler (grammars/lexsample.py); same
    contents and order as a fresh filter."""
    items = lex["cat3"]
    if transitivity is None:
        return items
    key = (id(items), transitivity)
    hit = _CAT3_BY_TR.get(key)
    if hit is None or hit[0] is not items:
        if len(_CAT3_BY_TR) > 16:
            _CAT3_BY_TR.clear()
        hit = (items, [x for x in items if x["transitivity"] == transitivity])
        _CAT3_BY_TR[key] = hit
    return hit[1]


def cat6_np_items(lex: dict) -> list:
    """CAT6 pool for the nominal zone. The pool is deliberately NOT functionally
    disjoint: every CAT6 item can appear in either zone (TRIVIALITY AUDIT §3c),
    so the zone is never lexically recoverable from the CAT6 form."""
    return lex["cat6"]


def cat6_vp_items(lex: dict) -> list:
    """CAT6 pool for the verbal zone. Same full pool as cat6_np_items — the
    zone is positional bookkeeping (recorded on the token), never lexical."""
    return lex["cat6"]


def cat9_items(lex: dict) -> list:
    """CAT9 subordinators (introduce a new positional substring, spec §5.1.7)."""
    return lex["cat9"]


def cat8_wh(lex: dict) -> dict:
    """The singleton CAT8 wh particle. Spec §2.2: CAT8 = {w}."""
    return lex["cat8"][0]


def cat3aux_item(lex: dict) -> dict:
    """The singleton CAT3AUX item. Spec §2.2: CAT3AUX = {n}."""
    return lex["cat3aux"][0]


def compatible_cat4(lex: dict, infl1_number: str) -> dict:
    """Return the single CAT4 item whose number matches the INFL1_number digit."""
    target = _NUM_TO_CAT4_NUM[infl1_number]
    return next(x for x in lex["cat4"] if x["number"] == target)


def cat1pron_item(lex: dict, subclass: str) -> dict:
    """Return the CAT1PRON item with the given subclass ('Pron' or 'Refl')."""
    for x in lex["cat1pron"]:
        if x["subclass"] == subclass:
            return x
    raise KeyError(subclass)


def pick(rng, items: list):
    """Sample one item using Zipf-Mandelbrot weighting for open-class categories.

    Identical logic to grammar_H/lexicon.py:pick — see that docstring for details.
    Both grammars share the same lexicon.json and the same ZIPF_S/ZIPF_Q config,
    so the sampling distribution is identical across all four grammar conditions.
    """
    from grammars.lexsample import zipf_pick  # noqa: PLC0415

    # Stream-identical fast path (cached cumulative weights): see lexsample.
    return zipf_pick(rng, items)
