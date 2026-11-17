"""Lexicon loader and lookup helpers for Grammar P."""

from __future__ import annotations

import json
from pathlib import Path

LEXICON_PATH: Path = Path(__file__).resolve().parents[2] / "lexicon.json"

_NUM_TO_CAT4_NUM: dict = {"3": "singular", "4": "plural"}


def load(path: Path = LEXICON_PATH) -> dict:
    """Load lexicon.json and return the raw JSON dict."""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def cat1_items(lex: dict) -> list:
    return lex["cat1"]


def cat3_items(lex: dict, transitivity: str | None = None) -> list:
    items = lex["cat3"]
    if transitivity is None:
        return items
    return [x for x in items if x["transitivity"] == transitivity]


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
    from lexicon.config import ZIPF_Q, ZIPF_S  # noqa: PLC0415

    if items and items[0].get("zipf_rank", 0) > 0:
        weights = [1.0 / (item["zipf_rank"] + ZIPF_Q) ** ZIPF_S for item in items]
        return rng.choices(items, weights=weights, k=1)[0]
    return rng.choice(items)
