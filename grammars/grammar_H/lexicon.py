"""Lexicon loader and lookup helpers for Grammar H."""

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
    """CAT6 pool for PP-under-NP. The pool is deliberately NOT functionally
    disjoint: every CAT6 item can head either attachment (TRIVIALITY AUDIT §3c),
    so attachment is never lexically recoverable from the CAT6 form."""
    return lex["cat6"]


def cat6_vp_items(lex: dict) -> list:
    """CAT6 pool for PP-under-VP. Same full pool as cat6_np_items — attachment
    is structural (recorded on the Type3 node), never lexical."""
    return lex["cat6"]


def cat9_items(lex: dict) -> list:
    """CAT9 subordinators (head Type4 relative clauses). Spec §2.2."""
    return lex["cat9"]


def cat8_wh(lex: dict) -> dict:
    """The singleton CAT8 wh particle. Spec §2.2: CAT8 = {w}."""
    return lex["cat8"][0]


def compatible_cat4(lex: dict, infl1_number: str) -> list:
    """Return CAT4 items whose number matches the given INFL1_number digit value."""
    target = _NUM_TO_CAT4_NUM[infl1_number]
    return [x for x in lex["cat4"] if x["number"] == target]


def infl_values(lex: dict, param_name: str) -> list:
    for p in lex["inflectional"]["parameters"]:
        if p["name"] == param_name:
            return list(p["values"])
    raise KeyError(param_name)


def cat1pron_item(lex: dict, subclass: str) -> dict:
    """Return the single CAT1PRON item with the given subclass ('Pron' or 'Refl')."""
    for x in lex["cat1pron"]:
        if x["subclass"] == subclass:
            return x
    raise KeyError(subclass)


def pick(rng, items: list):
    """Sample one item using Zipf-Mandelbrot weighting for open-class categories.

    Open-class items (CAT1, CAT2, CAT3, CAT5) carry zipf_rank > 0 after lexicon
    generation.  For these, item probability is proportional to 1/(rank+q)^s where
    rank=1 is the most frequent item (shortest stem; frequency decreases with rank).
    Closed-class items (CAT4, CAT6, CAT8, CAT9, CAT1PRON, CAT3AUX) have
    zipf_rank=0 and are sampled uniformly — their selection is governed by
    grammatical rule, not frequency.

    The items list may be a filtered subset (e.g. only transitive verbs, or only
    cat6_compatible nouns); the original corpus-wide ranks are preserved in the
    subset, so relative frequency ordering is respected even within subsets.

    ZIPF_S and ZIPF_Q are read from lexicon.config and match the values used to
    assign ranks at lexicon generation time, ensuring the same distribution governs
    both rank assignment and sampling.
    """
    from grammars.lexsample import zipf_pick  # noqa: PLC0415

    # Stream-identical fast path (cached cumulative weights): see lexsample.
    return zipf_pick(rng, items)
