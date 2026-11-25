"""Stem -> category lookup, shared by both grammar families.

All four grammars draw from ONE shared lexicon (verified: H's and P's loaders
return the same 4635 (stem -> category) pairs, zero conflicts) with an
injective stem -> category mapping (a bare wordform belongs to exactly one
category; CAT1PRON additionally carries a ``subclass`` — "Pron" or "Refl").
This means the category (and, for CAT1PRON, the subclass) of every surface
word is recoverable from the string ALONE, with no tree/token object and no
knowledge of which grammar produced it — exactly the information a
surface-only observer (e.g. a plain n-gram model) would have. This is the
ground truth the fine-grained co-occurrence analysis parses sentences with.

The null-morpheme gap trace (``#NULL#``, wh/aux movement) is not a lexicon
entry; it is recognised specially and categorised ``"GAP"``.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

NULL_WORD = "#NULL#"
_NON_ENTRY_KEYS = ("inflectional", "metadata")


def load_stem_map(lex: dict) -> Dict[str, Tuple[str, Optional[str]]]:
    """(lexical morpheme -> (category, subclass)) from a loaded lexicon dict.

    ``subclass`` is non-None only for CAT1PRON entries ("Pron" / "Refl").
    Lexicon v0.3.0: an open-class entry's ``form`` is a derived word
    ("ab#cde#fg"); EVERY lexical morpheme in it (stem and affixes) maps to the
    entry's category, and so does each item of the ``affixes`` inventory. Affixes
    are spelled from their category's own letter pool, so a word's first
    ``#``-segment — prefix or stem — still identifies its category uniquely.
    Raises if two entries disagree on a morpheme's category (would break every
    downstream surface-parse; caught once here rather than silently).
    """
    separator = lex.get("metadata", {}).get("config_snapshot", {}).get("SEPARATOR", "#")
    out: Dict[str, Tuple[str, Optional[str]]] = {}
    for key, entries in lex.items():
        if key in _NON_ENTRY_KEYS or not isinstance(entries, list):
            continue
        for item in entries:
            form, cat = item.get("form"), item.get("category")
            if form is None or cat is None:
                continue
            resolved = (cat, item.get("subclass"))
            for morph in form.split(separator):
                if not morph:
                    continue
                if morph in out and out[morph] != resolved:
                    raise ValueError(
                        f"morpheme {morph!r} maps to both {out[morph]} and {resolved}")
                out[morph] = resolved
    return out


def load_shared_stem_map() -> Dict[str, Tuple[str, Optional[str]]]:
    """The project's one shared stem map (H's and P's loaders agree; verified
    at generation-audit time — see docs/TRANSFORM_RULES.md §9)."""
    from grammars.grammar_H.lexicon import load as h_load
    return load_stem_map(h_load())


def categorize_sentence(
    sentence: str, stem_map: Dict[str, Tuple[str, Optional[str]]],
) -> List[Tuple[str, Optional[str]]]:
    """Surface string -> per-word ``(category, subclass)``, in word order.

    A gap trace (``#NULL#``) categorises as ``("GAP", None)``. Every other
    word's category is read off its first ``#``-separated segment (its stem, or
    its derivational prefix — both belong to the word's category).
    """
    out: List[Tuple[str, Optional[str]]] = []
    for word in sentence.split():
        if word == NULL_WORD:
            out.append(("GAP", None))
            continue
        stem = word.split("#", 1)[0]
        if stem not in stem_map:
            raise KeyError(
                f"stem {stem!r} (from word {word!r}) not in the shared lexicon")
        out.append(stem_map[stem])
    return out
