"""One worked minimal pair per violation, from the IOD and OOD probe corpora.

Reads ``corpora/<set>/structural_probe_{IOD,OOD}/<grammar>.jsonl`` and its
``.meta.jsonl`` sidecar and writes a single CSV holding, for every
``(split, grammar, construction, subcase, violation)`` cell, one grammatical /
ungrammatical pair side by side plus the word-level edit that separates them.

This is an exhibit, not a measurement: it reads the corpus only, never a model
score, and it follows the no-replay convention (metadata comes from the sidecar,
nothing is regenerated).

    python -m analysis.reports.corpus.export_probe_examples                # set_3, first pair
    python -m analysis.reports.corpus.export_probe_examples --set set_3 --pick shortest
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import re
from pathlib import Path

from analysis.category_labels import CATEGORY_DISPLAY, relabel
from analysis.report_common import (
    BIN_LABEL,
    CONSTRUCTION_DISPLAY,
    SPLIT_DISPLAY,
    SUBCASE_DISPLAY,
    SUBCASE_ORDER,
    VIOLATION_TO_BIN,
)
from analysis.stats import GRAMMAR_ORDER

REPO = Path(__file__).resolve().parents[3]
SPLITS = ("structural_probe_IOD", "structural_probe_OOD")
OUT_DIR = REPO / "results/analysis/descriptive/tables"

# Short display names. `stats.GRAMMAR_LABEL` spells the gloss out ("H\u00a0(hierarchical)"),
# which is too wide for a column repeated on every row; the data keys are kept
# in `grammar_key`, and Lprime displays as P\u2032 as everywhere else.
GRAMMAR_DISPLAY = {"H": "H", "P": "P", "Hprime": "H\u2032", "Lprime": "P\u2032"}

COLUMNS = [
    "split", "grammar", "construction", "subcase", "violation",
    "violation_display", "term", "bin",
    "bin_label", "pair_id", "n_pairs_in_cell", "length", "length_ungrammatical",
    "sentence_grammatical", "sentence_ungrammatical", "edit",
    "target_category", "target_word_index", "bind_site", "aux_site", "wh_site",
    "wh_clause", "n_clauses", "embedding_depth", "id_grammatical",
    "id_ungrammatical", "item_uid_grammatical", "item_uid_ungrammatical",
    "split_key", "grammar_key",
]


def term_of(subcase: str, violation: str) -> str:
    """The reporting term (``report_common.TERM_ORDER``) this violation sits in.

    ``positional_*`` collapses to ``neutral_positional`` exactly as
    ``evaluation.aggregate.to_pairs`` collapses it to ``positional``.
    """
    if subcase in ("refl", "pron"):
        return f"{subcase}:{violation}"
    return subcase


# `category_labels.relabel` deliberately leaves `_cat2` alone -- its underscore
# guard protects code constants such as `P_CAT2`. The violation slugs are labels,
# not constants, so they get the display names through their own substitution;
# the raw slug stays in the `violation` column.
_CAT_IN_SLUG = re.compile(r"(?<=_)(" + "|".join(
    sorted(CATEGORY_DISPLAY, key=len, reverse=True)) + r")(?![a-z0-9])", re.IGNORECASE)


def violation_display(violation: str) -> str:
    return _CAT_IN_SLUG.sub(lambda m: CATEGORY_DISPLAY[m.group(0).upper()].lower(), violation)


def bin_of(violation: str) -> str:
    key = "positional" if violation.startswith("positional") else violation
    return VIOLATION_TO_BIN[key]


def word_edit(good: str, bad: str) -> str:
    """Word-level diff, as ``[i] good -> bad`` chunks over the token lists."""
    a, b = good.split(), bad.split()
    parts = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        left = " ".join(a[i1:i2]) or "0"
        right = " ".join(b[j1:j2]) or "0"
        parts.append(f"[{i1}] {left} -> {right}")
    return " ; ".join(parts)


def load_split(corpus_root: Path, split: str, grammar: str) -> dict:
    """``pair_id -> {variant: record}``, with the sidecar's structural fields."""
    meta = {}
    with open(corpus_root / split / f"{grammar}.meta.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            meta[d["id"]] = d
    pairs: dict[str, dict] = {}
    with open(corpus_root / split / f"{grammar}.jsonl") as fh:
        for line in fh:
            d = json.loads(line)
            d["meta"] = meta[d["id"]]
            pairs.setdefault(d["pair_id"], {})[d["variant"]] = d
    return pairs


def cells(pairs: dict) -> dict:
    """Group pair ids by ``(construction, subcase, violation)``, file order kept."""
    out: dict[tuple, list] = {}
    for pair_id, variants in pairs.items():
        bad = variants.get("ungrammatical")
        if bad is None or "grammatical" not in variants:
            continue
        key = (bad["construction"], bad["subcase"], bad["violation"])
        out.setdefault(key, []).append(pair_id)
    return out


def row_for(split: str, grammar: str, key: tuple, pair: dict, n_in_cell: int) -> dict:
    construction, subcase, violation = key
    good, bad = pair["grammatical"], pair["ungrammatical"]
    m = good["meta"]
    targets = m.get("targets") or [{}]
    return {
        "split": SPLIT_DISPLAY[split],
        "grammar": GRAMMAR_DISPLAY[grammar],
        "construction": CONSTRUCTION_DISPLAY.get(construction, construction),
        "subcase": SUBCASE_DISPLAY.get(subcase, subcase),
        "violation": violation,
        "violation_display": violation_display(violation),
        "term": term_of(subcase, violation),
        "bin": bin_of(violation),
        "bin_label": BIN_LABEL[bin_of(violation)],
        "pair_id": good["pair_id"],
        "n_pairs_in_cell": n_in_cell,
        "length": good["length"],
        "length_ungrammatical": bad["length"],
        "sentence_grammatical": good["sentence"],
        "sentence_ungrammatical": bad["sentence"],
        "edit": word_edit(good["sentence"], bad["sentence"]),
        "target_category": relabel(targets[0].get("category", "")),
        "target_word_index": targets[0].get("word_index", ""),
        "bind_site": m.get("bind_site") or "",
        "aux_site": m.get("aux_site") or "",
        "wh_site": m.get("wh_site") or "",
        "wh_clause": m.get("wh_clause") or "",
        "n_clauses": m.get("n_clauses", ""),
        "embedding_depth": m.get("embedding_depth", ""),
        "id_grammatical": good["id"],
        "id_ungrammatical": bad["id"],
        "item_uid_grammatical": good["item_uid"],
        "item_uid_ungrammatical": bad["item_uid"],
        "split_key": split,
        "grammar_key": grammar,
    }


def build(corpus_root: Path, pick: str) -> list[dict]:
    rows = []
    for split in SPLITS:
        for grammar in GRAMMAR_ORDER:
            pairs = load_split(corpus_root, split, grammar)
            grouped = cells(pairs)
            for key in sorted(grouped, key=lambda k: (SUBCASE_ORDER.index(k[1]), k[2])):
                ids = grouped[key]
                if pick == "shortest":
                    chosen = min(ids, key=lambda p: (pairs[p]["grammatical"]["length"], p))
                else:
                    chosen = ids[0]
                rows.append(row_for(split, grammar, key, pairs[chosen], len(ids)))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="corpus_set", default="set_3",
                    help="corpus set under corpora/ (default: the reporting set)")
    ap.add_argument("--pick", choices=("first", "shortest"), default="first",
                    help="which pair of each cell to show (default: first in file order)")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <tables>/probe_minimal_pair_examples_<set>.csv")
    args = ap.parse_args()
    # The set goes in the filename so a run on another set cannot silently overwrite the
    # reporting set's table.
    out = args.out or OUT_DIR / f"probe_minimal_pair_examples_{args.corpus_set}.csv"

    rows = build(REPO / "corpora" / args.corpus_set, args.pick)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} minimal pairs ({args.corpus_set}, pick={args.pick}) -> {out}")


if __name__ == "__main__":
    main()
