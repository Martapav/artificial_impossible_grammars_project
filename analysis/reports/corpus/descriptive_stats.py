"""Descriptive statistics for the thesis: lexicon composition + transformation knobs.

Every number is read from source — the generated ``lexicon.json`` and the grammar
constant modules — never hardcoded here. Run from the repo root:

    python analysis/reports/corpus/descriptive_stats.py

Writes four files under ``analysis/descriptive/``:
    lexicon_stats.csv / .md
    transformation_stats.csv / .md

Each ``.md`` cites the source file for every value so the numbers can be spot-checked.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from analysis.category_labels import relabel  # noqa: E402  (display names for CAT/INFL)


def _relabel_rows(rows: list[dict]) -> list[dict]:
    """Substitute CAT/INFL tokens with linguistic display names in every string
    cell (data keys elsewhere are untouched; this is presentation only)."""
    return [{k: (relabel(v) if isinstance(v, str) else v) for k, v in r.items()}
            for r in rows]
OUT = REPO / "analysis" / "descriptive"
LEXICON_JSON = REPO / "lexicon.json"


# ─────────────────────────────────────────────────────────────────────────────
# LEXICON
# ─────────────────────────────────────────────────────────────────────────────
def _pct(n: int, total: int) -> str:
    return f"{100.0 * n / total:.2f}%" if total else "—"


def build_lexicon_rows() -> list[dict]:
    """Return per-feature breakdown rows read from the realized lexicon.json.

    Design probabilities are pulled from lexicon/config.py so the realized counts
    sit next to the nominal rate that generated them.
    """
    from lexicon import config

    lex = json.loads(LEXICON_JSON.read_text())
    meta = lex["metadata"]
    sizes = meta["category_sizes"]

    rows: list[dict] = []

    # 1) Category sizes (all ten categories).
    for cat in ("CAT1", "CAT1PRON", "CAT2", "CAT3", "CAT3AUX",
                "CAT4", "CAT5", "CAT6", "CAT8", "CAT9"):
        rows.append({
            "scope": cat,
            "feature": "category size (# stems/items)",
            "value": "",
            "count": sizes[cat],
            "share_of_group": "",
            "design_prob": "combinatorial" if cat in config.COMBINATORIAL_CATEGORIES else "fixed inventory",
        })

    # 2) CAT1 inherent features (the only randomized open class with features).
    cat1 = lex["cat1"]
    n1 = len(cat1)

    countability = Counter(x["countability"] for x in cat1)
    for val in ("countable", "proper_like"):
        rows.append({
            "scope": "CAT1",
            "feature": "countability",
            "value": val,
            "count": countability.get(val, 0),
            "share_of_group": _pct(countability.get(val, 0), n1),
            "design_prob": f"P_PROPER_LIKE={config.P_PROPER_LIKE} → proper_like; else countable",
        })

    gender = Counter(x["inherent_gender"] for x in cat1)
    for val in ("1", "2"):
        rows.append({
            "scope": "CAT1",
            "feature": "inherent_gender",
            "value": val,
            "count": gender.get(val, 0),
            "share_of_group": _pct(gender.get(val, 0), n1),
            "design_prob": "uniform choice of ('1','2')",
        })

    cat6c = Counter(bool(x["cat6_compatible"]) for x in cat1)
    for val in (True, False):
        rows.append({
            "scope": "CAT1",
            "feature": "cat6_compatible",
            "value": str(val),
            "count": cat6c.get(val, 0),
            "share_of_group": _pct(cat6c.get(val, 0), n1),
            "design_prob": f"P_CAT6_COMPATIBLE={config.P_CAT6_COMPATIBLE}",
        })

    # cat4_required is defined as (countability == 'countable'); report for completeness.
    cat4req = Counter(bool(x["cat4_required"]) for x in cat1)
    for val in (True, False):
        rows.append({
            "scope": "CAT1",
            "feature": "cat4_required",
            "value": str(val),
            "count": cat4req.get(val, 0),
            "share_of_group": _pct(cat4req.get(val, 0), n1),
            "design_prob": "= (countability == 'countable')",
        })

    # 3) CAT3 transitivity.
    cat3 = lex["cat3"]
    n3 = len(cat3)
    trans = Counter(x["transitivity"] for x in cat3)
    for val in ("transitive", "intransitive"):
        rows.append({
            "scope": "CAT3",
            "feature": "transitivity",
            "value": val,
            "count": trans.get(val, 0),
            "share_of_group": _pct(trans.get(val, 0), n3),
            "design_prob": "uniform choice of (transitive, intransitive)",
        })

    # 4) CAT6 attachment (fixed 2:1 partition — no randomness).
    cat6 = lex["cat6"]
    n6 = len(cat6)
    attach = Counter(x["attachment"] for x in cat6)
    for val in ("cat1_selecting", "cat3_selecting"):
        rows.append({
            "scope": "CAT6",
            "feature": "attachment",
            "value": val,
            "count": attach.get(val, 0),
            "share_of_group": _pct(attach.get(val, 0), n6),
            "design_prob": "fixed 2:1 (r,y under-NP; z under-VP)",
        })

    # 5) CAT4 number, CAT1PRON subclass (closed-class, fixed).
    for x in lex["cat4"]:
        rows.append({
            "scope": "CAT4", "feature": "number", "value": x["number"],
            "count": 1, "share_of_group": _pct(1, len(lex["cat4"])),
            "design_prob": "fixed ({u}=singular, {v}=plural)",
        })
    for x in lex["cat1pron"]:
        rows.append({
            "scope": "CAT1PRON", "feature": "subclass", "value": x["subclass"],
            "count": 1, "share_of_group": _pct(1, len(lex["cat1pron"])),
            "design_prob": "fixed ({m}=Pron, {mm}=Refl)",
        })

    return rows, meta


# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMATIONS
# ─────────────────────────────────────────────────────────────────────────────
def build_transform_rows() -> list[dict]:
    """Phenomenon mix + probability knobs, one row per knob, sourced from code.

    ``applies_to`` records which grammars a knob governs: 'all four' (shared),
    'H family (H, H')' or 'P family (P, P')'. H'/P' import their family's rules
    module unchanged, so a knob defined in grammar_H/rules.py or grammar_P/rules.py
    holds for the whole family.
    """
    from grammars import base_generator
    from grammars.grammar_H import rules as H
    from grammars.grammar_P import rules as P

    rows: list[dict] = []

    def add(name, value, applies, role, source):
        rows.append({"knob": name, "value": value, "applies_to": applies,
                     "role": role, "source": source})

    # 1) Phenomenon mix (identical across all four grammars; enforced as exact
    #    corpus quotas by CorpusBuilder via CONSTRUCTION_MIX).
    mix = base_generator.CONSTRUCTION_MIX
    add("neutral", f"{mix['neutral']:.2f}", "all four",
        "no transformation", "base_generator.py:CONSTRUCTION_MIX")
    add("anaphoric_binding (total)", f"{H.PHENOMENON_PROBS['anaphoric_binding']:.2f}", "all four",
        "binding phenomenon draw", "rules.py:PHENOMENON_PROBS")
    add("  ├─ anaphoric_binding_refl", f"{mix['anaphoric_binding_refl']:.2f}", "all four",
        "refl sub-draw = bernoulli(0.5) of binding", "base_generator.py:CONSTRUCTION_MIX; transforms.py bernoulli(rng,0.5)")
    add("  └─ anaphoric_binding_pron", f"{mix['anaphoric_binding_pron']:.2f}", "all four",
        "pron sub-draw = bernoulli(0.5) of binding", "base_generator.py:CONSTRUCTION_MIX; transforms.py bernoulli(rng,0.5)")
    add("auxiliary_movement", f"{mix['auxiliary_movement']:.2f}", "all four",
        "aux phenomenon draw", "rules.py:PHENOMENON_PROBS")
    add("wh_movement", f"{mix['wh_movement']:.2f}", "all four",
        "wh phenomenon draw", "rules.py:PHENOMENON_PROBS")

    # 2) Structural / site knobs shared across all four grammars (H mirrors P).
    add("Q_EMB_OBJECT", f"{H.Q_EMB_OBJECT}", "all four",
        "embedded-clause host = object vs subject; also refl/wh forced host",
        "grammar_H/rules.py, grammar_P/rules.py")
    add("Q_AUX_MATRIX", f"{H.Q_AUX_MATRIX}", "all four",
        "forced aux sits in matrix clause (else embedded-only: H skips, P fronts)",
        "grammar_H/rules.py, grammar_P/rules.py")
    add("P_AUX", f"{H.P_AUX}", "all four",
        "in-situ compound tense in the matrix clause",
        "grammar_H/rules.py, grammar_P/rules.py")
    add("P_AUX_EMBEDDED", f"{H.P_AUX_EMBEDDED}", "all four",
        "in-situ compound tense in an embedded clause",
        "grammar_H/rules.py, grammar_P/rules.py")
    add("P_CAT2", f"{H.P_CAT2}", "all four",
        "continuation prob. of another CAT2 modifier on a CAT1",
        "grammar_H/rules.py, grammar_P/rules.py")
    add("P_CAT5", f"{H.P_CAT5}", "all four",
        "continuation prob. of another CAT5 adverb on a CAT3",
        "grammar_H/rules.py, grammar_P/rules.py")
    add("P_TYPE4 / P_CAT9", f"{H.P_TYPE4}", "all four",
        "embedded clause (Type4/CAT9) draw, once per clause",
        "grammar_H/rules.py:P_TYPE4, grammar_P/rules.py:P_CAT9")
    add("PP-under-NP", f"{H.P_PP_UNDER_NP}", "all four",
        "PP attached under a nominal", "grammar_H/rules.py:P_PP_UNDER_NP, grammar_P/rules.py:P_CAT6_NP")
    add("PP-under-NP (complement)", f"{H.P_PP_UNDER_NP_COMP}", "all four",
        "PP nesting inside a complement NP (shallower)",
        "grammar_H/rules.py:P_PP_UNDER_NP_COMP, grammar_P/rules.py:P_CAT6_NP_COMP")
    add("PP-under-VP", f"{H.P_PP_UNDER_VP}", "all four",
        "PP attached under a verbal domain", "grammar_H/rules.py:P_PP_UNDER_VP, grammar_P/rules.py:P_CAT6_VP")

    # 3) H-family-only knob.
    add("P_PRON_FIRST", f"{H.P_PRON_FIRST}", "H family (H, H')",
        "pron substitutes the FIRST occurrence (majority: the second)",
        "grammar_H/rules.py")

    # 4) P-family-only positional licensing constants.
    add("REFL_OFFSET", f"{P.REFL_OFFSET}", "P family (P, P')",
        "licensed refl distance (surface items from antecedent)", "grammar_P/rules.py")
    add("PRON_OFFSETS", f"{P.PRON_OFFSETS}", "P family (P, P')",
        "licensed pron distances", "grammar_P/rules.py")
    add("REFL_ANT_ORDINALS", f"{P.REFL_ANT_ORDINALS}", "P family (P, P')",
        "licensed refl antecedent CAT1-position ordinal(s)", "grammar_P/rules.py")
    add("PRON_ANT_ORDINALS", f"{P.PRON_ANT_ORDINALS}", "P family (P, P')",
        "licensed pron antecedent CAT1-position ordinal(s)", "grammar_P/rules.py")
    add("WH_I_NOMINAL_ORDINAL", f"{P.WH_I_NOMINAL_ORDINAL}", "P family (P, P')",
        "wh clause (i) CAT1-position ordinal", "grammar_P/rules.py")
    add("WH_II_ANCHOR_CAT", f"{P.WH_II_ANCHOR_CAT}", "P family (P, P')",
        "wh clause (ii) anchor category", "grammar_P/rules.py")
    add("WH_II_ANCHOR_ORDINAL", f"{P.WH_II_ANCHOR_ORDINAL}", "P family (P, P')",
        "wh clause (ii) anchor occurrence ordinal", "grammar_P/rules.py")
    add("WH_II_NOMINAL_ORDINAL", f"{P.WH_II_NOMINAL_ORDINAL}", "P family (P, P')",
        "wh clause (ii) nominal ordinal after the anchor", "grammar_P/rules.py")

    return rows


# ─────────────────────────────────────────────────────────────────────────────
# WRITERS
# ─────────────────────────────────────────────────────────────────────────────
def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def write_md(path: Path, title: str, preamble: str, headers: list[str],
             keys: list[str], rows: list[dict]) -> None:
    lines = [f"# {title}", "", preamble, "",
             "| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        cells = [str(r.get(k, "")) for k in keys]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    path.write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    lex_rows, meta = build_lexicon_rows()
    lex_rows = _relabel_rows(lex_rows)
    lex_fields = ["scope", "feature", "value", "count", "share_of_group", "design_prob"]
    write_csv(OUT / "lexicon_stats.csv", lex_fields, lex_rows)
    write_md(
        OUT / "lexicon_stats.md",
        "Lexicon composition",
        f"Source: `lexicon.json` (seed {meta['seed']}, version {meta['version']}, "
        f"content_hash `{meta['content_hash'][:12]}…`), shared by all four grammars. "
        f"Design probabilities from `lexicon/config.py`. "
        f"`share_of_group` = percentage within the category. Realized counts are the "
        f"actual draws in this lexicon build.",
        ["Scope", "Feature", "Value", "Count", "Share of category", "Design rule"],
        lex_fields, lex_rows,
    )

    tr_rows = _relabel_rows(build_transform_rows())
    tr_fields = ["knob", "value", "applies_to", "role", "source"]
    write_csv(OUT / "transformation_stats.csv", ["knob", "value", "applies_to"], tr_rows)
    write_md(
        OUT / "transformation_stats.md",
        "Transformation phenomena — mix and probability knobs",
        "Design-time values read from `grammars/base_generator.py`, "
        "`grammars/grammar_H/rules.py`, and `grammars/grammar_P/rules.py`. "
        "The phenomenon mix (neutral/binding/aux/wh) is identical across all four "
        "grammars and enforced as exact per-split quotas by the corpus builder "
        "(`CONSTRUCTION_MIX`). H′ and P′ import their family's rules module unchanged, "
        "so `applies_to` names the whole family.",
        ["Knob", "Value", "Applies to", "Role", "Source"],
        tr_fields, tr_rows,
    )

    print(f"Wrote:\n  {OUT/'lexicon_stats.csv'}\n  {OUT/'lexicon_stats.md'}"
          f"\n  {OUT/'transformation_stats.csv'}\n  {OUT/'transformation_stats.md'}")
    print(f"\nLexicon rows: {len(lex_rows)}  |  Transformation rows: {len(tr_rows)}")


if __name__ == "__main__":
    main()
