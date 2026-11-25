"""Derivational morphology: lexicon targets vs corpus usage (lexicon v0.3.0).

experiment.md §2.5/§2.6 set every morphology target at the LEXICON level; the
corpus is a Zipf-weighted sample of the language, so its rates are REPORTED,
not targeted. This report puts the two side by side for one corpus set, and
checks the model-token budget (``n_positions`` = 256) on every split.

    THESIS_CORPUS_SET=set_4 python analysis/reports/corpus/report_morphology.py

Writes ``analysis/descriptive/morphology_<set>.md`` and prints it.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from analysis.category_labels import relabel  # noqa: E402
from lexicon.paths import resolve_lexicon_path  # noqa: E402

OPEN = ("CAT1", "CAT2", "CAT3", "CAT5")
N_POSITIONS = 256          # models/model.py GPT-2 context (BOS + morphemes + EOS)
OUT = REPO / "analysis" / "descriptive"


def _lexicon_tables(lex: dict):
    pools = lex["metadata"]["category_alphabets"]
    slot_of, cat_of = {}, {}
    for a in lex.get("affixes", []):
        slot_of[a["form"]] = a["slot"]
    for key, items in lex.items():
        if isinstance(items, list) and key != "affixes":
            for it in items:
                for m in it.get("form", "").split("#"):
                    if m:
                        cat_of[m] = it.get("category")
    return pools, slot_of, cat_of


def _lexicon_stats(lex: dict) -> list[dict]:
    pools = lex["metadata"]["category_alphabets"]
    rows = []
    for cat in OPEN:
        es = lex[cat.lower()]
        bare = [e for e in es if e.get("n_morphemes", 1) == 1]
        pre = sum(bool(e.get("prefix")) for e in es)
        suf = sum(bool(e.get("suffix_inner")) + bool(e.get("suffix_outer")) for e in es)
        mc = Counter(e.get("n_morphemes", 1) for e in es)
        rows.append({
            "category": cat, "level": "lexicon (entries)", "n": len(es),
            "stems": len(bare),
            "morph 1/2/3/4 %": "/".join(f"{100 * mc[k] / len(es):.1f}" for k in (1, 2, 3, 4)),
            "mean morphemes": f"{sum(e.get('n_morphemes', 1) for e in es) / len(es):.2f}",
            "suffix:prefix": f"{suf / pre:.2f}" if pre else "—",
            "mean stem len": f"{sum(len(e.get('stem', e['form'])) for e in bare) / len(bare):.2f}",
            "% stem > pool": f"{100 * sum(len(e.get('stem', e['form'])) > len(pools[cat]) for e in bare) / len(bare):.0f}",
        })
    return rows


def _corpus_stats(root: Path, grammar: str, lex: dict) -> list[dict]:
    pools, slot_of, cat_of = _lexicon_tables(lex)
    acc = {c: defaultdict(int) for c in OPEN}
    mc = {c: Counter() for c in OPEN}
    stem_len = {c: [] for c in OPEN}
    with open(root / "train" / f"{grammar}.jsonl", encoding="utf-8") as fh:
        for line in fh:
            for w in json.loads(line)["sentence"].replace("#NULL#", "").split():
                lexical = [m for m in w.split("#") if m and not m.isdigit()]
                if not lexical:
                    continue
                cat = cat_of.get(lexical[0])
                if cat not in OPEN:
                    continue
                stems = [m for m in lexical if m not in slot_of]
                a = acc[cat]
                a["n"] += 1
                a["pre"] += sum(slot_of.get(m) == "prefix" for m in lexical)
                a["suf"] += sum(slot_of.get(m, "").startswith("suffix") for m in lexical)
                mc[cat][len(lexical)] += 1
                stem_len[cat].append(len(stems[0]) if stems else 0)
    rows = []
    for cat in OPEN:
        a, n = acc[cat], acc[cat]["n"] or 1
        L = stem_len[cat] or [0]
        rows.append({
            "category": cat, "level": f"corpus train ({grammar}, occurrences)", "n": a["n"],
            "stems": "",
            "morph 1/2/3/4 %": "/".join(f"{100 * mc[cat][k] / n:.1f}" for k in (1, 2, 3, 4)),
            "mean morphemes": f"{sum(k * v for k, v in mc[cat].items()) / n:.2f}",
            "suffix:prefix": f"{a['suf'] / a['pre']:.2f}" if a["pre"] else "—",
            "mean stem len": f"{sum(L) / len(L):.2f}",
            "% stem > pool": f"{100 * sum(x > len(pools[cat]) for x in L) / len(L):.0f}",
        })
    return rows


def _token_budget(root: Path) -> list[dict]:
    rows = []
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(split_dir.glob("*.jsonl")):
            if f.name.endswith(".meta.jsonl"):
                continue
            mx, tot, n = 0, 0, 0
            with open(f, encoding="utf-8") as fh:
                for line in fh:
                    d = json.loads(line)
                    s = d.get("sentence")
                    if s is None:
                        continue
                    k = sum(1 for w in s.replace("#NULL#", "").split()
                            for m in w.split("#") if m) + 2        # + BOS/EOS
                    mx, tot, n = max(mx, k), tot + k, n + 1
            rows.append({"split": split_dir.name, "file": f.name, "sentences": n,
                         "mean tokens": f"{tot / max(n, 1):.1f}", "max tokens": mx,
                         "fits 256": "yes" if mx <= N_POSITIONS else "NO"})
    return rows


def _md(rows: list[dict]) -> str:
    if not rows:
        return "_(none)_\n"
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(relabel(str(r[c])) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def main() -> None:
    from models.dataload import CORPORA_ROOT
    root = CORPORA_ROOT
    lex_path = resolve_lexicon_path()
    lex = json.loads(lex_path.read_text(encoding="utf-8"))
    if not lex.get("affixes"):
        sys.exit(f"{lex_path} has no derivational morphology (lexicon v0.2.0).")
    grammars = sorted(p.stem for p in (root / "train").glob("*.jsonl")
                      if not p.name.endswith(".meta.jsonl"))
    rows = _lexicon_stats(lex)
    for g in grammars:
        rows += _corpus_stats(root, g, lex)
    text = (f"# Morphology: lexicon vs corpus — {root.name}\n\n"
            f"Lexicon: `{lex_path.relative_to(REPO) if lex_path.is_relative_to(REPO) else lex_path}` "
            f"(version {lex['metadata']['version']}, seed {lex['metadata']['seed']}). "
            "Targets are lexicon-level (experiment.md §2.6); corpus rows are "
            "Zipf-weighted occurrences in `train`, reported only.\n\n"
            "## Lexical morphemes, suffix preference, stem length\n\n" + _md(rows) +
            "\n## Model-token budget (BOS + morphemes + EOS vs n_positions=256)\n\n"
            + _md(_token_budget(root)))
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"morphology_{root.name}.md"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"-> {out}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
