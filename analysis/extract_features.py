"""Per-sentence feature tables for corpus descriptive statistics.

Reads the generation-time metadata sidecars
(``corpora/<split>/<grammar>.meta.jsonl`` — TRANSFORM_RULES §8) and derives the
per-sentence feature table cached to
``analysis/data/<grammar>_train_features.parquet`` (train split; other splits
get ``<grammar>_<split>_features.parquet``).

The sidecar is written at emission from the actual tree/token objects
(§8.5), so every structural field here is exact by construction and
describes the on-disk corpus line-for-line (shared ``id``). The old *replay*
harness — re-running the generation trajectory to recover the structs — is gone
(§8), and with it the replay/staleness caveat: if the sidecars exist,
they ARE the corpus's own record.

Field notes:
  * ``embedding_depth`` is populated for BOTH families (2026-08-04; it used to
    be ``NaN`` for P/L′ under §8.4 decision A). The positional family has no
    constituent tree, so its value is ``n_embedded_clauses`` — which equals the
    depth exactly, positional embedding being a non-branching chain. Sidecars
    written before that date still carry ``NaN`` there; ``embedding_depth`` and
    ``n_embedded_clauses`` are interchangeable for P/L′ either way.

Usage:
  python -m analysis.extract_features --grammar all --split train
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

# Resolved through `models.dataload`, the single place that knows about corpus
# SETS (corpora/set_N/...). Hardcoding PROJECT_ROOT / "corpora" here would point
# at the directory that CONTAINS the sets, so every split lookup would miss.
from models.dataload import CORPORA_ROOT as CORPORA_DIR  # noqa: E402
DATA_DIR = Path(__file__).resolve().parent / "data"

GRAMMARS = ("H", "P", "Hprime", "Lprime")
BINDING_CONSTRUCTIONS = {"anaphoric_binding_refl", "anaphoric_binding_pron"}

# Table column order (a superset of the old replay table minus ``tree_depth``,
# which is not part of the locked §8.4 field set; plus the site primitives).
COLUMNS: List[str] = [
    "id", "grammar", "construction", "length", "char_len",
    "morphemes_total", "morphemes_mean", "morphemes_max",
    "n_clauses", "embedding_depth", "n_embedded_clauses",
    "n_modifiers", "n_verbs", "expansion_sites",
    "has_aux", "has_gap", "has_pron",
    "is_binding",
    "bind_site", "aux_site", "wh_site", "wh_clause",
]


def sidecar_path(grammar: str, split: str, corpora_dir: Path = CORPORA_DIR) -> Path:
    return Path(corpora_dir) / split / f"{grammar}.meta.jsonl"


def extract_grammar(grammar: str, split: str = "train",
                    corpora_dir: Path = CORPORA_DIR) -> pd.DataFrame:
    """Load one grammar's sidecar for one split as a feature DataFrame."""
    path = sidecar_path(grammar, split, corpora_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The metadata sidecar is written at corpus "
            "generation (TRANSFORM_RULES §8.5); regenerate the corpora "
            "with scripts/build_all_corpora.py first (§11)."
        )
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            construction = m["construction"]
            rows.append({
                "id": m["id"],
                "grammar": m["grammar_type"],
                "construction": construction,
                "length": m["length"],
                "char_len": m["char_len"],
                "morphemes_total": m["morphemes_total"],
                "morphemes_mean": m["morphemes_mean"],
                "morphemes_max": m["morphemes_max"],
                "n_clauses": m["n_clauses"],
                "embedding_depth": m["embedding_depth"],  # None -> NaN (P, L')
                "n_embedded_clauses": m["n_embedded_clauses"],
                "n_modifiers": m["n_modifiers"],
                "n_verbs": m["n_verbs"],
                "expansion_sites": m["expansion_sites"],
                "has_aux": m["has_aux"],
                "has_gap": m["has_gap"],
                "has_pron": m["has_pron"],
                "is_binding": construction in BINDING_CONSTRUCTIONS,
                "bind_site": m["bind_site"],
                "aux_site": m["aux_site"],
                "wh_site": m["wh_site"],
                "wh_clause": m["wh_clause"],
            })
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["embedding_depth"] = pd.to_numeric(df["embedding_depth"])
    df["wh_clause"] = pd.to_numeric(df["wh_clause"])
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grammar", choices=[*GRAMMARS, "all"], default="all")
    p.add_argument("--split", default="train",
                   help="corpus split whose sidecar to read")
    p.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    p.add_argument("--out", type=Path, default=DATA_DIR)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    targets = list(GRAMMARS) if args.grammar == "all" else [args.grammar]
    for label in targets:
        df = extract_grammar(label, args.split, args.corpora_dir)
        stem = (f"{label}_train_features" if args.split == "train"
                else f"{label}_{args.split}_features")
        out_path = args.out / f"{stem}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"[{label}] {sidecar_path(label, args.split, args.corpora_dir)} "
              f"-> {out_path}  ({len(df)} rows, {df.shape[1]} cols)", flush=True)


if __name__ == "__main__":
    main()
