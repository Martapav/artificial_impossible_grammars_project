"""The grammar floor: from the raw derivation cost ``ℓ(x)`` to ``−log p_corpus(x)``.

``grammars/tracing.py`` records ``ℓ(x) = Σ −log π(draw)`` — the cost of the
derivation the generator actually took. That is **not** the quantity the
learnability claim needs. The model is trained on a rejection-sampled corpus, so
the distribution it saw is a reweighting of the raw generative one, and
``E[Δ] = D_KL(p‖q)`` only holds when ``p`` is the distribution ``q`` was trained
on (SURPRISAL_REFERENCE_POINTS.md §6.1).

The correction
--------------
Generation draws i.i.d. from ``p_raw`` and keeps a draw only if its label is
valid, its length is in window, and its label's quota is unfilled. Conditional on
label ``c`` the survivors are i.i.d. from ``p_raw(· | c, len ok)``, and the label
proportions are pinned to ``quota_c / n``. So for a string ``x`` of construction
``c``:

    p_corpus(x)  =  (quota_c / n) · p_raw(x) / P_raw(c, len ok)
    ℓ_corpus(x)  =  ℓ(x)  +  log P_raw(c, len ok)  −  log(quota_c / n)
                 =  ℓ(x)  +  correction_c

The correction is **constant within a construction**. It therefore cancels in any
within-construction comparison, and does NOT cancel across constructions — which
is exactly where a raw-``ℓ`` analysis would go wrong, since the quota moves some
labels by a factor of two or more (measured on H: wh is 5.1% of eligible draws
but 10% of the corpus).

That ``p_corpus`` is proper is easy to see: summing ``p_corpus`` over the eligible
strings of label ``c`` gives ``(quota_c/n) · P_raw(c, len ok) / P_raw(c, len ok)
= quota_c/n``, and those sum to 1 over ``c``.

Estimation
----------
``P_raw(c, len ok)`` comes from the per-split ``<grammar>.accept.json`` written at
generation, as ``eligible[c] / attempts``. Eligibility is recorded before the
quota cap can mask it — ``accepted[c]`` is pinned to ``quota_c`` by construction
and would measure the quota, not the grammar. The sample is sequentially stopped
(generation halts once every quota is filled), so this carries an
``O(1/attempts)`` bias: negligible at corpus sizes, but an estimate, not an
identity.

Usage
-----
    python -m analysis.entropy_baseline --grammar all --split validation
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

# Resolved through `models.dataload`, the single place that knows about corpus
# SETS (corpora/set_N/...). Hardcoding PROJECT_ROOT / "corpora" here would point
# at the directory that CONTAINS the sets, so every split lookup would miss.
from models.dataload import CORPORA_ROOT as CORPORA_DIR  # noqa: E402
DATA_DIR = Path(__file__).resolve().parent / "data"
GRAMMARS = ("H", "P", "Hprime", "Lprime")

# The only admissible join key back to model scores; see
# analysis/differential_surprisal.py. Defined here rather than imported to keep
# this module free of a circular dependency on its own consumer.
STABLE_KEY = "item_uid"


def _missing(path: Path, what: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{path} not found. {what} is written at corpus generation by the "
        "traced generators; regenerate with scripts/build_all_corpora.py "
        "(SURPRISAL_REFERENCE_POINTS.md §7 A1)."
    )


def load_derivations(grammar: str, split: str,
                     corpora_dir: Path = CORPORA_DIR) -> pd.DataFrame:
    """``item_uid``, ``id``, ``construction``, ``ell_raw``, ``n_draws``.

    ``item_uid`` is the JOIN KEY and is required (A4). ``id`` is carried along
    for file traceability only — it is ``{grammar}-{split}-{index}``, a dense
    index into the split file as it was written, so it identifies a LINE and
    re-points itself whenever the corpus is rebuilt in a different order. This
    function used to key its rows on ``id``, which meant every downstream join
    was one rebuild away from being silently wrong.

    Raises when the sidecar carries no ``item_uid``. That is the current state
    of the on-disk corpora: ``CorpusBuilder._write`` stamps ``item_uid = None``
    on corpus items, and the real identifier is written by task A3. Failing
    loudly here is the point — there is no positional fallback.
    """
    path = Path(corpora_dir) / split / f"{grammar}.meta.jsonl"
    if not path.exists():
        raise _missing(path, "The metadata sidecar")
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            m = json.loads(line)
            d = m.get("derivation")
            if d is None:
                raise ValueError(
                    f"{path} has no 'derivation' field (item {m.get('id')!r}). "
                    "The corpora predate the tracing layer; rerun generation."
                )
            uid = m.get(STABLE_KEY)
            if uid is None:
                raise ValueError(
                    f"{path}: item {m.get('id')!r} has no {STABLE_KEY!r} "
                    "(missing or null). ℓ(x) rows are joined to model scores on "
                    f"{STABLE_KEY!r} and on nothing else — 'id' indexes a line "
                    "in this file, not an item, so a rebuild re-points it and "
                    "every previously written score joins onto the wrong "
                    "record with no error anywhere. The corpus-side identifier "
                    "is written at generation time by task A3; these corpora "
                    "predate it and must be rebuilt."
                )
            rows.append({STABLE_KEY: uid, "id": m["id"],
                         "grammar": m["grammar_type"],
                         "split": split, "construction": m["construction"],
                         "ell_raw": d["neg_logp"], "n_draws": d["n_draws"]})
    return pd.DataFrame(rows)


def load_acceptance(grammar: str, split: str,
                    corpora_dir: Path = CORPORA_DIR) -> Dict:
    path = Path(corpora_dir) / split / f"{grammar}.accept.json"
    if not path.exists():
        raise _missing(path, "The acceptance record")
    return json.loads(path.read_text(encoding="utf-8"))


def corrections(accept: Dict) -> Dict[str, float]:
    """``correction_c = log P_raw(c, len ok) − log(quota_c / n)``, per construction."""
    p_elig = accept.get("p_eligible_by_construction") or {}
    prop = accept.get("corpus_proportion") or {}
    if not p_elig:
        raise ValueError(
            "acceptance record has no 'p_eligible_by_construction'; it was "
            "written by an older AcceptanceCounter that tracked only quota-capped "
            "accept counts, which cannot estimate P_raw(c, len ok)."
        )
    out = {}
    for c, share in prop.items():
        pe = p_elig.get(c)
        if not pe or not share:
            raise ValueError(f"construction {c!r}: degenerate acceptance record "
                             f"(p_eligible={pe}, corpus_proportion={share})")
        out[c] = math.log(pe) - math.log(share)
    return out


def ell_corpus(grammar: str, split: str,
               corpora_dir: Path = CORPORA_DIR) -> pd.DataFrame:
    """Per-item table with ``ell_raw``, ``correction``, and ``ell_corpus``."""
    df = load_derivations(grammar, split, corpora_dir)
    corr = corrections(load_acceptance(grammar, split, corpora_dir))
    unknown = set(df.construction) - set(corr)
    if unknown:
        raise ValueError(f"{grammar}/{split}: no correction for {sorted(unknown)}")
    df["correction"] = df.construction.map(corr)
    df["ell_corpus"] = df.ell_raw + df.correction
    return df


def emit(grammars: Sequence[str] = GRAMMARS,
         splits: Sequence[str] = ("validation", "test_indistribution"),
         corpora_dir: Path = CORPORA_DIR,
         out_dir: Path = DATA_DIR) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for g in grammars:
        for split in splits:
            df = ell_corpus(g, split, corpora_dir)
            frames.append(df)
            by = df.groupby("construction").agg(
                n=("ell_raw", "size"), raw=("ell_raw", "mean"),
                corr=("correction", "first"), corrected=("ell_corpus", "mean"))
            print(f"[{g}/{split}] n={len(df)}  "
                  f"H_hat(p)={df.ell_corpus.mean():.3f} nats "
                  f"(raw {df.ell_raw.mean():.3f})", flush=True)
            print(by.round(3).to_string(), flush=True)
    out = pd.concat(frames, ignore_index=True)
    path = out_dir / "entropy_baseline.parquet"
    out.to_parquet(path, index=False)
    print(f"\nwrote {path}  ({len(out)} rows)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grammar", choices=[*GRAMMARS, "all"], default="all")
    p.add_argument("--splits", nargs="+",
                   default=["validation", "test_indistribution"])
    p.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()
    targets = list(GRAMMARS) if args.grammar == "all" else [args.grammar]
    emit(targets, args.splits, args.corpora_dir, args.out_dir)


if __name__ == "__main__":
    main()
