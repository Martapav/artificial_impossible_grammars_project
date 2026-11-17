"""Prove a sweep read the corpus you think it read.

A checksum of the copied corpus proves it was *delivered*; it does not prove it
was *consumed*. A stale flat `corpora/train/` shadowing `corpora/<set>/` would let
a sweep train and probe-score on a different generation while the delivery check
passes. `models/dataload.resolve_corpora_root` no longer allows that shadowing,
but a run's manifest is still the only record of what was actually opened, and it
is worth reading rather than trusting.

What it compares
----------------
`run_manifest.jsonl` records, per invocation, `corpora_root` and a `corpus_sha256`
of every split file the sweep read — measured at sweep start with the same
resolver the loader uses. This compares those hashes against a local corpus set
and reports, per file: MATCH, DIFFERS, or MISSING.

It also flags a `corpora_root` with no `set_N` component. That is the fingerprint
of the shadowing failure: a run addressed by set that resolved to something that
is not a set.

Usage
-----
    python -m scripts.verify_corpus_binding results/pilot_search/run_manifest.jsonl
    python -m scripts.verify_corpus_binding <manifest> --set set_2
    python -m scripts.verify_corpus_binding <manifest> --corpora-dir /path/to/set

Exit status is 0 only if every recorded file matches. Wire it into any script
that consumes a sweep.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.dataload import resolve_corpora_root, sha256_file  # noqa: E402


def records(manifest: Path) -> list[dict]:
    """Records to verify, from either artefact that records what was consumed.

    Two producers, two shapes:

    * ``models.grid_search`` writes ``run_manifest.jsonl`` — one line per
      invocation, carrying ``corpus_sha256`` as ``{grammar: {split: sha}}``.
    * ``models.train`` writes no such file. It writes one
      ``<run_id>/run_metadata.json`` per run, whose ``corpus`` block carries the
      same evidence per run: ``{corpus_set, corpora_root, files: {split: sha}}``
      for the one grammar that run trained on. Pass the training OUTPUT
      DIRECTORY and each run is checked separately, which is stricter than the
      sweep case — 20 independent statements instead of one.

    The training shape is normalised into the sweep shape so everything below
    stays single-path.
    """
    manifest = Path(manifest)
    if manifest.is_dir():
        found = sorted(manifest.glob("*/run_metadata.json"))
        if not found:
            raise SystemExit(
                f"no */run_metadata.json under {manifest}. Point this at a "
                f"training output directory, or at a sweep's run_manifest.jsonl.")
        out = []
        for path in found:
            meta = json.loads(path.read_text(encoding="utf-8"))
            corpus = meta.get("corpus") or {}
            grammar = (meta.get("spec") or {}).get("grammar", path.parent.name)
            out.append({
                "started_utc": meta.get("started_utc", path.parent.name),
                "corpora_root": corpus.get("corpora_root", "?"),
                "corpus_sha256": {grammar: corpus.get("files") or {}},
            })
        return out
    with open(manifest, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def compare(record: dict, root: Path) -> tuple[list[str], list[str], list[str]]:
    """`(matched, differed, missing)` file labels for one manifest record."""
    matched: list[str] = []
    differed: list[str] = []
    missing: list[str] = []
    for grammar, splits in (record.get("corpus_sha256") or {}).items():
        for split, expected in splits.items():
            # "<split>.meta" addresses the sidecar, not a split of that name.
            if split.endswith(".meta"):
                path = root / split[:-5] / f"{grammar}.meta.jsonl"
            else:
                path = root / split / f"{grammar}.jsonl"
            label = f"{grammar}/{split}"
            if not path.exists():
                missing.append(f"{label}  ({path})")
            elif sha256_file(path) == expected:
                matched.append(label)
            else:
                differed.append(
                    f"{label}\n      run  {expected[:16]}…\n      local"
                    f" {sha256_file(path)[:16]}…")
    return matched, differed, missing


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path,
                    help="a sweep's run_manifest.jsonl, OR a models.train "
                         "output directory (its */run_metadata.json are read)")
    ap.add_argument("--set", dest="corpus_set", default=None,
                    help="corpus set to compare against (default: the resolved one)")
    ap.add_argument("--corpora-dir", type=Path, default=None,
                    help="compare against this directory directly, bypassing --set")
    args = ap.parse_args()

    root = args.corpora_dir or resolve_corpora_root(corpus_set=args.corpus_set)
    print(f"comparing against: {root}\n")

    ok = True
    for i, record in enumerate(records(args.manifest), 1):
        started = record.get("started_utc", "?")
        recorded_root = str(record.get("corpora_root", "?"))
        print(f"[{i}] started {started}")
        print(f"    run's corpora_root: {recorded_root}")

        # A run addressed by set must resolve to a directory named for one.
        if not Path(recorded_root).name.startswith("set_"):
            ok = False
            print(f"    !! resolved root is not a set directory. A flat corpus "
                  f"there SHADOWS the set you shipped (the 2026-08-07 failure). "
                  f"Point THESIS_CORPORA_ROOT at the set itself.")

        matched, differed, missing = compare(record, root)
        print(f"    match {len(matched)}   differ {len(differed)}   "
              f"missing {len(missing)}")
        for line in differed:
            ok = False
            print(f"    DIFFERS {line}")
        for line in missing:
            ok = False
            print(f"    MISSING {line}")
        if not record.get("corpus_sha256"):
            ok = False
            print("    !! this record has no corpus_sha256; nothing can be "
                  "verified about what it read")
        print()

    if ok:
        print("OK — every file this sweep read matches the local corpus.")
    else:
        print("FAIL — the sweep did not read this corpus. Anything computed "
              "locally against it (n-gram baselines, ell(x), analysis tables) "
              "does NOT describe the same sentences.")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
