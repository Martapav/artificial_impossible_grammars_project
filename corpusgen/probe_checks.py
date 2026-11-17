"""Post-build integrity checks for the three structural probes.

The probes are separated only by RNG seed offset (IOD +200_000, OOD +250_000,
SELECT +300_000 — see ``CorpusBuilder``), and nothing in the builder dedups:
``build_structural_probe`` loops until its quota is full without tracking what
it has already emitted. Distinct seeds make collisions unlikely, not
impossible, and the collision probability grows with the product of the sample
sizes — so raising ``--select-n-per-subcase`` silently erodes the margin.

That margin is load-bearing. SELECT chooses the recipe and IOD/OOD report the
result; a pair present in both turns the reported number into a number the
selection step already optimized. This module turns the implicit luck into an
explicit build-time failure.

Checked per grammar (never across grammars — H/H' and P/L' share a base
pipeline, so overlap between them is expected and meaningless):

  * no duplicate pair within a single probe,
  * no pair shared by two probes,
  * no grammatical member shared by two probes (half a shared item still
    couples selection to reporting).
"""

from __future__ import annotations

import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

PROBE_SPLITS = ("structural_probe_IOD", "structural_probe_OOD",
                "structural_probe_SELECT")

_MAX_EXAMPLES = 5  # how many offending pair_ids to name in an error message


class ProbeOverlapError(RuntimeError):
    """Raised when the probes share items, within a file or across files."""


def _load_pairs(path: Path) -> List[Dict]:
    """Read one probe file and fold its flat item list back into pairs."""
    by_pair: Dict[str, Dict[str, Dict]] = defaultdict(dict)
    order: List[str] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            pid = rec["pair_id"]
            if pid not in by_pair:
                order.append(pid)
            by_pair[pid][rec["variant"]] = rec

    pairs: List[Dict] = []
    for pid in order:
        members = by_pair[pid]
        missing = {"grammatical", "ungrammatical"} - members.keys()
        if missing:
            raise ProbeOverlapError(
                f"{path}: pair {pid} is missing its {sorted(missing)} member")
        gram = members["grammatical"]["sentence"]
        ungram = members["ungrammatical"]["sentence"]
        pairs.append({
            "pair_id": pid,
            "subcase": members["grammatical"]["subcase"],
            "gram": gram,
            "key": (gram, ungram),
        })
    return pairs


def _examples(pair_ids: List[str]) -> str:
    head = ", ".join(sorted(pair_ids)[:_MAX_EXAMPLES])
    extra = len(pair_ids) - _MAX_EXAMPLES
    return head + (f" (+{extra} more)" if extra > 0 else "")


def check_probe_disjointness(
    output_root: Path,
    grammar: str,
    probes: tuple = PROBE_SPLITS,
) -> Dict[str, int]:
    """Verify one grammar's probes are internally and mutually duplicate-free.

    Reads the written JSONL rather than the in-memory items, so it validates
    what evaluation will actually score. Returns a summary dict on success;
    raises ``ProbeOverlapError`` naming the offending pair_ids otherwise.
    Probes whose file is absent are skipped (partial rebuilds stay usable).
    """
    output_root = Path(output_root)
    loaded: Dict[str, List[Dict]] = {}
    for probe in probes:
        path = output_root / probe / f"{grammar}.jsonl"
        if path.exists():
            loaded[probe] = _load_pairs(path)

    problems: List[str] = []

    # 1. Within-probe duplicates.
    for probe, pairs in loaded.items():
        counts = Counter(p["key"] for p in pairs)
        dup_keys = {k for k, c in counts.items() if c > 1}
        if dup_keys:
            ids = [p["pair_id"] for p in pairs if p["key"] in dup_keys]
            problems.append(
                f"{grammar}/{probe}: {len(dup_keys)} pair(s) occur more than "
                f"once within the probe — {_examples(ids)}")

    # 2. Cross-probe overlap, on the full pair and on the grammatical member.
    for a, b in itertools.combinations(loaded, 2):
        pa, pb = loaded[a], loaded[b]
        for field, label in (("key", "identical pair"),
                             ("gram", "shared grammatical member")):
            shared = {p[field] for p in pa} & {p[field] for p in pb}
            if shared:
                ids = ([f"{a}:{p['pair_id']}" for p in pa if p[field] in shared]
                       + [f"{b}:{p['pair_id']}" for p in pb if p[field] in shared])
                problems.append(
                    f"{grammar}: {a} and {b} share {len(shared)} "
                    f"{label}(s) — {_examples(ids)}")

    if problems:
        raise ProbeOverlapError(
            "structural probes are not disjoint; selection would leak into the "
            "reported numbers:\n  - " + "\n  - ".join(problems))

    return {probe: len(pairs) for probe, pairs in loaded.items()}
