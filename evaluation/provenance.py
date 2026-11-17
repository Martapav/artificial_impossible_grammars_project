"""Corpus provenance carried inside every derived parquet.

Why this exists
---------------
Every join key in this project is **positional**. `id` is a dense line index,
`pair_id` is assigned in generation order, and `item_uid` is `{pair_id}/{variant}`
built from it. None of them is a content hash. So two artifacts computed from two
*different generations* of the corpora join cleanly, row for row, and produce a
table whose columns describe two different corpora — with no error anywhere.

This is a real failure mode: an n-gram baseline fit on one copy of the corpora,
joined onto probe scores computed on another copy where a stale flat corpus
directory shadowed the intended set, passes every row-level check while most
items are simply different sentences.

The fix is that an artifact must carry the identity of the corpus it was computed
from, and anything joining two artifacts must refuse when they disagree. The
record is `models.dataload.corpus_provenance`: the set name, the resolved root,
and a sha256 per corpus file and probe sidecar actually read.

Storage
-------
Parquet key-value schema metadata, under `thesis_corpus_provenance`. It survives
a pandas round trip (pandas does not preserve it, hence the explicit pyarrow
write here), travels with the file when it is copied, and costs a few hundred bytes.
A sidecar JSON was rejected: it separates from its parquet exactly when someone
copies "just the results" somewhere, which is the moment the guarantee is needed.

Usage
-----
    from evaluation.provenance import write_parquet, read_provenance
    write_parquet(df, path, corpus_provenance(grammars, splits))
    prov = read_provenance(path)          # None on a pre-2026-08-07 file
"""
from __future__ import annotations

import json
from pathlib import Path

METADATA_KEY = b"thesis_corpus_provenance"


def write_parquet(df, path: Path, provenance: dict | None = None) -> None:
    """`df.to_parquet(path)`, with `provenance` embedded in schema metadata."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    if provenance is not None:
        meta = dict(table.schema.metadata or {})
        meta[METADATA_KEY] = json.dumps(provenance).encode("utf-8")
        table = table.replace_schema_metadata(meta)
    pq.write_table(table, path)


def read_provenance(path: Path) -> dict | None:
    """The embedded record, or None if the file predates this convention.

    None is not "fine" — it means the file cannot be proven to match anything.
    Callers should say so rather than treating it as agreement.
    """
    import pyarrow.parquet as pq

    meta = pq.read_schema(Path(path)).metadata or {}
    raw = meta.get(METADATA_KEY)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def describe(provenance: dict | None) -> str:
    """One-line summary for logs and error messages."""
    if provenance is None:
        return "<no provenance recorded>"
    return (f"{provenance.get('corpus_set', '?')} "
            f"({len(provenance.get('files') or {})} files) "
            f"at {provenance.get('corpus_root', '?')}")
