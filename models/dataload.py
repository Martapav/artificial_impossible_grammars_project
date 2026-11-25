"""Loads generated corpora and prepares tokenized datasets for training.

Corpora live in corpora/{split}/{grammar}.jsonl, one JSON object per line
with the sentence under "sentence". Grammar files are discovered at runtime,
so the loader keeps working when corpora are regenerated or renamed.

Tokenization delegates to models.tokenize: one token per surface morpheme,
with whitespace (word boundary) and SEPARATOR (morpheme boundary) removed
and the null morpheme deleted. Run `python -m models.dataload --check` to
verify token/morpheme correspondence over the corpora.
"""
import hashlib
import json
import os
from pathlib import Path
from typing import Sequence
import argparse
import sys

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from models.tokenize import LEXICON_PATH, load_lexicon, tokenizer

# --- Corpus sets -----------------------------------------------------------
# Every generation of the corpora lives in its own set directory:
#
#     corpora/set_1/train/H.jsonl, corpora/set_1/structural_probe_IOD/H.jsonl, ...
#     corpora/set_2/...
#
# A regeneration never overwrites an earlier one (`scripts/build_all_corpora.py`
# refuses to write into an existing set), so a model can always be re-scored on
# the exact corpus it was trained on, and two sets can be compared.
#
# This is the ONE place the corpus location is resolved. `evaluation/*` and
# `analysis/*` all import `CORPORA_ROOT` from here, so switching sets is one
# environment variable and nothing downstream needs to know sets exist:
#
#     THESIS_CORPUS_SET=set_2 python -m models.train ...
#
# `THESIS_CORPORA_ROOT` still points at the directory that CONTAINS the sets
# (e.g. on a separate storage volume). Setting it to a set directory directly also
# works — a root that already holds split directories is used as given, which is
# what the pre-set-layout corpora and the per-split symlinks need.
CORPORA_BASE = Path(os.environ.get(
    "THESIS_CORPORA_ROOT", Path(__file__).resolve().parent.parent / "corpora"))
DEFAULT_CORPUS_SET = "set_3"
CORPUS_SET = os.environ.get("THESIS_CORPUS_SET", DEFAULT_CORPUS_SET)
# Whether the set was CHOSEN or defaulted. An explicit choice outranks the
# flat-root fallback below; a defaulted one does not. See `resolve_corpora_root`.
CORPUS_SET_IS_EXPLICIT = "THESIS_CORPUS_SET" in os.environ

# The five-split architecture (TRANSFORM_RULES §10). The structural probes
# are evaluation-only minimal-pair corpora; training consumes only "train".
SPLITS = ("train", "validation", "test_indistribution",
          "structural_probe_IOD", "structural_probe_OOD",
          "structural_probe_SELECT")
TEXT_FIELD = "sentence"


def is_split_root(path: Path) -> bool:
    """True if `path` holds the split directories themselves, not a set layer."""
    return (path / "train").is_dir()


def resolve_corpora_root(base: Path = CORPORA_BASE,
                         corpus_set: str | None = None) -> Path:
    """The directory holding the split directories, for one corpus set.

    Resolution order, most explicit first:

    1. `base/<set>` exists AND the set was named explicitly (an argument, or
       `THESIS_CORPUS_SET` in the environment) — a named set always wins.
    2. `base` already contains `train/` — a flat, pre-set-layout corpora
       directory (or `THESIS_CORPORA_ROOT` pointed straight at a set). Used as
       given, so nothing that already works breaks.
    3. `base/<set>` exists — the normal case, with the set defaulted.
    4. None of these — raise, listing the sets that do exist. A missing set must
       never degenerate into "some other corpus", which is the failure this
       layout exists to prevent.

    **Why rule 1 comes first.** If a `base` holding `train/` were returned
    before `THESIS_CORPUS_SET` is read, a stale flat pre-set-layout
    `corpora/train/` sitting beside `corpora/set_1/` would make a run launched
    with `THESIS_CORPUS_SET=set_1` silently train and score on the stale
    generation, while a checksum of `corpora/set_1/` still verifies clean. An
    explicitly named set must outrank a directory that merely happens to be
    there.
    """
    name = corpus_set or CORPUS_SET
    explicit = corpus_set is not None or CORPUS_SET_IS_EXPLICIT
    candidate = base / name

    # `base` contains the sets — the normal layout. A named set wins here, which
    # is what stops a flat corpus beside the sets from shadowing it.
    if is_split_root(candidate):
        return candidate

    if is_split_root(base):
        # `base` IS a split root. Either nobody named a set, or the name agrees
        # with the directory — `THESIS_CORPORA_ROOT=<...>/set_2` together with
        # `THESIS_CORPUS_SET=set_2` is a supported pattern and must
        # not be read as a request for `<...>/set_2/set_2`.
        if not explicit or base.name == name:
            return base
        raise FileNotFoundError(
            f"corpus set {name!r} was named explicitly, but {candidate} does "
            f"not exist and {base} is itself a corpus directory named "
            f"{base.name!r}. Using it would give you a different generation "
            f"under the name you asked for — the 2026-08-05 failure.\n"
            f"Point THESIS_CORPORA_ROOT at the set you mean, or move the flat "
            f"splits under {base}/<set_name>/."
        )
    available = sorted(p.name for p in base.glob("set_*") if is_split_root(p)) \
        if base.is_dir() else []
    raise FileNotFoundError(
        f"no corpus set {name!r} under {base} "
        f"(available: {available or 'none'}). Set THESIS_CORPUS_SET to one of "
        f"them, or build it with `python -m scripts.build_all_corpora --set {name}`."
    )


def _initial_root() -> Path:
    """`CORPORA_ROOT` at import time, tolerating an unbuilt set.

    Import must not fail just because the set is missing — `--help`, the unit
    tests and `build_all_corpora` itself all import this module before any
    corpus exists. The error surfaces on first use instead, via
    `resolve_corpora_root`.
    """
    try:
        return resolve_corpora_root()
    except FileNotFoundError:
        return CORPORA_BASE / CORPUS_SET


CORPORA_ROOT = _initial_root()


# --- Corpus provenance -----------------------------------------------------
# Every artifact computed FROM a corpus carries these, so two artifacts can be
# proven to describe the same generation before they are joined. Without it the
# join keys (`pair_id`, `item_uid`, `id`) are all positional and match happily
# across generations, producing numbers that are silently about two corpora.

def corpus_set_name(root: Path = CORPORA_ROOT) -> str:
    """The set this root IS, by manifest if it has one, else by directory name.

    A flat pre-set-layout root has neither, and reports `unset:<dirname>` rather
    than inventing a set name — the caller must be able to tell "set_1" from
    "a directory that is not addressed by set at all".
    """
    manifest = Path(root) / "manifest.json"
    if manifest.exists():
        try:
            name = json.loads(manifest.read_text(encoding="utf-8")).get("corpus_set")
            if name:
                return str(name)
        except json.JSONDecodeError:
            pass
    name = Path(root).name
    return name if name.startswith("set_") else f"unset:{name}"


def corpus_provenance(grammars: Sequence[str], splits: Sequence[str],
                      root: Path = CORPORA_ROOT) -> dict:
    """`{corpus_set, root, files: {"<split>/<grammar>.jsonl": sha256}}`.

    Hashes the corpus files AND their probe sidecars — the sidecar carries the
    critical spans and every label column, so a sidecar regenerated against an
    unchanged corpus changes the scores without changing the corpus hash.
    """
    files: dict[str, str] = {}
    for split in splits:
        for grammar in grammars:
            path = Path(root) / split / f"{grammar}.jsonl"
            if path.exists():
                files[f"{split}/{grammar}.jsonl"] = sha256_file(path)
            sidecar = path.with_suffix(".meta.jsonl")
            if sidecar.exists():
                files[f"{split}/{grammar}.meta.jsonl"] = sha256_file(sidecar)
    return {"corpus_set": corpus_set_name(root), "corpus_root": str(root),
            "files": files}


def set_scoped(base: Path, root: Path = CORPORA_ROOT) -> Path:
    """`base/<corpus set>` — where an artifact derived from `root` belongs.

    Derived files are as set-specific as the corpora themselves: an n-gram
    baseline, an `ell(x)` table, an aggregated term table and a probe score shard
    are all statements about ONE generation. Writing them to a shared path means
    the second set silently overwrites the first's, or worse, does not overwrite
    it and the stale file gets joined onto new scores.
    """
    return Path(base) / corpus_set_name(root)


def require_set_scoped(path: Path, root: Path = CORPORA_ROOT, *,
                       override: bool = False) -> Path:
    """`path`, if it is addressed by corpus set. Otherwise raise.

    Enforcement, not advice: a derived artifact whose path does not name a set
    cannot be told apart from the same artifact for another generation, and the
    join keys downstream are all positional, so the confusion is silent. See
    `evaluation/provenance.py`.
    """
    name = corpus_set_name(root)
    path = Path(path)
    if override or name in path.parts:
        return path
    suggested = path.parent / name / path.name
    raise SystemExit(
        f"\nERROR: {path} is not addressed by corpus set.\n"
        f"It is derived from {name!r}, and an artifact that does not name its "
        f"generation gets silently reused across generations — every join key "
        f"downstream (`id`, `pair_id`, `item_uid`) is positional and matches "
        f"anyway.\n"
        f"Write it to:  {suggested}\n"
        f"(or pass --allow-unscoped-out if this file genuinely spans sets)."
    )


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of one file."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_provenance(a: dict, b: dict, *, label_a: str = "A",
                       label_b: str = "B") -> list[str]:
    """Human-readable disagreements between two `corpus_provenance` records.

    Empty list means the two artifacts describe the same corpus bytes, for every
    file they have in common. Files present in only one are reported too: a
    baseline built over three splits and scores covering one is fine, but a
    baseline MISSING the split being scored is not, and the caller decides.
    """
    problems: list[str] = []
    if a.get("corpus_set") != b.get("corpus_set"):
        problems.append(
            f"corpus set differs: {label_a}={a.get('corpus_set')!r} "
            f"vs {label_b}={b.get('corpus_set')!r}")
    fa, fb = a.get("files") or {}, b.get("files") or {}
    shared = sorted(set(fa) & set(fb))
    if not shared:
        problems.append(
            f"{label_a} and {label_b} record no corpus file in common, so they "
            f"cannot be proven to describe the same generation")
    for name in shared:
        if fa[name] != fb[name]:
            problems.append(
                f"{name}: {label_a}={fa[name][:12]}… {label_b}={fb[name][:12]}…")
    return problems


def available_grammars(root: Path = CORPORA_ROOT) -> list[str]:
    """Grammar names present in every split directory (jsonl file stems).

    Skips the `<grammar>.meta.jsonl` metadata sidecars: they sit beside the
    sentence files and would otherwise register as grammars named "H.meta".
    """
    per_split = [
        {
            path.stem for path in (root / split).glob("*.jsonl")
            if not path.name.endswith(".meta.jsonl")
        }
        for split in SPLITS
    ]
    return sorted(set.intersection(*per_split))


def corpus_path(grammar: str, split: str, root: Path = CORPORA_ROOT) -> Path:
    path = root / split / f"{grammar}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No corpus for grammar {grammar!r}, split {split!r}: {path}"
        )
    return path


def load_corpus(grammar: str, root: Path = CORPORA_ROOT) -> DatasetDict:
    """All splits of one grammar's corpus as raw (untokenized) datasets."""
    return DatasetDict({
        split: Dataset.from_json(str(corpus_path(grammar, split, root)))
        for split in SPLITS
    })


def tokenize_corpus(corpus: DatasetDict) -> DatasetDict:
    """Maps sentences to input_ids/attention_mask, dropping raw columns."""

    def encode(batch: dict) -> dict:
        return tokenizer(batch[TEXT_FIELD])

    return corpus.map(
        encode,
        batched=True,
        remove_columns=corpus[SPLITS[0]].column_names,
    )


def assert_lexicon_matches(root: Path = CORPORA_ROOT) -> None:
    """Refuse to tokenize a corpus set with a lexicon other than the one it was
    built with (lexicon v0.3.0 changed the vocabulary). Compares the sha256 of
    the lexicon in force with every build recorded in the set's manifest; a set
    without a manifest or without recorded hashes is not checked."""
    manifest = root / "manifest.json"
    if not manifest.is_file():
        return
    builds = json.loads(manifest.read_text(encoding="utf-8")).get("builds", [])
    recorded = {b["lexicon_sha256"] for b in builds if b.get("lexicon_sha256")}
    if recorded and sha256_file(LEXICON_PATH) not in recorded:
        raise RuntimeError(
            f"Corpus set {root.name} was built with lexicon sha256 {sorted(recorded)}, "
            f"but the lexicon in force is {LEXICON_PATH} (sha256 "
            f"{sha256_file(LEXICON_PATH)}). Set THESIS_CORPUS_SET={root.name} so the "
            f"set's manifest selects its lexicon, or THESIS_LEXICON explicitly.")


def load_tokenized(grammar: str, root: Path = CORPORA_ROOT) -> DatasetDict:
    """Raw corpus loading plus tokenization in one step."""
    assert_lexicon_matches(root)
    return tokenize_corpus(load_corpus(grammar, root))


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    shuffle: bool = False,
) -> DataLoader:
    """Batches a tokenized split for causal-LM training.

    The collator pads dynamically and copies input_ids to labels with
    padding masked out (-100), matching next-token-prediction training.
    """
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collator,
    )


# --- Safety check: token <-> morpheme correspondence -----------------------

def expected_morphemes(sentence: str, separator: str, null_morpheme: str) -> list[str]:
    """Gold morpheme segmentation, independent of the tokenizers library.

    Mirrors the grammar's linearization convention: the null morpheme has no
    surface form, whitespace separates words, SEPARATOR separates morphemes.
    """
    cleaned = sentence.replace(null_morpheme, "")
    return [
        morpheme
        for word in cleaned.split()
        for morpheme in word.split(separator)
        if morpheme
    ]


def check_alignment(sentences: list[str]) -> list[dict]:
    """Compares tokenizer output against the gold segmentation.

    Returns one record per misaligned sentence: any UNK token, a missing
    BOS/EOS boundary, or an interior token sequence that differs from the
    sentence's morpheme sequence.
    """
    config = load_lexicon(LEXICON_PATH)["metadata"]["config_snapshot"]
    separator = config["SEPARATOR"]
    null_morpheme = config["NULL_MORPHEME"]

    encodings = tokenizer(sentences)["input_ids"]
    failures = []
    for sentence, ids in zip(sentences, encodings):
        tokens = tokenizer.convert_ids_to_tokens(ids)
        expected = (
            [tokenizer.bos_token]
            + expected_morphemes(sentence, separator, null_morpheme)
            + [tokenizer.eos_token]
        )
        if tokens != expected or tokenizer.unk_token in tokens:
            failures.append(
                {"sentence": sentence, "tokens": tokens, "expected": expected}
            )
    return failures


def run_safety_check(root: Path = CORPORA_ROOT, limit: int | None = None) -> bool:
    """Checks every corpus file; prints a report and returns overall success."""
    ok = True
    for grammar in available_grammars(root):
        for split in SPLITS:
            dataset = Dataset.from_json(str(corpus_path(grammar, split, root)))
            sentences = list(dataset[TEXT_FIELD][:limit] if limit else dataset[TEXT_FIELD])
            failures = check_alignment(sentences)
            status = "OK" if not failures else f"FAIL ({len(failures)} misaligned)"
            print(f"{grammar:24s} {split:22s} {len(sentences):6d} sentences  {status}")
            for failure in failures[:3]:
                print(f"  e.g. {failure['sentence']!r}")
                print(f"       tokens   {failure['tokens']}")
                print(f"       expected {failure['expected']}")
            ok = ok and not failures
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="run the token/morpheme correspondence check over all corpora",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="check only the first N sentences per file",
    )
    args = parser.parse_args()

    # Printed before anything else: every "wrong row count" incident so far has
    # been a corpora root pointing somewhere unexpected, and the set layer adds
    # one more way for that to happen silently.
    root = resolve_corpora_root()
    print(f"Corpora: {root}")

    if args.check:
        sys.exit(0 if run_safety_check(root=root, limit=args.limit) else 1)

    for grammar in available_grammars(root):
        corpus = load_tokenized(grammar, root)
        sizes = ", ".join(f"{split}={len(corpus[split])}" for split in SPLITS)
        print(f"{grammar}: {sizes}")


if __name__ == "__main__":
    main()
