"""Loads generated corpora and prepares tokenized datasets for training.

Corpora live in corpora/{split}/{grammar}.jsonl, one JSON object per line
with the sentence under "sentence". Grammar files are discovered at runtime,
so the loader keeps working when corpora are regenerated or renamed.

Tokenization delegates to models.tokenize: one token per surface morpheme,
with whitespace (word boundary) and SEPARATOR (morpheme boundary) removed
and the null morpheme deleted. Run `python -m models.dataload --check` to
verify token/morpheme correspondence over the corpora.
"""

import argparse
import sys
from pathlib import Path

from datasets import Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

from models.tokenize import LEXICON_PATH, load_lexicon, tokenizer

CORPORA_ROOT = Path(__file__).resolve().parent.parent / "corpora"
SPLITS = ("train", "validation", "test_indistribution", "test_generalization")
TEXT_FIELD = "sentence"


def available_grammars(root: Path = CORPORA_ROOT) -> list[str]:
    """Grammar names present in every split directory (jsonl file stems)."""
    per_split = [
        {path.stem for path in (root / split).glob("*.jsonl")}
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


def load_tokenized(grammar: str, root: Path = CORPORA_ROOT) -> DatasetDict:
    """Raw corpus loading plus tokenization in one step."""
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

    if args.check:
        sys.exit(0 if run_safety_check(limit=args.limit) else 1)

    for grammar in available_grammars():
        corpus = load_tokenized(grammar)
        sizes = ", ".join(f"{split}={len(corpus[split])}" for split in SPLITS)
        print(f"{grammar}: {sizes}")


if __name__ == "__main__":
    main()
