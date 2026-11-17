"""Tokenizer safety check: every token must be exactly one surface morpheme.

Samples the head of each corpus file for speed; run
`python -m models.dataload --check` for full-corpus coverage.
"""

import json

import pytest

from models.dataload import (
    CORPORA_ROOT,
    SPLITS,
    TEXT_FIELD,
    available_grammars,
    check_alignment,
    corpus_path,
)

SAMPLE_SIZE = 200


def sample_sentences(grammar: str, split: str) -> list[str]:
    sentences = []
    with open(corpus_path(grammar, split), encoding="utf-8") as fh:
        for line in fh:
            sentences.append(json.loads(line)[TEXT_FIELD])
            if len(sentences) >= SAMPLE_SIZE:
                break
    return sentences


def corpus_files() -> list[tuple[str, str]]:
    return [
        (grammar, split)
        for grammar in available_grammars(CORPORA_ROOT)
        for split in SPLITS
    ]


@pytest.mark.parametrize("grammar,split", corpus_files())
def test_tokens_match_morphemes(grammar: str, split: str) -> None:
    failures = check_alignment(sample_sentences(grammar, split))
    assert not failures, (
        f"{len(failures)} misaligned sentences in {grammar}/{split}, "
        f"first: {failures[0]}"
    )
