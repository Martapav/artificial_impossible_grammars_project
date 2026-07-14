"""CLI entry point for corpus construction across all four grammar conditions.

Instantiates the appropriate generator(s) and drives CorpusBuilder to produce
train / validation / test / generalization splits.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ensure project root is importable
sys.path.insert(0, str(PROJECT_ROOT))

from corpora.corpus_builder import CorpusBuilder
from grammars.grammar_H.generate import GrammarHGenerator
from grammars.grammar_H_prime.generator import GrammarHPrimeGenerator
from grammars.grammar_L_prime.generator import GrammarLPrimeGenerator
from grammars.grammar_P.generator import GrammarPGenerator


GRAMMAR_CHOICES = ["H", "P", "Hprime", "Lprime", "all"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build train / test / generalization corpora for the thesis grammars.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--grammar",
        choices=GRAMMAR_CHOICES,
        default="all",
        help="Which grammar to generate.  'all' runs all four conditions.",
    )
    p.add_argument(
        "--train-n",
        type=int,
        default=100_000,
        help="Number of sentences in the training split.",
    )
    p.add_argument(
        "--val-n",
        type=int,
        default=5_000,
        help="Number of sentences in the validation split.",
    )
    p.add_argument(
        "--test-n",
        type=int,
        default=5_000,
        help="Number of sentences in the in-distribution test split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.  Held-out splits use seed + 100000.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "corpora",
        help="Root output directory.  Sub-directories train/ etc. are created if missing.",
    )
    return p


def _make_generators(grammar: str, seed: int):
    """Yield (label, generator) pairs for the requested grammar condition(s).

    The label (e.g. "Hprime") becomes the output filename stem.
    """
    if grammar == "all":
        targets = ["H", "P", "Hprime", "Lprime"]
    else:
        targets = [grammar]

    factories = {
        "H": GrammarHGenerator,
        "P": GrammarPGenerator,
        "Hprime": GrammarHPrimeGenerator,
        "Lprime": GrammarLPrimeGenerator,
    }
    for g in targets:
        yield g, factories[g](seed=seed)


def run(args: argparse.Namespace) -> None:
    """Build corpora for each grammar condition and print the output paths."""
    for label, generator in _make_generators(args.grammar, args.seed):
        print(f"[{label}] building corpora -> {args.output_dir}")
        builder = CorpusBuilder(generator, output_root=args.output_dir)
        paths = builder.build_all(train_n=args.train_n, val_n=args.val_n, test_n=args.test_n)
        for split, path in paths.items():
            print(f"  {split}: {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
