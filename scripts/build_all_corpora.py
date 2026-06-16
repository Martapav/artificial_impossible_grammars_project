"""CLI entry point for corpus construction across all four grammar conditions.

Instantiates the appropriate generator(s) and drives CorpusBuilder to produce
train / test / generalization splits.  Mixed grammars (H_prime, L_prime) create
one generator per rule_type.
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


GRAMMAR_CHOICES = ["H", "H_prime", "L_prime", "P", "all"]
RULE_TYPE_CHOICES = ["hierarchical", "linear"]


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
        "--rule-type",
        choices=RULE_TYPE_CHOICES,
        default=None,
        help=(
            "For mixed grammars (H_prime, L_prime): which rule type to generate. "
            "If omitted when --grammar is 'all', both rule types are generated."
        ),
    )
    p.add_argument(
        "--train-n",
        type=int,
        default=50_000,
        help="Number of sentences in the training split.",
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


def _make_generators(grammar: str, rule_type: str | None, seed: int):
    """Yield (label, generator) pairs for the requested grammar condition(s).

    The label (e.g. "H_prime_hierarchical") becomes the output filename stem.
    Mixed grammars yield one generator per rule type.
    """
    if grammar == "all":
        targets = ["H", "H_prime", "L_prime", "P"]
    else:
        targets = [grammar]

    for g in targets:
        if g == "H":
            yield g, GrammarHGenerator(seed=seed)
        elif g == "P":
            yield g, GrammarPGenerator(seed=seed)
        elif g in ("H_prime", "L_prime"):
            cls = GrammarHPrimeGenerator if g == "H_prime" else GrammarLPrimeGenerator
            # If --rule-type was specified, use it; otherwise generate both variants.
            rule_types = [rule_type] if rule_type is not None else RULE_TYPE_CHOICES
            for rt in rule_types:
                yield f"{g}_{rt}", cls(rule_type=rt, seed=seed)
        else:  # pragma: no cover — argparse restricts choices, so unreachable
            raise ValueError(f"unknown grammar {g!r}")


def run(args: argparse.Namespace) -> None:
    """Build corpora for each grammar condition and print the output paths."""
    for label, generator in _make_generators(args.grammar, args.rule_type, args.seed):
        print(f"[{label}] building corpora -> {args.output_dir}")
        builder = CorpusBuilder(generator, output_root=args.output_dir)
        paths = builder.build_all(train_n=args.train_n, test_n=args.test_n)
        for split, path in paths.items():
            print(f"  {split}: {path}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
