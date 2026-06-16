"""Entry point for lexicon generation and verification.

Usage:
    python -m lexicon.cli --seed 42 --out lexicon.json [--max-stem-length 3]
    python -m lexicon.cli --verify lexicon.json
"""

import argparse
import sys
from pathlib import Path


def _print_summary(lexicon) -> None:
    print("Lexicon generated successfully.")
    print(f"  Content hash : {lexicon.metadata.content_hash}")
    print(f"  Seed         : {lexicon.metadata.seed}")
    print(f"  Timestamp    : {lexicon.metadata.generation_timestamp}")
    print("  Category sizes:")
    for cat, count in sorted(lexicon.metadata.category_sizes.items()):
        print(f"    {cat:<12} {count}")


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate, validate, and serialize a new lexicon."""
    from .generation import assemble_lexicon
    from .serialization import serialize
    from .validation import validate

    try:
        lexicon = assemble_lexicon(seed=args.seed, max_stem_length=args.max_stem_length)
        validate(lexicon)
        out = Path(args.out)
        serialize(lexicon, out)
        _print_summary(lexicon)
        print(f"  Output path  : {out.resolve()}")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def cmd_verify(args: argparse.Namespace) -> int:
    """Load and validate an existing lexicon without regenerating."""
    from .serialization import deserialize
    from .validation import LexiconValidationError

    try:
        lexicon = deserialize(Path(args.verify))
        print(f"Verified OK  : {Path(args.verify).resolve()}")
        print(f"  Content hash : {lexicon.metadata.content_hash}")
        return 0
    except LexiconValidationError as exc:
        print(f"Integrity error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch to generate or verify mode.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Exit code (0 = success, 1 = failure).
    """
    from .config import MAX_STEM_LENGTH

    parser = argparse.ArgumentParser(
        prog="python -m lexicon.cli",
        description="Generate or verify a lexicon artifact for the artificial grammar project.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Generate mode (default when --seed/--out are given at top level)
    gen = parser.add_argument_group("generate mode")
    gen.add_argument("--seed", type=int, default=None, help="RNG seed for generation")
    gen.add_argument("--out", type=str, default=None, help="Output JSON path")
    gen.add_argument(
        "--max-stem-length",
        type=int,
        default=MAX_STEM_LENGTH,
        dest="max_stem_length",
        help=f"Maximum combinatorial stem length (default: {MAX_STEM_LENGTH})",
    )

    # Verify mode
    parser.add_argument(
        "--verify",
        type=str,
        default=None,
        metavar="PATH",
        help="Verify an existing lexicon JSON without regenerating",
    )

    args = parser.parse_args(argv)

    if args.verify is not None:
        return cmd_verify(args)

    if args.seed is None or args.out is None:
        parser.print_help(sys.stderr)
        print("\nError: --seed and --out are required for generation mode.", file=sys.stderr)
        return 1

    return cmd_generate(args)


if __name__ == "__main__":
    sys.exit(main())
