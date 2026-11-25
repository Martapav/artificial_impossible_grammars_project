"""Entry point for lexicon generation and verification.

Usage:
    python -m lexicon.cli --seed 42 --out lexicons/lexicon_v0.3.0.json
    python -m lexicon.cli --lexicon-version 0.2.0 --seed 42 --out lexicon.json   # frozen legacy
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
    if lexicon.metadata.stem_counts:
        print(f"  Stems per open category: {lexicon.metadata.stem_counts}")
        print(f"  Affixes      : {len(lexicon.affixes)}")


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate, validate, and serialize a new lexicon."""
    from .serialization import serialize
    from .validation import validate

    try:
        if args.lexicon_version == "0.2.0":
            from .legacy_v020 import assemble_lexicon   # frozen v0.2.0 (set_3 and earlier)
        else:
            from .generation import assemble_lexicon    # current v0.3.0
        lexicon = assemble_lexicon(seed=args.seed)
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
        "--lexicon-version", choices=("0.3.0", "0.2.0"), default="0.3.0",
        help="0.3.0 = current sampled lexicon with derivational morphology "
             "(lexicon.generation); 0.2.0 = frozen enumerated lexicon that built "
             "set_3 and earlier (lexicon.legacy_v020).")

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
