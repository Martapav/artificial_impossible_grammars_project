"""CLI entry point for corpus construction across all four grammar conditions.

Instantiates the appropriate generator(s) and drives CorpusBuilder to produce
the six locked splits (train / validation / test_indistribution /
structural_probe_IOD / structural_probe_OOD / structural_probe_SELECT —
TRANSFORM_RULES §10).

Corpus sets
-----------
Every generation goes into its own set directory, `corpora/set_N/`, and **a set
is never written twice**: rebuilding into an existing set aborts unless
`--force`. A trained model's corpus therefore stays on disk, byte for byte, for
as long as the model is interesting — which is what makes a re-score, a second
training set, or a comparison between two generations possible at all. Without
it a rebuild silently redefines what every earlier result was measured on, and
because the corpus `id` is a dense line index, the old scores would still join,
onto the wrong items.

    python -m scripts.build_all_corpora --set set_2 --seed 43
    THESIS_CORPUS_SET=set_2 python -m models.train ...

**A new set needs a new `--seed`.** The generators are deterministic in it, so
`--set set_2` at the default seed reproduces `set_1` byte for byte: two set
directories, one generation, and every comparison between them vacuous.
`guard_seed_unique` refuses a seed a sibling set's manifest already records.

`manifest.json` in each set records what produced it (git SHA, lexicon hash,
seed, sizes, per-file sha256), so a set can be identified without trusting its
name. `models.dataload` resolves which set everything downstream reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # ensure project root is importable
sys.path.insert(0, str(PROJECT_ROOT))

from corpusgen.corpus_builder import CorpusBuilder
from corpusgen.probe_checks import check_probe_disjointness
from grammars.grammar_H.generate import GrammarHGenerator
from grammars.grammar_H_prime.generator import GrammarHPrimeGenerator
from grammars.grammar_L_prime.generator import GrammarLPrimeGenerator
from grammars.grammar_P.generate import GrammarPGenerator


GRAMMAR_CHOICES = ["H", "P", "Hprime", "Lprime", "all"]
CORPORA_BASE = PROJECT_ROOT / "corpora"
MANIFEST_FILENAME = "manifest.json"
SPLIT_NAMES = ("train", "validation", "test_indistribution",
               "structural_probe_IOD", "structural_probe_OOD",
               "structural_probe_SELECT")


def next_set_name(base: Path = CORPORA_BASE) -> str:
    """`set_N` for the lowest N not already present."""
    existing = {p.name for p in base.glob("set_*")} if base.is_dir() else set()
    n = 1
    while f"set_{n}" in existing:
        n += 1
    return f"set_{n}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def target_grammars(grammar: str) -> list[str]:
    """The grammar labels one invocation builds."""
    return ["H", "P", "Hprime", "Lprime"] if grammar == "all" else [grammar]


def would_overwrite(out_dir: Path, grammars: list[str],
                    probes_only: bool) -> list[str]:
    """The corpus files this invocation would write over, if any.

    Per FILE, not per split directory. A set is legitimately built up one
    grammar at a time (`--grammar H`, then `--grammar P` into the same set) —
    a directory-level check would abort the second invocation because `train/`
    exists, even though it holds only `H.jsonl` and nothing would be lost.
    """
    splits = [s for s in SPLIT_NAMES if s.startswith("structural_probe_")] \
        if probes_only else list(SPLIT_NAMES)
    return [f"{split}/{grammar}.jsonl"
            for split in splits for grammar in grammars
            if (out_dir / split / f"{grammar}.jsonl").exists()]


def seeds_in_use(base: Path = CORPORA_BASE,
                 exclude: Path | None = None) -> dict[int, str]:
    """`{seed: set_name}` over every sibling set's manifest.

    Only sets that record a manifest are visible here. A set built before
    manifests existed contributes nothing, which is a gap the guard reports
    rather than hides.
    """
    out: dict[int, str] = {}
    if not base.is_dir():
        return out
    for path in sorted(base.glob("set_*")):
        if exclude is not None and path.resolve() == exclude.resolve():
            continue
        manifest = path / MANIFEST_FILENAME
        if not manifest.exists():
            continue
        try:
            builds = json.loads(manifest.read_text(encoding="utf-8")).get("builds", [])
        except json.JSONDecodeError:
            continue
        for build in builds:
            seed = build.get("seed")
            if isinstance(seed, int):
                out.setdefault(seed, path.name)
    return out


def guard_seed_unique(out_dir: Path, seed: int, *, base: Path = CORPORA_BASE,
                      force: bool = False) -> None:
    """Refuse to build a set with a seed another set already used.

    The generators are deterministic in the seed, so `--set set_2` at the
    default seed does not produce a second generation — it reproduces the first
    one byte for byte. That defeats the entire purpose of the set layout while
    looking like it worked: two sets, one corpus, and any comparison between
    them is vacuous. (Observed 2026-08-07: set_1 and set_2 were identical across
    all 48 files, both at seed 42.)

    A set may legitimately be filled by several invocations, so a seed already
    used by THIS set is fine — only a collision with a *sibling* set is fatal.
    """
    if force:
        return
    clash = seeds_in_use(base, exclude=out_dir).get(seed)
    if clash is None:
        return
    raise SystemExit(
        f"seed {seed} was already used to build '{clash}'.\n"
        f"The generators are deterministic in the seed, so building "
        f"'{out_dir.name}' with it would reproduce '{clash}' byte for byte "
        f"rather than create a new generation.\n"
        f"Pass an unused --seed (e.g. --seed {suggest_seed(base)}), or --force "
        f"if you genuinely mean to duplicate '{clash}'."
    )


def suggest_seed(base: Path = CORPORA_BASE) -> int:
    """A seed no existing set has used, for the guard's error message."""
    used = set(seeds_in_use(base))
    seed = 42
    while seed in used:
        seed += 1
    return seed


def guard_output_dir(out_dir: Path, grammars: list[str], *,
                     probes_only: bool, force: bool) -> None:
    """Refuse to overwrite an existing set.

    The whole point of the set layout is that a generation is immutable once
    written, so this is an abort and not a warning.

    The guard looks at the splits this invocation would WRITE, which under
    `--probes-only` is the three probe splits and nothing else. Checking the
    others instead — as the first version of this function did — gets the
    protection exactly backwards: it aborts on train/validation/test, which
    `--probes-only` never touches, while letting it overwrite the probe splits
    in place. That overwrite is not harmless. `pair_id` is positional, so a
    rebuilt probe re-points every previously written probe score onto a
    different item, and `item_uid` (`{pair_id}/{variant}`) does not save you.
    """
    if force or not out_dir.exists():
        return
    clash = would_overwrite(out_dir, grammars, probes_only)
    if clash:
        shown = ", ".join(clash[:6]) + (f", +{len(clash) - 6} more"
                                        if len(clash) > 6 else "")
        raise SystemExit(
            f"{out_dir} already has {shown}.\n"
            f"A corpus set is never overwritten: models trained on it record "
            f"its hashes, and a rebuild would silently redefine what they were "
            f"measured on.\nBuild a new set (--set {next_set_name(out_dir.parent)}) "
            f"or pass --force if you genuinely mean to discard this one."
        )


def write_manifest(out_dir: Path, args: argparse.Namespace,
                   grammars: list[str]) -> Path:
    """Record what produced this set, so it can be identified without its name.

    A set can be filled one grammar at a time, so this is APPEND-shaped: the
    file list and hashes describe the whole set as it now stands, while `builds`
    keeps one entry per invocation that contributed to it. Recording only the
    last invocation would have the manifest claim `grammars: ["P"]` over a
    directory holding H and P — the seed and sizes would then describe some of
    the files and not others, with nothing saying which.
    """
    from models.tokenize import LEXICON_PATH

    files = {}
    grammars_present: set[str] = set()
    for split in SPLIT_NAMES:
        for path in sorted((out_dir / split).glob("*.jsonl")):
            files[f"{split}/{path.name}"] = {
                "sha256": sha256(path),
                "n_rows": sum(1 for _ in path.open(encoding="utf-8")),
            }
            if not path.name.endswith(".meta.jsonl"):
                grammars_present.add(path.stem)

    path = out_dir / MANIFEST_FILENAME
    previous = []
    if path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8")).get("builds", [])
        except json.JSONDecodeError:
            previous = []

    build = {
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "grammars": grammars,
        "seed": args.seed,
        "sizes": {"train": args.train_n, "validation": args.val_n,
                  "test_indistribution": args.test_n,
                  "probe_n_per_subcase": args.probe_n_per_subcase,
                  "select_n_per_subcase": args.select_n_per_subcase},
        "probes_only": bool(args.probes_only),
        "forced": bool(args.force),
        # Per build, not per set: two invocations into one set can run from
        # different commits or a different lexicon, and averaging that away
        # would hide exactly the inconsistency worth catching.
        "git_sha": _git_sha(),
        "lexicon_sha256": sha256(LEXICON_PATH),
        "lexicon_path": _repo_relative(LEXICON_PATH),
        "python": platform.python_version(),
    }

    manifest = {
        "corpus_set": out_dir.name,
        # The lexicon this set was built with; lexicon.paths.resolve_lexicon_path
        # reads it back, so THESIS_CORPUS_SET=<this set> loads the right
        # vocabulary for training and scoring without any flag.
        "lexicon_path": _repo_relative(LEXICON_PATH),
        "grammars_present": sorted(grammars_present),
        "builds": previous + [build],
        "files": files,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build the six corpus splits for the thesis grammars.",
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
        default=10_000,
        help="Number of sentences in the validation split.",
    )
    p.add_argument(
        "--test-n",
        type=int,
        default=15_000,
        help="Number of sentences in the in-distribution test split.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed.  Held-out splits use seed + 100000.  Must not "
             "collide with a seed another set already recorded: the generators "
             "are deterministic, so a repeated seed rebuilds that set rather "
             "than producing a new generation.",
    )
    p.add_argument(
        "--set",
        dest="corpus_set",
        default=None,
        help="Corpus set to write, e.g. 'set_2'.  Default: the lowest set_N "
             "not already present, so a plain invocation never overwrites.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Full output root, bypassing --set.  Sub-directories train/ etc. "
             "are created if missing.  Default: corpora/<set>.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite a set that already has corpora in it.  Discards a "
             "generation other results may depend on; think before using it.",
    )
    p.add_argument(
        "--probes-only",
        action="store_true",
        help="Rebuild only the structural probes (IOD / OOD / SELECT), leaving "
             "train, validation and test_indistribution untouched.  The probes "
             "are evaluation-only minimal-pair corpora, so rebuilding them does "
             "not invalidate any trained model.  Needs an explicit --set: on its "
             "own it would create a new set holding probes and no training data.",
    )
    p.add_argument(
        "--probe-n-per-subcase",
        type=int,
        default=300,
        help="Pairs per subcase in the IOD and OOD reporting probes. "
             "Total pairs = this x 7 subcases; items = pairs x 2.",
    )
    p.add_argument(
        "--lexicon",
        type=str,
        default=None,
        help="Lexicon JSON to build with (e.g. lexicons/lexicon_v0.3.0.json). "
             "Recorded in the set's manifest; default: the lexicon in force "
             "(THESIS_LEXICON, else repo-root lexicon.json).",
    )
    p.add_argument(
        "--select-n-per-subcase",
        type=int,
        default=300,
        help="Pairs per subcase in the SELECT probe (recipe selection only).",
    )
    return p


def _make_generators(grammar: str, seed: int):
    """Yield (label, generator) pairs for the requested grammar condition(s).

    The label (e.g. "Hprime") becomes the output filename stem.
    """
    targets = target_grammars(grammar)
    factories = {
        "H": GrammarHGenerator,
        "P": GrammarPGenerator,
        "Hprime": GrammarHPrimeGenerator,
        "Lprime": GrammarLPrimeGenerator,
    }
    for g in targets:
        yield g, factories[g](seed=seed)


def resolve_output_dir(args: argparse.Namespace) -> Path:
    """Where this invocation writes: `--output-dir`, else `corpora/<set>`."""
    if args.output_dir is not None:
        return args.output_dir
    if args.corpus_set is None and args.probes_only:
        # Auto-naming would invent an empty set and fill it with probes alone:
        # no train/validation/test, so `models.dataload` will not even see it as
        # a set. Rebuilding probes means rebuilding them INSIDE some existing
        # generation, which has to be named.
        raise SystemExit(
            "--probes-only needs an explicit --set (e.g. --set set_1).\n"
            "Without one it would build a new set containing probes and no "
            "training data. Note that rebuilding probes inside an existing set "
            "overwrites them, so it also needs --force — and it re-points every "
            "probe score already written against that set, because pair_id is "
            "positional."
        )
    return CORPORA_BASE / (args.corpus_set or next_set_name())


def run(args: argparse.Namespace) -> None:
    """Build corpora for each grammar condition and print the output paths."""
    out_dir = resolve_output_dir(args)
    guard_output_dir(out_dir, target_grammars(args.grammar),
                     probes_only=args.probes_only, force=args.force)
    guard_seed_unique(out_dir, args.seed, force=args.force)
    print(f"Corpus set: {out_dir}  (seed {args.seed})")
    _bind_lexicon(args, out_dir)

    built: list[str] = []
    for label, generator in _make_generators(args.grammar, args.seed):
        built.append(label)
        print(f"[{label}] building corpora -> {out_dir}")
        builder = CorpusBuilder(generator, output_root=out_dir)
        if args.probes_only:
            paths = {
                "structural_probe_IOD": builder.build_structural_probe_iod(
                    args.probe_n_per_subcase),
                "structural_probe_OOD": builder.build_structural_probe_ood(
                    args.probe_n_per_subcase),
                "structural_probe_SELECT": builder.build_structural_probe_select(
                    args.select_n_per_subcase),
            }
        else:
            paths = builder.build_all(
                train_n=args.train_n, val_n=args.val_n, test_n=args.test_n,
                probe_n_per_subcase=args.probe_n_per_subcase,
                select_n_per_subcase=args.select_n_per_subcase,
            )
        for split, path in paths.items():
            print(f"  {split}: {path}")

        # SELECT picks the recipe, IOD/OOD report the result: a pair shared
        # between them would make the reported number one that selection had
        # already optimized. Nothing in the builder dedups, so verify it.
        counts = check_probe_disjointness(out_dir, label)
        if counts:
            print("  probes disjoint: "
                  + ", ".join(f"{s.replace('structural_probe_', '')}={n}"
                              for s, n in counts.items()))

    manifest_path = write_manifest(out_dir, args, built)
    print(f"\nManifest: {manifest_path}")
    print(f"To train on this set: THESIS_CORPUS_SET={out_dir.name} "
          f"python -m models.train ...")


def _repo_relative(path: Path) -> str:
    """Repo-relative POSIX path when inside the repo, else absolute."""
    from lexicon.paths import REPO_ROOT
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _bind_lexicon(args: argparse.Namespace, out_dir: Path) -> None:
    """Pin the lexicon for this build (THESIS_LEXICON) and refuse to mix two
    lexicons inside one corpus set."""
    import os

    from lexicon.paths import ENV_LEXICON, manifest_lexicon_path, resolve_lexicon_path
    if args.lexicon:
        lex = Path(args.lexicon).resolve()
        if not lex.is_file():
            sys.exit(f"--lexicon {args.lexicon}: no such file")
        os.environ[ENV_LEXICON] = str(lex)
    chosen = resolve_lexicon_path()
    recorded = manifest_lexicon_path(out_dir.name)
    if recorded is not None and recorded.resolve() != chosen.resolve():
        sys.exit(f"Corpus set {out_dir.name} was built with lexicon {recorded}; "
                 f"this build would use {chosen}. One set = one lexicon.")
    print(f"Lexicon: {chosen}")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
