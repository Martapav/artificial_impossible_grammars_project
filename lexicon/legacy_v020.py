"""FROZEN lexicon v0.2.0 generator — kept only to reproduce ``lexicon.json``.

Lexicon v0.2.0 built every corpus up to set_3: open-class stems ENUMERATED
exhaustively (all strings up to length = letter-pool size), Zipf rank = stem
order (shortest = most frequent), no derivational morphology. It was replaced by
the sampled lexicon v0.3.0 (``lexicon.generation``, experiment.md §2.6, §3).

This module is a verbatim copy of ``lexicon/generation.py`` at commit 0c91285,
with one change: its configuration is the FROZEN snapshot recorded in
``lexicon.json`` (``CONFIG_V020`` below) instead of ``lexicon.config``, so later
edits to the live config or generator can never alter it. It shares only the
data model (``lexicon.schema``) and the hashing (``lexicon.serialization``),
whose v0.3.0 additions are omitted at their defaults, so a v0.2.0 lexicon
hashes exactly as before. ``tests/test_legacy_lexicon.py`` guards the
reproduction: ``assemble_lexicon(42)`` must equal ``lexicon.json`` (content hash
and every item).

    python -m lexicon.cli --lexicon-version 0.2.0 --seed 42 --out lexicon.json

Do not modify. New design work goes into ``lexicon.generation``.

Random draw order (canonical — changing this order breaks determinism):
  Phase 1 — CAT1 feature assignment (stems in stem-id order: length asc, then lex):
    draw 1: inherent_gender    (rng.choice(("1", "2")))
    draw 2: countability       (rng.random() < P_PROPER_LIKE → "proper_like", else "countable")
    draw 3: cat6_compatible    (rng.random() < P_CAT6_COMPATIBLE → True, else False)
  Phase 2 — CAT3 transitivity (stems in stem-id order):
    draw 1: transitivity       (rng.choice(("transitive", "intransitive")))

"""

import dataclasses
import itertools
import random
from datetime import datetime, timezone

# ── Frozen v0.2.0 configuration (verbatim: lexicon.json metadata.config_snapshot) ──
CONFIG_V020: dict = {
    "CAT6_PP_UNDER_NP_LETTERS": [
        "r",
        "y"
    ],
    "CAT6_PP_UNDER_VP_LETTERS": [
        "z"
    ],
    "CATEGORY_MAX_STEM_LENGTH": {
        "CAT1": 5,
        "CAT2": 4,
        "CAT3": 4,
        "CAT5": 3
    },
    "CATEGORY_SIZE_CAPS": {},
    "COMBINATORIAL_CATEGORIES": [
        "CAT1",
        "CAT2",
        "CAT3",
        "CAT5"
    ],
    "DIGIT_PARTITIONS": {
        "INFL1_gender": [
            "1",
            "2"
        ],
        "INFL1_number": [
            "3",
            "4"
        ],
        "INFL3_tense": [
            "7",
            "8",
            "9"
        ]
    },
    "LETTER_PARTITIONS": {
        "CAT1": [
            "a",
            "b",
            "c",
            "d",
            "e"
        ],
        "CAT1PRON": [
            "m",
            "mm"
        ],
        "CAT2": [
            "f",
            "g",
            "h",
            "x"
        ],
        "CAT3": [
            "i",
            "j",
            "k",
            "l"
        ],
        "CAT3AUX": [
            "n"
        ],
        "CAT4": [
            "u",
            "v"
        ],
        "CAT5": [
            "o",
            "p",
            "q"
        ],
        "CAT6": [
            "r",
            "y",
            "z"
        ],
        "CAT8": [
            "w"
        ],
        "CAT9": [
            "s",
            "t"
        ]
    },
    "MAX_STEM_LENGTH": 5,
    "NULL_MORPHEME": "#NULL#",
    "P_CAT6_COMPATIBLE": 0.5,
    "P_PROPER_LIKE": 0.1,
    "SEPARATOR": "#",
    "VERSION": "0.2.0",
    "ZIPF_CATEGORIES": [
        "CAT1",
        "CAT2",
        "CAT3",
        "CAT5"
    ],
    "ZIPF_Q": 0.0,
    "ZIPF_S": 1.0
}

VERSION: str = CONFIG_V020["VERSION"]
MAX_STEM_LENGTH: int = CONFIG_V020["MAX_STEM_LENGTH"]
P_PROPER_LIKE: float = CONFIG_V020["P_PROPER_LIKE"]
P_CAT6_COMPATIBLE: float = CONFIG_V020["P_CAT6_COMPATIBLE"]
CATEGORY_SIZE_CAPS: dict = CONFIG_V020["CATEGORY_SIZE_CAPS"]
LETTER_PARTITIONS: dict = {k: tuple(v) for k, v in CONFIG_V020["LETTER_PARTITIONS"].items()}
CATEGORY_MAX_STEM_LENGTH: dict = dict(CONFIG_V020["CATEGORY_MAX_STEM_LENGTH"])
CAT6_PP_UNDER_NP_LETTERS: tuple = tuple(CONFIG_V020["CAT6_PP_UNDER_NP_LETTERS"])
CAT6_PP_UNDER_VP_LETTERS: tuple = tuple(CONFIG_V020["CAT6_PP_UNDER_VP_LETTERS"])
DIGIT_PARTITIONS: dict = {k: tuple(v) for k, v in CONFIG_V020["DIGIT_PARTITIONS"].items()}


def assign_zipf_ranks(items: list) -> list:
    """zipf_rank = 1-based position (v0.2.0: generation order = frequency order)."""
    return [dataclasses.replace(item, zipf_rank=i + 1) for i, item in enumerate(items)]

from .schema import (
    CAT1Item,
    CAT1PRONItem,
    CAT2Item,
    CAT3AUXItem,
    CAT3Item,
    CAT4Item,
    CAT5Item,
    CAT6Item,
    CAT8Item,
    CAT9Item,
    InflectionalInventory,
    InflectionalParameter,
    Lexicon,
    LexiconMetadata,
)


def partition_alphabet() -> dict[str, tuple[str, ...]]:
    """Return the letter partition mapping from config, validated for disjointness.

    Returns:
        Mapping from category name to its letter tuple.

    Raises:
        ValueError: If any two categories share a letter, or if any entry is not
            a lowercase Latin letter.
    """
    partitions = LETTER_PARTITIONS
    all_letters: list[str] = []
    for cat, letters in partitions.items():
        for letter in letters:
            if not (letter.islower() and letter.isalpha()):
                raise ValueError(
                    f"Partition entry {letter!r} for {cat} is not a lowercase Latin letter."
                )
        all_letters.extend(letters)

    if len(all_letters) != len(set(all_letters)):
        seen: set[str] = set()
        for cat, letters in partitions.items():
            for letter in letters:
                if letter in seen:
                    raise ValueError(
                        f"Letter {letter!r} appears in more than one category partition."
                    )
                seen.add(letter)

    return partitions

def generate_combinatorial_stems(alphabet: tuple[str, ...], max_length: int) -> list[str]:
    """Generate all strings of lengths 1..max_length over alphabet, sorted (len, lex).

    Args:
        alphabet: The letter symbols available for this category.
        max_length: Maximum stem length (inclusive).

    Returns:
        Sorted list of all permutations with repetition of lengths 1..max_length.
    """
    stems = []
    for length in range(1, max_length + 1):
        for combo in itertools.product(alphabet, repeat=length):
            stems.append("".join(combo))
    stems.sort(key=lambda s: (len(s), s))
    return stems


def _make_stem_id(category_lower: str, index: int) -> str:
    """Format a stable stem ID as '{category_lower}_{index:04d}'.

    Args:
        category_lower: Lowercased category name (e.g. 'cat1').
        index: Zero-based position in the sorted stem list.

    Returns:
        ID string such as 'cat1_0003'.
    """
    return f"{category_lower}_{index:04d}"


def assign_cat1_features(stems: list[str], rng: random.Random) -> list[CAT1Item]:
    """Assign inherent features to CAT1 stems using the provided RNG instance.

    For each stem (in stem-id order): draws gender, then countability, then cat6_compatible.

    Args:
        stems: Sorted list of CAT1 stem strings.
        rng: Seeded Random instance; consumed in documented order.

    Returns:
        List of CAT1Item with all features assigned.
    """
    items: list[CAT1Item] = []
    for idx, form in enumerate(stems):
        gender = rng.choice(("1", "2"))
        countability = "proper_like" if rng.random() < P_PROPER_LIKE else "countable"
        cat6_compat = rng.random() < P_CAT6_COMPATIBLE
        items.append(
            CAT1Item(
                id=_make_stem_id("cat1", idx),
                form=form,
                inherent_gender=gender,
                countability=countability,
                cat4_required=(countability == "countable"),
                cat6_compatible=cat6_compat,
            )
        )
    return items


def build_cat2(stems: list[str]) -> list[CAT2Item]:
    """Build CAT2 items from stems with no randomized features."""
    return [CAT2Item(id=_make_stem_id("cat2", idx), form=form) for idx, form in enumerate(stems)]


def assign_cat3_transitivity(stems: list[str], rng: random.Random) -> list[CAT3Item]:
    """Assign transitivity to CAT3 stems using the provided RNG instance.

    For each stem (in stem-id order): draws transitivity uniformly.

    Args:
        stems: Sorted list of CAT3 stem strings.
        rng: Seeded Random instance; consumed in documented order.

    Returns:
        List of CAT3Item with transitivity assigned.
    """
    items: list[CAT3Item] = []
    for idx, form in enumerate(stems):
        transitivity = rng.choice(("transitive", "intransitive"))
        items.append(CAT3Item(id=_make_stem_id("cat3", idx), form=form, transitivity=transitivity))
    return items


def build_cat5(stems: list[str]) -> list[CAT5Item]:
    """Build CAT5 items from stems with no randomized features."""
    return [CAT5Item(id=_make_stem_id("cat5", idx), form=form) for idx, form in enumerate(stems)]


def build_cat4() -> list[CAT4Item]:
    """Build the two CAT4 determiner items.

    Spec §2.2: {u}=singular, {v}=plural.
    """
    return [
        CAT4Item(id="cat4_u", form="u", number="singular"),
        CAT4Item(id="cat4_v", form="v", number="plural"),
    ]


def build_cat1pron() -> list[CAT1PRONItem]:
    """Build the two hardcoded CAT1PRON pronominal items.

    Spec §2.2: {m}=Pron, {mm}=Refl.
    """
    return [
        CAT1PRONItem(id="cat1pron_pron", form="m", subclass="Pron"),
        CAT1PRONItem(id="cat1pron_refl", form="mm", subclass="Refl"),
    ]


def build_cat9() -> list[CAT9Item]:
    """Build the two hardcoded CAT9 subordinator items.

    Spec §2.2: {s, t}. Both head Type4 (relative clauses).
    """
    return [
        CAT9Item(id="cat9_s", form="s"),
        CAT9Item(id="cat9_t", form="t"),
    ]


def build_cat8() -> list[CAT8Item]:
    """Build the singleton CAT8 wh particle.

    Spec §2.2: {w}. Single item used only by wh-movement.
    """
    return [CAT8Item(id="cat8_w", form="w")]


def build_cat3aux() -> list[CAT3AUXItem]:
    """Build the singleton CAT3AUX auxiliary item.

    Spec §2.2: {n}.
    """
    return [CAT3AUXItem(id="cat3aux_n")]


def build_cat6() -> list[CAT6Item]:
    """Build the three CAT6 prepositional items with attachment partition.

    Spec §2.4: three items partitioned 2:1 — two CAT1-selecting (PP-under-NP),
    one CAT3-selecting (PP-under-VP). 
    """
    items: list[CAT6Item] = []
    for letter in CAT6_PP_UNDER_NP_LETTERS:
        items.append(CAT6Item(id=f"cat6_{letter}", form=letter, attachment="cat1_selecting"))
    for letter in CAT6_PP_UNDER_VP_LETTERS:
        items.append(CAT6Item(id=f"cat6_{letter}", form=letter, attachment="cat3_selecting"))
    return items


def build_inflectional_inventory() -> InflectionalInventory:
    """Build the complete inflectional inventory from the canonical parameter definitions."""
    parameters = (
        InflectionalParameter(
            name="INFL1_number",
            values=("3", "4"),
            dimension_label="number-like",
            applies_to=("CAT1", "CAT1PRON", "CAT2", "CAT3", "CAT3AUX"),
        ),
        InflectionalParameter(
            name="INFL1_gender",
            values=("1", "2"),
            dimension_label="gender-like",
            applies_to=("CAT1", "CAT1PRON", "CAT2", "CAT3", "CAT3AUX"),
        ),
        InflectionalParameter(
            name="INFL3_tense",
            values=("7", "8", "9"),
            dimension_label="tense",
            applies_to=("CAT3", "CAT3AUX"),
        ),
    )
    return InflectionalInventory(parameters=parameters)


def assemble_lexicon(
    seed: int,
    max_stem_length: int = MAX_STEM_LENGTH,
    category_max_stem_length: dict[str, int] | None = None,
) -> Lexicon:
    """The RNG is consumed in the order documented at the top of this module.
    Two calls with the same seed (and the same length settings) produce an
    identical Lexicon.

    Args:
        seed: Integer seed for the random.Random instance.
        max_stem_length: Global fallback max stem length, used for any
            combinatorial category absent from ``category_max_stem_length``.
        category_max_stem_length: Per-category max combinatorial stem length.
            Defaults to the frozen ``CATEGORY_MAX_STEM_LENGTH``. A category listed
            here overrides ``max_stem_length`` for that category.

    Returns:
        Fully assembled Lexicon with metadata and content hash.
    """
    from .serialization import compute_content_hash

    rng = random.Random(seed)

    # Resolve the per-category stem-length bound (per-category entry wins;
    # otherwise fall back to the global max_stem_length).
    cml = CATEGORY_MAX_STEM_LENGTH if category_max_stem_length is None else category_max_stem_length
    resolved_lengths = {
        cat: cml.get(cat, max_stem_length) for cat in ("CAT1", "CAT2", "CAT3", "CAT5")
    }

    # Generate combinatorial stems (each category at its own length bound).
    cat1_stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT1"], resolved_lengths["CAT1"])
    cat2_stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT2"], resolved_lengths["CAT2"])
    cat3_stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT3"], resolved_lengths["CAT3"])
    cat5_stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT5"], resolved_lengths["CAT5"])

    # Build all categories (order matters for RNG consumption).
    # Open-class categories (CAT1, CAT2, CAT3, CAT5) receive 1-based frequency
    # ranks via assign_zipf_ranks().  Stems are already in (length asc, lex asc)
    # order from generate_combinatorial_stems(), so rank 1 = shortest/first stem,
    # i.e. the most frequent word.  Closed-class categories keep zipf_rank=0.
    cat1 = tuple(assign_zipf_ranks(assign_cat1_features(cat1_stems, rng)))
    cat2 = tuple(assign_zipf_ranks(build_cat2(cat2_stems)))
    cat3 = tuple(assign_zipf_ranks(assign_cat3_transitivity(cat3_stems, rng)))
    cat4 = tuple(build_cat4())
    cat5 = tuple(assign_zipf_ranks(build_cat5(cat5_stems)))
    cat6 = tuple(build_cat6())
    cat8 = tuple(build_cat8())
    cat9 = tuple(build_cat9())
    cat1pron = tuple(build_cat1pron())
    cat3aux = tuple(build_cat3aux())
    inflectional = build_inflectional_inventory()

    # Build items dict (excluding metadata) for hashing
    from .serialization import _dataclass_to_dict  # noqa: PLC0415

    items_dict: dict = {
        "cat1":     [_dataclass_to_dict(x) for x in cat1],
        "cat1pron": [_dataclass_to_dict(x) for x in cat1pron],
        "cat2":     [_dataclass_to_dict(x) for x in cat2],
        "cat3":     [_dataclass_to_dict(x) for x in cat3],
        "cat3aux":  [_dataclass_to_dict(x) for x in cat3aux],
        "cat4":     [_dataclass_to_dict(x) for x in cat4],
        "cat5":     [_dataclass_to_dict(x) for x in cat5],
        "cat6":     [_dataclass_to_dict(x) for x in cat6],
        "cat8":     [_dataclass_to_dict(x) for x in cat8],
        "cat9":     [_dataclass_to_dict(x) for x in cat9],
        "inflectional": _dataclass_to_dict(inflectional),
    }
    content_hash = compute_content_hash(items_dict)

    import json as _json
    config_snapshot = _json.loads(_json.dumps(CONFIG_V020))

    category_sizes = {
        "CAT1":     len(cat1),
        "CAT1PRON": len(cat1pron),
        "CAT2":     len(cat2),
        "CAT3":     len(cat3),
        "CAT3AUX":  len(cat3aux),
        "CAT4":     len(cat4),
        "CAT5":     len(cat5),
        "CAT6":     len(cat6),
        "CAT8":     len(cat8),
        "CAT9":     len(cat9),
    }

    category_alphabets = {cat: list(letters) for cat, letters in LETTER_PARTITIONS.items()}

    metadata = LexiconMetadata(
        version=VERSION,
        seed=seed,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        max_stem_length=resolved_lengths,
        category_sizes=category_sizes,
        category_alphabets=category_alphabets,
        config_snapshot=config_snapshot,
        content_hash=content_hash,
    )

    return Lexicon(
        metadata=metadata,
        cat1=cat1,
        cat1pron=cat1pron,
        cat2=cat2,
        cat3=cat3,
        cat3aux=cat3aux,
        cat4=cat4,
        cat5=cat5,
        cat6=cat6,
        cat8=cat8,
        cat9=cat9,
        inflectional=inflectional,
    )
