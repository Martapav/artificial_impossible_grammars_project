"""Dataclasses defining the lexicon structure.

All dataclasses are frozen to prevent accidental mutation after construction.
Tuple fields are used throughout (not lists) to preserve hashability.

"""

from dataclasses import dataclass, field

# Lexicon v0.3.0 entry fields (experiment.md §2.6). Marked OMIT_DEFAULT so that a
# field holding its default value is left out of the serialized/hashed dict:
# a v0.2.0 lexicon (no derivational morphology) serializes and hashes exactly
# as before, so its stored content_hash still validates.
OMIT_DEFAULT = {"omit_default": True}


@dataclass(frozen=True)
class CAT1Item:
    """Noun-like category with inherent gender, countability, and modifier compatibility."""

    id: str
    form: str
    category: str = field(default="CAT1")
    inherent_gender: str = field(default="")        # "1" or "2"
    countability: str = field(default="")           # "countable" or "proper_like"
    cat4_required: bool = field(default=False)      # True iff countability == "countable"
    cat6_compatible: bool = field(default=False)    # fixed at generation time
    inflectional_slots: tuple = field(default=("INFL1_number", "INFL1_gender"))
    zipf_rank: int = field(default=0)               # 1-based frequency rank; 0 = unranked (closed-class sentinel)
    # v0.3.0 derived entry: `form` is the full derived string, e.g. "ab#cde#fg"
    stem: str = field(default="", metadata=OMIT_DEFAULT)
    stem_id: str = field(default="", metadata=OMIT_DEFAULT)
    prefix: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_inner: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_outer: str = field(default="", metadata=OMIT_DEFAULT)
    n_morphemes: int = field(default=0, metadata=OMIT_DEFAULT)   # 1-4; 0 = v0.2.0 item


@dataclass(frozen=True)
class CAT1PRONItem:
    """Pronominal category inflecting for gender and number only (no person distinction)."""

    id: str
    form: str
    category: str = field(default="CAT1PRON")  # person inflection omitted
    subclass: str = field(default="")              # "Pron" or "Refl"
    inflectional_slots: tuple = field(default=("INFL1_number", "INFL1_gender"))


@dataclass(frozen=True)
class CAT2Item:
    """Adjectival category agreeing in gender and number."""

    id: str
    form: str
    category: str = field(default="CAT2")
    agreement_targets: tuple = field(default=("INFL1_number", "INFL1_gender"))
    inflectional_slots: tuple = field(default=("INFL1_number", "INFL1_gender"))
    zipf_rank: int = field(default=0)
    # v0.3.0 derived entry: `form` is the full derived string, e.g. "ab#cde#fg"
    stem: str = field(default="", metadata=OMIT_DEFAULT)
    stem_id: str = field(default="", metadata=OMIT_DEFAULT)
    prefix: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_inner: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_outer: str = field(default="", metadata=OMIT_DEFAULT)
    n_morphemes: int = field(default=0, metadata=OMIT_DEFAULT)   # 1-4; 0 = v0.2.0 item


@dataclass(frozen=True)
class CAT3Item:
    """Verbal category with transitivity, agreement, and tense inflection."""

    id: str
    form: str
    category: str = field(default="CAT3")
    transitivity: str = field(default="")          # "transitive" or "intransitive"
    agreement_targets: tuple = field(default=("INFL1_number", "INFL1_gender"))
    free_features: tuple = field(default=("INFL3_tense",))
    inflectional_slots: tuple = field(default=("INFL1_number", "INFL1_gender", "INFL3_tense"))
    zipf_rank: int = field(default=0)
    # v0.3.0 derived entry: `form` is the full derived string, e.g. "ab#cde#fg"
    stem: str = field(default="", metadata=OMIT_DEFAULT)
    stem_id: str = field(default="", metadata=OMIT_DEFAULT)
    prefix: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_inner: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_outer: str = field(default="", metadata=OMIT_DEFAULT)
    n_morphemes: int = field(default=0, metadata=OMIT_DEFAULT)   # 1-4; 0 = v0.2.0 item


@dataclass(frozen=True)
class CAT3AUXItem:
    """Auxiliary verbal category; singleton with form 'n'."""

    id: str
    form: str = field(default="n")
    category: str = field(default="CAT3AUX")
    agreement_targets: tuple = field(default=("INFL1_number", "INFL1_gender"))
    free_features: tuple = field(default=("INFL3_tense",))
    inflectional_slots: tuple = field(default=("INFL1_number", "INFL1_gender", "INFL3_tense"))


@dataclass(frozen=True)
class CAT4Item:
    """Determiner; intrinsically number-marked, non-inflecting.

    Spec §2.2: exactly two items — {u}=singular, {v}=plural.
    """

    id: str
    form: str
    category: str = field(default="CAT4")
    number: str = field(default="")                 # "singular" or "plural"
    inflectional_slots: tuple = field(default=())


@dataclass(frozen=True)
class CAT5Item:
    """Adverbial category with no inflection."""

    id: str
    form: str
    category: str = field(default="CAT5")
    inflectional_slots: tuple = field(default=())
    zipf_rank: int = field(default=0)
    # v0.3.0 derived entry: `form` is the full derived string, e.g. "ab#cde#fg"
    stem: str = field(default="", metadata=OMIT_DEFAULT)
    stem_id: str = field(default="", metadata=OMIT_DEFAULT)
    prefix: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_inner: str = field(default="", metadata=OMIT_DEFAULT)
    suffix_outer: str = field(default="", metadata=OMIT_DEFAULT)
    n_morphemes: int = field(default=0, metadata=OMIT_DEFAULT)   # 1-4; 0 = v0.2.0 item


@dataclass(frozen=True)
class CAT6Item:
    """Prepositional category with attachment partition.

    Spec §2.4: three items partitioned 2:1. Two select CAT1 (PP-under-NP);
    one selects CAT3 (PP-under-VP).
    """

    id: str
    form: str
    category: str = field(default="CAT6")
    attachment: str = field(default="")            # "cat1_selecting" or "cat3_selecting"
    inflectional_slots: tuple = field(default=())


@dataclass(frozen=True)
class CAT8Item:
    """Wh particle; singleton.

    Spec §2.2: exactly one item — {w}.
    """

    id: str
    form: str = field(default="w")
    category: str = field(default="CAT8")
    inflectional_slots: tuple = field(default=())


@dataclass(frozen=True)
class CAT9Item:
    """Subordinator heading a relative clause (Type4).

    Spec §2.2: exactly two items — {s, t}.
    """

    id: str
    form: str
    category: str = field(default="CAT9")
    inflectional_slots: tuple = field(default=())


@dataclass(frozen=True)
class AffixItem:
    """A derivational affix (lexicon v0.3.0, experiment.md §2.6).

    Feature-free: carries only its category, its side and its position class.
    ``slot`` is "prefix", "suffix_inner" or "suffix_outer".
    """

    id: str
    form: str
    category: str      # the open category it attaches to (CAT1/CAT2/CAT3/CAT5)
    slot: str


@dataclass(frozen=True)
class InflectionalParameter:
    """A single inflectional dimension with its possible values and scope."""

    name: str
    values: tuple
    dimension_label: str
    applies_to: tuple                              # category names


@dataclass(frozen=True)
class InflectionalInventory:
    """The complete inflectional system: parameters and morpheme ordering rules.

    Spec §1: "for INFL1-bearing items, number morpheme precedes gender morpheme".
    """

    parameters: tuple                              # tuple of InflectionalParameter
    morpheme_order_INFL1: tuple = field(default=("INFL1_number", "INFL1_gender"))
    morpheme_order_INFL3: tuple = field(default=("INFL3_tense",))
    # Stem precedes all inflectional morphemes. INFL3 follows INFL1 on verbs.
    global_order: tuple = field(default=("INFL1", "INFL3"))


@dataclass(frozen=True)
class LexiconMetadata:
    """Provenance and integrity information for generated lexicon."""

    version: str
    seed: int
    generation_timestamp: str      # ISO 8601 UTC
    max_stem_length: dict          # per-category max combinatorial stem length, {cat: int}
    category_sizes: dict
    category_alphabets: dict
    config_snapshot: dict
    content_hash: str              # SHA-256 over canonical JSON of all items (not metadata)
    stem_counts: dict = field(default_factory=dict, metadata=OMIT_DEFAULT)  # v0.3.0: stems per open category


@dataclass(frozen=True)
class Lexicon:
    """The complete lexicon: all category inventories plus inflectional system."""

    metadata: LexiconMetadata
    cat1: tuple
    cat1pron: tuple
    cat2: tuple
    cat3: tuple
    cat3aux: tuple
    cat4: tuple
    cat5: tuple
    cat6: tuple
    cat8: tuple
    cat9: tuple
    inflectional: InflectionalInventory
    affixes: tuple = field(default=(), metadata=OMIT_DEFAULT)   # v0.3.0: AffixItem tuple
