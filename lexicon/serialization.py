"""JSON I/O and content hashing for Lexicon artifacts.

serialize() writes a Lexicon to disk.
deserialize() loads, reconstructs, and validates a Lexicon from disk.
compute_content_hash() produces a SHA-256 over the canonical items JSON.
"""

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

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


def _dataclass_to_dict(obj: Any) -> Any:
    """Recursively convert a dataclass to plain Python types.

    Tuples become lists (for JSON compatibility). Primitives pass through unchanged.

    Args:
        obj: A dataclass instance, tuple, list, dict, or primitive.

    Returns:
        A JSON-serializable value.
    """
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type): #If you need to know if a class is an instance of a dataclass (and not a dataclass itself), then add a further check for not isinstance(obj, type)
        return {k: _dataclass_to_dict(v) for k, v in dataclasses.asdict(obj).items()} #.asdict converst dataclass object in a dictionary. so this line: for each key in the converted dict, it associates the values. it's just an object type conversion step
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    return obj


def compute_content_hash(items_dict: dict) -> str:
    """Compute SHA-256 (hashing of the string to set length) over the canonical (sorted-keys, no-whitespace) JSON of items.  
    """
    canonical = json.dumps(items_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _reconstruct_inflectional_parameter(d: dict) -> InflectionalParameter:
    return InflectionalParameter(
        name=d["name"],
        values=tuple(d["values"]),
        dimension_label=d["dimension_label"],
        applies_to=tuple(d["applies_to"]),
    )


def _reconstruct_inflectional_inventory(d: dict) -> InflectionalInventory:
    return InflectionalInventory(
        parameters=tuple(_reconstruct_inflectional_parameter(p) for p in d["parameters"]),
        morpheme_order_INFL1=tuple(d["morpheme_order_INFL1"]),
        morpheme_order_INFL1PRON=tuple(d["morpheme_order_INFL1PRON"]),
        morpheme_order_INFL3=tuple(d["morpheme_order_INFL3"]),
        global_order=tuple(d["global_order"]),
    )


def _reconstruct_lexicon(data: dict) -> Lexicon:
    """Reconstruct a Lexicon from a parsed JSON dict.

    Args:
        data: Parsed JSON dict as produced by serialize().

    Returns:
        Fully reconstructed Lexicon dataclass.
    """
    m = data["metadata"]
    metadata = LexiconMetadata(
        version=m["version"],
        seed=m["seed"],
        generation_timestamp=m["generation_timestamp"],
        max_stem_length=m["max_stem_length"],
        category_sizes=m["category_sizes"],
        category_alphabets=m["category_alphabets"],
        config_snapshot=m["config_snapshot"],
        content_hash=m["content_hash"],
    )

    cat1 = tuple(
        CAT1Item(
            id=x["id"], form=x["form"], category=x["category"],
            inherent_gender=x["inherent_gender"], countability=x["countability"],
            cat4_required=x["cat4_required"], cat6_compatible=x["cat6_compatible"],
            inflectional_slots=tuple(x["inflectional_slots"]),
            zipf_rank=x.get("zipf_rank", 0),
        )
        for x in data["cat1"]
    )
    cat1pron = tuple(
        CAT1PRONItem(
            id=x["id"], form=x["form"], category=x["category"],
            subclass=x["subclass"], inflectional_slots=tuple(x["inflectional_slots"]),
        )
        for x in data["cat1pron"]
    )
    cat2 = tuple(
        CAT2Item(
            id=x["id"], form=x["form"], category=x["category"],
            agreement_targets=tuple(x["agreement_targets"]),
            inflectional_slots=tuple(x["inflectional_slots"]),
            zipf_rank=x.get("zipf_rank", 0),
        )
        for x in data["cat2"]
    )
    cat3 = tuple(
        CAT3Item(
            id=x["id"], form=x["form"], category=x["category"],
            transitivity=x["transitivity"],
            agreement_targets=tuple(x["agreement_targets"]),
            free_features=tuple(x["free_features"]),
            inflectional_slots=tuple(x["inflectional_slots"]),
            zipf_rank=x.get("zipf_rank", 0),
        )
        for x in data["cat3"]
    )
    cat3aux = tuple(
        CAT3AUXItem(
            id=x["id"], form=x["form"], category=x["category"],
            agreement_targets=tuple(x["agreement_targets"]),
            free_features=tuple(x["free_features"]),
            inflectional_slots=tuple(x["inflectional_slots"]),
        )
        for x in data["cat3aux"]
    )
    cat4 = tuple(
        CAT4Item(
            id=x["id"], form=x["form"], category=x["category"],
            number=x["number"],
            inflectional_slots=tuple(x["inflectional_slots"]),
        )
        for x in data["cat4"]
    )
    cat5 = tuple(
        CAT5Item(
            id=x["id"], form=x["form"], category=x["category"],
            inflectional_slots=tuple(x["inflectional_slots"]),
            zipf_rank=x.get("zipf_rank", 0),
        )
        for x in data["cat5"]
    )
    cat6 = tuple(
        CAT6Item(
            id=x["id"], form=x["form"], category=x["category"],
            attachment=x["attachment"],
            inflectional_slots=tuple(x["inflectional_slots"]),
        )
        for x in data["cat6"]
    )
    cat8 = tuple(
        CAT8Item(
            id=x["id"], form=x["form"], category=x["category"],
            inflectional_slots=tuple(x["inflectional_slots"]),
        )
        for x in data["cat8"]
    )
    cat9 = tuple(
        CAT9Item(
            id=x["id"], form=x["form"], category=x["category"],
            inflectional_slots=tuple(x["inflectional_slots"]),
        )
        for x in data["cat9"]
    )
    inflectional = _reconstruct_inflectional_inventory(data["inflectional"])

    return Lexicon(
        metadata=metadata,
        cat1=cat1, cat1pron=cat1pron, cat2=cat2, cat3=cat3, cat3aux=cat3aux,
        cat4=cat4, cat5=cat5, cat6=cat6, cat8=cat8, cat9=cat9,
        inflectional=inflectional,
    )


def serialize(lexicon: Lexicon, path: Path) -> None:
    """Write a Lexicon to a JSON file with sorted keys and 2-space indent.

    Args:
        lexicon: The Lexicon to serialize.
        path: Destination file path.
    """
    data = _dataclass_to_dict(lexicon)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, sort_keys=True, indent=2, ensure_ascii=False)


def deserialize(path: Path) -> Lexicon:
    """Load and reconstruct a Lexicon from a JSON file, then validate it.

    Runs the full validation pass after reconstruction. Raises LexiconIntegrityError
    if the stored content_hash does not match the recomputed hash.

    Args:
        path: Source file path.

    Returns:
        Validated Lexicon.

    Raises:
        LexiconIntegrityError: If the hash check fails.
        LexiconValidationError: If any other validation check fails.
    """
    from .validation import validate  # noqa: PLC0415 (avoid circular import at module level)

    with open(Path(path), encoding="utf-8") as fh:
        data = json.load(fh)
    lexicon = _reconstruct_lexicon(data)
    validate(lexicon)
    return lexicon
