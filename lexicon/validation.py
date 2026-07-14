"""Post-generation integrity checks for lexicon.

validate() runs all checks in order and raises on the first failure.

Spec reference: CLAUDE.md §2.
"""

import dataclasses
from typing import Any

from .schema import CAT1Item, Lexicon


class LexiconValidationError(Exception):
    """Raised when any lexicon validation check fails."""


class LexiconIntegrityError(LexiconValidationError):
    """Raised specifically when the stored content_hash does not match the recomputed hash."""


def _all_items(lexicon: Lexicon) -> list[Any]:
    """Collect every item from every category into a flat list."""
    return list(
        lexicon.cat1 + lexicon.cat1pron + lexicon.cat2 + lexicon.cat3
        + lexicon.cat3aux + lexicon.cat4 + lexicon.cat5 + lexicon.cat6
        + lexicon.cat8 + lexicon.cat9
    )


def validate(lexicon: Lexicon) -> None:
    """Run all integrity checks on the Lexicon.

    Checks are run in order; the first failure raises immediately.

    Args:
        lexicon: The Lexicon to validate.

    Raises:
        LexiconIntegrityError: If the content_hash check fails.
        LexiconValidationError: If any other check fails.
    """
    _check_1_alphabet_disjoint(lexicon)
    _check_2_letter_digit_exclusivity(lexicon)
    _check_3_id_uniqueness(lexicon)
    _check_4_form_uniqueness_within_category(lexicon)
    _check_5_cat1pron_structure(lexicon)
    _check_6_cat4_structure(lexicon)
    _check_7_cat6_partition(lexicon)
    _check_8_cat8_singleton(lexicon)
    _check_9_cat9_structure(lexicon)
    _check_10_inflectional_slots_coverage(lexicon)
    _check_11_cat4_required_consistency(lexicon)
    _check_12_content_hash(lexicon)
    _check_13_category_sizes_metadata(lexicon)


# ── Individual checks ────────────────────────────────────────────────────────


def _check_1_alphabet_disjoint(lexicon: Lexicon) -> None:
    from .config import LETTER_PARTITIONS  # noqa: PLC0415

    seen: dict[str, str] = {}
    for cat, letters in LETTER_PARTITIONS.items():
        for letter in letters:
            if not (letter.islower() and letter.isalpha()):
                raise LexiconValidationError(
                    f"Check 1 failed: {letter!r} in {cat} is not a lowercase Latin letter."
                )
            if letter in seen:
                raise LexiconValidationError(
                    f"Check 1 failed: letter {letter!r} appears in both {seen[letter]} and {cat}."
                )
            seen[letter] = cat


def _check_2_letter_digit_exclusivity(lexicon: Lexicon) -> None:
    from .config import DIGIT_PARTITIONS  # noqa: PLC0415

    all_digit_values: set[str] = set()
    for values in DIGIT_PARTITIONS.values():
        for v in values:
            if not v.isdigit():
                raise LexiconValidationError(
                    f"Check 2 failed: inflectional value {v!r} is not a digit."
                )
            all_digit_values.add(v)

    for item in _all_items(lexicon):
        form = item.form
        for ch in form:
            if ch.isdigit():
                raise LexiconValidationError(
                    f"Check 2 failed: stem form {form!r} (id={item.id!r}) contains digit {ch!r}."
                )


def _check_3_id_uniqueness(lexicon: Lexicon) -> None:
    seen: dict[str, str] = {}
    for item in _all_items(lexicon):
        if item.id in seen:
            raise LexiconValidationError(
                f"Check 3 failed: duplicate ID {item.id!r} in "
                f"{item.category} and {seen[item.id]}."
            )
        seen[item.id] = item.category


def _check_4_form_uniqueness_within_category(lexicon: Lexicon) -> None:
    category_groups: dict[str, list] = {}
    for item in _all_items(lexicon):
        category_groups.setdefault(item.category, []).append(item)

    for cat, items in category_groups.items():
        seen_forms: set[str] = set()
        for item in items:
            if item.form in seen_forms:
                raise LexiconValidationError(
                    f"Check 4 failed: duplicate form {item.form!r} within {cat}."
                )
            seen_forms.add(item.form)


def _check_5_cat1pron_structure(lexicon: Lexicon) -> None:
    if len(lexicon.cat1pron) != 2:
        raise LexiconValidationError(
            f"Check 5 failed: CAT1PRON must have exactly 2 items, got {len(lexicon.cat1pron)}."
        )
    subclasses = {item.subclass for item in lexicon.cat1pron}
    if subclasses != {"Pron", "Refl"}:
        raise LexiconValidationError(
            f"Check 5 failed: CAT1PRON subclasses must be {{'Pron', 'Refl'}}, got {subclasses}."
        )


def _check_6_cat4_structure(lexicon: Lexicon) -> None:
    """Spec §2.2: CAT4 has exactly two items — singular and plural."""
    if len(lexicon.cat4) != 2:
        raise LexiconValidationError(
            f"Check 6 failed: CAT4 must have exactly 2 items, got {len(lexicon.cat4)}."
        )
    numbers = {item.number for item in lexicon.cat4}
    if numbers != {"singular", "plural"}:
        raise LexiconValidationError(
            f"Check 6 failed: CAT4 numbers must be {{'singular', 'plural'}}, got {numbers}."
        )


def _check_7_cat6_partition(lexicon: Lexicon) -> None:
    """Spec §2.4: CAT6 has three items partitioned 2:1 (NP-selecting : VP-selecting)."""
    if len(lexicon.cat6) != 3:
        raise LexiconValidationError(
            f"Check 7 failed: CAT6 must have exactly 3 items, got {len(lexicon.cat6)}."
        )
    np_count = sum(1 for x in lexicon.cat6 if x.attachment == "cat1_selecting")
    vp_count = sum(1 for x in lexicon.cat6 if x.attachment == "cat3_selecting")
    if (np_count, vp_count) != (2, 1):
        raise LexiconValidationError(
            f"Check 7 failed: CAT6 attachment partition must be 2 cat1_selecting : 1 cat3_selecting, "
            f"got {np_count}:{vp_count}."
        )


def _check_8_cat8_singleton(lexicon: Lexicon) -> None:
    """Spec §2.2: CAT8 is the singleton wh particle {w}."""
    if len(lexicon.cat8) != 1:
        raise LexiconValidationError(
            f"Check 8 failed: CAT8 must have exactly 1 item, got {len(lexicon.cat8)}."
        )


def _check_9_cat9_structure(lexicon: Lexicon) -> None:
    """Spec §2.2: CAT9 has exactly two subordinator items {s, t}."""
    if len(lexicon.cat9) != 2:
        raise LexiconValidationError(
            f"Check 9 failed: CAT9 must have exactly 2 items, got {len(lexicon.cat9)}."
        )


def _check_10_inflectional_slots_coverage(lexicon: Lexicon) -> None:
    known_params = {p.name for p in lexicon.inflectional.parameters}
    for item in _all_items(lexicon):
        slots = getattr(item, "inflectional_slots", ())
        for slot in slots:
            if slot not in known_params:
                raise LexiconValidationError(
                    f"Check 10 failed: item {item.id!r} has slot {slot!r} "
                    f"with no matching InflectionalParameter."
                )


def _check_11_cat4_required_consistency(lexicon: Lexicon) -> None:
    for item in lexicon.cat1:
        expected = item.countability == "countable"
        if item.cat4_required != expected:
            raise LexiconValidationError(
                f"Check 11 failed: CAT1 item {item.id!r} has countability={item.countability!r} "
                f"but cat4_required={item.cat4_required} (expected {expected})."
            )


def _check_12_content_hash(lexicon: Lexicon) -> None:
    from .serialization import _dataclass_to_dict, compute_content_hash  # noqa: PLC0415

    items_dict = {
        "cat1":         [_dataclass_to_dict(x) for x in lexicon.cat1],
        "cat1pron":     [_dataclass_to_dict(x) for x in lexicon.cat1pron],
        "cat2":         [_dataclass_to_dict(x) for x in lexicon.cat2],
        "cat3":         [_dataclass_to_dict(x) for x in lexicon.cat3],
        "cat3aux":      [_dataclass_to_dict(x) for x in lexicon.cat3aux],
        "cat4":         [_dataclass_to_dict(x) for x in lexicon.cat4],
        "cat5":         [_dataclass_to_dict(x) for x in lexicon.cat5],
        "cat6":         [_dataclass_to_dict(x) for x in lexicon.cat6],
        "cat8":         [_dataclass_to_dict(x) for x in lexicon.cat8],
        "cat9":         [_dataclass_to_dict(x) for x in lexicon.cat9],
        "inflectional": _dataclass_to_dict(lexicon.inflectional),
    }
    recomputed = compute_content_hash(items_dict)
    if recomputed != lexicon.metadata.content_hash:
        raise LexiconIntegrityError(
            f"Check 12 failed: stored hash {lexicon.metadata.content_hash!r} "
            f"does not match recomputed hash {recomputed!r}."
        )


def _check_13_category_sizes_metadata(lexicon: Lexicon) -> None:
    actual = {
        "CAT1":     len(lexicon.cat1),
        "CAT1PRON": len(lexicon.cat1pron),
        "CAT2":     len(lexicon.cat2),
        "CAT3":     len(lexicon.cat3),
        "CAT3AUX":  len(lexicon.cat3aux),
        "CAT4":     len(lexicon.cat4),
        "CAT5":     len(lexicon.cat5),
        "CAT6":     len(lexicon.cat6),
        "CAT8":     len(lexicon.cat8),
        "CAT9":     len(lexicon.cat9),
    }
    recorded = lexicon.metadata.category_sizes
    for cat, count in actual.items():
        if recorded.get(cat) != count:
            raise LexiconValidationError(
                f"Check 13 failed: metadata records {cat}={recorded.get(cat)} "
                f"but actual count is {count}."
            )
