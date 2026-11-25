"""Post-generation integrity checks for lexicon.

validate() runs all checks in order and raises on the first failure.

Spec reference: experiment.md §2.
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
    if lexicon.affixes:                       # lexicon v0.3.0 (experiment.md §2.6)
        _check_14_affix_inventory(lexicon)
        _check_15_entry_structure(lexicon)
        _check_16_stem_consistency(lexicon)
        _check_17_morph_quotas(lexicon)
        _check_18_zipf_ranks(lexicon)


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
    from .generation import items_dict  # noqa: PLC0415
    from .serialization import compute_content_hash  # noqa: PLC0415

    parts = {f: getattr(lexicon, f) for f in (
        "cat1", "cat1pron", "cat2", "cat3", "cat3aux", "cat4", "cat5", "cat6",
        "cat8", "cat9", "inflectional", "affixes")}
    recomputed = compute_content_hash(items_dict(parts))
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


# ── Lexicon v0.3.0 checks (derivational morphology, experiment.md §2.6) ──────

_OPEN = (("CAT1", "cat1"), ("CAT2", "cat2"), ("CAT3", "cat3"), ("CAT5", "cat5"))


def _check_14_affix_inventory(lexicon: Lexicon) -> None:
    """Sizes per slot, 2-3 letters, inside the category pool, globally unique,
    and never equal to any stem."""
    from .config import AFFIX_INVENTORY, AFFIX_LENGTHS, LETTER_PARTITIONS  # noqa: PLC0415

    forms: dict[str, str] = {}
    for a in lexicon.affixes:
        if len(a.form) not in AFFIX_LENGTHS:
            raise LexiconValidationError(
                f"Check 14 failed: affix {a.form!r} has length {len(a.form)}, not in {AFFIX_LENGTHS}.")
        if set(a.form) - set(LETTER_PARTITIONS[a.category]):
            raise LexiconValidationError(
                f"Check 14 failed: affix {a.form!r} uses letters outside the {a.category} pool.")
        if a.form in forms:
            raise LexiconValidationError(
                f"Check 14 failed: affix form {a.form!r} appears twice ({forms[a.form]}, {a.id}).")
        forms[a.form] = a.id
    for cat, slots in AFFIX_INVENTORY.items():
        for slot, k in slots.items():
            got = sum(1 for a in lexicon.affixes if a.category == cat and a.slot == slot)
            if got != k:
                raise LexiconValidationError(
                    f"Check 14 failed: {cat} {slot} has {got} affixes, expected {k}.")
    for _, key in _OPEN:
        for e in getattr(lexicon, key):
            if e.stem in forms:
                raise LexiconValidationError(
                    f"Check 14 failed: stem {e.stem!r} ({e.id}) is a reserved affix string.")


def _check_15_entry_structure(lexicon: Lexicon) -> None:
    """form == PRE#STEM#S_inner#S_outer; each affix is of the entry's category and
    in its slot; slot limits; n_morphemes matches."""
    from .config import SEPARATOR  # noqa: PLC0415

    slot_of = {(a.category, a.form): a.slot for a in lexicon.affixes}
    for cat, key in _OPEN:
        for e in getattr(lexicon, key):
            parts = [m for m in (e.prefix, e.stem, e.suffix_inner, e.suffix_outer) if m]
            if e.form != SEPARATOR.join(parts):
                raise LexiconValidationError(
                    f"Check 15 failed: {e.id} form {e.form!r} != joined morphemes {parts}.")
            if e.n_morphemes != len(parts):
                raise LexiconValidationError(
                    f"Check 15 failed: {e.id} n_morphemes={e.n_morphemes}, has {len(parts)}.")
            for slot in ("prefix", "suffix_inner", "suffix_outer"):
                v = getattr(e, slot)
                if v and slot_of.get((cat, v)) != slot:
                    raise LexiconValidationError(
                        f"Check 15 failed: {e.id} has {v!r} in {slot}, which is not a {cat} {slot}.")


def _check_16_stem_consistency(lexicon: Lexicon) -> None:
    """Each stem has ENTRIES_PER_STEM entries, exactly one bare (form == stem),
    no duplicate entry, and all its entries share the stem's features."""
    from .config import ENTRIES_PER_STEM  # noqa: PLC0415

    feature_fields = ("inherent_gender", "countability", "cat4_required",
                      "cat6_compatible", "transitivity")
    for cat, key in _OPEN:
        by_stem: dict[str, list] = {}
        for e in getattr(lexicon, key):
            by_stem.setdefault(e.stem_id, []).append(e)
        if len(by_stem) != lexicon.metadata.stem_counts.get(cat):
            raise LexiconValidationError(
                f"Check 16 failed: {cat} has {len(by_stem)} stems, metadata says "
                f"{lexicon.metadata.stem_counts.get(cat)}.")
        for sid, es in by_stem.items():
            if len(es) != ENTRIES_PER_STEM:
                raise LexiconValidationError(
                    f"Check 16 failed: stem {sid} has {len(es)} entries, expected {ENTRIES_PER_STEM}.")
            if sum(1 for e in es if e.n_morphemes == 1) != 1:
                raise LexiconValidationError(f"Check 16 failed: stem {sid} needs exactly one bare entry.")
            if len({e.stem for e in es}) != 1:
                raise LexiconValidationError(f"Check 16 failed: stem {sid} entries disagree on the stem.")
            for f in feature_fields:
                if len({getattr(e, f, None) for e in es}) != 1:
                    raise LexiconValidationError(
                        f"Check 16 failed: stem {sid} entries disagree on feature {f}.")


def _check_17_morph_quotas(lexicon: Lexicon) -> None:
    """Entry shares by lexical-morpheme count match generation.morph_quota exactly."""
    from .generation import morph_quota  # noqa: PLC0415

    for cat, key in _OPEN:
        entries = getattr(lexicon, key)
        expected = morph_quota(lexicon.metadata.stem_counts[cat])
        got = {k: sum(1 for e in entries if e.n_morphemes == k) for k in expected}
        if got != expected:
            raise LexiconValidationError(
                f"Check 17 failed: {cat} morpheme-count quotas {got} != expected {expected}.")


def _check_18_zipf_ranks(lexicon: Lexicon) -> None:
    """Ranks are a permutation of 1..N within each open category."""
    for cat, key in _OPEN:
        ranks = sorted(e.zipf_rank for e in getattr(lexicon, key))
        if ranks != list(range(1, len(ranks) + 1)):
            raise LexiconValidationError(
                f"Check 18 failed: {cat} zipf ranks are not a permutation of 1..{len(ranks)}.")
