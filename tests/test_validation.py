"""Tests for lexicon/validation.py."""

import dataclasses

import pytest

from lexicon import assemble_lexicon
from lexicon.generation import partition_alphabet
from lexicon.validation import LexiconValidationError, validate


@pytest.fixture(scope="module")
def lexicon():
    return assemble_lexicon(seed=42)


def test_alphabet_disjoint():
    """partition_alphabet() should pass disjointness check without raising."""
    partitions = partition_alphabet()
    all_letters = [l for letters in partitions.values() for l in letters]
    assert len(all_letters) == len(set(all_letters))


def test_letter_digit_exclusivity(lexicon):
    """No stem form may contain a digit; no inflectional value may be a letter."""
    from lexicon.config import DIGIT_PARTITIONS

    all_digit_values = {v for values in DIGIT_PARTITIONS.values() for v in values}
    all_forms = [item.form for cat in (
        lexicon.cat1, lexicon.cat1pron, lexicon.cat2, lexicon.cat3,
        lexicon.cat3aux, lexicon.cat4, lexicon.cat5, lexicon.cat6,
        lexicon.cat8, lexicon.cat9,
    ) for item in cat]
    for form in all_forms:
        for ch in form:
            assert not ch.isdigit(), f"Form {form!r} contains digit {ch!r}"
    for v in all_digit_values:
        assert v.isdigit(), f"Inflectional value {v!r} is not a digit"


def test_id_uniqueness_passes(lexicon):
    """All item IDs across the lexicon must be unique."""
    all_ids = [item.id for cat in (
        lexicon.cat1, lexicon.cat1pron, lexicon.cat2, lexicon.cat3,
        lexicon.cat3aux, lexicon.cat4, lexicon.cat5, lexicon.cat6,
        lexicon.cat8, lexicon.cat9,
    ) for item in cat]
    assert len(all_ids) == len(set(all_ids))


def test_form_uniqueness_within_category(lexicon):
    """Surface forms must be unique within each category."""
    for cat in (
        lexicon.cat1, lexicon.cat1pron, lexicon.cat2, lexicon.cat3,
        lexicon.cat3aux, lexicon.cat4, lexicon.cat5, lexicon.cat6,
        lexicon.cat8, lexicon.cat9,
    ):
        forms = [item.form for item in cat]
        assert len(forms) == len(set(forms)), (
            f"Duplicate forms in category: {[f for f in forms if forms.count(f) > 1]}"
        )


def test_validate_passes_on_correct_lexicon(lexicon):
    """validate() must not raise on a correctly generated lexicon."""
    validate(lexicon)  # should not raise


def test_validate_catches_duplicate_id(lexicon):
    """Injecting a duplicate ID into cat2 must trigger check 3."""
    first_cat2 = lexicon.cat2[0]
    # Duplicate: replace the second cat2 item with one that has the same ID as the first
    duplicate = dataclasses.replace(lexicon.cat2[1], id=first_cat2.id)
    corrupted = dataclasses.replace(lexicon, cat2=(first_cat2, duplicate) + lexicon.cat2[2:])
    with pytest.raises(LexiconValidationError, match="Check 3"):
        validate(corrupted)


def test_validate_catches_bad_cat4_required(lexicon):
    """Flipping cat4_required on a CAT1 item must trigger the cat4_required consistency check."""
    item = lexicon.cat1[0]
    corrupted_item = dataclasses.replace(item, cat4_required=not item.cat4_required)
    corrupted = dataclasses.replace(lexicon, cat1=(corrupted_item,) + lexicon.cat1[1:])
    with pytest.raises(LexiconValidationError, match="Check 11"):
        validate(corrupted)
