"""Tests for lexicon/generation.py.

Spec reference: experiment.md §2 (Lexicon).
"""

import random

from lexicon.generation import (
    assign_cat1_features,
    build_cat1pron,
    build_cat4,
    build_cat6,
    build_cat8,
    build_cat9,
    generate_combinatorial_stems,
)
from lexicon.config import LETTER_PARTITIONS


def test_combinatorial_stems_cat1_count():
    # CAT1 = {a,b,c,d,e}: 5^1 + 5^2 + 5^3 = 155
    stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT1"], max_length=3)
    assert len(stems) == 155


def test_combinatorial_stems_cat2_count():
    # CAT2 = {f,g,h,x}: 4^1 + 4^2 + 4^3 = 84
    stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT2"], max_length=3)
    assert len(stems) == 84


def test_combinatorial_stems_cat3_count():
    # CAT3 = {i,j,k,l}: 4^1 + 4^2 + 4^3 = 84
    stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT3"], max_length=3)
    assert len(stems) == 84


def test_combinatorial_stems_cat5_count():
    # CAT5 = {o,p,q}: 3^1 + 3^2 + 3^3 = 39
    stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT5"], max_length=3)
    assert len(stems) == 39


def test_stems_sorted_order():
    stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT1"], max_length=3)
    assert stems[0] == "a"
    assert stems[-1] == "eee"
    assert stems == sorted(stems, key=lambda s: (len(s), s))


def test_cat4_two_items_numbered():
    """Spec §2.2: CAT4 has exactly two items, {u}=sg and {v}=pl, no definiteness."""
    items = build_cat4()
    assert len(items) == 2
    numbers = {item.number for item in items}
    assert numbers == {"singular", "plural"}
    forms = {item.form for item in items}
    assert forms == {"u", "v"}


def test_cat1pron_pron_and_refl():
    items = build_cat1pron()
    assert len(items) == 2
    subclasses = {item.subclass for item in items}
    assert subclasses == {"Pron", "Refl"}
    forms = {item.form for item in items}
    assert forms == {"m", "mm"}


def test_cat6_three_items_partitioned():
    """Spec §2.4: 3 CAT6 items partitioned 2:1 (NP-selecting : VP-selecting)."""
    items = build_cat6()
    assert len(items) == 3
    np_count = sum(1 for x in items if x.attachment == "cat1_selecting")
    vp_count = sum(1 for x in items if x.attachment == "cat3_selecting")
    assert (np_count, vp_count) == (2, 1)


def test_cat8_singleton_w():
    """Spec §2.2: CAT8 = {w}, single wh-particle."""
    items = build_cat8()
    assert len(items) == 1
    assert items[0].form == "w"


def test_cat9_two_subordinators():
    """Spec §2.2: CAT9 = {s, t}, both head Type4 (relative clauses)."""
    items = build_cat9()
    assert len(items) == 2
    assert {item.form for item in items} == {"s", "t"}


def test_cat1_cat4_required_consistent():
    stems = generate_combinatorial_stems(LETTER_PARTITIONS["CAT1"], max_length=3)
    items = assign_cat1_features(stems, random.Random(42))
    for item in items:
        assert item.cat4_required == (item.countability == "countable"), (
            f"Item {item.id}: cat4_required={item.cat4_required} "
            f"but countability={item.countability!r}"
        )
