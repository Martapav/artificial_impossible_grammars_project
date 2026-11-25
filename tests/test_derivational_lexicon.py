"""Lexicon v0.3.0: sampled stems, derivational affixes, Zipf ranks, and their
integration with the generators, tokenizer and metadata.

Spec: experiment.md §2.1, §2.2, §2.5, §2.6, §3, §10; TRANSFORM_RULES §3, §8.1.
"""
from __future__ import annotations

import random
from collections import Counter

import pytest

from lexicon import assemble_lexicon, deserialize, serialize, validate
from lexicon import generation as G
from lexicon.config import (
    AFFIX_INVENTORY,
    ENTRIES_PER_STEM,
    LETTER_PARTITIONS,
    MORPH_SHARES,
    OPEN_CATEGORIES,
    STEM_TARGET,
    WORDNET_30_UNIQUE_STRINGS,
)

OPEN_KEYS = {"CAT1": "cat1", "CAT2": "cat2", "CAT3": "cat3", "CAT5": "cat5"}


@pytest.fixture(scope="module")
def lex():
    return assemble_lexicon(seed=42)


@pytest.fixture(scope="module")
def lex_path(lex, tmp_path_factory):
    path = tmp_path_factory.mktemp("lex") / "lexicon_v0.3.0.json"
    serialize(lex, path)
    return path


# ── Lexicon structure ────────────────────────────────────────────────────────

def test_validates(lex):
    validate(lex)


def test_affix_inventory_sizes_lengths_pools_reservation(lex):
    for cat, slots in AFFIX_INVENTORY.items():
        for slot, k in slots.items():
            got = [a for a in lex.affixes if a.category == cat and a.slot == slot]
            assert len(got) == k, (cat, slot)
            for a in got:
                assert len(a.form) in (2, 3)                       # never a single letter
                assert set(a.form) <= set(LETTER_PARTITIONS[cat])  # own pool only
    forms = [a.form for a in lex.affixes]
    assert len(forms) == len(set(forms)) == 84                     # one class each
    stems = {e.stem for k in OPEN_KEYS.values() for e in getattr(lex, k)}
    assert not stems & set(forms)                                  # reserved


def test_stem_counts_follow_wordnet_and_sum_to_target(lex):
    counts = lex.metadata.stem_counts
    assert sum(counts.values()) == STEM_TARGET
    z = sum(WORDNET_30_UNIQUE_STRINGS.values())
    for cat in OPEN_CATEGORIES:
        assert abs(counts[cat] - STEM_TARGET * WORDNET_30_UNIQUE_STRINGS[cat] / z) < 1


def test_entries_per_stem_and_exact_morph_quotas(lex):
    for cat, key in OPEN_KEYS.items():
        entries = getattr(lex, key)
        n = lex.metadata.stem_counts[cat]
        assert len(entries) == n * ENTRIES_PER_STEM
        per_stem = Counter(e.stem_id for e in entries)
        assert set(per_stem.values()) == {ENTRIES_PER_STEM}
        shares = Counter(e.n_morphemes for e in entries)
        for k, share in MORPH_SHARES.items():
            assert abs(shares[k] / len(entries) - share) < 1e-3, (cat, k)


def test_template_order_and_slot_limits(lex):
    slot_of = {(a.category, a.form): a.slot for a in lex.affixes}
    for cat, key in OPEN_KEYS.items():
        for e in getattr(lex, key):
            morphs = e.form.split("#")
            i = morphs.index(e.stem)
            assert morphs[:i] == ([e.prefix] if e.prefix else [])          # ≤1 prefix, before stem
            after = [m for m in (e.suffix_inner, e.suffix_outer) if m]
            assert morphs[i + 1:] == after                                # inner before outer
            for m in morphs[:i]:
                assert slot_of[(cat, m)] == "prefix"
            if e.suffix_inner:
                assert slot_of[(cat, e.suffix_inner)] == "suffix_inner"
            if e.suffix_outer:
                assert slot_of[(cat, e.suffix_outer)] == "suffix_outer"


def test_suffix_preference_close_to_three_to_one(lex):
    for key in OPEN_KEYS.values():
        es = getattr(lex, key)
        pre = sum(bool(e.prefix) for e in es)
        suf = sum(bool(e.suffix_inner) + bool(e.suffix_outer) for e in es)
        assert 2.7 < suf / pre < 3.3, key


def test_four_morpheme_entries_are_prefix_plus_two_suffixes(lex):
    for key in OPEN_KEYS.values():
        for e in getattr(lex, key):
            if e.n_morphemes == 4:
                assert e.prefix and e.suffix_inner and e.suffix_outer


def test_stem_length_histogram_matches_truncated_curve(lex):
    for cat, key in OPEN_KEYS.items():
        reserved = {a.form for a in lex.affixes if a.category == cat}
        expected = G.stem_length_counts(cat, lex.metadata.stem_counts[cat], reserved)
        got = Counter(len(e.stem) for e in getattr(lex, key) if e.n_morphemes == 1)
        assert dict(got) == expected
        # stems may exceed the pool size (no pool-size cap)
        assert max(got) > len(LETTER_PARTITIONS[cat])


def test_entries_of_a_stem_share_its_features(lex):
    by_stem = {}
    for e in lex.cat1:
        by_stem.setdefault(e.stem_id, set()).add(
            (e.inherent_gender, e.countability, e.cat6_compatible))
    assert all(len(v) == 1 for v in by_stem.values())


def test_zipf_ranks_are_a_permutation_and_affixes_drift_rarer(lex):
    for key in OPEN_KEYS.values():
        es = getattr(lex, key)
        assert sorted(e.zipf_rank for e in es) == list(range(1, len(es) + 1))
        mean_rank = {k: sum(e.zipf_rank for e in es if e.n_morphemes == k)
                     / sum(1 for e in es if e.n_morphemes == k) for k in (1, 2, 3, 4)}
        assert mean_rank[1] < mean_rank[2] < mean_rank[3] < mean_rank[4]


def test_capacity_guard_fires():
    with pytest.raises(G.LexiconCapacityError):
        G.stem_length_counts("CAT5", 10**7, reserved=set())


def test_phases_are_independent(monkeypatch, lex):
    """Changing a phase-E parameter leaves phases A-D untouched."""
    monkeypatch.setattr(G, "ZIPF_ALPHA", 2.0)
    other = assemble_lexicon(seed=42)
    assert other.affixes == lex.affixes
    assert [e.form for e in other.cat1] == [e.form for e in lex.cat1]
    assert [e.inherent_gender for e in other.cat1] == [e.inherent_gender for e in lex.cat1]
    assert [e.zipf_rank for e in other.cat1] != [e.zipf_rank for e in lex.cat1]


def test_serialization_round_trip(lex, lex_path):
    back = deserialize(lex_path)
    assert back.cat1 == lex.cat1 and back.affixes == lex.affixes
    assert back.metadata.content_hash == lex.metadata.content_hash


def test_v020_lexicon_still_validates_with_its_stored_hash():
    old = deserialize("lexicon.json")
    assert old.metadata.version == "0.2.0" and not old.affixes


# ── Sampling stays stream-identical ──────────────────────────────────────────

def test_zipf_pick_matches_original_weighted_choices():
    from grammars.lexsample import zipf_pick
    from grammars.tracing import TracedRandom

    items = [{"form": f"x{i}", "zipf_rank": r} for i, r in
             enumerate(random.Random(0).sample(range(1, 501), 500), start=0)]
    weights = [1.0 / it["zipf_rank"] for it in items]
    a, b = random.Random(5), random.Random(5)
    assert [zipf_pick(a, items) for _ in range(300)] == \
           [b.choices(items, weights=weights, k=1)[0] for _ in range(300)]
    ta, tb = TracedRandom(9), TracedRandom(9)
    out_a = [zipf_pick(ta, items) for _ in range(100)]
    out_b = [tb.choices(items, weights=weights, k=1)[0] for _ in range(100)]
    assert out_a == out_b
    assert ta.trace.draws == tb.trace.draws          # identical logged log-probs


# ── Generators, tokenizer, metadata on the v0.3.0 lexicon ────────────────────

@pytest.fixture()
def use_v030(monkeypatch, lex_path):
    monkeypatch.setenv("THESIS_LEXICON", str(lex_path))
    return lex_path


def _word_ok(word: str, morph_cat: dict, slots: dict) -> bool:
    lexical = [m for m in word.split("#") if m and not m.isdigit()]
    if not lexical:
        return True
    cats = {morph_cat.get(m) for m in lexical}
    if len(cats) != 1 or None in cats:
        return False
    kinds = [slots.get(m, "stem") for m in lexical]
    order = {"prefix": 0, "stem": 1, "suffix_inner": 2, "suffix_outer": 3}
    return kinds.count("stem") == 1 and kinds == sorted(kinds, key=order.__getitem__)


def test_generated_words_follow_the_template_and_tokenize_cleanly(use_v030, lex):
    from grammars.grammar_H.generate import GrammarHGenerator
    from grammars.grammar_P.generate import GrammarPGenerator
    from models.tokenize import build_tokenizer

    tok = build_tokenizer(use_v030)
    morph_cat = {}
    for key in ("cat1", "cat1pron", "cat2", "cat3", "cat3aux", "cat4", "cat5",
                "cat6", "cat8", "cat9"):
        for it in getattr(lex, key):
            for m in it.form.split("#"):
                morph_cat[m] = it.category
    slots = {a.form: a.slot for a in lex.affixes}
    closed = {"CAT1PRON", "CAT3AUX", "CAT4", "CAT6", "CAT8", "CAT9"}
    for Gen in (GrammarHGenerator, GrammarPGenerator):
        g = Gen(seed=3)
        for _ in range(60):
            s, _m = g.generate_string()
            for w in s.replace("#NULL#", "").split():
                assert _word_ok(w, morph_cat, slots), w
                lexical = [m for m in w.split("#") if m and not m.isdigit()]
                if morph_cat[lexical[0]] in closed:
                    assert len(lexical) == 1, w           # closed classes never affixed
            ids = tok(s)["input_ids"]
            assert tok.unk_token_id not in ids


def test_p_binding_duplicate_is_the_whole_derived_word(use_v030):
    from grammars.grammar_P.generate import _one_item
    from grammars.grammar_P.lexicon import load

    lex = load()
    rng = random.Random(1)
    seen = 0
    for _ in range(400):
        _s, phen, toks = _one_item(rng, lex)
        if not phen.startswith("anaphoric_binding") or phen.endswith("_skipped"):
            continue
        prons = [t for t in toks if t.cat == "CAT1PRON"]
        assert len(prons) == 1 and prons[0].coref_form
        forms = [t.lex["form"] for t in toks if t.cat == "CAT1"]
        assert prons[0].coref_form in forms        # antecedent = identical derived entry
        seen += 1
    assert seen > 5


def test_stem_span_skips_a_prefix():
    from grammars.metadata import stem_span

    class Ref:
        def __init__(self, lex):
            self.lex = lex
    pre, bare = Ref({"form": "ab#cde", "prefix": "ab"}), Ref({"form": "cde"})
    spans = {id(pre): (4, 8), id(bare): (8, 11)}
    assert stem_span(pre, spans) == [5, 6]
    assert stem_span(bare, spans) == [8, 9]


def test_resolver_order(monkeypatch, tmp_path, lex_path):
    from lexicon.paths import DEFAULT_LEXICON_PATH, resolve_lexicon_path

    monkeypatch.delenv("THESIS_LEXICON", raising=False)
    monkeypatch.delenv("THESIS_CORPUS_SET", raising=False)
    assert resolve_lexicon_path() == DEFAULT_LEXICON_PATH
    monkeypatch.setenv("THESIS_CORPUS_SET", "set_3")          # no lexicon_path recorded
    assert resolve_lexicon_path() == DEFAULT_LEXICON_PATH
    monkeypatch.setenv("THESIS_LEXICON", str(lex_path))
    assert resolve_lexicon_path() == lex_path
