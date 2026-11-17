"""Harness tests for the metadata sidecar (TRANSFORM_RULES §8).

Covers: corpus↔sidecar id/sentence alignment; target/control span round-trips
against the real tokenizer; EOS-fallback correctness; annotated-token role and
licensing-status invariants (incl. consistency with ``grammars.verdicts``);
and generation determinism (double build byte-identical). The structural-probe
splits have their own battery in test_structural_probes.py.

The on-disk drift check (§10) is opt-in: run
``CORPUS_DETERMINISM=1 pytest tests/test_metadata_sidecar.py`` AFTER the step-8
regeneration to assert the committed corpora are exactly what HEAD produces.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from corpusgen.corpus_builder import CorpusBuilder
from grammars import metadata as M
from grammars import verdicts as V
from grammars.base_generator import CONSTRUCTION_MIX
from grammars.grammar_H.generate import GrammarHGenerator
from grammars.grammar_H_prime.generator import GrammarHPrimeGenerator
from grammars.grammar_L_prime.generator import GrammarLPrimeGenerator
from grammars.grammar_P.generate import GrammarPGenerator
from models.tokenize import build_tokenizer

GRAMMARS = ["H", "P", "Hprime", "Lprime"]


def _registry():
    """grammar -> (_one_item, family, lexicon loader) for in-memory draws."""
    from grammars.grammar_H.generate import _one_item as h_item
    from grammars.grammar_H.lexicon import load as h_lex
    from grammars.grammar_H_prime.generate import _one_item as hp_item
    from grammars.grammar_L_prime.generate import _one_item as lp_item
    from grammars.grammar_P.generate import _one_item as p_item
    from grammars.grammar_P.lexicon import load as p_lex
    return {
        "H": (h_item, M.FAMILY_TREE, h_lex),
        "Hprime": (hp_item, M.FAMILY_TREE, h_lex),
        "P": (p_item, M.FAMILY_TOKENS, p_lex),
        "Lprime": (lp_item, M.FAMILY_TOKENS, p_lex),
    }
GEN_CLASSES = {
    "H": GrammarHGenerator, "P": GrammarPGenerator,
    "Hprime": GrammarHPrimeGenerator, "Lprime": GrammarLPrimeGenerator,
}
NULL = "#NULL#"
TRAIN_N = 120


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory) -> Path:
    """Small train corpora for all grammars (the structural-probe splits have
    their own test battery in test_structural_probes.py)."""
    root = tmp_path_factory.mktemp("sidecar_corpora")
    for g in GRAMMARS:
        builder = CorpusBuilder(GEN_CLASSES[g](seed=42), root)
        builder.build_train(TRAIN_N, constructions=CONSTRUCTION_MIX)
    return root


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


def _read(root: Path, split: str, grammar: str):
    corpus = [json.loads(l) for l in
              (root / split / f"{grammar}.jsonl").open(encoding="utf-8")]
    metas = [json.loads(l) for l in
             (root / split / f"{grammar}.meta.jsonl").open(encoding="utf-8")]
    return corpus, metas


def _splits(root: Path, grammar: str):
    yield "train"


def _stem(word: str):
    parts = [m for m in word.replace(NULL, "").split("#") if m]
    return parts[0] if parts else None


# ---------------------------------------------------------------------------
# Alignment & schema
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_id_alignment(corpus_root, grammar):
    """Every corpus line has a sidecar line with the same id and sentence."""
    for split in _splits(corpus_root, grammar):
        corpus, metas = _read(corpus_root, split, grammar)
        assert len(corpus) == len(metas) > 0
        for i, (item, meta) in enumerate(zip(corpus, metas)):
            assert item["id"] == meta["id"] == f"{grammar}-{split}-{i:05d}"
            assert item["sentence"] == meta["sentence"]
            assert item["construction"] == meta["construction"]


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_structural_fields(corpus_root, grammar):
    positional = grammar in ("P", "Lprime")
    for split in _splits(corpus_root, grammar):
        _, metas = _read(corpus_root, split, grammar)
        for m in metas:
            for f in ("n_clauses", "n_embedded_clauses", "n_modifiers",
                      "n_verbs", "expansion_sites", "length",
                      "morphemes_mean", "n_morphemes", *M.SITE_FIELDS):
                assert f in m, f"{m['id']}: missing {f}"
            assert m["expansion_sites"] == m["n_embedded_clauses"] + m["n_modifiers"]
            assert m["embedding_depth"] is not None, f"{m['id']}: null depth"
            assert m["embedding_depth"] >= 0
            if positional:
                # 2026-08-04, reversing §8.4 decision A: the positional family
                # has no tree, but its clause chain is non-branching, so
                # `n_embedded_clauses` IS the depth and the generator knows it.
                # This asserts the identity rather than the old `is None`.
                assert m["embedding_depth"] == m["n_embedded_clauses"]
            # `structural_type` is the axis `family` used to be conflated with.
            assert m["structural_type"] == ("positional" if positional
                                            else "hierarchical")
            assert m["phenomenon_family"] is not None


# ---------------------------------------------------------------------------
# Targets & spans (round-trip against the real tokenizer)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_targets_and_span_roundtrip(corpus_root, tokenizer, grammar):
    for split in _splits(corpus_root, grammar):
        _, metas = _read(corpus_root, split, grammar)
        for m in metas:
            toks = tokenizer.convert_ids_to_tokens(tokenizer.encode(m["sentence"]))
            assert toks[0] == "[BOS]" and toks[-1] == "[EOS]"
            assert m["n_morphemes"] == len(toks) - 2
            words = m["sentence"].split()
            if m["construction"] == "neutral":
                assert m["targets"] == []
            else:
                assert len(m["targets"]) == 1, m["id"]
            for entry in m["targets"] + m["annotated_tokens"]:
                s, e = entry["model_token_span"]
                if entry["category"] == "EOS":
                    # §8.1 sentence-final-gap fallback (or its pair-base mirror)
                    assert entry["word_index"] is None
                    assert toks[s] == "[EOS]" and e == s + 1
                elif s == e:
                    continue  # zero-width (gap) span
                else:
                    assert toks[s] == _stem(words[entry["word_index"]]), \
                        f"{m['id']}: span {s} != word {entry['word_index']}"


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_annotated_roles_and_status(corpus_root, grammar):
    """Exactly one target-role annotation iff a phenomenon fired; every
    licensing_status carries the four grammar keys with legal values."""
    legal = {"licensed", "unlicensed", "na"}
    for split in _splits(corpus_root, grammar):
        _, metas = _read(corpus_root, split, grammar)
        for m in metas:
            n_target_roles = 0
            for a in m["annotated_tokens"]:
                assert a["role"] in ("target", "licensed_unrealized",
                                    "unlicensed", "category_baseline")
                st = a["licensing_status"]
                assert set(st) == set(GRAMMARS)
                assert set(st.values()) <= legal
                if a["role"] == "target":
                    n_target_roles += 1
                    # the fired position is licensed under its own grammar
                    assert st[grammar] == "licensed"
            if m["construction"] == "neutral":
                assert n_target_roles == 0
            else:
                assert n_target_roles == 1, m["id"]


# ---------------------------------------------------------------------------
# Off-frame status (2026-07-20): a candidate's own-rule verdict for phenomena
# OTHER than the sentence's active one — e.g. a CAT3AUX in a wh-movement
# sentence still gets an aux-licensing verdict.
# ---------------------------------------------------------------------------

_NOMINAL_OFF_FRAME = ("anaphoric_binding_refl", "anaphoric_binding_pron", "wh_movement")
_AUX_OFF_FRAME = "auxiliary_movement"


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_off_frame_status_schema(corpus_root, grammar):
    """Every annotated_tokens entry carries off_frame_status keyed by exactly
    the phenomena relevant to its category, MINUS whichever is active here;
    every value is a well-formed four-key licensing_status dict."""
    legal = {"licensed", "unlicensed", "na"}
    for split in _splits(corpus_root, grammar):
        _, metas = _read(corpus_root, split, grammar)
        for m in metas:
            construction = m["construction"]
            for a in m["annotated_tokens"]:
                assert "off_frame_status" in a, m["id"]
                off = a["off_frame_status"]
                if a["category"] in ("CAT1", "CAT1PRON"):
                    expected_keys = {p for p in _NOMINAL_OFF_FRAME if p != construction}
                elif a["category"] == "CAT3AUX":
                    expected_keys = ({_AUX_OFF_FRAME} if construction != _AUX_OFF_FRAME
                                     else set())
                else:
                    expected_keys = set()
                assert set(off) == expected_keys, (m["id"], a["category"], construction)
                for status in off.values():
                    assert set(status) == set(GRAMMARS)
                    assert set(status.values()) <= legal


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_off_frame_status_matches_direct_computation(grammar):
    """On NEUTRAL (un-transformed) draws, off_frame_status for every phenomenon
    matches directly recomputing that phenomenon's own licensing primitives on
    the same struct — the strongest available correctness check, independent
    of metadata.py's own off-frame machinery."""
    from grammars import violations as X
    from grammars.licensing import (
        aux_hfamily_licenses, aux_pfamily_licenses, binding_hfamily_licensed_ids,
        binding_pfamily_licensed_indices, neutral_from_tokens, neutral_from_tree,
    )

    one_item, family, load_lex = _registry()[grammar]
    tree = family == M.FAMILY_TREE
    rng = random.Random(9090)
    lex = load_lex()
    checked = 0
    for _ in range(3000):
        if checked >= 60:
            break
        sites = {}
        surf, label, struct = one_item(rng, lex, sites_out=sites)
        if label != "neutral":
            continue
        meta = M.build_meta_record(surf, grammar, family, label, struct, sites=sites)
        refs_and_words, *_ = M._refs_and_constants(struct, family)

        if tree:
            # Coreference-aware, matching the sidecar (TRANSFORM_RULES §2): a
            # candidate is licensed only as part of a genuine duplicated-CAT1 pair.
            lic_refl = binding_hfamily_licensed_ids(struct, "Refl", require_coref=True)
            lic_pron = binding_hfamily_licensed_ids(struct, "Pron", require_coref=True)
            # binding_hfamily_licensed_ids keys by the WRAPPER's (Type1/CAT1PRON)
            # .node_id, not the head terminal's -- refs_and_words gives head
            # terminals (span-bearing), so re-derive terminal-id -> wrapper-id.
            wrapper_node_id = {}

            def _walk_wrappers(node):
                if node.label == "Type1" and node.role != "fronted":
                    head = next((c for c in node.children if c.role == "head"), None)
                    if head is not None:
                        wrapper_node_id[id(head)] = node.node_id
                elif node.label == "CAT1PRON" and node.lex is not None and node.role != "fronted":
                    wrapper_node_id[id(node)] = node.node_id
                for c in node.children:
                    _walk_wrappers(c)
            _walk_wrappers(struct)
            own_wh, prime_wh = "H", "Hprime"
            from grammars.grammar_H.transforms import _wh_targetable_nominals
            own_sites = X.wh_sites(struct, family, own_wh)
            prime_sites = X.wh_sites(struct, family, prime_wh)
            noms = _wh_targetable_nominals(struct)
            wh_by_ref_id = {
                id(M._head_terminal_of(nom["node"])): (own, prime)
                for nom, own, prime in zip(noms, own_sites, prime_sites)
                if M._head_terminal_of(nom["node"]) is not None
            }
        else:
            base_cats = ["CAT1" if t.cat == "CAT1PRON" else t.cat for t in struct]
            base_forms = [(t.coref_form if t.cat == "CAT1PRON"
                           else (t.lex.get("form") if t.lex is not None else None))
                          for t in struct]
            lic_refl = binding_pfamily_licensed_indices(base_cats, "Refl", base_forms=base_forms)
            lic_pron = binding_pfamily_licensed_indices(base_cats, "Pron", base_forms=base_forms)
            own_wh, prime_wh = "P", "Lprime"
            own_sites = X.wh_sites(struct, family, own_wh)
            prime_sites = X.wh_sites(struct, family, prime_wh)
            wh_by_ref_id = {
                id(struct[own["site"]]): (own, prime)
                for own, prime in zip(own_sites, prime_sites)
            }
        neutral = neutral_from_tree(struct) if tree else neutral_from_tokens(struct)

        for wi, (word, ref) in enumerate(refs_and_words):
            entry = next((a for a in meta["annotated_tokens"] if a["word_index"] == wi), None)
            if entry is None:
                continue
            off = entry["off_frame_status"]
            if entry["category"] in ("CAT1", "CAT1PRON"):
                if tree:
                    node_id = wrapper_node_id.get(id(ref))
                    exp_refl = "licensed" if node_id in lic_refl else "unlicensed"
                    exp_pron = "licensed" if node_id in lic_pron else "unlicensed"
                else:
                    exp_refl = "licensed" if wi in lic_refl else "unlicensed"
                    exp_pron = "licensed" if wi in lic_pron else "unlicensed"
                assert off["anaphoric_binding_refl"][grammar] == exp_refl, surf
                assert off["anaphoric_binding_pron"][grammar] == exp_pron, surf
                if id(ref) in wh_by_ref_id:
                    own, prime = wh_by_ref_id[id(ref)]
                    exp_own = "licensed" if (own["licensed_i"] or own["licensed_ii"]) else "unlicensed"
                    exp_prime = "licensed" if (prime["licensed_i"] or prime["licensed_ii"]) else "unlicensed"
                    assert off["wh_movement"][own_wh] == exp_own, surf
                    assert off["wh_movement"][prime_wh] == exp_prime, surf
            elif entry["category"] == "CAT3AUX":
                h = aux_hfamily_licenses(neutral, wi)
                p = aux_pfamily_licenses(neutral, wi)
                exp = {"H": h, "Hprime": h, "P": p, "Lprime": p}
                for g in GRAMMARS:
                    want = "licensed" if exp[g] else "unlicensed"
                    assert off["auxiliary_movement"][g] == want, surf
            checked += 1
    assert checked >= 60


@pytest.mark.parametrize("grammar", ["H", "Hprime"])
def test_wh_licensing_status_matches_the_pre_transform_base(grammar):
    """Every wh candidate's sidecar ``licensing_status`` equals the verdict the
    grammar's own site table gives on the base BEFORE fronting.

    The strong form of the check: truth is snapshotted on the un-transformed
    tree, then the transform is applied and the meta record built, so it tests
    ``_wh_facts_tree``'s base reconstruction against something computed
    independently of it.

    This is what the target-only check against ``verdicts.py`` cannot see.
    Clause (i) is E3 — matrix object AND it hosts an embedded clause — and a
    fronted matrix object is always E3-licensed (it could not have fronted
    otherwise), so a stale plain-``matrix_object`` clause still labels the TARGET
    correctly while mislabelling the in-situ CONTROLS. That is exactly the defect
    this pins (2026-07-31): 5.4% of candidates, 31.7% of H / 40.3% of H′ wh
    sentences, all of them non-hosting matrix objects reported ``licensed``.
    """
    from grammars import violations as X
    from grammars.grammar_H.build import build_type0, attach_forced_type4
    from grammars.grammar_H.features import assign
    from grammars.grammar_H.lexicon import load as h_lex
    from grammars.grammar_H.transforms import (
        front_nominal, has_nucleus_duplicate, _wh_targetable_nominals,
    )

    rng = random.Random(21)
    lex = h_lex()
    family = M.FAMILY_TREE
    checked = 0
    for _ in range(3000):
        if checked >= 40:
            break
        counter = [0]
        # The wh plan's forced geometry (grammar_H/generate): transitive matrix
        # plus one embedded clause, host MIXED — so both hosting and non-hosting
        # matrix objects occur, which is what makes E3's second conjunct testable.
        tree = build_type0(rng, lex, counter, force_transitive=True)
        attach_forced_type4(tree, rng, lex, counter,
                            force_transitive_inner=False, mix_host=True)
        if has_nucleus_duplicate(tree):
            continue
        assign(tree, lex, rng)

        noms = _wh_targetable_nominals(tree)
        own = X.wh_sites(tree, family, grammar)
        truth, discriminating = {}, set()
        for site, nom in zip(own, noms):
            ht = M._head_terminal_of(nom["node"])
            if ht is None:
                continue
            truth[id(ht)] = site["licensed_i"] or site["licensed_ii"]
            # The ONLY configuration that separates E3 from plain "matrix
            # object": a matrix object that hosts nothing. Under the stale rule
            # it reads licensed; under E3 it is a negative.
            if nom["matrix_object"] and not nom["hosts_type4"]:
                discriminating.add(id(ht))

        # Front ANY licensed site, clause (ii) included. Restricting to clause
        # (i) would front the hosting matrix object every time and exclude it
        # from the comparison, leaving the discriminating case unreachable —
        # which is how the first version of this test passed against the bug.
        licensed = [(c["node"], c["parent"], c["idx"]) for c in noms
                    if truth.get(id(M._head_terminal_of(c["node"])), False)]
        if not licensed:
            continue
        target, parent, idx = rng.choice(licensed)
        front_nominal(tree, target, parent, idx, lex, counter)

        surf = " ".join(w for w, _ in M._refs_and_constants(tree, family)[0])
        meta = M.build_meta_record(surf, grammar, family, "wh_movement", tree)
        refs_and_words, *_ = M._refs_and_constants(tree, family)

        for entry in meta["annotated_tokens"]:
            if entry["category"] not in ("CAT1", "CAT1PRON"):
                continue
            ref = refs_and_words[entry["word_index"]][1]
            if id(ref) not in truth or ref is M._head_terminal_of(target):
                continue          # the fronted target is checked by its own test
            want = "licensed" if truth[id(ref)] else "unlicensed"
            assert entry["licensing_status"][grammar] == want, (surf, grammar)
            if id(ref) in discriminating:
                checked += 1
    # ``checked`` counts ONLY in-situ non-hosting matrix objects, so the test
    # cannot silently stop covering the case it exists for.
    assert checked >= 40


@pytest.mark.parametrize("grammar", ["P", "Lprime"])
def test_offframe_binding_is_evaluated_on_the_base_string(grammar):
    """P-family binding licensing is two counts over the BASE string — the
    antecedent→target distance in items and the antecedent's CAT1-position
    ordinal (TRANSFORM_RULES §1, §3.2) — so the off-frame annotation of a
    MOVEMENT sentence must splice the fronted element back before counting.

    Two things go wrong on the derived list and this pins both:
      * the GAP occupies an index while carrying no lexical item, so it would be
        counted as an intervening item (it is a bare ``#NULL#`` slot, not a
        lexeme with #-attached inflections — D4);
      * wh fronting moves the block to absolute position 0, shifting every
        distance that spans the extraction site and every ordinal before it.

    Aux fronting happens to preserve both counts (the aux leaves a GAP at its own
    slot and everything shifts by exactly +1), so wh is where this bites — but
    the invariant is asserted for both rather than relying on that coincidence.
    """
    one_item, family, load_lex = _registry()[grammar]
    assert family == M.FAMILY_TOKENS
    rng = random.Random(4242)
    lex = load_lex()
    seen = {"wh_movement": 0, "auxiliary_movement": 0}
    for _ in range(4000):
        if all(v >= 25 for v in seen.values()):
            break
        surf, label, struct = one_item(rng, lex)
        if label not in seen:
            continue
        seen[label] += 1

        base = M.base_order_tokens(struct)
        assert base is not struct, surf              # a movement item HAS a base
        assert not any(t.cat == "GAP" for t in base), surf
        assert len(base) == len(struct) - (2 if label == "wh_movement" else 1), surf

        # The verdicts the sidecar reports equal the verdicts computed on that
        # base — i.e. nothing is read off derived order.
        for subtype in ("Refl", "Pron"):
            derived_call = M._off_frame_binding_status(struct, family, subtype)
            base_call = M._off_frame_binding_status(base, family, subtype)
            assert derived_call == base_call, (surf, subtype)
    assert all(v >= 25 for v in seen.values()), seen


def test_base_order_tokens_is_identity_without_movement():
    """No GAP means the list already IS the base, so the recovery must be a
    no-op — this is what keeps the ON-frame binding annotation (a binding
    sentence never contains a gap) byte-identical to before the fix."""
    from grammars.grammar_P.generate import _one_item
    from grammars.grammar_P.lexicon import load as p_lex

    rng = random.Random(31)
    lex = p_lex()
    checked = 0
    for _ in range(2000):
        _surf, label, struct = _one_item(rng, lex)
        if label in ("wh_movement", "auxiliary_movement"):
            continue
        assert M.base_order_tokens(struct) is struct, label
        checked += 1
    assert checked >= 500


@pytest.mark.parametrize("grammar", ["H", "Hprime"])
def test_off_frame_aux_labels_embedded_only_blocking_case(grammar):
    """The motivating case: an H-family sentence whose only aux sits in an
    embedded clause (no matrix aux at all) is H's blocking configuration for
    aux movement, regardless of the sentence's actual active phenomenon —
    verify it is now labeled 'unlicensed' for H/Hprime and 'licensed' for
    P/Lprime (P fronts first-in-scan, so the same configuration licenses it)."""
    from grammars import violations as X
    from corpusgen.structural_probes import _draw_base, _registry as probe_registry

    family, load_lex = probe_registry()[grammar]
    lex = load_lex()
    rng = random.Random(2024)
    found = False
    for _ in range(8000):
        base = _draw_base(family, rng, lex)
        sites = X.aux_sites(base, family)
        if sites and not any(s["licensed"] for s in sites) and any(not s["licensed"] for s in sites):
            found = True
            refs, *_ = M._refs_and_constants(base, family)
            sentence = " ".join(w for w, _ in refs)
            rec = M.build_meta_record(sentence, grammar, family, "neutral", base)
            aux_entries = [t for t in rec["annotated_tokens"] if t["category"] == "CAT3AUX"]
            assert aux_entries
            for t in aux_entries:
                off = t["off_frame_status"]["auxiliary_movement"]
                assert off["H"] == "unlicensed"
                assert off["Hprime"] == "unlicensed"
                assert off["P"] == "licensed"
                assert off["Lprime"] == "licensed"
            break
    assert found


# ---------------------------------------------------------------------------
# Licensing consistency with grammars.verdicts (wh clause-(i) axis)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_wh_status_consistent_with_verdicts(grammar):
    """The clause-(i) verdicts recovered by verdicts.py imply the sidecar's
    licensing_status for the fired wh target (verdicts are single clauses;
    the status is the full disjunction, so implication is the exact relation).

    tree family:   hprime_verdict (3rd CAT1)  => Hprime licensed
                   lprime_verdict (E3)       => H licensed
    token family:  hprime_verdict (3rd CAT1)  => P licensed
                   lprime_verdict (E3)       => Lprime licensed

    NOTE this speaks only to the fired TARGET, and a movement target is licensed
    by construction — so it cannot detect a stale licensing clause applied to the
    controls. That blind spot is covered by
    ``test_wh_licensing_status_matches_the_pre_transform_base``.
    """
    one_item, family, load_lex = _registry()[grammar]
    rng = random.Random(1234)
    lex = load_lex()
    tree = family == M.FAMILY_TREE
    checked = 0
    for _ in range(4000):
        if checked >= 40:
            break
        sites = {}
        surf, label, struct = one_item(rng, lex, sites_out=sites)
        if label != "wh_movement":
            continue
        meta = M.build_meta_record(surf, grammar, family, label, struct,
                                   sites=sites)
        status = next(a["licensing_status"] for a in meta["annotated_tokens"]
                      if a["role"] == "target")
        v = V.wh_verdicts_tree(struct) if tree else V.wh_verdicts_token(struct)
        assert v is not None
        if v["hprime_verdict"] == "license":
            assert status["Hprime" if tree else "P"] == "licensed"
        if v["lprime_verdict"] == "license":
            assert status["H" if tree else "Lprime"] == "licensed"
        checked += 1
    assert checked >= 40


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_build_determinism(tmp_path, grammar):
    """Two builds from the same seed are byte-identical (corpus AND sidecar)."""
    paths = []
    for sub in ("a", "b"):
        builder = CorpusBuilder(GEN_CLASSES[grammar](seed=7), tmp_path / sub)
        builder.build_train(60, constructions=CONSTRUCTION_MIX)
        paths.append(tmp_path / sub)
    for suffix in (".jsonl", ".meta.jsonl"):
        f = Path("train") / f"{grammar}{suffix}"
        assert (paths[0] / f).read_bytes() == (paths[1] / f).read_bytes()


@pytest.mark.skipif(
    not os.environ.get("CORPUS_DETERMINISM"),
    reason="opt-in on-disk drift check (§10): set CORPUS_DETERMINISM=1 after "
           "the step-8 regeneration",
)
@pytest.mark.parametrize("grammar", GRAMMARS)
def test_on_disk_corpora_match_head(tmp_path, grammar):
    """The committed corpora are exactly what HEAD's generators produce.

    Rebuilds every split at the on-disk sizes (splits share one sequential RNG
    stream, so the full build_all sequence is replayed) and byte-compares each
    file that exists on disk. Closes the §10 drift once step 8 has run.
    """
    root = Path(__file__).resolve().parent.parent / "corpora"
    counts = {}
    for split in ("train", "validation", "test_indistribution"):
        f = root / split / f"{grammar}.jsonl"
        if not f.exists():
            pytest.skip(f"{f} not on disk")
        counts[split] = sum(1 for _ in f.open(encoding="utf-8"))
    probe_f = root / "structural_probe_IOD" / f"{grammar}.jsonl"
    probe_n = (sum(1 for _ in probe_f.open(encoding="utf-8")) // 14
               if probe_f.exists() else 30)

    builder = CorpusBuilder(GEN_CLASSES[grammar](seed=42), tmp_path)
    builder.build_all(train_n=counts["train"], val_n=counts["validation"],
                      test_n=counts["test_indistribution"],
                      probe_n_per_subcase=probe_n)
    for split_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for suffix in (".jsonl", ".meta.jsonl"):
            on_disk = split_dir / f"{grammar}{suffix}"
            if not on_disk.exists():
                continue
            rebuilt = tmp_path / split_dir.name / f"{grammar}{suffix}"
            assert rebuilt.exists(), f"HEAD did not produce {rebuilt}"
            assert on_disk.read_bytes() == rebuilt.read_bytes(), \
                f"{on_disk} drifts from HEAD output"
