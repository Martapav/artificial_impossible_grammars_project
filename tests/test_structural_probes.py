"""Tests for the IOD/OOD structural probes (TRANSFORM_RULES §9).

Pins: pair counts and subcase quotas; single-violation minimality per subcase;
the licensed/unlicensed asymmetry between members (via the sidecar's own
licensing annotation); EOS exclusion for movement targets; target-span
round-trips against the real tokenizer in BOTH members; wh verdict stamps;
neutral divergence targets; the OOD window; and byte-determinism.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

from corpusgen.corpus_builder import CorpusBuilder
from corpusgen.structural_probes import SUBCASES, build_structural_probe
from grammars.grammar_H.generate import GrammarHGenerator
from grammars.grammar_H_prime.generator import GrammarHPrimeGenerator
from grammars.grammar_L_prime.generator import GrammarLPrimeGenerator
from grammars.grammar_P.generate import GrammarPGenerator
from models.tokenize import build_tokenizer

GRAMMARS = ["H", "P", "Hprime", "Lprime"]
GEN_CLASSES = {
    "H": GrammarHGenerator, "P": GrammarPGenerator,
    "Hprime": GrammarHPrimeGenerator, "Lprime": GrammarLPrimeGenerator,
}
NULL = "#NULL#"
N = 3  # pairs per subcase in the test build


@pytest.fixture(scope="module")
def probe_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("probe_corpora")
    for g in GRAMMARS:
        CorpusBuilder(GEN_CLASSES[g](seed=42), root) \
            .build_structural_probe_iod(n_per_subcase=N)
    return root


@pytest.fixture(scope="module")
def tokenizer():
    return build_tokenizer()


def _read(root: Path, grammar: str, split: str = "structural_probe_IOD"):
    corpus = [json.loads(l) for l in
              (root / split / f"{grammar}.jsonl").open(encoding="utf-8")]
    metas = [json.loads(l) for l in
             (root / split / f"{grammar}.meta.jsonl").open(encoding="utf-8")]
    return corpus, metas


def _pairs(root: Path, grammar: str):
    corpus, metas = _read(root, grammar)
    assert len(corpus) == len(metas)
    for i in range(0, len(corpus), 2):
        yield corpus[i], corpus[i + 1], metas[i], metas[i + 1]


def _stem(word: str):
    parts = [m for m in word.replace(NULL, "").split("#") if m]
    return parts[0] if parts else None


def _morphemes(sentence: str):
    out = []
    for w in sentence.split():
        out.extend(m for m in w.replace(NULL, "").split("#") if m)
    return out


# ---------------------------------------------------------------------------

@pytest.mark.parametrize("grammar", GRAMMARS)
def test_counts_and_pair_structure(probe_root, grammar):
    corpus, metas = _read(probe_root, grammar)
    assert len(corpus) == N * len(SUBCASES) * 2
    by = Counter((it["subcase"], it["variant"]) for it in corpus)
    for s in SUBCASES:
        assert by[(s, "grammatical")] == by[(s, "ungrammatical")] == N
    for gc, uc, gm, um in _pairs(probe_root, grammar):
        assert gc["pair_id"] == uc["pair_id"] == gm["pair_id"] == um["pair_id"]
        assert (gc["variant"], uc["variant"]) == ("grammatical", "ungrammatical")
        assert gc["violation"] is None and uc["violation"] is not None
        assert gc["construction"] == uc["construction"]
        assert gc["id"] != uc["id"] and gc["id"] == gm["id"]
    # binding modes split (§7.2): wrong_item and wrong_site both present
    for sub in ("refl", "pron"):
        modes = Counter(it["violation"] for it in corpus
                        if it["subcase"] == sub and it["violation"])
        assert modes["wrong_item"] >= 1 and modes["wrong_site"] >= 1


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_single_violation_minimality(probe_root, grammar):
    """The two members are identical except the rule's application (§1)."""
    for gc, uc, gm, um in _pairs(probe_root, grammar):
        sub, viol = gc["subcase"], uc["violation"]
        gw, uw = gc["sentence"].split(), uc["sentence"].split()
        if sub in ("wh_i", "wh_ii", "aux") or viol == "positional_cat9_displaced" \
                or viol.startswith("positional"):
            # movement / relocation: the SAME words, rearranged
            assert Counter(gw) == Counter(uw), gc["pair_id"]
            assert gw != uw
        elif viol == "wrong_item":
            # single-word difference: the pron subtype swap at one site
            assert len(gw) == len(uw)
            diff = [(a, b) for a, b in zip(gw, uw) if a != b]
            assert len(diff) == 1, gc["pair_id"]
            a, b = diff[0]
            assert a.split("#")[1:] == b.split("#")[1:]  # same inflections
            assert a.split("#")[0] != b.split("#")[0]    # different stem
        elif viol == "agreement":
            # single-morpheme difference
            gms, ums = _morphemes(gc["sentence"]), _morphemes(uc["sentence"])
            assert len(gms) == len(ums)
            assert sum(1 for x, y in zip(gms, ums) if x != y) == 1, gc["pair_id"]
        elif viol == "no_substitution":
            # licensed configuration left bare: the reflexive is replaced by a
            # second copy of its antecedent, so one word differs and the bare
            # site repeats a stem occurring earlier in the same string.
            assert len(gw) == len(uw)
            diff = [i for i, (a, b) in enumerate(zip(gw, uw)) if a != b]
            assert len(diff) == 1, gc["pair_id"]
            i = diff[0]
            assert gw[i].split("#")[0] == "mm", gc["pair_id"]
            assert uw[i].split("#")[0] in {w.split("#")[0] for w in uw[:i]}, \
                gc["pair_id"]
        else:  # wrong_site binding: substitution at a different position
            assert viol == "wrong_site"
            assert gc["sentence"] != uc["sentence"]


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_licensing_asymmetry(probe_root, grammar):
    """The sidecar's own annotation confirms the design: the fired site is
    licensed in the grammatical member and unlicensed in the ungrammatical one
    (except wrong_item, where the SITE stays licensed and the pron subtype is
    the violation)."""
    for gc, uc, gm, um in _pairs(probe_root, grammar):
        if gc["construction"] == "neutral":
            continue
        if uc["violation"] == "no_substitution":
            # KNOWN GAP: the annotator marks the construction's CAT1PRON as the
            # target, and this member deliberately has none, so its
            # `annotated_tokens` carries no target role at all. The `targets`
            # list IS populated (via `_divergence_targets`), which is what the
            # scorer reads, so nothing downstream is affected — but the
            # licensing annotation cannot be checked here.
            # See docs/RECIPE_SELECTION.md Step 2.
            assert not [a for a in um["annotated_tokens"] if a["role"] == "target"]
            assert um["targets"], "the scorer's target must still be present"
            continue
        g_status = next(a["licensing_status"] for a in gm["annotated_tokens"]
                        if a["role"] == "target")
        u_status = next(a["licensing_status"] for a in um["annotated_tokens"]
                        if a["role"] == "target")
        assert g_status[grammar] == "licensed", gc["pair_id"]
        if uc["violation"] == "unlicensed_site" or uc["violation"] == "wrong_site":
            assert u_status[grammar] == "unlicensed", uc["pair_id"]
        else:  # wrong_item: positional licensing intact by design
            assert u_status[grammar] == "licensed"


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_targets_no_eos_and_roundtrip(probe_root, tokenizer, grammar):
    """Movement targets are contentful (EOS exclusion, §1) and every stored
    target span round-trips against the real tokenizer in BOTH members."""
    for gc, uc, gm, um in _pairs(probe_root, grammar):
        for c, m in ((gc, gm), (uc, um)):
            assert len(m["targets"]) == 1, m["id"]
            t = m["targets"][0]
            assert t["category"] != "EOS"
            toks = tokenizer.convert_ids_to_tokens(tokenizer.encode(m["sentence"]))
            s, e = t["model_token_span"]
            assert 0 < s < e <= len(toks) - 1
            if t["word_index"] is not None:
                assert toks[s] == _stem(m["sentence"].split()[t["word_index"]])
            else:
                # Divergence target (first differing model token): every neutral
                # member, plus the ungrammatical member of a no_substitution
                # pair, which has no CAT1PRON for the target-finder to locate.
                assert (m["construction"] == "neutral"
                        or c.get("violation") == "no_substitution"), m["id"]
        if gc["construction"] == "neutral":
            gms = _morphemes(gc["sentence"])
            ums = _morphemes(uc["sentence"])
            d = next(i for i, (x, y) in enumerate(zip(gms, ums)) if x != y)
            assert gm["targets"][0]["model_token_span"][0] == d + 1  # +BOS
            assert um["targets"][0]["model_token_span"][0] == d + 1


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_wh_verdict_stamps(probe_root, grammar):
    for gc, uc, gm, um in _pairs(probe_root, grammar):
        if gc["subcase"] in ("wh_i", "wh_ii"):
            for m in (gm, um):
                assert m["hprime_verdict"] in ("license", "not")
                assert m["lprime_verdict"] in ("license", "not")
                assert isinstance(m["divergence"], bool)


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_base_conformity_is_stamped_and_shared_within_a_pair(probe_root, grammar):
    """`base_conformity` is a property of the pair's shared BASE (§9).

    Both members are built from one deep-copied base, so a per-member verdict
    would mean the field was computed after the transform rather than on the
    base — which is the whole point of computing it at generation time.
    """
    for gc, _uc, gm, um in _pairs(probe_root, grammar):
        for m in (gm, um):
            c = m["base_conformity"]
            assert isinstance(c["matched"], bool)
            assert isinstance(c["conjuncts"], dict)
            assert all(isinstance(v, bool) for v in c["conjuncts"].values())
            assert c["matched"] == all(c["conjuncts"].values())
        assert gm["base_conformity"] == um["base_conformity"], gc["pair_id"]


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_base_conformity_names_the_right_conjuncts(probe_root, grammar):
    """Which conjuncts exist is decided by what the CORPUS conditions on.

    Pinning the key sets, not just their values: a silent rename or a dropped
    conjunct would leave `matched` looking healthy while measuring less than it
    claims to. `aux` additionally carries the `density` block, because its
    residual is low mass rather than zero mass and `matched` cannot see it.
    """
    tree = grammar in ("H", "Hprime")
    expected = {
        "refl": {"embedded_transitive_nucleus"} if tree
                else {"refl_offset_ordinal_geometry"},
        "pron": set() if tree else {"pron_offset_ordinal_geometry"},
        "wh_i": {"union_matrix_transitive", "union_has_embedded_clause"},
        "wh_ii": {"union_matrix_transitive", "union_has_embedded_clause"},
        "aux": {"corpus_aux_site"},
        "neutral_agreement": set(),
        "neutral_positional": set(),
    }
    seen = set()
    for gc, _uc, gm, _um in _pairs(probe_root, grammar):
        sc = gc["subcase"]
        seen.add(sc)
        c = gm["base_conformity"]
        assert set(c["conjuncts"]) == expected[sc], (sc, c)
        if sc == "aux":
            assert set(c["density"]) == {"n_aux", "n_matrix_aux"}
            # The probe pair needs two auxes; the corpus conditions on one.
            assert c["density"]["n_aux"] >= 2, c["density"]
    assert seen == set(SUBCASES)


@pytest.mark.parametrize("grammar", GRAMMARS)
def test_refl_conformity_holds_after_the_minimal_conditioning_refactor(
        probe_root, grammar):
    """Refl bases must lie on the corpus's conditioning path — 100%.

    Before the §1c refactor the corpus forced matrix transitivity for refl while
    probe bases were natural draws, so a large fraction of refl probe items were
    corpus-unreachable (measured `out_of_support`, mechanism confirmed at a
    perfect 179/0 contingency against matrix transitivity). C2 replaced that
    forcing with rejection sampling on rule A′'s own predicate, and the probe
    keeps a base only if it admits a refl pair — which entails a licensed site,
    which IS that predicate. So the two now coincide exactly.

    Mutation check: reinstating the build-time forcing does not move this test,
    but reinstating it while ALSO letting `base_conformity` read the forced
    geometry would — which is why the predicate is imported from
    `transforms.refl_licensing_nuclei` rather than restated here.
    """
    for gc, _uc, gm, _um in _pairs(probe_root, grammar):
        if gc["subcase"] != "refl":
            continue
        assert gm["base_conformity"]["matched"], (
            f"{gc['pair_id']}: refl base off the corpus path — "
            f"{gm['base_conformity']['conjuncts']}")


def test_ood_window_and_shared_builder():
    """The OOD probe is the same builder at [26, 48] — nothing else differs."""
    items = build_structural_probe("P", seed=99, min_length=26, max_length=48,
                                   n_per_subcase=2)
    assert len(items) == 2 * len(SUBCASES) * 2
    for it in items:
        if it["variant"] == "grammatical":
            assert 26 <= it["length"] <= 48, it["pair_id"]


def test_build_determinism(tmp_path):
    for sub in ("a", "b"):
        CorpusBuilder(GEN_CLASSES["Hprime"](seed=5), tmp_path / sub) \
            .build_structural_probe_iod(n_per_subcase=2)
    for suffix in (".jsonl", ".meta.jsonl"):
        a = (tmp_path / "a" / "structural_probe_IOD" / f"Hprime{suffix}").read_bytes()
        b = (tmp_path / "b" / "structural_probe_IOD" / f"Hprime{suffix}").read_bytes()
        assert a == b
