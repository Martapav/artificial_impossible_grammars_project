"""Top-level entry point for Grammar H sentence generation.

Exposes:
  generate(n, seed)              → List[str]
  generate_with_metadata(n, seed) → List[Tuple[str, str, Node]]
  GrammarHGenerator              → BaseGrammarGenerator subclass

Spec reference: §8.1 (Hierarchical generation pipeline), §9.1
(70% neutral, 10% per transformation).

Construction labels returned by generate_string():
  "neutral"                     (~70%)
  "anaphoric_binding_refl"      (~5%)
  "anaphoric_binding_pron"      (~5%)
  "auxiliary_movement"          (~10%)
  "wh_movement"                 (~10%)

A small fraction of items may be labeled with the *_skipped suffix when the
pre-selected phenomenon cannot be applied to the drawn base structure (e.g.
wh-movement on a sentence with no licit extraction site). These are filtered
out by BaseGrammarGenerator.generate_batch via its slot quota.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.base_generator import BaseGrammarGenerator
from .lexicon import load as load_lex
from .nodes import Node
from .rules import (
    PHENOMENA, PHENOMENON_PROBS, P_AUX, P_BIND_EMB, Q_AUX_MATRIX,
)
from .build import (
    build_type0, build_type1, build_type2, build_terminal, fresh_id,
    attach_forced_type4, _insert_embedded_aux,
)
from .features import assign
from .transforms import (
    apply as apply_transform, apply_background_binding,
    has_nucleus_duplicate, _embedded_type0s,
)
from .linearize import to_string


class GrammarHGenerator(BaseGrammarGenerator):
    """String generator for Grammar H (fully hierarchical)."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="H", seed=seed)
        self._rng = random.Random(seed)
        self._lex = load_lex()

    def load_lexicon(self) -> Dict[str, List[str]]:
        """Return {category: [form, ...]} for BaseGrammarGenerator compatibility."""
        raw = load_lex()
        return {
            cat: [item["form"] for item in items]
            for cat, items in raw.items()
            if isinstance(items, list)
        }

    def generate_string(self) -> Tuple[str, Dict]:
        surface, phenomenon, _ = _one_item(self._rng, self._lex)
        return surface, {
            "grammar_type": "H",
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, min_length, max_length)

def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return a list of n grammatical surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, Node]]:
    """Return (surface_string, phenomenon_label, tree) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]


def _draw_phenomenon(rng: random.Random) -> str:
    """Draw a phenomenon label from the spec proportions."""
    r = rng.random()
    acc = 0.0
    for name in PHENOMENA:
        acc += PHENOMENON_PROBS[name]
        if r < acc:
            return name
    return PHENOMENA[-1]


_MAX_SCREEN_ATTEMPTS = 20  # chance-duplicate resampling (rate ~2%/nominal pair)


def _one_item(rng: random.Random, lex: dict,
              apply_fn=None) -> Tuple[str, str, Node]:
    """Generate one sentence following the hierarchical pipeline (spec §8.1).

    Steps:
      1. Transformation pre-selection (binding refined to refl/pron; the refl
         nucleus site and the aux site are drawn here too).
      2. Base sentence + expansions (build_type0), plus any structure the plan
         requires (embedded transitive nucleus, embedded-only aux).
      3. Duplicate screen: a chance same-stem subject/object pair in one
         nucleus is licensed only as a reflexive intermediate state — resample.
      4. CAT3AUX insertion; inflection (features.assign).
      5. Background binding (non-binding items only), then the transformation.
      6. Surface linearization.

    ``apply_fn`` is the transform dispatcher (default: Grammar H's
    ``transforms.apply``). Grammar H' passes its own ``apply`` here, so the
    mixed grammar shares this exact generation plan — same draws, same forced
    structures, same screens — and differs ONLY in the transform module.
    """
    if apply_fn is None:
        apply_fn = apply_transform
    phenomenon = _draw_phenomenon(rng)
    if phenomenon == "anaphoric_binding":
        phenomenon = ("anaphoric_binding_refl" if rng.random() < 0.5
                      else "anaphoric_binding_pron")
    bind_site = None
    if phenomenon == "anaphoric_binding_refl":
        bind_site = "embedded" if rng.random() < P_BIND_EMB else "matrix"
    aux_site = None
    if phenomenon == "auxiliary_movement":
        aux_site = "matrix" if rng.random() < Q_AUX_MATRIX else "embedded"
    # A wh/binding sentence may carry an in-situ compound-tense aux; only
    # *fronting* both wh and aux is barred; one phenomenon applies per sentence.
    has_aux = (aux_site == "matrix") or (aux_site is None and rng.random() < P_AUX)

    for _ in range(_MAX_SCREEN_ATTEMPTS):
        counter = [0]
        tree = build_type0(rng, lex, counter,
                           force_transitive=(bind_site == "matrix"))
        if bind_site == "embedded":
            attach_forced_type4(tree, rng, lex, counter,
                                force_transitive_inner=True)
        if aux_site == "embedded":
            _force_embedded_aux(tree, rng, lex, counter)
        if not has_nucleus_duplicate(tree):
            break
    else:
        raise RuntimeError("duplicate screen exhausted — check CAT1 pool")

    if has_aux:
        _insert_aux(tree, lex, counter)
    assign(tree, lex, rng)
    if not phenomenon.startswith("anaphoric_binding"):
        apply_background_binding(tree, rng, lex)
    phenomenon = apply_fn(tree, phenomenon, rng, lex, counter,
                          bind_site=bind_site)
    surface = to_string(tree)
    return surface, phenomenon, tree


def _force_embedded_aux(tree: Node, rng: random.Random, lex: dict,
                        counter: list) -> None:
    """Guarantee an in-situ CAT3AUX inside some embedded clause.

    Used by the embedded-only aux branch: the matrix clause stays simple
    tense, so H's aux-movement must skip (structural blocking) while P fronts
    the embedded aux (divergence). Attaches a Type4 first if none exists.
    """
    inners = _embedded_type0s(tree)
    if not inners:
        inners = [attach_forced_type4(tree, rng, lex, counter)]
    without_aux = [
        t0 for t0 in inners
        if not any(
            c.label == "CAT3AUX"
            for t2 in t0.children if t2.label == "Type2"
            for c in t2.children
        )
    ]
    if without_aux:
        _insert_embedded_aux(rng.choice(without_aux), lex, counter)


def _insert_aux(tree: Node, lex: dict, counter: list) -> None:
    """Insert a CAT3AUX terminal immediately after the matrix CAT3 head.

    Spec §4.1: CAT3AUX immediately follows CAT3 in the compound-tense form.
    Only the matrix Type2 is affected; CAT3AUX inside any embedded Type4 must
    be inserted (or not) by its own recursive pipeline step — which is fine
    here because Type4 only embeds neutral Type0s built without aux.
    """
    type2 = next(c for c in tree.children if c.label == "Type2")
    head_idx = next(i for i, c in enumerate(type2.children) if c.role == "head")
    aux_node = build_terminal("CAT3AUX", lex["cat3aux"][0], "aux", counter)
    type2.children.insert(head_idx + 1, aux_node)


_MAX_GEN_ITEM_ATTEMPTS = 1000  # per item; handles right-tail length filter


def _generalization_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Generate generalization items for Grammar H.

    Spec §9.3: structural-complexity probes — vary embedding depth and
    dependency length. For each transformation we generate items where a
    structural rule must be consulted (PP-under-NP intervening, Type4
    embedded, multiple CAT2 modifiers).

    Each item is re-generated until its surface length falls in
    [min_length, max_length], placing gen items in the upper half of the
    shared N(25, 8) distribution.  With min_cat2=2 and force_pp, most
    sentences are 10-20 tokens; reaching ≥25 relies on the geometric
    expansion tails and may require O(10-50) retries per item.
    """
    items: List[Dict] = []

    # 25 items per non-neutral phenomenon, each with extra structural depth.
    for construction in ("anaphoric_binding", "auxiliary_movement", "wh_movement"):
        accepted = 0
        attempts = 0
        while accepted < 25 and attempts < 25 * _MAX_GEN_ITEM_ATTEMPTS:
            attempts += 1
            counter = [0]
            # Force a complex subject Type1: ≥2 CAT2 modifiers + a PP-under-NP.
            subj = build_type1(
                rng, lex, counter,
                role="subject", min_cat2=2, force_pp=True,
            )
            vp = build_type2(rng, lex, counter, phenomenon=construction)
            tree = Node(
                label="Type0", head_cat="CAT3", lex=None, feats={},
                children=[subj, vp], role="root",
                licensor_id=None, node_id=fresh_id(counter),
            )
            # Forced Type4 nesting (depth 2 or 3), mirroring Grammar P's
            # force_cat9_depth grid: embedding is clause-level now, so gen
            # items must force the depth the probes are about (chance draws
            # would rarely reach the [25, 48] window).
            inner = tree
            for _ in range(rng.choice((2, 3))):
                inner = attach_forced_type4(inner, rng, lex, counter)

            if has_nucleus_duplicate(tree):
                continue  # forbidden chance duplicate — resample

            has_aux = (construction == "auxiliary_movement") or (rng.random() < P_AUX)
            if construction == "wh_movement":
                has_aux = False
            if has_aux:
                _insert_aux(tree, lex, counter)

            assign(tree, lex, rng)
            phenomenon = apply_transform(tree, construction, rng, lex, counter)
            if phenomenon.endswith("_skipped"):
                continue  # no licensed geometry — resample
            surface = to_string(tree)
            length = len(surface.split())
            if not (min_length <= length <= max_length):
                continue
            items.append({
                "sentence": surface,
                "grammar_type": "H",
                "construction": phenomenon,
                "length": length,
                "split": "generalization",
            })
            accepted += 1

        if accepted < 25:
            raise RuntimeError(
                f"Grammar H gen items for '{construction}': only {accepted}/25 "
                f"fell in [{min_length}, {max_length}] after {attempts} attempts. "
                "Consider increasing min_cat2 or relaxing the length window."
            )

    return items
