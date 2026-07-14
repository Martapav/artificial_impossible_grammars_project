"""Top-level entry point for Grammar P (positional) sentence generation.

Exposes:
  generate(n, seed)                → List[str]
  generate_with_metadata(n, seed)  → List[Tuple[str, str, list]]
  GrammarPGenerator                → BaseGrammarGenerator subclass

Pipeline (experiment.md §8.2):
  1. transformation pre-selection
  2-6. positional base + inflection + obligatory items + expansions (build.py)
  7. syntactic transformation (transforms.py)
  → surface linearization (linearize.py)

Construction labels returned by generate_string():
  "neutral"                (~70%)
  "anaphoric_binding_refl" (~5%)
  "anaphoric_binding_pron" (~5%)
  "auxiliary_movement"     (~10%)
  "wh_movement"            (~10%)

A small fraction of items may carry a "*_skipped" label when the pre-selected
phenomenon cannot apply to the drawn base structure; BaseGrammarGenerator
filters these out via its slot quota.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.base_generator import BaseGrammarGenerator
from .tokens import Token
from .lexicon import load as load_lex, cat3aux_item
from .rules import (
    PHENOMENA, PHENOMENON_PROBS, P_AUX, P_BG_BIND, Q_AUX_MATRIX,
)
from .build import build_sentence
from .transforms import (
    apply as apply_transform, _binding, has_licensed_duplicate,
)
from .linearize import to_string


class GrammarPGenerator(BaseGrammarGenerator):
    """String generator for Grammar P (fully positional).

    Wraps the positional pipeline modules and satisfies the
    BaseGrammarGenerator interface.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="P", seed=seed)
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
            "grammar_type": "P",
            "length": len(surface.split()),
            "construction": phenomenon,
        }

    def get_generalization_items(self, min_length: int = 25, max_length: int = 48) -> List[Dict]:
        return _generalization_items(self._rng, self._lex, min_length, max_length)

def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return a list of n positionally well-formed surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, List[Token]]]:
    """Return (surface_string, phenomenon_label, token_list) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]


def _draw_phenomenon(rng: random.Random) -> str:
    """Draw a phenomenon label from the spec §9.1 proportions."""
    r = rng.random()
    acc = 0.0
    for name in PHENOMENA:
        acc += PHENOMENON_PROBS[name]
        if r < acc:
            return name
    return PHENOMENA[-1]


_MAX_SCREEN_ATTEMPTS = 20  # chance-duplicate resampling (rate ~2%/nominal pair)


def _one_item(rng: random.Random, lex: dict,
              apply_fn=None) -> Tuple[str, str, List[Token]]:
    """Generate one positional sentence (spec §8.2, reformalized).

    Pipeline: phenomenon pre-selection (binding refined to refl/pron, aux
    site drawn), positional build, duplicate screen (a chance same-stem CAT1
    pair at a licensed binding offset must never surface unsubstituted —
    resample), background binding (non-binding items only), transformation.

    ``apply_fn`` is the transform dispatcher (default: Grammar P's
    ``transforms.apply``). Grammar L' passes its own ``apply`` here, so the
    mixed grammar shares this exact generation plan — same draws, same forced
    structures, same screens — and differs ONLY in the transform module.
    """
    if apply_fn is None:
        apply_fn = apply_transform
    phenomenon = _draw_phenomenon(rng)
    if phenomenon == "anaphoric_binding":
        phenomenon = ("anaphoric_binding_refl" if rng.random() < 0.5
                      else "anaphoric_binding_pron")
    aux_site = None
    if phenomenon == "auxiliary_movement":
        aux_site = "matrix" if rng.random() < Q_AUX_MATRIX else "embedded"
    # A wh/binding sentence may carry an in-situ compound-tense aux; only
    # *fronting* both wh and aux is barred; one phenomenon applies per sentence.
    has_aux = (aux_site == "matrix") or (aux_site is None and rng.random() < P_AUX)

    for _ in range(_MAX_SCREEN_ATTEMPTS):
        counter = [0]
        toks = build_sentence(
            rng, lex, counter, phenomenon=phenomenon, has_aux=has_aux,
            force_cat9_depth=(1 if aux_site == "embedded" else None),
        )
        if aux_site == "embedded":
            _ensure_embedded_aux(toks, rng, lex)
        if not has_licensed_duplicate(toks):
            break
    else:
        raise RuntimeError("duplicate screen exhausted — check CAT1 pool")

    if not phenomenon.startswith("anaphoric_binding") and rng.random() < P_BG_BIND:
        subtype = "Refl" if rng.random() < 0.5 else "Pron"
        _binding(toks, rng, lex, subtype=subtype, embedded_only=True)

    phenomenon = apply_fn(toks, phenomenon, rng, lex)
    return to_string(toks), phenomenon, toks


def _ensure_embedded_aux(toks: List[Token], rng: random.Random, lex: dict) -> None:
    """Guarantee an in-situ CAT3AUX inside some embedded (CAT9) clause.

    Used by the embedded-only aux branch: the matrix clause stays simple
    tense, so P's scan-based fronting targets the embedded aux (H, given the
    same plan, must skip — the divergence direction). Converts one embedded
    simple-tense verb to compound tense: the verb's inflections move onto the
    inserted aux, mirroring the build-time compound-tense routing.
    """
    if any(t.cat == "CAT3AUX" and t.clause_id != 0 for t in toks):
        return
    embedded_verbs = [
        i for i, t in enumerate(toks)
        if t.cat == "CAT3" and t.clause_id != 0 and "INFL1_number" in t.feats
    ]
    if not embedded_verbs:
        return  # force_cat9_depth=1 guarantees a clause; guarded for safety
    i = rng.choice(embedded_verbs)
    verb = toks[i]
    aux = Token("CAT3AUX", cat3aux_item(lex), role="aux", clause_id=verb.clause_id)
    aux.feats = dict(verb.feats)
    verb.feats = {}
    toks.insert(i + 1, aux)


_MAX_GEN_ITEM_ATTEMPTS = 1000  # per item; handles right-tail length filter


def _generalization_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Generate the Grammar P generalization test set.

    Spec §9.3: for Grammars H and P, structural complexity is varied along
    embedding depth and dependency length (rather than H'/L' rule divergence).
    We build a grid over CAT9 nesting depth (embedding) and subject-CAT2 count
    (dependency length between subject and verb), beyond the training caps, and
    apply each transformation to it.

    Each item is re-generated until its surface length falls in
    [min_length, max_length].  With depth ≥ 2 and cat2_count ≥ 4, most items
    naturally reach ≥25 tokens; shallower/shorter cells may require more retries.
    """
    items: List[Dict] = []
    depths = [2, 3]      # CAT9 embedding depth (training distribution is geometric)
    cat2_counts = [2, 4] # forced subject CAT2 modifiers (dependency length)

    for construction in ("anaphoric_binding", "auxiliary_movement", "wh_movement"):
        for depth in depths:
            for cat2_count in cat2_counts:
                accepted = 0
                attempts = 0
                while accepted < 5 and attempts < 5 * _MAX_GEN_ITEM_ATTEMPTS:
                    attempts += 1
                    counter = [0]
                    has_aux = construction == "auxiliary_movement"
                    toks = build_sentence(
                        rng, lex, counter,
                        phenomenon=construction, has_aux=has_aux,
                        force_cat2_subject=cat2_count, force_cat9_depth=depth,
                    )
                    if has_licensed_duplicate(toks):
                        continue  # forbidden chance duplicate — resample
                    label = apply_transform(toks, construction, rng, lex)
                    if label.endswith("_skipped"):
                        continue  # no licensed geometry — resample
                    surface = to_string(toks)
                    tok_len = len(surface.split())
                    if not (min_length <= tok_len <= max_length):
                        continue
                    items.append({
                        "sentence": surface,
                        "grammar_type": "P",
                        "construction": label,
                        "length": tok_len,
                        "split": "generalization",
                        "embedding_depth": depth,
                        "dependency_length": cat2_count,
                    })
                    accepted += 1

                if accepted < 5:
                    raise RuntimeError(
                        f"Grammar P gen items (depth={depth}, cat2={cat2_count}, "
                        f"construction='{construction}'): only {accepted}/5 fell in "
                        f"[{min_length}, {max_length}] after {attempts} attempts."
                    )
    return items
