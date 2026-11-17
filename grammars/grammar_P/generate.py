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

from grammars.tracing import (
    TracedRandom, bernoulli, categorical, derivation_record,
    reset_trace, trace_mark, trace_rewind)
from typing import Dict, List, Optional, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.metadata import FAMILY_TOKENS
from .tokens import Token
from .lexicon import load as load_lex, cat3aux_item
from .rules import (
    PHENOMENA, PHENOMENON_PROBS, P_AUX, Q_AUX_MATRIX,
)
from .build import build_sentence
from . import binding
from .transforms import (
    apply as apply_transform, has_licensed_duplicate,
)
from .linearize import to_string


class GrammarPGenerator(BaseGrammarGenerator):
    """String generator for Grammar P (fully positional).

    Wraps the positional pipeline modules and satisfies the
    BaseGrammarGenerator interface.
    """

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="P", seed=seed)
        # TracedRandom is stream-identical to random.Random (tests/
        # test_tracing.py), so this records ell(x) without changing
        # a single generated character.
        self._rng = TracedRandom(seed)
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
        # One call = one generation ATTEMPT. generate_batch may reject it
        # (quota/length/skip), so each attempt starts a fresh record.
        reset_trace(self._rng)
        sites: Dict = {}
        surface, phenomenon, toks = _one_item(self._rng, self._lex,
                                              sites_out=sites)
        return surface, {
            "grammar_type": "P",
            "length": len(surface.split()),
            "construction": phenomenon,
            # Transient bundle for the sidecar meta record (§8.5);
            # popped by metadata.finalize_item on accepted items, never written.
            "_payload": {"struct": toks, "family": FAMILY_TOKENS, "sites": sites,
                          "derivation": derivation_record(self._rng)},
        }

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
    """Draw a phenomenon label from the spec §9.1 proportions.

    The dict is built in ``PHENOMENA`` order so the cumulative scan — and
    hence the RNG stream and the outcome — is identical to the inline loop
    this replaced.
    """
    return categorical(
        rng, {name: PHENOMENON_PROBS[name] for name in PHENOMENA},
        "phenomenon")


_MAX_SCREEN_ATTEMPTS = 20  # chance-duplicate resampling (rate ~2%/nominal pair)

# Binding items condition on the licensing geometry (2026-07-30) by RESAMPLING,
# which is rare in a draw: ~6.5% of bases can host refl B and ~20% pron N9
# (TRANSFORM_RULES §3.2). Conditioning is done by resampling the whole base,
# not by repairing one — the positional builder is scan-based with no look-ahead
# (build.py), so splicing items in to make a distance come out right is exactly
# the kind of step it may not take, and it would stamp a positional regularity on
# every binding item. Resampling instead draws from the natural base distribution
# CONDITIONED on hosting the rule, which is the population whose length/#CAT1
# profile was matched against Grammar H's forced refl/pron bases.
#
# ⚠ The previously measured cost (offline post-shipping check, 3k items each: refl 2.81
# base draws per item, pron 4.83) is SUPERSEDED for refl. It was cheap only
# because ``force_embed_here`` guaranteed an embedded transitive clause, lifting
# refl-B availability from 6.45% to ~36%. That forcing is gone (§3.2: "No
# embedding is forced"), so the refl cost reverts toward what 6.45% availability
# implies, ~15 draws per item. Pron forced nothing then and forces nothing now, so
# its 4.83 tracks the 19.85% figure and is unaffected. Both must be re-measured at
# rebuild (D10). If the cap is approached it is RAISED, never met by
# reintroducing build-time forcing.
_MAX_BIND_ATTEMPTS = 500

_BIND_SUBTYPE = {"anaphoric_binding_refl": "Refl",
                 "anaphoric_binding_pron": "Pron"}


def _one_item(rng: random.Random, lex: dict,
              apply_fn=None, sites_out: Optional[Dict] = None,
              base_out: Optional[Dict] = None) -> Tuple[str, str, List[Token]]:
    """Generate one positional sentence (spec §8.2, reformalized).

    Pipeline: phenomenon pre-selection (binding refined to refl/pron, aux
    site drawn), positional build, duplicate screen (a chance same-stem CAT1
    pair in a licensed binding configuration must never surface unsubstituted —
    resample), binding-geometry forcing (a binding item's base must host the
    subtype's offset + antecedent-ordinal geometry — resample; see
    ``_MAX_BIND_ATTEMPTS``), transformation.

    ``apply_fn`` is the transform dispatcher (default: Grammar P's
    ``transforms.apply``). Grammar L' passes its own ``apply`` here, so the
    mixed grammar shares this exact generation plan — same draws, same forced
    structures, same screens — and differs ONLY in the transform module.

    ``sites_out``: optional dict the site-primitive draws are recorded into
    (``aux_site`` — TRANSFORM_RULES §8.4; P draws no ``bind_site``, so the
    fired binding site is derived from the token list at finalize). Other
    callers keep the unchanged 3-tuple API.

    ``base_out``: optional dict receiving a ``metadata.capture_base`` snapshot
    of the pre-transform base — the − variant source for minimal pairs
    (§9). Only captured for non-neutral draws.
    """
    if apply_fn is None:
        apply_fn = apply_transform
    phenomenon = _draw_phenomenon(rng)
    if phenomenon == "anaphoric_binding":
        phenomenon = ("anaphoric_binding_refl"
                      if bernoulli(rng, 0.5, "bind_subtype")
                      else "anaphoric_binding_pron")
    aux_site = None
    if phenomenon == "auxiliary_movement":
        aux_site = ("matrix" if bernoulli(rng, Q_AUX_MATRIX, "aux_site")
                    else "embedded")
    # A wh/binding sentence may carry an in-situ compound-tense aux; only
    # *fronting* both wh and aux is barred; one phenomenon applies per sentence.
    has_aux = ((aux_site == "matrix")
               or (aux_site is None and bernoulli(rng, P_AUX, "insitu_aux")))

    bind_subtype = _BIND_SUBTYPE.get(phenomenon)
    attempts = _MAX_BIND_ATTEMPTS if bind_subtype else _MAX_SCREEN_ATTEMPTS
    screened = None  # last base that cleared the duplicate screen
    # Draws burnt on a screen/binding retry are not part of the surviving
    # derivation, so rewind to just after pre-selection each pass.
    _screen_mark = trace_mark(rng)
    for _ in range(attempts):
        trace_rewind(rng, _screen_mark)
        counter = [0]
        # Exactly two plans force anything at build time, and neither is binding.
        #
        # wh — the ONE documented exception to §1c, the UNION forcing. The four
        # grammars' clause-(i) predicates are not co-satisfiable at a common
        # minimum: H and L′ need a matrix object that HOSTS an embedded clause
        # (E3), H′ and P need only a third CAT1-position, which entails neither
        # transitivity nor embedding. The matched-procedure constraint requires
        # ONE base plan serving all four, and matrix-transitive-plus-embedding is
        # strictly the stronger predicate, so the plan forces the union: a
        # matrix-transitive clause plus one embedded clause. Availability in a
        # neutral draw is 2.25%. This is the only place in the grammar family
        # where forcing exceeds a grammar's own licensing predicate
        # (TRANSFORM_RULES §1c, §6).
        #
        #   NAMED CONSEQUENCE — P and H′ are OVER-CONDITIONED here. Their clause
        #   (i) requires neither conjunct. Any analysis of P or H′ wh items must
        #   treat matrix transitivity as a CONSTANT OF THE SLICE, not as a
        #   property the positional rule could be exploiting, and must check
        #   whether it correlates with anything the positional clause can read.
        #   Reporting a P or H′ wh result without that check states more than the
        #   design supports.
        #
        # The host is DRAWN, never pinned: pinning it to the object makes every
        # matrix object a host, which renders E3's second conjunct vacuous inside
        # the wh slice and collapses the rule back into plain "matrix object"
        # (measured 3000/3000, bounded-memory U back to 0.565 / AUC 0.917).
        # Inner transitivity stays natural — the union constrains only that the
        # embedded clause exists.
        #
        # embedded-only aux — §4 forces the clause and the compound verb, and
        # nothing else; the host and the nesting below it are drawn.
        #
        # refl and pron force NOTHING. Both P binding rules are purely positional
        # (an offset and an antecedent ordinal) and name no clause structure, so
        # their entire conditioning is the geometry resample below — rejection
        # sampling on the rule itself, §1c's canonical minimal form.
        toks = build_sentence(
            rng, lex, counter, phenomenon=phenomenon, has_aux=has_aux,
            force_embed_here=(phenomenon == "wh_movement"
                              or aux_site == "embedded"),
            force_matrix_transitive=(phenomenon == "wh_movement"),
        )
        if aux_site == "embedded":
            _ensure_embedded_aux(toks, rng, lex)
        if has_licensed_duplicate(toks):
            continue        # chance duplicate in a licensed configuration
        screened = toks
        if bind_subtype is not None and not binding.has_licensed_pair(
                [t.cat for t in toks], bind_subtype):
            continue        # geometry forcing: this base cannot host the rule
        break
    else:
        if screened is None:
            raise RuntimeError("duplicate screen exhausted — check CAT1 pool")
        # The geometry never appeared in _MAX_BIND_ATTEMPTS draws (~1e-15 for
        # refl). Fall back to a duplicate-screened base and let the transform
        # label the item *_skipped, which the corpus builder drops.
        toks = screened

    if base_out is not None and phenomenon != "neutral":
        from grammars.metadata import capture_base
        capture_base(toks, FAMILY_TOKENS, base_out)
    phenomenon = apply_fn(toks, phenomenon, rng, lex)
    if sites_out is not None:
        sites_out.update(bind_site=None, aux_site=aux_site)
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
        return  # force_embed_here guarantees a clause; guarded for safety
    i = rng.choice(embedded_verbs)
    verb = toks[i]
    aux = Token("CAT3AUX", cat3aux_item(lex), role="aux", clause_id=verb.clause_id)
    aux.feats = dict(verb.feats)
    verb.feats = {}
    toks.insert(i + 1, aux)
