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

from grammars.tracing import (
    TracedRandom, bernoulli, categorical, derivation_record,
    reset_trace, trace_mark, trace_rewind)
from typing import Dict, List, Optional, Tuple

from grammars.base_generator import BaseGrammarGenerator
from grammars.metadata import FAMILY_TREE
from .lexicon import load as load_lex
from .nodes import Node
from .rules import (
    PHENOMENA, PHENOMENON_PROBS, P_AUX, Q_AUX_MATRIX,
)
from .build import (
    build_type0, build_terminal, attach_forced_type4, _insert_embedded_aux,
)
from .features import assign
from .transforms import (
    apply as apply_transform,
    has_nucleus_duplicate, has_refl_licensing_nucleus, _embedded_type0s,
)
from .linearize import to_string


class GrammarHGenerator(BaseGrammarGenerator):
    """String generator for Grammar H (fully hierarchical)."""

    def __init__(self, seed: int = 42) -> None:
        super().__init__(grammar_type="H", seed=seed)
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
        surface, phenomenon, tree = _one_item(self._rng, self._lex,
                                              sites_out=sites)
        return surface, {
            "grammar_type": "H",
            "length": len(surface.split()),
            "construction": phenomenon,
            # Transient bundle for the sidecar meta record (§8.5);
            # popped by metadata.finalize_item on accepted items, never written.
            "_payload": {"struct": tree, "family": FAMILY_TREE, "sites": sites,
                          "derivation": derivation_record(self._rng)},
        }

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
    """Draw a phenomenon label from the spec proportions.

    The dict is built in ``PHENOMENA`` order so the cumulative scan — and
    hence the RNG stream and the outcome — is identical to the inline loop
    this replaced.
    """
    return categorical(
        rng, {name: PHENOMENON_PROBS[name] for name in PHENOMENA},
        "phenomenon")


_MAX_SCREEN_ATTEMPTS = 20  # chance-duplicate resampling (rate ~2%/nominal pair)

# Refl items additionally resample onto rule A′'s licensing predicate — an
# EMBEDDED nucleus with both a subject and an object, both full Type1
# (``transforms.refl_licensing_nuclei``). That predicate is rare in a natural
# draw: embedding lands at ~14.7% and the embedded clause is transitive at
# ~48.5%, so availability is ~7% and the expected cost is ~14 base draws per
# item. Held at the same cap Grammar P uses for its own binding resample, so the
# two families' conditioning mechanisms are identical in form as well as in
# effect. If the cap is ever approached it is RAISED, never met by reintroducing
# build-time forcing (TRANSFORM_RULES §1c, §7 step 5).
_MAX_REFL_ATTEMPTS = 500


def _one_item(rng: random.Random, lex: dict,
              apply_fn=None, sites_out: Optional[Dict] = None,
              base_out: Optional[Dict] = None) -> Tuple[str, str, Node]:
    """Generate one sentence following the hierarchical pipeline (spec §8.1).

    Steps:
      1. Transformation pre-selection (binding refined to refl/pron; the aux
         site is drawn here too — the refl nucleus site no longer is, see A′).
      2. Base sentence + expansions (build_type0), plus the only structure any
         plan still forces: the wh UNION (§6), and an embedded clause for the
         embedded-only aux branch. Refl forces nothing (see step 3b).
      3. Duplicate screen: a chance same-stem subject/object pair in an
         EMBEDDED nucleus is licensed only as a reflexive intermediate state —
         resample. Matrix same-nucleus duplicates are left alone: under A′ they
         are unlicensed by both sub-rules and surface as blocking evidence.
      3b. Refl geometry resample: keep the drawn base only if it satisfies rule
         A′'s licensing predicate. Rejection sampling, not construction — §1c.
      4. CAT3AUX insertion; inflection (features.assign).
      5. The transformation.
      6. Surface linearization.

    ``apply_fn`` is the transform dispatcher (default: Grammar H's
    ``transforms.apply``). Grammar H' passes its own ``apply`` here, so the
    mixed grammar shares this exact generation plan — same draws, same forced
    structures, same screens — and differs ONLY in the transform module.

    ``sites_out``: optional dict the site-primitive draws are recorded into
    (``bind_site``, ``aux_site`` — TRANSFORM_RULES §8.4).
    These decisions live on neither the tree nor the surface, so the emission
    path passes a dict here; other callers keep the unchanged 3-tuple API.

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
    # Rule A′ (TRANSFORM_RULES §3.1): the refl nucleus is ALWAYS embedded, so
    # there is no site to draw any more — P_BIND_EMB is retired. The field is
    # kept in the sidecar (it still varies for pron and for Grammar P) and is
    # simply constant for H-family refl items.
    bind_site = "embedded" if phenomenon == "anaphoric_binding_refl" else None
    aux_site = None
    if phenomenon == "auxiliary_movement":
        aux_site = ("matrix" if bernoulli(rng, Q_AUX_MATRIX, "aux_site")
                    else "embedded")
    # A wh/binding sentence may carry an in-situ compound-tense aux; only
    # *fronting* both wh and aux is barred; one phenomenon applies per sentence.
    has_aux = ((aux_site == "matrix")
               or (aux_site is None and bernoulli(rng, P_AUX, "insitu_aux")))

    # ── wh: the ONE documented exception to §1c — the UNION forcing ───────────
    #
    # The four grammars' clause-(i) predicates are not co-satisfiable at a common
    # minimum. H and L′ require a matrix object that HOSTS an embedded clause
    # (E3, grammars/wh_structural.py); H′ and P require only a third
    # CAT1-position, which entails neither transitivity nor embedding. Because
    # the matched-procedure constraint requires ONE base plan serving all four
    # grammars, and matrix-transitive-plus-embedding is strictly the stronger
    # predicate, the wh plan forces the UNION of the four predicates: a
    # matrix-transitive clause plus one embedded clause. This is the minimum
    # compatible with matching, and it is the only place in the grammar family
    # where forcing exceeds a grammar's own licensing predicate
    # (TRANSFORM_RULES §1c "One documented exception", §6 "Forcing: the union
    # case"). Availability in a neutral draw is 2.25%, which is why it is forced
    # at all.
    #
    # NAMED CONSEQUENCE — H′ and P are OVER-CONDITIONED here, and this must be
    # carried into every analysis of their wh items. For H′ and P, wh items are
    # conditioned on matrix transitivity and embedding, which their clause (i)
    # does not require. Any analysis of H′ or P wh items must therefore treat
    # matrix transitivity as a CONSTANT OF THE SLICE — not as a property the
    # positional rule could be exploiting — and must check whether it correlates
    # with anything the positional clause can read. Reporting an H′ or P wh
    # result without that check states more than the design supports.
    #
    # The HOST IS MIXED, not pinned to the object, and that is load-bearing —
    # measured, not assumed. Pinning it makes every matrix object a host, so E3's
    # second conjunct becomes VACUOUS inside the wh slice and the rule collapses
    # back into plain "matrix object", the very rule E3 replaces: 3000/3000
    # matrix objects licensed, 0 non-hosting negatives, and bounded-memory
    # recoverability re-inflates to U 0.565 / AUC 0.917 (comparable, τ=0.5).
    # Mixing restores the contrast and returns the rulewise bounded-memory
    # observer to chance (U −0.007 / AUC 0.500), at the cost of clause (i) being
    # available in ~53% of wh items rather than all.
    #
    #   General lesson: forcing a conjunct makes that conjunct vacuous unless it
    #   still varies across candidates within one sentence.
    #
    # Inner transitivity is left natural: the union constrains only that the
    # embedded clause EXISTS. Grammar H′ inherits this plan (it shares this
    # function).
    force_wh = (phenomenon == "wh_movement")

    # ── refl: rejection sampling on rule A′'s predicate (§1c) ────────────────
    #
    # NOTHING is forced at build time for refl. The base is drawn naturally and
    # kept only if it satisfies A′ — an embedded nucleus with both a subject and
    # an object, both full Type1 (``has_refl_licensing_nucleus``). The acceptance
    # test IS the rule, which is §1c's canonical minimal form, and it is the same
    # mechanism Grammar P uses for its own binding geometry.
    #
    # This replaced a build-time construction (force_transitive on the matrix
    # clause + attach_forced_type4(force_transitive_inner=True, mix_host=True)).
    # That construction forced FOUR properties A′ never names — matrix
    # transitivity, the Type4's host, its attachment level, and its depth (A′
    # licenses an embedded nucleus at ANY depth, and building one at depth 1 pins
    # it) — each of which then correlated with the refl label at a rate the
    # grammar does not specify. Rejection sampling releases all four at once.
    #
    # EXPECTED AND ACCEPTED: the residual counting signature is now stronger, not
    # weaker. With a subject host the embedded clause closes before the matrix
    # verb, so a licensed object has one verb to its left and an unlicensed one
    # two; an object host reverses that, and the two signatures cancel only to the
    # extent the host mix is balanced. Under natural matrix transitivity (~48.5%)
    # the mix is subject-weighted, so the cancellation is PARTIAL. Per §1c that
    # residual is measured on the correctly conditioned distribution and handled
    # in probe design and analysis — it is never cancelled by distorting
    # generation, which is exactly what forcing the matrix transitive was doing.
    force_refl = (bind_site == "embedded")

    # Draws burnt on a screen or geometry retry are not part of the surviving
    # derivation, so rewind to just after pre-selection each pass.
    attempts = _MAX_REFL_ATTEMPTS if force_refl else _MAX_SCREEN_ATTEMPTS
    screened = None  # last base that cleared the duplicate screen
    _screen_mark = trace_mark(rng)
    for _ in range(attempts):
        trace_rewind(rng, _screen_mark)
        counter = [0]
        tree = build_type0(rng, lex, counter, force_transitive=force_wh)
        if force_wh:
            attach_forced_type4(tree, rng, lex, counter,
                                force_transitive_inner=False, mix_host=True)
        if aux_site == "embedded":
            _force_embedded_aux(tree, rng, lex, counter)
        if has_nucleus_duplicate(tree):
            continue        # chance duplicate in a licensed configuration
        screened = tree
        if force_refl and not has_refl_licensing_nucleus(tree):
            continue        # geometry resample: this base cannot host rule A′
        break
    else:
        if screened is None:
            raise RuntimeError("duplicate screen exhausted — check CAT1 pool")
        # A′'s geometry never appeared in _MAX_REFL_ATTEMPTS draws. Fall back to
        # a duplicate-screened base and let the transform label the item
        # *_skipped, which the corpus builder drops — the same resolution
        # Grammar P applies to its own binding resample.
        tree = screened

    if has_aux:
        _insert_aux(tree, lex, counter)
    assign(tree, lex, rng)
    if base_out is not None and phenomenon != "neutral":
        from grammars.metadata import capture_base
        capture_base(tree, FAMILY_TREE, base_out)
    phenomenon = apply_fn(tree, phenomenon, rng, lex, counter)
    if sites_out is not None:
        sites_out.update(bind_site=bind_site, aux_site=aux_site)
    surface = to_string(tree)
    return surface, phenomenon, tree


def _force_embedded_aux(tree: Node, rng: random.Random, lex: dict,
                        counter: list) -> None:
    """Guarantee an in-situ CAT3AUX inside some embedded clause.

    Used by the embedded-only aux branch: the matrix clause stays simple
    tense, so H's aux-movement must skip (structural blocking) while P fronts
    the embedded aux (divergence). Attaches a Type4 first if none exists.

    The host of that Type4 is DRAWN, ``mix_host=True`` — the object at
    ``Q_EMB_OBJECT`` when the matrix clause has one, the subject otherwise,
    exactly as natural embedding draws it. It used to be pinned to the subject
    (``mix_host`` left at its default), which neither aux rule asks for: H reads
    the matrix Type2 and P scans left-to-right, so neither consults the embedded
    clause's attachment at all. Measured under the pin: 100% subject-hosted, and
    half of those draws had a matrix object available (TRANSFORM_RULES §4, §7,
    §1c).
    """
    inners = _embedded_type0s(tree)
    if not inners:
        inners = [attach_forced_type4(tree, rng, lex, counter, mix_host=True)]
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
