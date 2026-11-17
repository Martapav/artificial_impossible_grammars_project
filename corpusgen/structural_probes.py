"""IOD / OOD structural-probe construction (TRANSFORM_RULES §9).

Grammatical-vs-ungrammatical minimal pairs probing the model's inferred
hypothesis space. Per grammar and probe: ``n_per_subcase`` pairs for each
transformation subcase (refl, pron, wh_i, wh_ii, aux — both members TRANSFORMED
on one shared base, licensed vs unlicensed site) plus that many
agreement-violation and positional-violation neutral pairs (single-edit).
Seven subcases, two members each: ``n_per_subcase`` x 14 items per grammar per
probe (4200 at the default 300).

Binding subcases split their pairs across the modes in `BINDING_MODES`: refl
three ways (wrong-item / wrong-site / no-substitution), pron two (wrong-item /
wrong-site). See that constant for why no-substitution is refl-only.

Bases are drawn **exactly as the corpus draws them for the same phenomenon**, then
kept only if they admit the pair — one licensed and one unlicensed site for the
subcase's rule.

For refl, pron, aux and the neutrals that means a natural draw: after the §1c
refactor (2026-08-05) the corpus conditions each of those on its own licensing
predicate, and the pair requirement entails a licensed site, which entails the
predicate. For **wh_i and wh_ii** it means the §6 **union** is forced here too
(``UNION_FORCED_SUBCASES``), because the corpus forces it for every wh item — see
``_draw_base``. Probe conditioning reproducing corpus conditioning is the
requirement §9 states, not a §1c exception.

One residual survives, recorded per item in ``base_conformity`` rather than
assumed away: **aux**, where the corpus conditions on one aux while the pair needs
two. That is a low-mass region of the training distribution, not a region outside
it, so it is carried in ``base_conformity.density`` rather than as a support
failure. See ``base_conformity`` below.

The two probes share
this builder verbatim — the ONLY difference is the length window
(IOD [2, 25], OOD [26, 48]; §2/§7.1). Movement pairs reject any member whose
surprisal target would be the EOS fallback (the gap must be followed by a
contentful token in BOTH members). wh pairs are stamped with both clause-(i)
verdicts (grammars/verdicts.py), so H′↔L′-divergence-relevant pairs are
identifiable from the records alone.

Every member carries the standard §8 sidecar meta record; neutral members get
their targets overridden with the pair's first-divergence model token, plus two
probe-level fields: ``item_uid`` (the generator's own item key, independent of
where the line lands in the written file) and ``rule_type`` (which of H′/L′'s
two wh clauses licenses the item, and whether the two grammars disagree about
it — ``grammars/metadata.licensing_rule_type``).
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from grammars import metadata as M
from grammars import verdicts as V
from grammars import violations as X
from grammars.tracing import bernoulli, reset_trace, trace_mark, trace_rewind


def _derivation_of(rng):
    """The replayable script for what ``rng`` just served, when it records one."""
    fn = getattr(rng, "derivation", None)
    return fn() if fn is not None else None

TRANSFORM_SUBCASES = ("refl", "pron", "wh_i", "wh_ii", "aux")
NEUTRAL_SUBCASES = ("neutral_agreement", "neutral_positional")
SUBCASES = TRANSFORM_SUBCASES + NEUTRAL_SUBCASES

# Ungrammaticality modes per binding subcase, allocated in equal blocks over
# `n_per_subcase`. Three ways for a substitution rule to be violated:
#
#   wrong_item       licensed site, the OTHER CAT1PRON form (Refl <-> Pron)
#   wrong_site       correct form, at a site its rule does not license
#   no_substitution  licensed configuration left bare — two coreferential full
#                    CAT1s where the grammar requires substitution
#
# `no_substitution` is **Refl only**. For P/L′ both subtypes are obligatory
# (`has_licensed_duplicate` screens them out of training), but H/H′ Pron
# substitution is *optional at the grammar level* — TRANSFORM_RULES §3.1 —
# and unsubstituted cross-nucleus duplicates legitimately surface in H's neutral
# items at ~2% per nominal pair, as H's blocking-contrast evidence. Building the
# violation for P-pron but not H-pron would make the two families' pron terms
# non-comparable, and the H-vs-P contrast is the experiment; so it is withheld
# from pron in BOTH families and refl carries it symmetrically.
#
# Revisit if H pron is ever forced (the open H-side narrowing): at that point the
# mode becomes valid for all eight grammar x subtype cells.
BINDING_MODES = {
    "refl": ("wrong_item", "wrong_site", "no_substitution"),
    "pron": ("wrong_item", "wrong_site"),
}

_CONSTRUCTION = {
    "refl": "anaphoric_binding_refl",
    "pron": "anaphoric_binding_pron",
    "wh_i": "wh_movement",
    "wh_ii": "wh_movement",
    "aux": "auxiliary_movement",
    "neutral_agreement": "neutral",
    "neutral_positional": "neutral",
}

_MAX_ATTEMPTS_PER_PAIR = 6000


def _registry():
    """grammar -> (family, lexicon loader). Draw machinery is family-level."""
    from grammars.grammar_H.lexicon import load as h_lex
    from grammars.grammar_P.lexicon import load as p_lex
    return {
        "H": (M.FAMILY_TREE, h_lex),
        "Hprime": (M.FAMILY_TREE, h_lex),
        "P": (M.FAMILY_TOKENS, p_lex),
        "Lprime": (M.FAMILY_TOKENS, p_lex),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Natural base drawing (same rules and probabilities as training)
# ─────────────────────────────────────────────────────────────────────────────

# Subcases whose bases are drawn under the corpus's §6 UNION conditioning rather
# than naturally (2026-08-05).
#
# WHY THIS IS NOT A §1c EXCEPTION. §1c constrains how the CORPUS conditions a
# phenomenon slice; the requirement on a probe is different and in fact opposite —
# §9 states that the IOD probe tests configurations matching the model's training
# exposure, so probe conditioning is supposed to REPRODUCE corpus conditioning.
# The corpus forces the union for every wh item (§6). A probe that drew wh bases
# naturally was therefore the thing out of line, not the corpus: it was testing the
# rule on configurations training never presented in that slice.
#
# Measured before the change (150 pairs/subcase/grammar): `base_conformity` on
# `wh_i` read 100 / 28.7 / 20.0 / 100% for H / H′ / P / L′, and on `wh_ii`
# 33.3 / 42.0 / 25.3 / 17.3% — off-path in all four, since the `wh_ii` subcase
# selects bases by clause (ii), which entails neither union conjunct in either
# family.
#
# Stratifying on conformity instead was rejected: at production size the matched
# stratum is ~300 pairs for H against ~86 for H′, so the central H-vs-H′ clause-(i)
# contrast would compare unequal power, and H′'s matched subset is not a random
# sample of its licensing conditions but exactly those third-CAT1-position bases
# that happen to carry union geometry. The unmatched stratum is not usable as
# frame-generalization evidence either — it is an accident of the builder sampled
# at grammar-dependent rates, not a designed axis. Frame generalization, if wanted,
# becomes a deliberate third probe set.
#
# Availability improves in all sixteen (grammar x subcase x window) cells, by
# 3-25x: the worst union cell costs 13.3 base draws per pair against 187.5 for the
# worst natural cell, and OOD stops being the expensive window because a union base
# is longer and lands in [26,48] far more often.
UNION_FORCED_SUBCASES = ("wh_i", "wh_ii")


def _draw_base(family: str, rng: random.Random, lex: dict,
               force_union: bool = False):
    """One base. ``force_union`` reproduces the corpus's §6 wh conditioning:
    a matrix-transitive clause plus one embedded clause, host MIXED, inner
    transitivity left natural — term for term what ``grammar_H/generate`` and
    ``grammar_P/generate`` do for a wh item."""
    if family == M.FAMILY_TREE:
        from grammars.grammar_H.build import build_type0, attach_forced_type4
        from grammars.grammar_H.features import assign
        from grammars.grammar_H.generate import _insert_aux
        from grammars.grammar_H.rules import P_AUX
        from grammars.grammar_H.transforms import has_nucleus_duplicate
        # Draws burnt on a screen retry are not part of the surviving
        # derivation, so rewind to the top of the loop each pass — the same
        # discipline the corpus generator applies to its own screen
        # (grammar_H/generate.py). No variates are consumed by mark/rewind, so
        # this is invisible to the stream and affects only the record.
        _screen_mark = trace_mark(rng)
        for _ in range(20):
            trace_rewind(rng, _screen_mark)
            counter = [0]
            tree = build_type0(rng, lex, counter, force_transitive=force_union)
            if force_union:
                attach_forced_type4(tree, rng, lex, counter,
                                    force_transitive_inner=False, mix_host=True)
            if not has_nucleus_duplicate(tree):
                break
        else:
            raise RuntimeError("duplicate screen exhausted")
        # Traced form of `rng.random() < P_AUX`. `bernoulli` consumes exactly
        # one variate, so the stream — and therefore every probe item ever
        # built from a given seed — is unchanged; the decision simply stops
        # being invisible to a recorder. The label matches the corpus
        # generator's own name for this decision (`_one_item`), because scoring
        # a probe derivation under π_corpus matches sites by label.
        if bernoulli(rng, P_AUX, "insitu_aux"):
            _insert_aux(tree, lex, counter)
        assign(tree, lex, rng)
        return tree

    from grammars.grammar_P.build import build_sentence
    from grammars.grammar_P.rules import P_AUX
    from grammars.grammar_P.transforms import has_licensed_duplicate
    _screen_mark = trace_mark(rng)
    for _ in range(20):
        trace_rewind(rng, _screen_mark)
        counter = [0]
        # As above. The argument is evaluated before the call, so the draw keeps
        # its position in the stream as well as its variate count.
        toks = build_sentence(rng, lex, counter, phenomenon=None,
                              has_aux=bernoulli(rng, P_AUX, "insitu_aux"),
                              force_embed_here=force_union,
                              force_matrix_transitive=force_union)
        if not has_licensed_duplicate(toks):
            break
    else:
        raise RuntimeError("duplicate screen exhausted")
    return toks


# ─────────────────────────────────────────────────────────────────────────────
# Base conformity (TRANSFORM_RULES §9, "Conformity field")
# ─────────────────────────────────────────────────────────────────────────────
#
# Does this probe base lie on the CORPUS's conditioning path for its phenomenon?
# Computed here, at generation time, by importing the very predicates the corpus
# generator conditions on — never inferred from a scoring reason field, which is
# bounded rather than exact.
#
# After the §1c refactor (2026-08-05) the corpus conditions every phenomenon on
# its own licensing predicate, and the probe keeps a natural base only if it
# admits the pair — which entails a licensed site, which entails the predicate.
# So the two coincide everywhere EXCEPT the two documented residuals:
#
#   wh in H′ and P   the corpus forces the §6 UNION (matrix transitive plus
#                    embedding); their clause (i) requires only a third
#                    CAT1-position, so a probe base can license wh without
#                    carrying the union geometry. H and L′ are unaffected —
#                    their clause (i) entails the union.
#   aux              the corpus conditions on ONE aux; the probe pair requires
#                    TWO (one licensed, one not). Two-aux bases are IN SUPPORT
#                    but rare, since a matrix Type2 holds at most one aux by
#                    construction (§4) — a low-mass region, not a zero-mass one,
#                    which is exactly what `out_of_support` cannot detect.
#
# Recording it per item makes those a STRATIFICATION rather than a silent
# mixture, so "has the rule been learned in the frame it was taught" and "does it
# generalize beyond that frame" are separable post hoc from one probe build.


def _matrix_transitive(base, family: str) -> bool:
    if family == M.FAMILY_TREE:
        t2 = next((c for c in base.children if c.label == "Type2"), None)
        return t2 is not None and any(c.role == "object" for c in t2.children)
    return any(t.cat == "CAT1" and t.role == "object" and t.clause_id == 0
               for t in base)


def _has_embedded_clause(base, family: str) -> bool:
    if family == M.FAMILY_TREE:
        def walk(n):
            return any(c.label == "Type4" or walk(c) for c in n.children)
        return walk(base)
    return any(t.cat == "CAT9" for t in base)


def _aux_sites_present(base, family: str):
    """(matrix aux present, embedded aux present)."""
    if family == M.FAMILY_TREE:
        t2 = next((c for c in base.children if c.label == "Type2"), None)
        matrix = t2 is not None and any(c.label == "CAT3AUX"
                                        for c in t2.children)

        def walk(n, inside):
            found = False
            for c in n.children:
                if c.label == "CAT3AUX" and inside:
                    found = True
                found = walk(c, inside or c.label == "Type4") or found
            return found
        return matrix, walk(base, False)
    return (any(t.cat == "CAT3AUX" and t.clause_id == 0 for t in base),
            any(t.cat == "CAT3AUX" and t.clause_id != 0 for t in base))


def _n_aux(base, family: str) -> int:
    if family == M.FAMILY_TREE:
        def walk(n):
            return sum((c.label == "CAT3AUX") + walk(c) for c in n.children)
        return walk(base)
    return sum(1 for t in base if t.cat == "CAT3AUX")


def base_conformity(base, family: str, grammar: str, subcase: str) -> Dict:
    """Whether ``base`` lies on the corpus's conditioning path for ``subcase``.

    Returns ``{"matched": bool, "conjuncts": {name: bool}, "density": {...}}``.
    ``conjuncts`` is empty where the corpus conditions on nothing at build time,
    which is then trivially matched; otherwise every conjunct is named, so a
    False entry says exactly WHICH one failed rather than only that conformity
    failed.

    ``matched`` is a **support** predicate: it answers "can the corpus produce a
    base like this at all", i.e. is the probability non-zero. That is the right
    question for the wh residual, where an H′/P base genuinely lies off the
    corpus's forced path. It is the WRONG question for the aux residual, which
    §9 describes as "in support but low mass — which `out_of_support` cannot
    detect, since it detects zero probability, not small probability". A two-aux
    base IS reachable (a matrix aux plus an embedded clause drawing its own at
    ``P_AUX``), so aux conformity is correctly but uninformatively 100%.

    ``density`` therefore carries the counts the low-mass stratum has to be
    selected on — for aux, how many auxes the base actually has, and where. It
    is descriptive, never folded into ``matched``: conflating "unreachable" with
    "rare" is precisely the distinction the field exists to keep.
    """
    conj: Dict[str, bool] = {}
    density: Dict[str, int] = {}

    if subcase == "refl":
        if family == M.FAMILY_TREE:
            # Rule A′, the same predicate generation resamples onto.
            from grammars.grammar_H.transforms import has_refl_licensing_nucleus
            conj["embedded_transitive_nucleus"] = bool(
                has_refl_licensing_nucleus(base))
        else:
            from grammars.grammar_P import binding
            conj["refl_offset_ordinal_geometry"] = bool(
                binding.has_licensed_pair([t.cat for t in base], "Refl"))

    elif subcase == "pron":
        if family == M.FAMILY_TOKENS:
            from grammars.grammar_P import binding
            conj["pron_offset_ordinal_geometry"] = bool(
                binding.has_licensed_pair([t.cat for t in base], "Pron"))
        # H-family pron is UNFORCED in the corpus (§3.1), so any base conforms.

    elif subcase in ("wh_i", "wh_ii"):
        # The §6 union, forced identically in both families. For H and L′ this is
        # entailed by their own clause (i); for H′ and P it is the documented
        # over-conditioning, and these two conjuncts are how it is detected.
        conj["union_matrix_transitive"] = _matrix_transitive(base, family)
        conj["union_has_embedded_clause"] = _has_embedded_clause(base, family)

    elif subcase == "aux":
        # The corpus draws an aux SITE (§4): matrix at Q_AUX_MATRIX, else
        # embedded-only with the matrix left simple. A base conforms if it looks
        # like either branch's product. Note this is satisfiable by a two-aux
        # base (matrix + embedded), which is why the aux residual is low mass
        # rather than out of support.
        matrix, embedded = _aux_sites_present(base, family)
        conj["corpus_aux_site"] = matrix or embedded
        # The low-mass stratum: the corpus conditions on ONE aux, the pair needs
        # TWO. Recorded as counts, not as a conformity failure.
        density["n_aux"] = _n_aux(base, family)
        density["n_matrix_aux"] = int(matrix)

    # neutral_agreement / neutral_positional: the corpus forces nothing.

    return {"matched": all(conj.values()), "conjuncts": conj,
            "density": density}


# ─────────────────────────────────────────────────────────────────────────────
# Member finishing: surface, meta, guards
# ─────────────────────────────────────────────────────────────────────────────

def _surface(struct, family: str) -> str:
    refs, _, _ = M._refs_and_constants(struct, family)
    return " ".join(w for w, _ in refs)


def _morpheme_stream(struct, family: str) -> Tuple[List[str], List[object]]:
    """(flat model-morpheme list, owning source ref per morpheme)."""
    refs, sep, null = M._refs_and_constants(struct, family)
    stream: List[str] = []
    owners: List[object] = []
    for w, ref in refs:
        morphs = M.word_morphemes(w, separator=sep, null_morpheme=null)
        stream.extend(morphs)
        owners.extend([ref] * len(morphs))
    return stream, owners


def _member_meta(struct, family: str, grammar: str,
                 phenomenon: str) -> Tuple[str, Dict]:
    sentence = _surface(struct, family)
    return sentence, M.build_meta_record(
        sentence, grammar, family, phenomenon, struct)


def _movement_target_ok(meta: Dict) -> bool:
    """EOS exclusion (§1/§4): the gap must be followed by a contentful token."""
    return bool(meta["targets"]) and meta["targets"][0]["category"] != "EOS"


def _stamp_wh_verdicts(meta: Dict, struct, family: str) -> None:
    v = (V.wh_verdicts_tree(struct) if family == M.FAMILY_TREE
         else V.wh_verdicts_token(struct))
    meta["hprime_verdict"] = v["hprime_verdict"] if v else None
    meta["lprime_verdict"] = v["lprime_verdict"] if v else None
    meta["divergence"] = v["divergence"] if v else None


# ─────────────────────────────────────────────────────────────────────────────
# Pair constructors (return (gram_struct, ungram_struct, violation) or None)
# ─────────────────────────────────────────────────────────────────────────────

def _make_wh_pair(base, family, grammar, clause, rng, lex):
    sites = X.wh_sites(base, family, grammar)
    if clause == "i":
        gram = [s for s in sites if s["licensed_i"] and not s["licensed_ii"]]
    else:
        gram = [s for s in sites if s["licensed_ii"] and not s["licensed_i"]]
    ungram = [s for s in sites
              if not s["licensed_i"] and not s["licensed_ii"]]
    if not gram or not ungram:
        return None
    g_site = rng.choice(gram)["site"]
    u_site = rng.choice(ungram)["site"]
    g = copy.deepcopy(base)
    u = copy.deepcopy(base)
    X.apply_wh(g, family, g_site, lex, [1_000_000])
    X.apply_wh(u, family, u_site, lex, [1_000_000])
    return g, u, "unlicensed_site"


def _make_aux_pair(base, family, rng):
    sites = X.aux_sites(base, family)
    gram = [s for s in sites if s["licensed"]]
    ungram = [s for s in sites if not s["licensed"]]
    if not gram or not ungram:
        return None
    g = copy.deepcopy(base)
    u = copy.deepcopy(base)
    X.apply_aux(g, family, rng.choice(gram)["site"])
    X.apply_aux(u, family, rng.choice(ungram)["site"])
    return g, u, "unlicensed_site"


def _make_binding_pair(base, family, subtype, mode, rng, lex):
    sites = X.binding_sites(base, family, subtype)
    gram = [s for s in sites if s["licensed"] and s["has_antecedent"]]
    if mode == "no_substitution":
        # Licensed configuration left bare: both members are coindexed at the
        # SAME licensed site and differ only in whether the CAT1PRON is written.
        # Refl only — see BINDING_MODES for why H/H′ Pron cannot host this.
        if subtype != "Refl":
            raise ValueError("no_substitution is defined for Refl only")
        if not gram:
            return None
        site = rng.choice(gram)["site"]
        g = copy.deepcopy(base)
        u = copy.deepcopy(base)
        X.apply_binding(g, family, site, subtype, lex, licensed=True)
        X.apply_binding(u, family, site, subtype, lex, licensed=True,
                        substitute=False)
        return g, u, "no_substitution"
    if mode == "wrong_item":
        # subtype-EXCLUSIVE licensed site (§7.2a): the swapped pron must not be
        # independently licensed there.
        gram = [s for s in gram if not s["licensed_other"]]
        if not gram:
            return None
        site = rng.choice(gram)["site"]
        other = "Pron" if subtype == "Refl" else "Refl"
        g = copy.deepcopy(base)
        u = copy.deepcopy(base)
        X.apply_binding(g, family, site, subtype, lex, licensed=True)
        X.apply_binding(u, family, site, other, lex, licensed=True)
        return g, u, "wrong_item"
    # wrong_site: correct subtype at a site its rule does not license.
    ungram = [s for s in sites
              if not s["licensed"] and s["has_antecedent"]]
    if not gram or not ungram:
        return None
    g = copy.deepcopy(base)
    u = copy.deepcopy(base)
    X.apply_binding(g, family, rng.choice(gram)["site"], subtype, lex,
                    licensed=True)
    X.apply_binding(u, family, rng.choice(ungram)["site"], subtype, lex,
                    licensed=False)
    return g, u, "wrong_site"


def _make_neutral_pair(base, family, kind, rng):
    u = copy.deepcopy(base)
    if kind == "neutral_agreement":
        violation = X.break_agreement(u, family, rng)
    else:
        violation = X.break_position(u, family, rng)
    if violation is None:
        return None
    return base, u, violation


# ─────────────────────────────────────────────────────────────────────────────
# Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_structural_probe(
    grammar: str,
    seed: int,
    min_length: int,
    max_length: int,
    n_per_subcase: int = 300,
    exclude_grammatical: Optional[set] = None,
    rng: Optional[random.Random] = None,
    derivations_out: Optional[List[Dict]] = None,
) -> List[Dict]:
    """Build one grammar's structural probe: flat item list, grammatical member
    immediately followed by its ungrammatical partner, quota-balanced over
    SUBCASES. The length window applies to the grammatical member.

    Distinct seeds do NOT imply distinct items. The short tail of the natural
    length distribution is a small finite space — at length 2 a
    ``neutral_agreement`` pair is one of a handful of feature combinations — so
    at production quotas the same pair recurs both within a probe and across
    probes. Two guards, because a repeat is never wanted:

    * an internal ``seen`` set rejects a pair this probe has already emitted;
    * ``exclude_grammatical`` rejects any pair whose grammatical member belongs
      to a previously built probe. Callers pass the reporting probes' members
      when building SELECT, so recipe selection cannot reuse an item that IOD
      or OOD will later report. Excluding on the grammatical member alone is
      deliberately stricter than excluding identical pairs: a shared
      grammatical member still couples the two probes' margin statistics.

    Both rejections resample rather than fail, so the quota is always met; the
    cost is that ultra-short items stop repeating, which is the point.

    ``rng`` overrides the seeded ``random.Random``; pass a
    ``scoring.DerivationRecorder`` to capture the derivation of each item. It is
    stream-identical to ``random.Random``, so the probe built is the same one
    (``tests/test_probe_scoring.py`` asserts this), and the default ``None``
    leaves every existing caller untouched.

    ``derivations_out`` receives one record per ACCEPTED pair: the base
    derivation, the subcase, and the pair id. Only accepted pairs, because the
    attempts this loop rejects (length window, movement target, uniqueness,
    exclusion) affect acceptance, not the derivation — they are part of
    p_probe's normalization and have no place in ℓ(x) under π_corpus.
    """
    family, load_lex = _registry()[grammar]
    if rng is None:
        rng = random.Random(seed)
    lex = load_lex()

    excluded = exclude_grammatical or set()
    seen: set = set()

    items: List[Dict] = []
    n_pairs = 0

    for subcase in SUBCASES:
        made = 0
        attempts = 0
        while made < n_per_subcase:
            if attempts >= n_per_subcase * _MAX_ATTEMPTS_PER_PAIR:
                raise RuntimeError(
                    f"{grammar}/{subcase}: only {made}/{n_per_subcase} pairs "
                    f"after {attempts} attempts in [{min_length}, {max_length}]"
                )
            attempts += 1
            # One pass = one ATTEMPT, so each starts a fresh record: draws burnt
            # on an attempt this loop goes on to reject are not part of the
            # emitted item's derivation. No-op on an untraced rng.
            reset_trace(rng)
            base = _draw_base(family, rng, lex,
                              force_union=subcase in UNION_FORCED_SUBCASES)
            base_derivation = _derivation_of(rng)
            # cheap prefilter: members differ from the base by ≤2 words
            if not (min_length - 2 <= len(_surface(base, family).split())
                    <= max_length + 2):
                continue

            if subcase in ("wh_i", "wh_ii"):
                pair = _make_wh_pair(base, family, grammar,
                                     subcase[-1] if subcase == "wh_i" else "ii",
                                     rng, lex)
            elif subcase == "aux":
                pair = _make_aux_pair(base, family, rng)
            elif subcase in ("refl", "pron"):
                modes = BINDING_MODES[subcase]
                mode = modes[(made * len(modes)) // n_per_subcase]
                subtype = "Refl" if subcase == "refl" else "Pron"
                pair = _make_binding_pair(base, family, subtype, mode, rng, lex)
            else:
                pair = _make_neutral_pair(base, family, subcase, rng)
            if pair is None:
                continue
            g_struct, u_struct, violation = pair

            construction = _CONSTRUCTION[subcase]
            g_sent, g_meta = _member_meta(g_struct, family, grammar, construction)
            u_sent, u_meta = _member_meta(u_struct, family, grammar, construction)

            if not (min_length <= g_meta["length"] <= max_length):
                continue
            if subcase in ("wh_i", "wh_ii", "aux"):
                if not (_movement_target_ok(g_meta)
                        and _movement_target_ok(u_meta)):
                    continue
            if subcase in ("wh_i", "wh_ii"):
                _stamp_wh_verdicts(g_meta, g_struct, family)
                _stamp_wh_verdicts(u_meta, u_struct, family)
            if construction == "neutral":
                target_pair = _divergence_targets(
                    g_struct, u_struct, family)
                if target_pair is None:
                    continue  # the edit was surface-neutral — resample
                g_meta["targets"], u_meta["targets"] = target_pair
            elif violation == "no_substitution":
                # The sidecar's target-finder locates the construction's
                # CAT1PRON, and the unsubstituted member has none — so it comes
                # back with `targets: []`, the critical span is NaN, and the
                # whole term silently drops out of the objective downstream.
                # Only the UNGRAMMATICAL member needs the divergence target: the
                # grammatical one already carries a richer native target (with
                # `word_index` and category CAT1PRON) at the same span, since the
                # members are index aligned and differ at exactly one token.
                target_pair = _divergence_targets(g_struct, u_struct, family)
                if target_pair is None:
                    continue
                u_meta["targets"] = target_pair[1]

            # Uniqueness is the last gate: everything above may still reject,
            # and a pair must not be booked as seen unless it is kept.
            if g_sent in excluded:
                continue
            if (g_sent, u_sent) in seen:
                continue
            seen.add((g_sent, u_sent))

            pair_id = f"{grammar}-sp-{n_pairs:05d}"
            # A property of the shared BASE, so both members carry the same
            # verdict — computed once, before either member was transformed.
            conformity = base_conformity(base, family, grammar, subcase)
            if derivations_out is not None:
                derivations_out.append({
                    "pair_id": pair_id,
                    "subcase": subcase,
                    "construction": construction,
                    "grammatical": g_sent,
                    "ungrammatical": u_sent,
                    "base_derivation": base_derivation,
                })
            for meta, sent, variant, viol in (
                    (g_meta, g_sent, "grammatical", None),
                    (u_meta, u_sent, "ungrammatical", violation)):
                meta["pair_id"] = pair_id
                meta["variant"] = variant
                meta["subcase"] = subcase
                meta["violation"] = viol
                meta["base_conformity"] = conformity
                # Generator-assigned, position-independent item key. The corpus
                # `id` CorpusBuilder stamps is a dense index into the written
                # file, so any rebuild that changes the item order silently
                # re-points every score joined on it. This one is derived from
                # what the builder decided, not from where the line landed.
                meta["item_uid"] = f"{pair_id}/{variant}"
                # Which of the grammar's two wh clauses licenses the item, and
                # whether H′ and L′ disagree about it. Stamped from the
                # GRAMMATICAL member's verdicts for both members, so a pair is
                # one rule-type unit: the ungrammatical member is fronted from an
                # unlicensed site and has no licensing clause of its own.
                meta["rule_type"] = M.licensing_rule_type(
                    grammar, subcase, g_meta.get("divergence"))
                items.append({
                    "sentence": sent,
                    "grammar_type": grammar,
                    "construction": construction,
                    "subcase": subcase,
                    "violation": viol,
                    "length": meta["length"],
                    "pair_id": pair_id,
                    "variant": variant,
                    "item_uid": meta["item_uid"],
                    "_meta": meta,
                })
            made += 1
            n_pairs += 1

    return items


def _divergence_targets(g_struct, u_struct, family) -> Optional[Tuple]:
    """Neutral pairs: target = the first model token at which the members'
    token streams diverge (the flipped inflection / the relocated token),
    localized in EACH member (§4). None if the streams are identical.
    The catalog's edits preserve token count, so the streams are equal-length;
    ``category`` is the owning source token's category per member."""
    gs, g_owners = _morpheme_stream(g_struct, family)
    us, u_owners = _morpheme_stream(u_struct, family)
    d = next((i for i, (a, b) in enumerate(zip(gs, us)) if a != b), None)
    if d is None:
        return None  # surface-neutral edit (or pure length change) — resample
    span = [M.BOS_OFFSET + d, M.BOS_OFFSET + d + 1]

    def entry(owners):
        return [{"role": "target",
                 "category": M._ref_cat(owners[d]),
                 "model_token_span": list(span),
                 "word_index": None}]
    return entry(g_owners), entry(u_owners)
