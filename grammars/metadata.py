"""Single source of truth for per-sentence structural metadata (TRANSFORM_RULES §8).

This module owns the *family-aware structural extractor*: given the actual object
that produced an emitted sentence — the constituent ``Node`` tree for the
hierarchical family (H, H') or the flat ``Token`` list for the positional family
(P, L') — it computes the locked §8.4 structural field set, exactly and by
construction (nothing is parsed from the surface or recovered by replay).

Design contract (why this file exists):
  * ``extract_features.py`` previously computed these fields inside a *replay*
    harness (re-running ``_one_item`` under a fresh seeded RNG). Replay is the
    thing §8 removes; the *computation* is reusable, so it is lifted here to be
    invoked inside the real emission path (``generate_string`` / ``generate_batch``).
  * The phenomenon-site primitives the surprisal targets need — ``bind_site``,
    ``aux_site``, and the wh extraction site/clause — are *decisions made in*
    ``_one_item`` (see grammar_H/generate.py and grammar_P/generate.py). They
    live on neither the tree nor the token list, so they CANNOT be recovered
    from ``struct`` here. They are accepted as pass-through inputs (``sites=``)
    and default to ``None``. This module never infers them from surface
    heuristics such as "a pronoun is present" — that inference is exactly the
    approximation §8 exists to eliminate.

One locked divergence from the old ``extract_features`` implementation: the site
primitives are threaded, not inferred (above).

(A second divergence — positional ``embedding_depth`` forced to ``None`` — was
reverted on 2026-08-04. For the positional family the clause chain is
non-branching, so ``n_embedded_clauses`` *is* the depth; see ``_token_features``.)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# The tokenizer prepends a single [BOS] token (models/tokenize.py,
# TemplateProcessing "[BOS] $A [EOS]"), so model-token index 0 is [BOS] and the
# first surface morpheme sits at index 1.
BOS_OFFSET = 1

# --------------------------------------------------------------------------
# Family tags. "tree" -> constituent Node tree (H, H'); the walk is recursive.
# "tokens" -> flat positional Token list (P, L'); the walk is a left-to-right scan.
# --------------------------------------------------------------------------

FAMILY_TREE = "tree"
FAMILY_TOKENS = "tokens"

BINDING_CONSTRUCTIONS = {"anaphoric_binding_refl", "anaphoric_binding_pron"}

# --------------------------------------------------------------------------
# The two label axes an analysis actually groups on (2026-08-04)
# --------------------------------------------------------------------------
# These used to be conflated. The sidecar wrote ``family`` for the *walk* tag
# ("tree"/"tokens") and ``grammar_type`` for the *grammar name* ("H"), and the
# scorer passed both through under those names — so the probe parquet's
# ``family`` column was a structural constant and its ``grammar_type`` column
# duplicated ``grammar``. Neither column named a phenomenon, and neither could
# be grouped on. The two axes are now explicit and separately named:
#
#   phenomenon_family — WHICH RULE the item probes (the linguistic axis)
#   structural_type   — HOW that grammar states its rules (the design axis)
#
# The walk tag stays ``family`` inside this module because that is what selects
# the walk; it is written to the sidecar as ``structural_type`` as well, under
# the name that says what it means.

PHENOMENON_FAMILY = {
    "anaphoric_binding_refl": "anaphoric_binding",
    "anaphoric_binding_pron": "anaphoric_binding",
    "wh_movement": "wh_movement",
    "auxiliary_movement": "auxiliary_movement",
    "neutral": "agreement",
}

STRUCTURAL_TYPE = {
    FAMILY_TREE: "hierarchical",
    FAMILY_TOKENS: "positional",
}

# Grammar label -> structural type, for callers that have the grammar but not
# the walk family (the scorer reading an old sidecar, mainly).
GRAMMAR_STRUCTURAL_TYPE = {
    "H": "hierarchical", "Hprime": "hierarchical",
    "P": "positional", "Lprime": "positional",
}


def phenomenon_family(construction: Optional[str]) -> Optional[str]:
    """Phenomenon family for a construction label.

    ``neutral`` maps to ``agreement``: the neutral subcases are the local
    agreement / linear-order controls, and calling that family "neutral" would
    name the *role* the items play rather than the phenomenon they probe.

    A ``*_skipped`` label keeps its base family. Those items are annotated before
    the quota filter drops them (``base_generator.generate_batch``), so raising
    on the suffix would break generation rather than catch a mistake.

    An unknown construction raises — a silent ``None`` would put items into an
    unlabelled bucket that every groupby then drops.
    """
    if construction is None:
        return None
    base = construction[:-len("_skipped")] if construction.endswith("_skipped") \
        else construction
    try:
        return PHENOMENON_FAMILY[base]
    except KeyError:
        raise ValueError(
            f"no phenomenon family for construction {construction!r}; add it to "
            f"PHENOMENON_FAMILY rather than letting it group as null."
        ) from None


def structural_type(family_or_grammar: Optional[str]) -> Optional[str]:
    """``hierarchical`` / ``positional`` from either a walk family or a grammar
    label, so a caller holding only one of the two never has to know which."""
    if family_or_grammar is None:
        return None
    if family_or_grammar in STRUCTURAL_TYPE:
        return STRUCTURAL_TYPE[family_or_grammar]
    if family_or_grammar in GRAMMAR_STRUCTURAL_TYPE:
        return GRAMMAR_STRUCTURAL_TYPE[family_or_grammar]
    raise ValueError(
        f"unknown family/grammar {family_or_grammar!r}; expected one of "
        f"{sorted(set(STRUCTURAL_TYPE) | set(GRAMMAR_STRUCTURAL_TYPE))}"
    )


# --------------------------------------------------------------------------
# Licensing rule type — the H'/L' probe axis
# --------------------------------------------------------------------------
# H′ and L′ each state wh-fronting with TWO clauses (grammars/verdicts.py):
# clause (i), the swap axis on which the two grammars deliberately disagree, and
# clause (ii), inherited unchanged from the parent grammar. An analysis of the
# H′-vs-L′ contrast has to separate three kinds of item, and until now could not:
#
#   clause_i_divergent — clause (i) licenses it AND the two grammars' clause-(i)
#                        verdicts disagree. These are the discriminating items:
#                        the only ones on which H′ and L′ make different
#                        predictions about the same string.
#   clause_i_shared    — clause (i) licenses it and both grammars agree. Same
#                        rule, no discriminating power: the within-rule control.
#   clause_ii          — licensed by the inherited clause. The across-rule
#                        control.
#
# One field rather than two because the question "which rule licenses this item"
# and the question "is this item discriminating" have the same answer here:
# only clause (i) can diverge, so a three-valued rule type partitions divergence
# items from BOTH kinds of control without a second column.
RULE_TYPE_CLAUSE_I_DIVERGENT = "clause_i_divergent"
RULE_TYPE_CLAUSE_I_SHARED = "clause_i_shared"
RULE_TYPE_CLAUSE_II = "clause_ii"

# Only the two mixed grammars have a swap axis. H and P each state one wh rule
# of their own; labelling their items with H′/L′ clause names would invite a
# crosstab that reads as a contrast when it is not one.
RULE_TYPE_GRAMMARS = ("Hprime", "Lprime")


def licensing_rule_type(grammar: str, subcase: Optional[str],
                        divergence: Optional[bool]) -> Optional[str]:
    """Which of ``grammar``'s two wh clauses licenses this item, or ``None``.

    ``None`` for every non-wh construction and for H/P (see RULE_TYPE_GRAMMARS).

    ``subcase`` is authoritative for the clause: ``build_structural_probe``
    selects the grammatical member's site as "licensed by clause (i) and NOT by
    clause (ii)" for ``wh_i`` and the converse for ``wh_ii``, so the subcase IS
    the licensing clause by construction. ``divergence`` then splits clause (i).
    """
    if grammar not in RULE_TYPE_GRAMMARS or subcase not in ("wh_i", "wh_ii"):
        return None
    if subcase == "wh_ii":
        return RULE_TYPE_CLAUSE_II
    return (RULE_TYPE_CLAUSE_I_DIVERGENT if divergence
            else RULE_TYPE_CLAUSE_I_SHARED)

# Clause-projecting label in the H constituent tree. Type0 is a clause (matrix or
# embedded): we count Type0 occurrences for clause count and their maximum
# root-to-leaf nesting for embedding depth.
_H_CLAUSE_LABELS = {"Type0"}
_H_MODIFIER_ROLES = {"np_adjunct", "vp_adjunct", "pp", "rel_clause"}
_H_VERB_CATS = {"CAT3", "CAT3AUX"}

_P_MODIFIER_CATS = {"CAT2", "CAT6"}
_P_VERB_CATS = {"CAT3", "CAT3AUX"}

# Site-primitive fields (§8.4 "raw primitives the targets need"). Threaded from
# _one_item at emission; null when this module is called outside that path.
SITE_FIELDS = ("bind_site", "aux_site", "wh_site", "wh_clause")


# --------------------------------------------------------------------------
# Surface-level features (family-independent)
# --------------------------------------------------------------------------

def surface_features(sentence: str) -> Dict:
    """Length and morpheme statistics read straight off the surface string.

    A word is ``stem#num#gen[#tense]``; ``morphemes_mean`` averages the ``#``-field
    count across words. These are descriptive extras beyond the locked §8.4 set.
    """
    words = sentence.split()
    morphs = [w.split("#") for w in words]
    counts = [len(m) for m in morphs] or [0]
    return {
        "length": len(words),
        "char_len": len(sentence),
        "morphemes_total": sum(counts),
        "morphemes_mean": sum(counts) / len(counts),
        "morphemes_max": max(counts),
    }


# --------------------------------------------------------------------------
# Structural walk — hierarchical family (Node tree)
# --------------------------------------------------------------------------

def _tree_features(root) -> Dict:
    """Recursive walk over an H/H' constituent tree.

    ``embedding_depth`` is the maximum Type0 nesting minus one (matrix == depth 0);
    ``n_embedded_clauses`` is every Type0 beyond the matrix.
    """
    n_clauses = 0
    n_modifiers = 0
    n_verbs = 0
    has_aux = False
    has_gap = False
    has_pron = False
    max_depth = 0
    max_clause_depth = 0

    def walk(node, depth, clause_depth):
        nonlocal n_clauses, n_modifiers, n_verbs, has_aux, has_gap, has_pron
        nonlocal max_depth, max_clause_depth
        max_depth = max(max_depth, depth)
        label = node.label
        role = getattr(node, "role", "")
        head = getattr(node, "head_cat", "")
        if label in _H_CLAUSE_LABELS:
            n_clauses += 1
            clause_depth += 1
            max_clause_depth = max(max_clause_depth, clause_depth)
        if role in _H_MODIFIER_ROLES:
            n_modifiers += 1
        if label == "GAP" or role == "gap":
            has_gap = True
        if label == "CAT3AUX" or head == "CAT3AUX":
            has_aux = True
        # Terminal categories are carried on head_cat for leaves.
        is_leaf = not getattr(node, "children", None)
        if is_leaf and head in _H_VERB_CATS:
            n_verbs += 1
        if is_leaf and head == "CAT1PRON":
            has_pron = True
        for child in getattr(node, "children", []) or []:
            walk(child, depth + 1, clause_depth)

    walk(root, 0, 0)
    return {
        "n_clauses": n_clauses,
        "embedding_depth": max(max_clause_depth - 1, 0),
        "n_embedded_clauses": max(n_clauses - 1, 0),
        "n_modifiers": n_modifiers,
        "n_verbs": n_verbs,
        "tree_depth": max_depth,
        "has_aux": has_aux,
        "has_gap": has_gap,
        "has_pron": has_pron,
    }


# --------------------------------------------------------------------------
# Structural walk — positional family (Token list)
# --------------------------------------------------------------------------

def _token_features(tokens: List) -> Dict:
    """Left-to-right scan over a P/L' token list.

    Positional grammars have no constituent tree, so there is no tree depth to
    read; ``tree_depth`` stays ``None``. Clauses are delimited by CAT9 and
    tracked on each token's ``clause_id`` (matrix == 0).

    ``embedding_depth`` IS populated here, as ``n_embedded_clauses`` (2026-08-04,
    reversing §8.4 decision A). Positional embedding is a strict non-branching
    chain — ``_maybe_attach_cat9`` attaches exactly one sub-clause per host and
    recurses into it (grammar_P/build.py) — so ``n_embedded_clauses`` *equals*
    the embedding depth, exactly and not approximately (verified: 0/1500
    divergence, P and L'). The generator therefore knows the depth, and writing
    ``None`` was withholding a value it had.

    Decision A's reasoning was that duplicating the field invites a reader to
    forget the two families derive it differently. What it actually produced was
    worse: an all-null column that silently dropped P and L' out of every
    depth-conditioned groupby, with nothing in the output saying so. The
    derivation stays recorded here; the value is now present.
    """
    clause_ids = {getattr(t, "clause_id", 0) for t in tokens}
    n_clauses = (max(clause_ids) + 1) if clause_ids else 1
    n_modifiers = sum(1 for t in tokens if t.cat in _P_MODIFIER_CATS)
    n_verbs = sum(1 for t in tokens if t.cat in _P_VERB_CATS)
    return {
        "n_clauses": n_clauses,
        # Exact for a non-branching chain — see the docstring.
        "embedding_depth": max(n_clauses - 1, 0),
        "n_embedded_clauses": max(n_clauses - 1, 0),
        "n_modifiers": n_modifiers,
        "n_verbs": n_verbs,
        "tree_depth": None,
        "has_aux": any(t.cat == "CAT3AUX" for t in tokens),
        "has_gap": any(t.cat == "GAP" for t in tokens),
        "has_pron": any(t.cat == "CAT1PRON" for t in tokens),
    }


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------

def _walk(struct, family: str) -> Dict:
    if family == FAMILY_TREE:
        return _tree_features(struct)
    if family == FAMILY_TOKENS:
        return _token_features(struct)
    raise ValueError(f"unknown family {family!r}; expected {FAMILY_TREE!r} or {FAMILY_TOKENS!r}")


def structural_fields(
    struct,
    family: str,
    *,
    sites: Optional[Dict] = None,
) -> Dict:
    """Locked §8.4 structural record for one sentence, exact by construction.

    ``struct`` is the emitted object (Node tree or Token list); ``family`` selects
    the walk. ``sites`` optionally carries the phenomenon-site primitives threaded
    from ``_one_item`` (any of ``SITE_FIELDS``); unset primitives are ``None`` —
    this function never infers them from the surface.

    ``expansion_sites`` (the cross-family comparison axis of §8.4) is
    ``n_embedded_clauses + n_modifiers`` in both families.
    """
    return assemble_structural(_walk(struct, family), family, sites)


def assemble_structural(feats: Dict, family: str, sites: Optional[Dict] = None) -> Dict:
    """Assemble the §8.4 record from an already-computed family walk.

    Split out of ``structural_fields`` so a walk captured BEFORE a mutating
    transform (``capture_base``, the minimal-pair − variant) can be assembled
    after the struct has changed.
    """
    record = {
        # ``family`` is the walk tag, kept for the readers that already key on
        # it; ``structural_type`` is the same fact under the name that says what
        # it means, and is what the probe tables group on.
        "family": family,
        "structural_type": structural_type(family),
        "n_clauses": feats["n_clauses"],
        "embedding_depth": feats["embedding_depth"],
        "n_embedded_clauses": feats["n_embedded_clauses"],
        "n_modifiers": feats["n_modifiers"],
        "n_verbs": feats["n_verbs"],
        "expansion_sites": feats["n_embedded_clauses"] + feats["n_modifiers"],
        "has_aux": feats["has_aux"],
        "has_gap": feats["has_gap"],
        "has_pron": feats["has_pron"],
    }
    # Site primitives: pass-through, defaulting to null. Never inferred here.
    sites = sites or {}
    for field in SITE_FIELDS:
        record[field] = sites.get(field)
    return record


# --------------------------------------------------------------------------
# Span localization (§8.2) — surface word -> model-token span [start, end)
# --------------------------------------------------------------------------

def word_morphemes(word: str, *, separator: str, null_morpheme: str) -> List[str]:
    """Model tokens contributed by one surface word.

    Mirrors ``models/dataload.py`` ``expected_morphemes`` at the single-word
    granularity, and MUST keep its operation order: strip the null morpheme as a
    unit FIRST, then split on the separator, then drop empties. The null morpheme
    contains the separator (``#NULL#`` vs ``#``), so splitting first would shatter
    it and corrupt every downstream index. A GAP word (the bare null morpheme)
    yields ``[]`` — zero model tokens — matching the tokenizer's null removal.
    """
    stripped = word.replace(null_morpheme, "")
    return [m for m in stripped.split(separator) if m]


def localize_all(
    refs_and_words: List[Tuple[str, object]],
    *,
    separator: str,
    null_morpheme: str,
    bos_offset: int = BOS_OFFSET,
) -> Tuple[Dict[int, Tuple[int, int]], int]:
    """Map every source ref to its ``model_token_span = [start, end)``.

    ``refs_and_words`` is the ``(surface_word, source_ref)`` list from a
    linearizer identity hook (``to_tokens_with_refs``), in derived
    (post-transform) order. Returns ``(spans_by_id, n_morphemes)`` where
    ``spans_by_id`` keys object identity (``id(ref)``) to its half-open span into
    the tokenized sequence, and ``n_morphemes`` is the sentence's morpheme count
    (excluding ``[BOS]``/``[EOS]``). A zero-width span (``start == end``) marks a
    ref that contributes no model tokens (a gap). Object identity is the key: the
    ``Node``/``Token`` refs stay alive in the struct, and dataclass ``==`` is not
    unique, so spans are located by ``is``-identity, never by value.
    """
    spans: Dict[int, Tuple[int, int]] = {}
    idx = bos_offset
    for word, ref in refs_and_words:
        n = len(word_morphemes(word, separator=separator, null_morpheme=null_morpheme))
        spans[id(ref)] = (idx, idx + n)
        idx += n
    return spans, idx - bos_offset


def eos_index(n_morphemes: int, *, bos_offset: int = BOS_OFFSET) -> int:
    """Model-token index of the ``[EOS]`` token — the sentence-final-gap target
    (§8.1). ``[EOS]`` sits immediately after the last surface morpheme."""
    return bos_offset + n_morphemes


# --------------------------------------------------------------------------
# Target selection (§8.1) — the surprisal target token per phenomenon
# --------------------------------------------------------------------------

def _ref_cat(ref) -> Optional[str]:
    """Category of a source ref, uniform across families. Token carries ``.cat``;
    Node carries ``.label``/``.head_cat`` (a GAP node has ``label == 'GAP'``)."""
    cat = getattr(ref, "cat", None)
    if cat is not None:
        return cat
    if getattr(ref, "label", None) == "GAP":
        return "GAP"
    return getattr(ref, "head_cat", None)


def target_of(
    refs_and_words: List[Tuple[str, object]],
    phenomenon: str,
    spans: Dict[int, Tuple[int, int]],
    n_morphemes: int,
    *,
    bos_offset: int = BOS_OFFSET,
) -> Optional[Dict]:
    """The surprisal target token for one sentence (§8.1), or ``None``.

    ``refs_and_words`` is the derived-order ``(surface_word, source_ref)`` list
    from a linearizer identity hook; ``spans``/``n_morphemes`` come from
    ``localize_all`` over that same list. The target is the STEM (first morpheme)
    of the target word, so its span is a single model token ``[start, start + 1)``.

    Selection, exact from the struct (no threading, no inference — see the
    uniqueness facts verified at build time):
      * binding (refl/pron) → the sentence's unique CAT1PRON (a binding sentence
        has exactly one substituted pronoun).
      * wh / auxiliary movement → the sentence's unique GAP, then the first
        contentful word after it in derived order (a gap contributes zero model
        tokens). If the gap is sentence-final, the target is ``[EOS]`` (§8.1
        fallback) — its ``category`` is ``"EOS"`` and ``word_index`` is ``None``.
      * neutral or any ``*_skipped`` phenomenon → ``None`` (no transformation
        fired, so there is no target).

    Returns ``{role, category, model_token_span, word_index}`` (``role`` is always
    ``"target"``; the control roles of §8.3 are assigned by the enumeration step).
    """
    if phenomenon == "neutral" or phenomenon.endswith("_skipped"):
        return None

    if phenomenon.startswith("anaphoric_binding"):
        prons = [(i, ref) for i, (_, ref) in enumerate(refs_and_words)
                 if _ref_cat(ref) == "CAT1PRON"]
        if len(prons) != 1:
            return None  # defensive: uniqueness is guaranteed for binding items
        word_index, target = prons[0]
        start, _ = spans[id(target)]
        return {"role": "target", "category": "CAT1PRON",
                "model_token_span": [start, start + 1], "word_index": word_index}

    if phenomenon in ("wh_movement", "auxiliary_movement"):
        gap_positions = [i for i, (_, ref) in enumerate(refs_and_words)
                         if _ref_cat(ref) == "GAP"]
        if len(gap_positions) != 1:
            return None  # defensive: exactly one gap per movement sentence
        g = gap_positions[0]
        for word_index in range(g + 1, len(refs_and_words)):
            _, ref = refs_and_words[word_index]
            start, end = spans[id(ref)]
            if end > start:  # first contentful word (a gap is zero-width)
                return {"role": "target", "category": _ref_cat(ref),
                        "model_token_span": [start, start + 1],
                        "word_index": word_index}
        # Sentence-final gap: the terminator carries the surprisal (§8.1).
        eos = eos_index(n_morphemes, bos_offset=bos_offset)
        return {"role": "target", "category": "EOS",
                "model_token_span": [eos, eos + 1], "word_index": None}

    return None


# --------------------------------------------------------------------------
# Candidate/control enumeration (§8.3 annotated_tokens)
# --------------------------------------------------------------------------

NOMINAL_CATS = ("CAT1", "CAT1PRON")
_AUX_CAT = "CAT3AUX"

# Which grammar labels share a binding/aux rule family (H'=H, L'=P).
_H_FAMILY = {"H", "Hprime"}
_P_FAMILY = {"P", "Lprime"}
_ALL_GRAMMARS = ("H", "P", "Hprime", "Lprime")


def _frame(phenomenon: str) -> Optional[str]:
    """The active dependency frame, or None (neutral / skipped).

    A ``*_skipped`` phenomenon fired no transformation, so it has no active
    dependency and must fall through to the baseline — checked first, because
    ``anaphoric_binding_skipped`` would otherwise match the binding prefix.
    """
    if phenomenon.endswith("_skipped"):
        return None
    if phenomenon.startswith("anaphoric_binding"):
        return "binding"
    if phenomenon == "wh_movement":
        return "wh"
    if phenomenon == "auxiliary_movement":
        return "aux"
    return None


_OFF_FRAME_NOMINAL_PHENOMENA = (
    "anaphoric_binding_refl", "anaphoric_binding_pron", "wh_movement")
_OFF_FRAME_AUX_PHENOMENON = "auxiliary_movement"


def annotate_tokens(
    struct,
    family: str,
    grammar: str,
    phenomenon: str,
    refs_and_words: List[Tuple[str, object]],
    spans: Dict[int, Tuple[int, int]],
) -> List[Dict]:
    """§8.3 annotated_tokens for one sentence — the unified entry point.

    Routes to the phenomenon-specific annotator; neutral / ``*_skipped``
    sentences (no active dependency) get every nominal/aux candidate as
    ``category_baseline``. See the per-annotator docstrings for the role and
    licensing_status semantics.

    Every entry ALSO gets ``off_frame_status`` (locked decision, 2026-07-20):
    a candidate's own-rule licensing verdict for whichever of the OTHER
    (non-active) phenomena its category is relevant to — e.g. a CAT3AUX
    sitting in a wh-movement sentence gets its aux-licensing verdict even
    though aux isn't this sentence's active phenomenon. Without this, a
    naturally-occurring blocking/licensing configuration for a phenomenon
    that ISN'T the sentence's own label was invisible (swallowed into
    ``category_baseline``, ``licensing_status`` all ``"na"``) — e.g. an H
    sentence whose only aux sits in an embedded clause (no matrix aux at
    all) is H's aux-blocking configuration regardless of which phenomenon
    actually fired here, but was previously unlabeled as such. See
    ``_off_frame_status`` for the per-phenomenon computation.
    """
    frame = _frame(phenomenon)
    if frame == "binding":
        subtype = "Refl" if phenomenon.endswith("refl") else "Pron"
        entries = annotate_binding(struct, family, grammar, subtype,
                                   refs_and_words, spans)
    elif frame == "aux":
        entries = annotate_aux(struct, family, grammar, refs_and_words, spans)
    elif frame == "wh":
        entries = annotate_wh(struct, family, grammar, refs_and_words, spans)
    else:
        entries = _annotate_baseline(refs_and_words, spans)
    _attach_off_frame_status(struct, family, phenomenon, refs_and_words, entries)
    return entries


def _attach_off_frame_status(
    struct, family: str, phenomenon: str,
    refs_and_words: List[Tuple[str, object]], entries: List[Dict],
) -> None:
    """Mutate ``entries`` in place, adding ``off_frame_status`` to each.

    Computed once per relevant off-frame phenomenon (not per-candidate) and
    fanned out by ``id(ref)``, reusing the SAME site/licensing primitives the
    on-frame annotators and ``corpusgen/structural_probes.py`` already use —
    every off-frame computation here operates on candidates that are, by
    construction, in their natural (un-transformed) position for THAT
    phenomenon: the active phenomenon is the only transform ever applied to
    a sentence, so a candidate off-frame for phenomenon X was never moved or
    substituted by X. (Binding/wh candidacy walks additionally guard against
    a DIFFERENT phenomenon's transform leaving a relabeled-but-still-typed
    node behind — see the ``role != "fronted"`` notes on
    ``_full_nominals_tree`` / ``_h_nominal_positions`` /
    ``_binding_candidates_tree``.)
    """
    needed_nominal = [p for p in _OFF_FRAME_NOMINAL_PHENOMENA if p != phenomenon]
    needed_aux = phenomenon != _OFF_FRAME_AUX_PHENOMENON

    status_by_phenomenon: Dict[str, Dict[int, Dict[str, str]]] = {}
    if "anaphoric_binding_refl" in needed_nominal:
        status_by_phenomenon["anaphoric_binding_refl"] = \
            _off_frame_binding_status(struct, family, "Refl")
    if "anaphoric_binding_pron" in needed_nominal:
        status_by_phenomenon["anaphoric_binding_pron"] = \
            _off_frame_binding_status(struct, family, "Pron")
    if "wh_movement" in needed_nominal:
        status_by_phenomenon["wh_movement"] = _off_frame_wh_status(struct, family)
    if needed_aux:
        status_by_phenomenon[_OFF_FRAME_AUX_PHENOMENON] = \
            _off_frame_aux_status(struct, family, refs_and_words)

    na = {g: "na" for g in _ALL_GRAMMARS}
    for entry in entries:
        wi = entry.get("word_index")
        rid = id(refs_and_words[wi][1]) if wi is not None else None
        off: Dict[str, Dict[str, str]] = {}
        if entry["category"] in NOMINAL_CATS:
            for name in _OFF_FRAME_NOMINAL_PHENOMENA:
                if name in status_by_phenomenon:
                    off[name] = status_by_phenomenon[name].get(rid, na)
        elif entry["category"] == _AUX_CAT and _OFF_FRAME_AUX_PHENOMENON in status_by_phenomenon:
            off[_OFF_FRAME_AUX_PHENOMENON] = status_by_phenomenon[_OFF_FRAME_AUX_PHENOMENON].get(rid, na)
        entry["off_frame_status"] = off


def _off_frame_binding_status(struct, family: str, subtype: str) -> Dict[int, Dict[str, str]]:
    """id(head-terminal ref) -> 4-key licensing_status under ``subtype``'s
    binding rule, for every nominal candidate — regardless of whether binding
    is this sentence's active phenomenon. Reuses the exact on-frame candidate
    collectors, discarding their target-detection (meaningless off-frame:
    nothing of THIS rule fired here)."""
    if family == FAMILY_TREE:
        cands, licensed_ids, _ = _binding_candidates_tree(struct, subtype)
    else:
        cands, licensed_ids, _ = _binding_candidates_tokens(struct, subtype)
    return {id(ref): _binding_status(None, family, id(ref) in licensed_ids)
            for ref, cat in cands if cat != _AUX_CAT}


def _off_frame_wh_status(struct, family: str) -> Dict[int, Dict[str, str]]:
    """id(head-terminal ref) -> 4-key licensing_status under the wh rule
    (clause (i) OR (ii)), for every wh-eligible nominal candidate, computed
    on the candidate's NATURAL (un-fronted) position via
    ``grammars.violations.wh_sites`` — the same site table
    ``corpusgen/structural_probes.py`` uses. Safe to call whenever wh is NOT
    active (the struct is then guaranteed not wh-fronted); own+prime exact,
    cross-family ``na``, matching the on-frame ``annotate_wh`` policy.
    """
    from grammars import violations as X
    if family == FAMILY_TREE:
        from grammars.grammar_H.transforms import _wh_targetable_nominals
        own_sites = X.wh_sites(struct, family, "H")
        prime_sites = X.wh_sites(struct, family, "Hprime")
        noms = _wh_targetable_nominals(struct)
        out = {}
        for own, prime, nom in zip(own_sites, prime_sites, noms):
            ref = _head_terminal_of(nom["node"])
            if ref is None:
                continue
            out[id(ref)] = {
                "H": _v(own["licensed_i"] or own["licensed_ii"]),
                "Hprime": _v(prime["licensed_i"] or prime["licensed_ii"]),
                "P": "na", "Lprime": "na",
            }
        return out
    own_sites = X.wh_sites(struct, family, "P")
    prime_sites = X.wh_sites(struct, family, "Lprime")
    out = {}
    for own, prime in zip(own_sites, prime_sites):
        ref = struct[own["site"]]
        out[id(ref)] = {
            "P": _v(own["licensed_i"] or own["licensed_ii"]),
            "Lprime": _v(prime["licensed_i"] or prime["licensed_ii"]),
            "H": "na", "Hprime": "na",
        }
    return out


def _off_frame_aux_status(
    struct, family: str, refs_and_words: List[Tuple[str, object]],
) -> Dict[int, Dict[str, str]]:
    """id(ref) -> 4-key licensing_status under the aux rule (all four exact),
    for every CAT3AUX candidate — regardless of whether aux is this
    sentence's active phenomenon. No base-recovery/splice-back needed (unlike
    the on-frame ``annotate_aux``): an off-frame sentence was never aux-
    transformed, so every aux is already in its natural position.
    """
    from grammars.licensing import (
        neutral_from_tree, neutral_from_tokens, aux_hfamily_licenses, aux_pfamily_licenses,
    )
    neutral = (neutral_from_tree(struct) if family == FAMILY_TREE
              else neutral_from_tokens(struct))
    refs = [ref for _, ref in refs_and_words]
    out = {}
    for i, (nt, ref) in enumerate(zip(neutral, refs)):
        if nt.cat != _AUX_CAT:
            continue
        h = aux_hfamily_licenses(neutral, i)
        p = aux_pfamily_licenses(neutral, i)
        out[id(ref)] = {"H": _v(h), "Hprime": _v(h), "P": _v(p), "Lprime": _v(p)}
    return out


def _annotate_baseline(refs_and_words, spans) -> List[Dict]:
    """All nominal/aux candidates as ``category_baseline`` (no active dependency)."""
    entries = []
    for wi, (word, ref) in enumerate(refs_and_words):
        cat = _ref_cat(ref)
        if cat not in NOMINAL_CATS and cat != _AUX_CAT:
            continue
        start, end = spans[id(ref)]
        span = [start, start + 1] if end > start else [start, end]
        entries.append({
            "role": "category_baseline", "category": cat,
            "licensing_status": {g: "na" for g in _ALL_GRAMMARS},
            "model_token_span": span, "word_index": wi,
        })
    return entries


def _binding_status(grammar: str, own_family: str, licensed: bool) -> Dict[str, str]:
    """Four-key licensing_status for a binding candidate: own family exact,
    cross-family 'na' (H-family binding is structural, not evaluable on a
    positional sentence — the locked decision)."""
    verdict = "licensed" if licensed else "unlicensed"
    fam = _H_FAMILY if own_family == "tree" else _P_FAMILY
    return {g: (verdict if g in fam else "na") for g in _ALL_GRAMMARS}


def annotate_binding(
    struct,
    family: str,
    grammar: str,
    subtype: str,
    refs_and_words: List[Tuple[str, object]],
    spans: Dict[int, Tuple[int, int]],
) -> List[Dict]:
    """§8.3 annotated_tokens for a binding-phenomenon sentence.

    Nominal candidates (CAT1/CAT1PRON) get binding-frame roles; auxes get
    ``category_baseline`` (off-frame). The target-role candidate is the
    substituted CAT1PRON (== the §8.1 surprisal target for binding). Licensing is
    the own-family exact set (cross-family 'na').
    """
    word_index_by_id = {id(ref): wi for wi, (_, ref) in enumerate(refs_and_words)}

    # Own-family licensed head-terminal identities, plus the fired-target id.
    if family == FAMILY_TREE:
        cands, licensed_ids, target_id = _binding_candidates_tree(struct, subtype)
    else:
        cands, licensed_ids, target_id = _binding_candidates_tokens(struct, subtype)

    entries: List[Dict] = []
    for ref, cat in cands:
        start, end = spans[id(ref)]
        span = [start, start + 1] if end > start else [start, end]
        wi = word_index_by_id.get(id(ref))
        if cat == _AUX_CAT:
            role, licensed = "category_baseline", False
        elif id(ref) == target_id:
            # The fired position: on corpus items its site is always licensed
            # (fired targets are own-licensed by construction); computed rather
            # than assumed so the structural probes' VIOLATING substitutions
            # (unlicensed site) annotate truthfully.
            role, licensed = "target", id(ref) in licensed_ids
        elif id(ref) in licensed_ids:
            role, licensed = "licensed_unrealized", True
        else:
            role, licensed = "unlicensed", False
        status = (_binding_status(grammar, family, licensed)
                  if cat != _AUX_CAT
                  else {g: "na" for g in _ALL_GRAMMARS})
        entries.append({
            "role": role, "category": cat,
            "licensing_status": status,
            "model_token_span": span, "word_index": wi,
        })
    return entries


def _wh_target_eligible(cat: str, lex) -> bool:
    """A wh candidate can front iff it is a CAT1 or a CAT1PRON-Pron (Refl never
    fronts)."""
    if cat == "CAT1":
        return True
    return cat == "CAT1PRON" and lex is not None and lex.get("subclass") == "Pron"


def _wh_block_end(toks, start: int = 1) -> int:
    """Index just past the fronted nominal block starting at ``toks[start]``.

    The block is the head CAT1/CAT1PRON plus its contiguous same-clause CAT4/CAT2
    — the granularity every grammar fronts (TRANSFORM_RULES §6).
    """
    head = toks[start]
    end = start + 1
    while (end < len(toks) and toks[end].cat in ("CAT4", "CAT2")
           and toks[end].clause_id == head.clause_id):
        end += 1
    return end


def base_order_tokens(toks: List) -> List:
    """A P-family token list in BASE order: the fronted element spliced back at
    its GAP, the wh CAT8 dropped. Returns ``toks`` unchanged if nothing fronted.

    Both P-family licensing rules that read linear structure — wh (ordinals over
    CAT1-positions) and binding (an item distance plus an antecedent ordinal) —
    are defined on the BASE string; transformations compose afterwards
    (TRANSFORM_RULES §1, "derivational rules, surface composition"). Any caller
    holding a possibly-transformed list must therefore recover the base first,
    and this is the one place that does it.

    Two derived shapes exist and they are exhaustive: only wh emits a CAT8, only
    aux fronting puts a CAT3AUX at absolute position 0 (a base always begins with
    the matrix subject CAT1), and exactly one phenomenon applies per sentence. A
    list with no GAP is a base already, so this is the identity on every
    untransformed and every binding-transformed sentence.

    The SAME Token objects are returned, only reordered, so callers keying
    results by ``id(token)`` map straight back onto the derived enumeration —
    which is what the surface spans and word indices must keep referring to.

    Note the GAP is REMOVED, not merely skipped: it renders as ``#NULL#`` and so
    occupies a whitespace-delimited slot, but it carries no lexical item and must
    not be counted as one (D4 — an "item" is a lexeme with its #-attached
    inflections, never a bare slot).
    """
    if len(toks) < 2:
        return toks
    if toks[0].cat == "CAT8":
        end = _wh_block_end(toks, 1)
        block, rest = toks[1:end], toks[end:]
    elif toks[0].cat == "CAT3AUX":
        block, rest = toks[0:1], toks[1:]
    else:
        return toks
    gi = next((i for i, t in enumerate(rest) if t.cat == "GAP"), None)
    if gi is None:
        return toks
    return rest[:gi] + block + rest[gi + 1:]


def _wh_facts_tokens(toks) -> Optional[Dict]:
    """Base recovery + per-nominal P/L' wh verdicts for a fronted token list.

    Splices the fronted block back at the GAP to restore base order
    (``base_order_tokens``), then reads the positional clauses from
    ``grammar_P.wh`` (which owns them: 3rd CAT1-position; 2nd CAT1-position after
    the 2nd CAT6 — clause (ii) revised 2026-07-30) and the structural clause
    (i'') from ``grammars.wh_structural`` (the matrix object that HOSTS an
    embedded clause — proposal E3, 2026-07-30). Returns
    ``{id(token): {"P": bool, "Lprime": bool}, "_target_id": id(fronted head)}``.
    """
    if not toks or toks[0].cat != "CAT8":
        return None
    head = toks[1]
    base = base_order_tokens(toks)
    if base is toks:            # no GAP: not a fronted wh item after all
        return None

    from grammars.grammar_P import wh as p_wh
    from grammars import wh_structural
    cats = [t.cat for t in base]
    positions = p_wh.nominal_positions(cats)
    third = p_wh.clause_i_index(cats)
    anchored = p_wh.clause_ii_index(cats)

    facts = {"_target_id": id(head)}
    for p in positions:
        t = base[p]
        eligible = _wh_target_eligible(t.cat, t.lex)
        third_pos = (p == third)
        clause_ii = (p == anchored)
        matrix_obj = wh_structural.is_licensed_token(t)   # clause (i''), E3
        facts[id(t)] = {
            "P": eligible and (third_pos or clause_ii),
            "Lprime": eligible and (matrix_obj or clause_ii),
        }
    return facts


def annotate_wh(
    struct,
    family: str,
    grammar: str,
    refs_and_words: List[Tuple[str, object]],
    spans: Dict[int, Tuple[int, int]],
) -> List[Dict]:
    """§8.3 annotated_tokens for a wh-movement sentence.

    Nominal candidates get wh-frame roles; auxes get ``category_baseline``. The
    target-role candidate is the FRONTED nominal (distinct from §8.1's post-gap
    surprisal target). Own + same-family prime verdicts are exact; cross-family is
    ``na`` (the other family's clause-(ii) fact is representation-specific).
    """
    if family == FAMILY_TREE:
        facts = _wh_facts_tree(struct)
        own_keys, na_keys = ("H", "Hprime"), ("P", "Lprime")
    else:
        facts = _wh_facts_tokens(struct)
        own_keys, na_keys = ("P", "Lprime"), ("H", "Hprime")
    if facts is None:
        return []
    target_id = facts.get("_target_id")

    entries: List[Dict] = []
    for wi, (word, ref) in enumerate(refs_and_words):
        cat = _ref_cat(ref)
        if cat not in NOMINAL_CATS and cat != _AUX_CAT:
            continue
        start, end = spans[id(ref)]
        span = [start, start + 1] if end > start else [start, end]
        fv = facts.get(id(ref)) if cat in NOMINAL_CATS else None
        if fv is None:
            role = "category_baseline"
            status = {g: "na" for g in _ALL_GRAMMARS}
        else:
            if id(ref) == target_id:
                role = "target"
            elif fv[grammar]:
                role = "licensed_unrealized"
            else:
                role = "unlicensed"
            status = {g: _v(fv[g]) for g in own_keys}
            status.update({g: "na" for g in na_keys})
        entries.append({
            "role": role, "category": cat,
            "licensing_status": status,
            "model_token_span": span, "word_index": wi,
        })
    return entries


def _head_terminal_of(nom):
    """The span-bearing head terminal of an H-family nominal (Type1 → its 'head'
    child; CAT1PRON → itself)."""
    if nom.label == "CAT1PRON":
        return nom
    return next((c for c in nom.children if c.role == "head"), None)


def _wh_facts_tree(tree):
    """Per-nominal H/L' wh verdicts for a fronted tree, keyed by head-terminal id.

    Evaluated on the BASE: the fronted constituent is conceptually spliced back at
    its GAP. A single in-order pass over the tree body (skipping CAT8 and the
    fronted node) EXPANDS the GAP into the fronted subtree at the gap's structural
    context, so every nominal gets its base clause-(i)/(ii) facts and its base
    linear position (for the 3rd CAT1-position fact). The fronted head's own
    ``role`` was overwritten to "fronted" by the transform, so its
    ``matrix_object`` is read from the gap's parent (a Type2 direct-child nominal
    is the object). Cross-family (P/L') is 'na', so the P-family clause (ii) is
    not needed here.

    **Clause (i) is E3** (fixed 2026-07-31): ``matrix_object AND hosts_type4``,
    the rule ``grammars/wh_structural.py`` owns. This function previously used
    plain ``matrix_object`` — the pre-2026-07-30 rule — and so reported every
    non-hosting matrix object as H-licensed. The fired *target* was unaffected (a
    matrix object can only have BEEN fronted if E3 licensed it), which is why the
    target-only consistency check against ``verdicts.py`` stayed green; the damage
    was to the in-situ CONTROLS, i.e. exactly the ``unlicensed`` negatives E3
    exists to create. Measured before the fix: 5.4% of candidates mislabelled,
    touching 31.7% of H and 40.3% of H′ wh sentences.

    The two readings of the host conjunct differ by construction:
      * an IN-SITU nominal keeps its children, so the fact is ``any Type4 child``;
      * the FRONTED nominal does not — ``front_nominal`` strands its Type3/Type4
        as the GAP's following siblings, so the fact is read at the EXTRACTION
        SITE. That is the rule's own "the licensor stays at the gap, not on the
        filler" property (wh_structural.py) showing up in recovery, and it is the
        same read ``verdicts._stranded_has_type4`` performs — imported rather than
        restated, since a second copy is the drift this consolidation removes.
    """
    from grammars.verdicts import _stranded_has_type4

    if len(tree.children) < 2 or tree.children[1].role != "fronted":
        return None
    fronted = tree.children[1]
    htf = _head_terminal_of(fronted)
    if htf is None:
        return None

    # head-terminal id -> (matrix_object, hosts_type4, in_type3_under_t4, ht_node)
    struct = {}
    order = []    # base linear order of nominal head-terminal ids

    def record(node, in_lt3, mo, hosts):
        ht = _head_terminal_of(node)
        if ht is not None:
            struct[id(ht)] = (mo, hosts, in_lt3, ht)
            order.append(id(ht))

    def visit(node, parent, idx, in_lt3, inside_t4, clause_matrix):
        if node.label == "GAP":
            # The gap is the fronted constituent's base slot. Record the fronted
            # head from the gap's context (role was overwritten -> position-based
            # matrix_object; the hosted Type4 was stranded here, so E3's second
            # conjunct is read off the gap's siblings), then its nested nominals
            # at the same context.
            record(fronted, in_lt3,
                   clause_matrix and parent is not None
                   and parent.label == "Type2",
                   _stranded_has_type4(parent, idx))
            for i, c in enumerate(fronted.children):
                visit(c, fronted, i,
                      in_lt3 or (fronted.label == "Type3" and inside_t4),
                      inside_t4 or fronted.label == "Type4",
                      clause_matrix and fronted.label != "Type4")
            return
        is_nom = node.label == "Type1" or (
            node.label == "CAT1PRON" and node.lex is not None)
        if is_nom and parent is not None and node is not fronted:
            record(node, in_lt3,
                   node.role == "object" and clause_matrix
                   and parent.label == "Type2",
                   any(c.label == "Type4" for c in node.children))
        for i, c in enumerate(node.children):
            visit(c, node, i,
                  in_lt3 or (node.label == "Type3" and inside_t4),
                  inside_t4 or node.label == "Type4",
                  clause_matrix and node.label != "Type4")

    for i, child in enumerate(tree.children[2:], start=2):
        visit(child, tree, i, False, False, True)
    third = order[2] if len(order) >= 3 else None

    facts = {"_target_id": id(htf)}
    for htid, (mo, hosts, in_lt3, ht_node) in struct.items():
        # Eligibility: the head terminal must be CAT1 or CAT1PRON-Pron.
        eligible = _wh_target_eligible(ht_node.head_cat, ht_node.lex)
        third_pos = (htid == third)
        facts[htid] = {
            # clause (i) = E3 (matrix object AND it hosts an embedded clause),
            # clause (ii) = inside a Type3 dominated by a Type4.
            "H": eligible and ((mo and hosts) or in_lt3),
            "Hprime": eligible and (third_pos or in_lt3),
        }
    return facts


def _binding_candidates_tree(tree, subtype):
    """(candidates, licensed head-terminal ids, target id) for an H-family tree.

    A nominal candidate is a Type1 (represented by its head CAT1 terminal — the
    span-bearing token) or a substituted CAT1PRON leaf; the licensed set from
    ``licensing.py`` is keyed by nominal node_id and remapped to the head
    terminal's object identity so it matches ``refs_and_words``/``spans``.

    Excludes ``role == "fronted"``: a wh-fronted nominal keeps its Type1/
    CAT1PRON label (relabeled+repositioned by ``front_nominal``, not removed —
    the vacated slot gets a separate GAP), so it would otherwise double as a
    phantom candidate outside its clausal slot. This function is now also
    called off-frame (a sentence's active phenomenon may be wh, not binding),
    so this guard matters even though it's a no-op when binding is active.
    """
    from grammars.grammar_H.nodes import is_terminal
    from grammars.licensing import binding_hfamily_licensed_ids

    # Enforce coreference (TRANSFORM_RULES §2): a candidate is a licensed binding
    # target only if it forms a genuine duplicated-CAT1 pair, reconstructed from
    # the substituted pron's ``coref_form`` — not merely the right structure.
    lic_nom_ids = binding_hfamily_licensed_ids(tree, subtype, require_coref=True)
    cands, licensed_ids, target_id = [], set(), None

    def head_terminal(nom):
        if nom.label == "CAT1PRON":
            return nom
        return next((c for c in nom.children if c.role == "head"), None)

    def walk(node):
        nonlocal target_id
        is_nominal = (node.label == "Type1" or (
            node.label == "CAT1PRON" and node.lex is not None)
        ) and node.role != "fronted"
        if is_nominal:
            ht = head_terminal(node)
            if ht is not None:
                cands.append((ht, ht.head_cat))
                if node.node_id in lic_nom_ids:
                    licensed_ids.add(id(ht))
                if node.label == "CAT1PRON":
                    target_id = id(ht)
        elif is_terminal(node) and node.head_cat == _AUX_CAT:
            cands.append((node, _AUX_CAT))
        for c in node.children:
            walk(c)

    walk(tree)
    return cands, licensed_ids, target_id


def _v(licensed: bool) -> str:
    return "licensed" if licensed else "unlicensed"


def annotate_aux(
    struct,
    family: str,
    grammar: str,
    refs_and_words: List[Tuple[str, object]],
    spans: Dict[int, Tuple[int, int]],
) -> List[Dict]:
    """§8.3 annotated_tokens for an auxiliary-movement sentence.

    Aux candidates (CAT3AUX) get aux-frame roles; nominals get
    ``category_baseline``. The target-role candidate is the FRONTED aux (at
    absolute position 0), distinct from §8.1's post-gap surprisal target.
    Licensing is evaluated on the base order (fronted aux spliced back at its
    GAP); all four keys are exact (H=H' matrix rule, P=L' first-in-scan rule).
    """
    from grammars.licensing import (
        neutral_from_tree, neutral_from_tokens,
        aux_hfamily_licenses, aux_pfamily_licenses,
    )
    neutral = (neutral_from_tree(struct) if family == FAMILY_TREE
               else neutral_from_tokens(struct))
    refs = [ref for _, ref in refs_and_words]
    # neutral is the same in-order terminal walk as refs_and_words: 1:1 aligned.
    pairs = list(zip(neutral, refs))

    # Base recovery: the fronted aux sits at index 0; splice it back at the GAP.
    # Its clause_id is taken from the GAP, not from ``fronted_nt`` — for the tree
    # family, ``neutral_from_tree`` derives clause_id from the CURRENT (post-
    # fronting) tree position, which for a fronted embedded aux (H-family
    # ungrammatical probe sites) reads as clause 0 (top of tree) rather than its
    # true base clause; the GAP, walked in place, carries the correct clause_id.
    fronted_nt, fronted_ref = pairs[0]
    rest = pairs[1:]
    gi = next((i for i, (nt, _) in enumerate(rest) if nt.cat == "GAP"), None)
    if gi is None or fronted_nt.cat != _AUX_CAT:
        base_pairs, target_ref_id = pairs, None   # defensive (e.g. _skipped)
    else:
        base_nt = fronted_nt._replace(clause_id=rest[gi][0].clause_id)
        base_pairs = rest[:gi] + [(base_nt, fronted_ref)] + rest[gi + 1:]
        target_ref_id = id(fronted_ref)
    base_neutral = [nt for nt, _ in base_pairs]
    base_index = {id(ref): i for i, (_, ref) in enumerate(base_pairs)}

    entries: List[Dict] = []
    for wi, (word, ref) in enumerate(refs_and_words):
        cat = _ref_cat(ref)
        if cat not in NOMINAL_CATS and cat != _AUX_CAT:
            continue
        start, end = spans[id(ref)]
        span = [start, start + 1] if end > start else [start, end]
        if cat == _AUX_CAT and id(ref) in base_index:
            bi = base_index[id(ref)]
            h = aux_hfamily_licenses(base_neutral, bi)
            p = aux_pfamily_licenses(base_neutral, bi)
            if id(ref) == target_ref_id:
                role = "target"
            elif (h if family == FAMILY_TREE else p):
                role = "licensed_unrealized"
            else:
                role = "unlicensed"
            status = {"H": _v(h), "Hprime": _v(h), "P": _v(p), "Lprime": _v(p)}
        else:
            role = "category_baseline"
            status = {g: "na" for g in _ALL_GRAMMARS}
        entries.append({
            "role": role, "category": cat,
            "licensing_status": status,
            "model_token_span": span, "word_index": wi,
        })
    return entries


def _binding_candidates_tokens(toks, subtype):
    """(candidates, licensed token ids, target id) for a P-family token list.

    Tokens are flat and stable under binding (in-place substitution), so a
    candidate's object identity is the token itself.

    Licensing is computed on the BASE order (``base_order_tokens``), never on
    ``toks`` directly. Both of P's binding conjuncts are counts over the base
    string — the antecedent→target distance in items and the antecedent's
    CAT1-position ordinal (TRANSFORM_RULES §1, §3.2) — so a transformed list
    gives the wrong answer twice over: the GAP occupies an index while carrying
    no lexical item, and wh fronting moves the block to the front, shifting every
    distance that spans the extraction site and every ordinal before it. This is
    the identity on a binding sentence (one phenomenon per sentence, so no GAP);
    it matters on the OFF-FRAME call, where this predicate is applied to a wh- or
    aux-transformed sentence.

    Candidate ENUMERATION stays on ``toks``: ``word_index`` and
    ``model_token_span`` must keep referring to derived surface positions.
    Splicing only reorders the same Token objects, so the licensed set — carried
    by ``id(token)`` — maps back onto that enumeration with no index translation.
    """
    from grammars.licensing import binding_pfamily_licensed_indices

    base = base_order_tokens(toks)
    base_cats = ["CAT1" if t.cat == "CAT1PRON" else t.cat for t in base]
    # Parallel base-order forms, with each binding pron reverted to its
    # antecedent's form (Token.coref_form) — reconstructs the coreferential
    # duplicate so licensing enforces coreference, not mere category distance.
    base_forms = [
        (t.coref_form if t.cat == "CAT1PRON"
         else (t.lex.get("form") if t.lex is not None else None))
        for t in base
    ]
    lic_idx = binding_pfamily_licensed_indices(base_cats, subtype, base_forms=base_forms)
    licensed_tok_ids = {id(base[i]) for i in lic_idx}

    cands, licensed_ids, target_id = [], set(), None
    for t in toks:
        if t.cat in NOMINAL_CATS:
            cands.append((t, t.cat))
            if id(t) in licensed_tok_ids:
                licensed_ids.add(id(t))
            if t.cat == "CAT1PRON":
                target_id = id(t)
        elif t.cat == _AUX_CAT:
            cands.append((t, _AUX_CAT))
    return cands, licensed_ids, target_id


# --------------------------------------------------------------------------
# Meta-record assembly (§8.5) — the sidecar record for one sentence
# --------------------------------------------------------------------------

def _refs_and_constants(struct, family):
    """(refs_and_words, SEPARATOR, NULL_MORPHEME) via the family's linearizer
    identity hook. Lazy imports keep this module free of top-level grammar
    dependencies."""
    if family == FAMILY_TREE:
        from grammars.grammar_H.linearize import to_tokens_with_refs
        from grammars.grammar_H.rules import SEPARATOR, NULL_MORPHEME
    else:
        from grammars.grammar_P.linearize import to_tokens_with_refs
        from grammars.grammar_P.rules import SEPARATOR, NULL_MORPHEME
    return to_tokens_with_refs(struct), SEPARATOR, NULL_MORPHEME


def _complete_sites(struct, family, phenomenon, sites: Dict) -> Dict:
    """Fill the fired-phenomenon site fields from the emitted struct, exactly.

    Threaded values (the draws made in ``_one_item``: ``bind_site``,
    ``aux_site``) always win; this only fills what was not
    drawn, reading object-level facts off the struct via the neutral rep — the
    same mechanism ``target_of``/``annotate_*`` use, never the surface:
      * binding → the clause of the sentence's unique CAT1PRON (the substituted
        one — a binding sentence has exactly one);
      * aux → the clause of the unique GAP (the fronted aux's base slot);
      * wh → ``wh_clause`` = the GAP's clause id, ``wh_site`` = matrix/embedded.
    Neutral / ``*_skipped`` sentences pass through unchanged (no fired site).
    """
    frame = _frame(phenomenon)
    if frame is None:
        return sites
    from grammars.licensing import neutral_from_tree, neutral_from_tokens
    neutral = (neutral_from_tree(struct) if family == FAMILY_TREE
               else neutral_from_tokens(struct))

    def clause_of(cat: str) -> Optional[int]:
        return next((nt.clause_id for nt in neutral if nt.cat == cat), None)

    def site_label(clause: Optional[int]) -> Optional[str]:
        return None if clause is None else ("matrix" if clause == 0 else "embedded")

    if frame == "binding" and sites.get("bind_site") is None:
        sites["bind_site"] = site_label(clause_of("CAT1PRON"))
    elif frame == "aux" and sites.get("aux_site") is None:
        sites["aux_site"] = site_label(clause_of("GAP"))
    elif frame == "wh":
        clause = clause_of("GAP")
        if clause is not None:
            sites["wh_clause"] = clause
            sites["wh_site"] = site_label(clause)
    return sites


def build_meta_record(
    sentence: str,
    grammar: str,
    family: str,
    phenomenon: str,
    struct,
    sites: Optional[Dict] = None,
    derivation: Optional[Dict] = None,
) -> Dict:
    """The full sidecar record for one emitted sentence (§8.5).

    Called on the exact object that produced ``sentence`` (accept/reject
    survivors only — annotation consumes no RNG, so computing after acceptance
    rides the same trajectory). ``id`` is stamped later by ``CorpusBuilder``
    (it owns the split name). Raises if the identity hook does not reproduce
    the emitted surface — the exactness guarantee of §8.
    """
    refs_and_words, separator, null_morpheme = _refs_and_constants(struct, family)
    rebuilt = " ".join(w for w, _ in refs_and_words)
    if rebuilt != sentence:
        raise ValueError(
            f"identity hook mismatch for {grammar}: linearizer reproduced "
            f"{rebuilt!r} but the emitted sentence is {sentence!r}"
        )
    spans, n_morphemes = localize_all(
        refs_and_words, separator=separator, null_morpheme=null_morpheme)
    sites = _complete_sites(struct, family, phenomenon, dict(sites or {}))
    target = target_of(refs_and_words, phenomenon, spans, n_morphemes)
    return {
        # ``grammar_type`` is the grammar LABEL ("H", "Lprime") — the corpus
        # filename stem, keyed on throughout the builder. The structural axis it
        # used to be confused with is ``structural_type``, written by
        # ``assemble_structural`` below; the phenomenon axis is
        # ``phenomenon_family``.
        "grammar_type": grammar,
        "construction": phenomenon,
        "phenomenon_family": phenomenon_family(phenomenon),
        **structural_fields(struct, family, sites=sites),
        **surface_features(sentence),
        "n_morphemes": n_morphemes,
        "targets": [target] if target is not None else [],
        "annotated_tokens": annotate_tokens(
            struct, family, grammar, phenomenon, refs_and_words, spans),
        # ell(x): the RAW derivation cost, in nats, of the trajectory that
        # produced THIS string (entropy plan §3.1). Omitted when the generator
        # ran on an untraced RNG. NOT -log p_corpus: the corpus is
        # rejection-sampled, so a per-(grammar, construction) correction from
        # the acceptance record is required -- see grammars/tracing.py.
        **({"derivation": derivation} if derivation is not None else {}),
    }


def finalize_item(item: Dict) -> None:
    """Convert a generator item's transient ``_payload`` into its ``_meta`` record.

    ``_payload`` is the non-serializable bundle attached by ``generate_string``
    / the generalization builders: ``{"struct", "family", "sites"}``. It is
    popped here (never written to disk); the resulting ``_meta`` dict is the
    sidecar record, later separated from the corpus record by
    ``CorpusBuilder._write``. Items without a payload are left untouched.
    """
    payload = item.pop("_payload", None)
    if payload is None:
        return
    item["_meta"] = build_meta_record(
        sentence=item["sentence"],
        grammar=item["grammar_type"],
        family=payload["family"],
        phenomenon=item["construction"],
        struct=payload["struct"],
        sites=payload.get("sites"),
        derivation=payload.get("derivation"),
    )


def capture_base(struct, family: str, out: Dict) -> None:
    """Snapshot the pre-transform base for minimal-pair emission (§9).

    MUST be called immediately BEFORE the phenomenon transform mutates the
    struct. Everything mutation-sensitive is materialized now: surface words are
    rendered to strings, per-ref categories are recorded (binding replaces the
    nominal with a new CAT1PRON object, and agreement fixup can re-render a
    verb), and the structural walk runs on the intact base. The refs
    themselves stay live inside the (about to be transformed) struct, which is
    what lets the − variant map the + variant's target/control positions back
    onto the base linearization by object identity.
    """
    refs_and_words, separator, null_morpheme = _refs_and_constants(struct, family)
    out["refs_and_words"] = refs_and_words
    out["cats"] = {id(ref): _ref_cat(ref) for _, ref in refs_and_words}
    out["features"] = _walk(struct, family)
    out["separator"] = separator
    out["null_morpheme"] = null_morpheme
