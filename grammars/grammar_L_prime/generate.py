"""Sentence generation for Grammar Lprime.

Lprime SHARES Grammar P's generation plan verbatim: ``_one_item`` below is
Grammar P's own ``_one_item`` (same phenomenon/site draws, forced structures,
duplicate screen, background binding) with the Lprime transform module passed
in. The two grammars differ in wh clause (i) only (matrix object from
generation history instead of third CAT1-position — see ``transforms.py``);
any change to Grammar P's pipeline propagates here automatically.

The generalization set combines two probe families for Lprime's single grammar:
clause-(i) divergence items sampled from NATURAL generation (no hand-built
probes — boundedness decisions) and labeled with both grammars' verdicts, and
depth probes of the rules Lprime keeps positional (clause (ii), binding, aux).

Construction labels: "neutral", "anaphoric_binding_refl",
"anaphoric_binding_pron", "auxiliary_movement", "wh_movement" (+ "*_skipped").
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import load as load_lex
from grammars.grammar_P.build import build_sentence
from grammars.grammar_P.linearize import to_string
from grammars.grammar_P.generate import _one_item as _p_one_item
from grammars.verdicts import wh_verdicts_token

from .transforms import apply as apply_transform


def _one_item(rng: random.Random, lex: dict) -> Tuple[str, str, List[Token]]:
    """One L' item: Grammar P's generation plan, L' transform module."""
    return _p_one_item(rng, lex, apply_fn=apply_transform)


def generate(n: int, seed: Optional[int] = None) -> List[str]:
    """Return n positionally well-formed L' surface strings."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex)[0] for _ in range(n)]


def generate_with_metadata(
    n: int, seed: Optional[int] = None
) -> List[Tuple[str, str, List[Token]]]:
    """Return (surface, phenomenon_label, token_list) for each of n items."""
    rng = random.Random(seed)
    lex = load_lex()
    return [_one_item(rng, lex) for _ in range(n)]


# ── Generalization items ──────────────────────────────────────────────────────


_MAX_GEN_ITEM_ATTEMPTS = 1000  # per item; handles right-tail length filter


def _generalization_items(
    rng: random.Random, lex: dict,
    min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Return the Lprime generalization set (both probe families combined).

    Both families are sampled from natural generation. Positional-depth items
    are re-generated until their surface length falls in
    [min_length, max_length]; clause-(i) divergence items are unconstrained in
    length (the divergent configurations carry their own length distribution).
    """
    return (
        _clause_i_divergence_items(rng, lex)
        + _positional_depth_items(rng, lex, min_length, max_length)
    )


def _positional_depth_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Probes of the rules L' keeps positional (wh clause (ii), binding, aux)
    under extra embedding depth and dependency length. Mirrors Grammar P's
    generalization set, routed through the L' transform module.

    Items are re-generated until surface length falls in [min_length, max_length].
    """
    items: List[Dict] = []
    depths = [2, 3]
    cat2_counts = [2, 4]
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
                    label = apply_transform(toks, construction, rng, lex)
                    if label.endswith("_skipped"):
                        continue  # no licensed geometry — resample (mirrors P)
                    surface = to_string(toks)
                    tok_len = len(surface.split())
                    if not (min_length <= tok_len <= max_length):
                        continue
                    items.append({
                        "sentence": surface,
                        "grammar_type": "Lprime",
                        "construction": label,
                        "length": tok_len,
                        "split": "generalization",
                        "probe": "positional_depth",
                        "embedding_depth": depth,
                        "dependency_length": cat2_count,
                    })
                    accepted += 1

                if accepted < 5:
                    raise RuntimeError(
                        f"Grammar L' positional-depth items (depth={depth}, "
                        f"cat2={cat2_count}, construction='{construction}'): only "
                        f"{accepted}/5 fell in [{min_length}, {max_length}] after "
                        f"{attempts} attempts."
                    )
    return items


def _clause_i_divergence_items(
    rng: random.Random, lex: dict,
    n_divergent: int = 30, n_coherent: int = 30,
) -> List[Dict]:
    """Clause-(i) divergence probes sampled from NATURAL L' generation.

    Draw items from the ordinary L' pipeline, keep the fronted wh ones, and
    label each with both clause-(i) verdicts recovered from the token list
    (H': is the target the third CAT1-position of the base string; L': is it
    the matrix object). Items where the verdicts disagree are the divergence
    cases; verdict-agreeing items are kept as coherent controls. No
    hand-built strings (boundedness decisions).
    """
    items: List[Dict] = []
    n_div = n_coh = 0
    attempts = 0
    while (n_div < n_divergent or n_coh < n_coherent) \
            and attempts < _MAX_GEN_ITEM_ATTEMPTS * (n_divergent + n_coherent):
        attempts += 1
        surface, label, toks = _one_item(rng, lex)
        if label != "wh_movement":
            continue
        v = wh_verdicts_token(toks)
        if v is None:
            continue
        if v["divergence"]:
            if n_div >= n_divergent:
                continue
            n_div += 1
            probe = "clause_i_divergence"
        else:
            if n_coh >= n_coherent:
                continue
            n_coh += 1
            probe = "clause_i_coherent"
        items.append({
            "sentence": surface,
            "grammar_type": "Lprime",
            "construction": label,
            "length": len(surface.split()),
            "split": "generalization",
            "probe": probe,
            "hprime_verdict": v["hprime_verdict"],
            "lprime_verdict": v["lprime_verdict"],
            "divergence": v["divergence"],
            "target_index": v["target_index"],
        })
    if n_div < n_divergent or n_coh < n_coherent:
        raise RuntimeError(
            f"Grammar L' clause-(i) divergence items: only {n_div}/{n_divergent} "
            f"divergent and {n_coh}/{n_coherent} coherent after {attempts} draws."
        )
    return items
