"""Sentence generation for Grammar L'.

L' reuses Grammar P's whole pipeline (build / linearize / lexicon); only the
wh-movement complex-NP island differs — see ``transforms.py``. Train and
in-distribution splits are produced exactly as for Grammar P with the L'
transform module swapped in.

``rule_type`` selects only which rule the generalization split probes; it does
not change the training distribution. L' has one irreducibly structural rule (the
wh complex-NP generation-history block):
  - "hierarchical" → divergence probes targeting that structural rule.
  - "linear"       → probes of the rules L' keeps positional (second-CAT1 /
                     post-CAT9 blocks), with extra depth and length.

Construction labels: "neutral", "anaphoric_binding_refl",
"anaphoric_binding_pron", "auxiliary_movement", "wh_movement" (+ "*_skipped").
"""

from __future__ import annotations

import copy
import random
from typing import Dict, List, Optional, Tuple

from grammars.grammar_P.tokens import Token
from grammars.grammar_P.lexicon import (
    load as load_lex,
    cat1_items,
    cat3_items,
    cat6_np_items,
    cat6_vp_items,
    cat8_wh,
    compatible_cat4,
    pick,
)
from grammars.grammar_P.rules import (
    PHENOMENA, PHENOMENON_PROBS, P_AUX,
    INFL1_NUMBER_VALUES, INFL3_TENSE_VALUES,
)
from grammars.grammar_P.build import build_sentence
from grammars.grammar_P.transforms import _extract_nominal_block
from grammars.grammar_P.linearize import to_string

from .transforms import apply as apply_transform, _wh_candidates


def _draw_phenomenon(rng: random.Random) -> str:
    """Draw a phenomenon label from the 70/10/10/10 proportions."""
    r = rng.random()
    acc = 0.0
    for name in PHENOMENA:
        acc += PHENOMENON_PROBS[name]
        if r < acc:
            return name
    return PHENOMENA[-1]


def _one_item(rng: random.Random, lex: dict) -> Tuple[str, str, List[Token]]:
    """Generate one L' sentence (Grammar-P pipeline, L' transform)."""
    phenomenon = _draw_phenomenon(rng)
    has_aux = (phenomenon == "auxiliary_movement") or (rng.random() < P_AUX)
    if phenomenon == "wh_movement":
        has_aux = False  # wh and aux are mutually exclusive

    counter = [0]
    toks = build_sentence(rng, lex, counter, phenomenon=phenomenon, has_aux=has_aux)
    phenomenon = apply_transform(toks, phenomenon, rng, lex)
    return to_string(toks), phenomenon, toks


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
    rng: random.Random, lex: dict, rule_type: str,
    min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Return the L' generalization set for the active ``rule_type``.

    Positional-depth items (rule_type="linear") are re-generated until their
    surface length falls in [min_length, max_length].

    Structural-divergence items (rule_type="hierarchical") are hand-built
    minimal probes and are exempt from the length constraint — they are
    returned as-is regardless of the bounds.
    """
    if rule_type == "hierarchical":
        return _structural_divergence_items(rng, lex)
    return _positional_depth_items(rng, lex, min_length, max_length)


def _positional_depth_items(
    rng: random.Random, lex: dict, min_length: int = 25, max_length: int = 48,
) -> List[Dict]:
    """Probes of the rules L' keeps positional (second-CAT1 / post-CAT9 blocks,
    binding, aux) under extra embedding depth and dependency length. Mirrors
    Grammar P's generalization set, routed through the L' transform module.

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
                    surface = to_string(toks)
                    tok_len = len(surface.split())
                    if not (min_length <= tok_len <= max_length):
                        continue
                    items.append({
                        "sentence": surface,
                        "grammar_type": "L_prime",
                        "rule_type": "linear",
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


def _structural_divergence_items(rng: random.Random, lex: dict) -> List[Dict]:
    """Divergence probes for the L' structural complex-NP rule.

    Each probe is a hand-built string with a fixed extraction target; we front
    exactly that target and label the result with both grammars' verdicts:

      "np_adjacent"     CAT1 CAT6_np CAT1_x               L' block, H' block  (control)
      "cat2_intervener" CAT1 CAT4 CAT2 CAT6_np CAT1_x     L' block, H' permit
      "vp_after_object" CAT1 CAT3 CAT1_obj CAT6_vp CAT1_x L' permit, H' block

    L' uses the generation-history rule (``licensor_attachment``); H' is the
    surface scan applied to the same string. Items where the verdicts disagree
    are the divergence cases.
    """
    builders = [
        ("np_adjacent", _probe_np_adjacent),
        ("cat2_intervener", _probe_cat2_intervener),
        ("vp_after_object", _probe_vp_after_object),
    ]
    items: List[Dict] = []
    for probe, build in builders:
        for _ in range(20):
            toks, target_idx = build(rng, lex)
            lprime_blocks = target_idx not in _wh_candidates(toks)
            hprime_blocks = _surface_complex_np_block(toks, target_idx)

            fronted = copy.deepcopy(toks)
            _force_front(fronted, target_idx, lex)
            surface = to_string(fronted)
            items.append({
                "sentence": surface,
                "grammar_type": "L_prime",
                "rule_type": "hierarchical",
                "construction": "wh_movement",
                "length": len(surface.split()),
                "split": "generalization",
                "probe": probe,
                "lprime_blocks": lprime_blocks,
                "hprime_blocks": hprime_blocks,
                "lprime_grammatical": not lprime_blocks,
                "divergence": lprime_blocks != hprime_blocks,
            })
    return items


def _surface_complex_np_block(toks: list[Token], idx: int) -> bool:
    """H's surface complex-NP rule applied to a token list: blocks iff the word
    before ``idx`` is a CAT6 and the word before that is a CAT1."""
    return (
        idx >= 2
        and toks[idx - 1].cat == "CAT6"
        and toks[idx - 2].cat == "CAT1"
    )


def _force_front(toks: list[Token], target_idx: int, lex: dict) -> None:
    """Front the CAT1 nominal block at ``target_idx`` after a CAT8 marker,
    regardless of candidacy (used to build probe extraction attempts)."""
    block = _extract_nominal_block(toks, target_idx)
    gap = Token("GAP", None, role="gap", clause_id=block[0].clause_id)
    toks.insert(target_idx, gap)
    cat8 = Token("CAT8", cat8_wh(lex), role="wh", clause_id=0)
    toks[0:0] = [cat8, *block]


# ── Hand-built probe token lists ──────────────────────────────────────────────


def _infl_cat1(rng: random.Random, item: dict) -> dict:
    return {"INFL1_number": rng.choice(INFL1_NUMBER_VALUES),
            "INFL1_gender": item["inherent_gender"]}


def _proper(lex: dict) -> list:
    return [x for x in cat1_items(lex) if x["countability"] == "proper_like"]


def _countable(lex: dict) -> list:
    return [x for x in cat1_items(lex) if x["countability"] == "countable"
            and x.get("cat4_required")]


def _verb(rng: random.Random, lex: dict, transitivity: str, subj_feats: dict) -> Token:
    """A CAT3 verb agreeing with the subject (INFL1 copied)."""
    item = pick(rng, cat3_items(lex, transitivity=transitivity))
    feats = {"INFL1_number": subj_feats["INFL1_number"],
             "INFL1_gender": subj_feats["INFL1_gender"],
             "INFL3_tense": rng.choice(INFL3_TENSE_VALUES)}
    return Token("CAT3", item, feats=feats, role="verb", clause_id=0)


def _probe_np_adjacent(rng: random.Random, lex: dict) -> tuple[list[Token], int]:
    """CAT1(proper) CAT6_np CAT1_x CAT3 — both grammars block (control)."""
    subj_item = pick(rng, _proper(lex))
    subj_feats = _infl_cat1(rng, subj_item)
    subj = Token("CAT1", subj_item, feats=subj_feats, role="subject", clause_id=0)
    cat6 = Token("CAT6", pick(rng, cat6_np_items(lex)), role="prep",
                 clause_id=0, attachment="np")
    comp_item = pick(rng, _proper(lex))
    comp = Token("CAT1", comp_item, feats=_infl_cat1(rng, comp_item),
                 role="complement", clause_id=0, licensor_attachment="np")
    verb = _verb(rng, lex, "intransitive", subj_feats)
    toks = [subj, cat6, comp, verb]
    return toks, toks.index(comp)


def _probe_cat2_intervener(rng: random.Random, lex: dict) -> tuple[list[Token], int]:
    """CAT1(countable) CAT4 CAT2 CAT6_np CAT1_x CAT3 — L' blocks, H' permits."""
    subj_item = pick(rng, _countable(lex))
    subj_feats = _infl_cat1(rng, subj_item)
    subj = Token("CAT1", subj_item, feats=subj_feats, role="subject", clause_id=0)
    cat4 = Token("CAT4", compatible_cat4(lex, subj_feats["INFL1_number"]),
                 role="det", clause_id=0)
    cat2 = Token("CAT2", pick(rng, lex["cat2"]),
                 feats=dict(subj_feats), role="modifier", clause_id=0)
    cat6 = Token("CAT6", pick(rng, cat6_np_items(lex)), role="prep",
                 clause_id=0, attachment="np")
    comp_item = pick(rng, _proper(lex))
    comp = Token("CAT1", comp_item, feats=_infl_cat1(rng, comp_item),
                 role="complement", clause_id=0, licensor_attachment="np")
    verb = _verb(rng, lex, "intransitive", subj_feats)
    toks = [subj, cat4, cat2, cat6, comp, verb]
    return toks, toks.index(comp)


def _probe_vp_after_object(rng: random.Random, lex: dict) -> tuple[list[Token], int]:
    """CAT1 CAT3 CAT1_obj CAT6_vp CAT1_x — L' permits, H' blocks."""
    subj_item = pick(rng, _proper(lex))
    subj_feats = _infl_cat1(rng, subj_item)
    subj = Token("CAT1", subj_item, feats=subj_feats, role="subject", clause_id=0)
    verb = _verb(rng, lex, "transitive", subj_feats)
    obj_item = pick(rng, _proper(lex))
    obj = Token("CAT1", obj_item, feats=_infl_cat1(rng, obj_item),
                role="object", clause_id=0)
    cat6 = Token("CAT6", pick(rng, cat6_vp_items(lex)), role="prep",
                 clause_id=0, attachment="vp")
    comp_item = pick(rng, _proper(lex))
    comp = Token("CAT1", comp_item, feats=_infl_cat1(rng, comp_item),
                 role="complement", clause_id=0, licensor_attachment="vp")
    toks = [subj, verb, obj, cat6, comp]
    return toks, toks.index(comp)
