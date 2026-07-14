"""Tree construction for Grammar H.

Builds Type0/1/2/3/4 constituent trees from the lexicon; feature assignment is in features.py.
Type3 has two variants: PP-under-NP (cat6_np_items, CAT1-selecting) and PP-under-VP
(cat6_vp_items, CAT3-selecting). role="object"/"pp" in build_type1 builds a Dependent Type1
with the same internal structure as the subject Type1.
"""

from __future__ import annotations

import random

from .nodes import Node
from .lexicon import (
    cat1_items,
    cat3_items,
    cat6_np_items,
    cat6_vp_items,
    cat9_items,
    pick,
)
from .rules import (
    P_CAT2,
    P_CAT5,
    P_AUX,
    P_PP_UNDER_NP,
    P_PP_UNDER_NP_COMP,
    P_PP_UNDER_VP,
    P_TYPE4,
    Q_EMB_OBJECT,
)


def fresh_id(counter: list) -> int:
    counter[0] += 1
    return counter[0]


def build_terminal(label: str, lex_item: dict, role: str, counter: list) -> Node:
    return Node(
        label=label, head_cat=label, lex=lex_item, feats={},
        children=[], role=role, licensor_id=None, node_id=fresh_id(counter),
    )


def build_type3_under_np(rng: random.Random, lex: dict, counter: list,
                         parent_licensor_id: int | None = None) -> Node:
    """Type3 as a CAT1 dependent (PP-under-NP): CAT6_NP + T1dep.

    licensor_id records the enclosing CAT1 head — unused in Grammar H but read by Grammar L'.
    """
    cat6 = pick(rng, cat6_np_items(lex))
    head = build_terminal("CAT6", cat6, "head", counter)
    pp_np = build_type1(rng, lex, counter, role="pp")
    return Node(
        label="Type3", head_cat="CAT6", lex=None, feats={"attachment": "np"},
        children=[head, pp_np], role="np_adjunct",
        licensor_id=parent_licensor_id, node_id=fresh_id(counter),
    )


def build_type3_under_vp(rng: random.Random, lex: dict, counter: list,
                         parent_licensor_id: int | None = None) -> Node:
    """Type3 as a CAT3 adjunct (PP-under-VP): CAT6_VP + T1dep."""
    cat6 = pick(rng, cat6_vp_items(lex))
    head = build_terminal("CAT6", cat6, "head", counter)
    pp_np = build_type1(rng, lex, counter, role="pp")
    return Node(
        label="Type3", head_cat="CAT6", lex=None, feats={"attachment": "vp"},
        children=[head, pp_np], role="vp_adjunct",
        licensor_id=parent_licensor_id, node_id=fresh_id(counter),
    )


def _insert_embedded_aux(inner: Node, lex: dict, counter: list) -> None:
    """Insert an in-situ CAT3AUX after the CAT3 head of an embedded Type0's Type2.

    Mirrors the matrix ``_insert_aux`` (grammar_H/generate.py) but is defined here
    to avoid a build↔generate import cycle. A no-op if the embedded VP has no head.
    """
    type2 = next((c for c in inner.children if c.label == "Type2"), None)
    if type2 is None:
        return
    head_idx = next((i for i, c in enumerate(type2.children) if c.role == "head"), None)
    if head_idx is None:
        return
    aux_node = build_terminal("CAT3AUX", lex["cat3aux"][0], "aux", counter)
    type2.children.insert(head_idx + 1, aux_node)


def build_type4(rng: random.Random, lex: dict, counter: list,
                parent_licensor_id: int | None = None,
                force_transitive_inner: bool = False) -> Node:
    """Type4 (relative clause): CAT9 + embedded Type0.

    The embedded Type0 may realize compound tense in situ (an embedded CAT3AUX),
    drawn independently at P_AUX. The aux is never fronted (aux-movement is a
    matrix operation). ``force_transitive_inner`` forces a transitive embedded
    nucleus (used when a reflexive item's licensing nucleus is embedded).
    """
    cat9 = pick(rng, cat9_items(lex))
    head = build_terminal("CAT9", cat9, "head", counter)
    inner = build_type0(rng, lex, counter,
                        force_transitive=force_transitive_inner)
    if rng.random() < P_AUX:
        _insert_embedded_aux(inner, lex, counter)
    return Node(
        label="Type4", head_cat="CAT9", lex=None, feats={},
        children=[head, inner], role="rel_clause",
        licensor_id=parent_licensor_id, node_id=fresh_id(counter),
    )


def attach_forced_type4(tree: Node, rng: random.Random, lex: dict, counter: list,
                        force_transitive_inner: bool = False) -> Node:
    """Attach a Type4 to the matrix subject Type1 and return its inner Type0.

    Used when an item's pre-drawn plan requires embedded structure (embedded
    reflexive nucleus, embedded-only auxiliary) that the probabilistic draws
    did not produce. Type4s are positioned finally within their host Type1.
    """
    subject = next(c for c in tree.children if c.role == "subject")
    head = next(c for c in subject.children if c.role == "head")
    t4 = build_type4(rng, lex, counter, parent_licensor_id=head.node_id,
                     force_transitive_inner=force_transitive_inner)
    subject.children.append(t4)
    return t4.children[1]

def build_type1(rng: random.Random, lex: dict, counter: list,
                role: str = "subject",
                min_cat2: int = 0,
                force_pp: bool = False) -> Node:
    """Build a Type1 node (role="subject" → matrix subject; "object"/"pp" → T1dep).

    [+cnt] schema: CAT1 CAT4 (CAT2)* (Type3)?
    [-cnt] schema: CAT1 (Type3)?  — proper-like CAT1 takes no CAT4/CAT2.

    A subject Type1 may additionally receive one clause-level Type4,
    positioned finally (attached by build_type0, not here).
    min_cat2 and force_pp are used by generalization items to force extra complexity.
    """
    cat1 = pick(rng, cat1_items(lex))
    head = build_terminal("CAT1", cat1, "head", counter)
    children = [head]

    is_countable = cat1["countability"] == "countable"

    # CAT4 placeholder: only countable CAT1s license a determiner.
    if cat1["cat4_required"]:
        slot = Node(
            label="CAT4_SLOT", head_cat="CAT4", lex=None, feats={},
            children=[], role="det", licensor_id=None, node_id=fresh_id(counter),
        )
        children.append(slot)

    # CAT2 modifiers: only countable CAT1 admits them (spec §2.4).
    if is_countable:
        n_forced = 0
        while n_forced < min_cat2 or rng.random() < P_CAT2:
            cat2 = pick(rng, lex["cat2"])
            children.append(build_terminal("CAT2", cat2, "modifier", counter))
            n_forced += 1

    # PP-under-NP: ONE draw per nominal (multiplicity parity with Grammar P,
    # 2026-07-14); a complement nominal (role="pp") nests at the shallower
    # P_PP_UNDER_NP_COMP rate, mirroring P's P_CAT6_NP_COMP. The lexical
    # cat6_compatible gate is dropped (the feature stops doing work).
    p_pp = P_PP_UNDER_NP_COMP if role == "pp" else P_PP_UNDER_NP
    if force_pp or rng.random() < p_pp:
        children.append(
            build_type3_under_np(rng, lex, counter, parent_licensor_id=head.node_id)
        )

    # Type4 attachment is clause-level (build_type0), not per nominal host:
    # one draw per clause, subject-anchored — parity with Grammar P's
    # _maybe_attach_cat9 (length-calibration decision, 2026-07-14).

    return Node(
        label="Type1", head_cat="CAT1", lex=None, feats={},
        children=children, role=role, licensor_id=None, node_id=fresh_id(counter),
    )


def build_type2(rng: random.Random, lex: dict, counter: list,
                phenomenon: str | None = None,
                force_transitive: bool = False) -> Node:
    """Build a Type2 (VP): CAT3 (CAT5)* (Type3_vp)* (T1dep)?.

    CAT3AUX is inserted by generate.py after this returns, so feature assignment
    can decide whether inflections sit on CAT3 or on CAT3AUX.
    Transitivity is forced only when the reflexive licensing nucleus is this
    clause (``force_transitive``; the legacy ``phenomenon`` spelling is kept
    for Grammar H'). Pron and wh items are NOT forced transitive.
    """
    needs_transitive = force_transitive or phenomenon == "anaphoric_binding"
    transitivity = "transitive" if needs_transitive else None
    cat3 = pick(rng, cat3_items(lex, transitivity=transitivity))
    head = build_terminal("CAT3", cat3, "head", counter)
    children = [head]

    while rng.random() < P_CAT5:
        cat5 = pick(rng, lex["cat5"])
        children.append(build_terminal("CAT5", cat5, "adjunct", counter))

    # PP-under-VP: ONE draw per clause (multiplicity parity with Grammar P).
    if rng.random() < P_PP_UNDER_VP:
        children.append(
            build_type3_under_vp(rng, lex, counter, parent_licensor_id=head.node_id)
        )

    if cat3["transitivity"] == "transitive":
        # Object position uses a Dependent Type1 (same structure as Type1).
        obj = build_type1(rng, lex, counter, role="object")
        children.append(obj)

    return Node(
        label="Type2", head_cat="CAT3", lex=None, feats={},
        children=children, role="vp", licensor_id=None, node_id=fresh_id(counter),
    )


def build_type0(rng: random.Random, lex: dict, counter: list,
                phenomenon: str | None = None,
                force_transitive: bool = False) -> Node:
    """Build a complete Type0 sentence tree: Type1(subject) + Type2(VP).

    Embedding is drawn ONCE per clause at P_TYPE4 and attaches a single Type4
    to a host nominal — the object with probability Q_EMB_OBJECT when the
    clause is transitive, else the subject — positioned finally within the
    host (shared rule with Grammar P's `_maybe_attach_cat9`; length/placement
    parity, 2026-07-14; the old per-Type1-host draw compounded with the
    nominal count and inflated H's embedding rate and length tail). Nesting
    still arises recursively: the embedded Type0 performs its own draw.
    Forced embedding (`attach_forced_type4`) stays subject-anchored.
    """
    subject = build_type1(rng, lex, counter, role="subject")
    vp = build_type2(rng, lex, counter, phenomenon=phenomenon,
                     force_transitive=force_transitive)

    if rng.random() < P_TYPE4:
        host = subject
        obj = next((c for c in vp.children if c.role == "object"), None)
        if obj is not None and rng.random() < Q_EMB_OBJECT:
            host = obj
        head = next(c for c in host.children if c.role == "head")
        host.children.append(
            build_type4(rng, lex, counter, parent_licensor_id=head.node_id)
        )

    return Node(
        label="Type0", head_cat="CAT3", lex=None, feats={},
        children=[subject, vp], role="root",
        licensor_id=None, node_id=fresh_id(counter),
    )
