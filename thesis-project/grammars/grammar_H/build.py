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
    P_PP_UNDER_NP,
    P_PP_UNDER_VP,
    P_TYPE4,
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


def build_type4(rng: random.Random, lex: dict, counter: list,
                parent_licensor_id: int | None = None) -> Node:
    """Type4 (relative clause): CAT9 + embedded Type0."""
    cat9 = pick(rng, cat9_items(lex))
    head = build_terminal("CAT9", cat9, "head", counter)
    inner = build_type0(rng, lex, counter, phenomenon="neutral")
    return Node(
        label="Type4", head_cat="CAT9", lex=None, feats={},
        children=[head, inner], role="rel_clause",
        licensor_id=parent_licensor_id, node_id=fresh_id(counter),
    )

def build_type1(rng: random.Random, lex: dict, counter: list,
                role: str = "subject",
                min_cat2: int = 0,
                force_pp: bool = False) -> Node:
    """Build a Type1 node (role="subject" → matrix subject; "object"/"pp" → T1dep).

    [+cnt] schema: CAT1 CAT4 (CAT2)* (Type3)* (Type4)*
    [-cnt] schema: CAT1 (Type3)* (Type4)*  — proper-like CAT1 takes no CAT4/CAT2.

    min_cat2 and force_pp are used by generalization items to force extra complexity.
    """
    pool = cat1_items(lex)
    if force_pp:
        pool = [x for x in pool if x["cat6_compatible"]]

    cat1 = pick(rng, pool)
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

    # PP-under-NP: licensed only on cat6_compatible CAT1; force_pp guarantees ≥1.
    if cat1["cat6_compatible"]:
        attached = False
        while (force_pp and not attached) or rng.random() < P_PP_UNDER_NP:
            children.append(
                build_type3_under_np(rng, lex, counter, parent_licensor_id=head.node_id)
            )
            attached = True

    # Type4 (relative clauses): zero or more, positioned finally.
    while rng.random() < P_TYPE4:
        children.append(
            build_type4(rng, lex, counter, parent_licensor_id=head.node_id)
        )

    return Node(
        label="Type1", head_cat="CAT1", lex=None, feats={},
        children=children, role=role, licensor_id=None, node_id=fresh_id(counter),
    )


def build_type2(rng: random.Random, lex: dict, counter: list,
                phenomenon: str | None = None) -> Node:
    """Build a Type2 (VP): CAT3 (CAT5)* (Type3_vp)* (T1dep)?.

    CAT3AUX is inserted by generate.py after this returns, so feature assignment
    can decide whether inflections sit on CAT3 or on CAT3AUX.
    Transitivity is forced for anaphoric_binding and wh_movement.
    """
    needs_transitive = phenomenon in ("anaphoric_binding", "wh_movement")
    transitivity = "transitive" if needs_transitive else None
    cat3 = pick(rng, cat3_items(lex, transitivity=transitivity))
    head = build_terminal("CAT3", cat3, "head", counter)
    children = [head]

    while rng.random() < P_CAT5:
        cat5 = pick(rng, lex["cat5"])
        children.append(build_terminal("CAT5", cat5, "adjunct", counter))

    while rng.random() < P_PP_UNDER_VP:
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
                phenomenon: str | None = None) -> Node:
    """Build a complete Type0 sentence tree: Type1(subject) + Type2(VP)."""
    subject = build_type1(rng, lex, counter, role="subject")
    vp = build_type2(rng, lex, counter, phenomenon=phenomenon)

    return Node(
        label="Type0", head_cat="CAT3", lex=None, feats={},
        children=[subject, vp], role="root",
        licensor_id=None, node_id=fresh_id(counter),
    )
