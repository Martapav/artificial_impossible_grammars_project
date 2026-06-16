"""Flat token dataclass for Grammar P surface strings.

A sentence is an ordered list of Token objects; all well-formedness conditions
are a left-to-right scan over this list (no tree). Each Token carries one lexical
item's inflectional features in ``feats`` (rendered with ``#`` by linearize.py),
so the surface format and vocabulary match Grammar H.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Token:
    cat: str                    # "CAT1","CAT3","CAT3AUX","CAT4","CAT2","CAT5",
                                # "CAT6","CAT9","CAT8","CAT1PRON","GAP"
    lex: dict | None            # lexicon item dict; None only for GAP
    feats: dict = field(default_factory=dict)   # inflectional features
    role: str = ""              # "subject","object","verb","aux","complement",
                                # "wh","gap","" (positional bookkeeping only)
    clause_id: int = 0          # positional clause index (matrix == 0),
                                # delimited by CAT9; the binding domain
    attachment: str | None = None   # for CAT6: "np" or "vp" surface zone
    licensor_attachment: str | None = None  # for a PP-complement CAT1: the zone
                                #   ("np"/"vp") of the CAT6 that licensed it.
                                #   Generation-history record read by Grammar L'.


def is_gap(tok: Token) -> bool:
    return tok.cat == "GAP"
