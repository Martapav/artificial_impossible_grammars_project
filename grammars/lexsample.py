"""Fast, stream-identical Zipf sampling of lexicon items (shared by H and P).

Lexicon v0.3.0 has ~76k CAT1 entries (experiment.md §2.6). The original
``pick`` rebuilt the weight list and ``random.choices`` re-accumulated it on
every draw, and ``TracedRandom.choices`` then scanned the whole population with
dict equality to record the draw's probability — O(N) three times per word.

Here each item list gets a cached sampler (cumulative weights, weights, exact
``math.fsum`` total, identity index). The draw is
``random.Random.choices(items, cum_weights=cw, k=1)``: the stdlib turns
``weights`` into ``cum_weights = list(accumulate(weights))`` and draws from those,
so passing the identical precomputed list consumes the same single variate and
returns the same item — the RNG stream and every generated character are
unchanged. The traced probability is ``w_i / fsum(w)``, exactly what
``TracedRandom.choices`` logged (lexicon items are distinct dicts, so the
"sum of equal values" it computed is the single item's weight).
"""
from __future__ import annotations

import math
import random
from collections import OrderedDict
from itertools import accumulate

_CACHE_SIZE = 64
_SAMPLERS: "OrderedDict[int, tuple]" = OrderedDict()


def _sampler(items: list) -> tuple:
    """(items_ref, cum_weights, weights, total, index_by_id), cached per list
    object. The cached tuple holds a reference to ``items``, so a hit is only
    accepted when it is the very same list (ids of dead lists can be reused)."""
    from lexicon.config import ZIPF_Q, ZIPF_S  # noqa: PLC0415

    key = id(items)
    hit = _SAMPLERS.get(key)
    if hit is not None and hit[0] is items and len(hit[2]) == len(items):
        _SAMPLERS.move_to_end(key)
        return hit
    weights = [1.0 / (item["zipf_rank"] + ZIPF_Q) ** ZIPF_S for item in items]
    entry = (items, list(accumulate(weights)), weights, math.fsum(weights),
             {id(x): i for i, x in enumerate(items)})
    _SAMPLERS[key] = entry
    if len(_SAMPLERS) > _CACHE_SIZE:
        _SAMPLERS.popitem(last=False)
    return entry


def zipf_pick(rng, items: list):
    """One item, Zipf-weighted for ranked (open-class) items, uniform otherwise.

    Same RNG consumption and same result as the original
    ``rng.choices(items, weights=[1/(rank+q)^s ...], k=1)[0]`` /
    ``rng.choice(items)``.
    """
    if not (items and items[0].get("zipf_rank", 0) > 0):
        return rng.choice(items)
    _, cum, weights, total, index = _sampler(items)
    log = getattr(rng, "_log", None)
    if log is None:                       # plain random.Random
        return rng.choices(items, cum_weights=cum, k=1)[0]
    out = random.Random.choices(rng, items, cum_weights=cum, k=1)[0]
    log("choices", weights[index[id(out)]] / total, "choices", len(items))
    return out
