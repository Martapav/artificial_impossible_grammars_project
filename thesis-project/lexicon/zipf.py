"""Zipfian frequency distribution for lexical sampling.
"""

import dataclasses
import random
from typing import TypeVar

T = TypeVar("T")  # any frozen dataclass with a zipf_rank: int field


# Rank assignment

def assign_zipf_ranks(items: list) -> list:
    """Return a new list with zipf_rank set to 1-based position.

    Assumes items arrive in the desired frequency order (generation order:
    length asc, then lex asc).  Does not consume any RNG draws.
    """
    return [dataclasses.replace(item, zipf_rank=i + 1) for i, item in enumerate(items)]

# weight computation

def zipf_weights(items: list, s: float, q: float = 0.0) -> list[float]:
    """Return unnormalized Zipf-Mandelbrot weights for a ranked item list.

    Args:
        items: Items with zipf_rank > 0.
        s:     Frequency-rank exponent (canonical Zipf uses s=1.0).
        q:     Mandelbrot offset; 0.0 = plain Zipf.

    Raises:
        ValueError: If any item has zipf_rank == 0 (closed-class sentinel).
    """
    for item in items:
        if item.zipf_rank == 0:
            raise ValueError(
                f"Item {item!r} has zipf_rank=0 (closed-class sentinel). "
                "Only open-class items may be frequency-sampled."
            )
    return [1.0 / (item.zipf_rank + q) ** s for item in items]


# Weighted sampling

def zipf_sample(items: list, rng: random.Random, s: float, q: float = 0.0):
    """Draw one item using Zipf-Mandelbrot weights.

    Args:
        items: Non-empty list of ranked open-class items.
        rng:   Seeded Random instance (shared with all other generation draws).
        s:     Zipf exponent.
        q:     Mandelbrot offset.

    Returns:
        One item drawn proportionally to 1/(rank+q)^s.
    """
    weights = zipf_weights(items, s=s, q=q)
    return rng.choices(items, weights=weights, k=1)[0]


def zipf_sample_n(
    items: list,
    n: int,
    rng: random.Random,
    s: float,
    q: float = 0.0,
    replace: bool = True,
) -> list:
    """Draw n items with (replace=True) or without (replace=False) replacement.

    replace=True is the default and matches natural language token sampling.
    """
    weights = zipf_weights(items, s=s, q=q)
    if replace:
        return rng.choices(items, weights=weights, k=n)
    # sampling without replacement: draw one at a time, updating weights
    pool = list(items)
    pool_weights = list(weights)
    result = []
    for _ in range(n):
        chosen = rng.choices(pool, weights=pool_weights, k=1)[0]
        idx = pool.index(chosen)
        result.append(pool.pop(idx))
        pool_weights.pop(idx)
    return result

# Diagnostic utility (optional, not called at generation time)

def true_zipf_probabilities(n: int, s: float, q: float = 0.0) -> list[float]:
    """Return exact Zipf-Mandelbrot PMF for ranks 1..n.

    Uses scipy.special.zeta for the normalization constant H(N, s, q).
    Only needed for analysis/visualization, not for corpus generation.
    """
    from scipy.special import zeta  # noqa: PLC0415

    if q == 0.0:
        normalizer = zeta(s, 1) - zeta(s, n + 1)  # partial sum via Hurwitz zeta
    else:
        normalizer = sum(1.0 / (r + q) ** s for r in range(1, n + 1))
    return [1.0 / ((r + q) ** s * normalizer) for r in range(1, n + 1)]
