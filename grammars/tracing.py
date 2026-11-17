"""Draw-probability tracing for the true-entropy baseline (entropy plan §2.2, §3).

The grammar-derived baseline needs ``ℓ(x) = Σ_steps −log π(step | state)`` — the
negative log-probability of the derivation the generator actually took. That sum
only exists if EVERY stochastic draw exposes its probability at the moment it is
made. Before this module the generation path made 61 draws through a threaded
``random.Random`` and logged none of them.

Design
------
Every live draw in the generation path goes through one threaded ``rng`` object
(``self._rng = random.Random(seed)``, passed down into build/features/transforms);
``np.random`` is only seeded, and its one consumer — ``BaseGrammarGenerator.
sample_length`` — is dead code. So subclassing ``random.Random`` captures the
whole derivation without touching call sites, except where the *decision*
probability is invisible to the callee (``rng.random() < P``): those are
converted to :meth:`TracedRandom.bernoulli` / :meth:`TracedRandom.categorical`.

Stream preservation (load-bearing)
----------------------------------
Every method here delegates to ``super()`` and consumes exactly the same number
of underlying variates, in the same order, as the uninstrumented call it
replaces. Tracing therefore CANNOT change what is generated: the corpora stay
byte-identical and only gain a sidecar field. ``tests/test_tracing.py`` asserts
this against an untraced ``random.Random`` on the same seed.

Marginalization
---------------
``choice``/``choices`` log the probability of the drawn *value*, summing over
duplicate entries in the population, not the probability of an index. For
``ℓ(x)`` that is the correct factor: two population slots holding the same form
are the same surface outcome.

What this module does NOT decide
--------------------------------
``ℓ(x)`` is the RAW derivation log-probability. The corpus the model trains on is
a REJECTION-SAMPLED reweighting of that distribution (construction quota, length
window, ``*_skipped`` discards, and the in-generator duplicate screen), so
``p_corpus ≠ p_raw``. The correction is a per-(grammar, construction) scalar; see
``AcceptanceCounter`` and the note in ``ENTROPY_BASELINE.md``. This module emits
the pieces; it does not silently pick one normalization.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

# A draw whose probability underflows to 0 would make ℓ(x) infinite; that always
# indicates a bug (an impossible outcome was drawn), so we fail loudly.
_MIN_P = 1e-300


@dataclass
class Draw:
    """One logged stochastic decision."""

    label: str
    logp: float          # natural log of the probability of the drawn outcome
    kind: str            # "bernoulli" | "categorical" | "choice" | "choices"
    n_outcomes: Optional[int] = None


@dataclass
class Trace:
    """The accumulated derivation record for one generated item."""

    draws: List[Draw] = field(default_factory=list)

    @property
    def logp(self) -> float:
        """log p(derivation) in nats (≤ 0)."""
        return math.fsum(d.logp for d in self.draws)

    @property
    def neg_logp(self) -> float:
        """ℓ(x) = −log p(derivation) in nats (≥ 0)."""
        return -self.logp

    @property
    def n_draws(self) -> int:
        return len(self.draws)

    def by_label(self) -> Dict[str, float]:
        """−log p aggregated per label, for attributing where the mass sits."""
        out: Dict[str, float] = {}
        for d in self.draws:
            out[d.label] = out.get(d.label, 0.0) - d.logp
        return out


class TracedRandom(random.Random):
    """``random.Random`` that records −log p for every draw it serves.

    Stream-identical to its base class: each method consumes the same variates
    in the same order, so swapping this in never changes generated output.
    """

    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self.trace = Trace()
        self._recording = True

    # --- trace control ----------------------------------------------------

    def reset_trace(self) -> None:
        """Start a fresh derivation record.

        Called at the top of each generation attempt so that draws burnt on a
        REJECTED attempt (the duplicate screen's resample loop) do not leak into
        the accepted item's ℓ(x). Only the surviving trajectory is the
        derivation of the emitted string.
        """
        self.trace = Trace()

    def take_trace(self) -> Trace:
        """Return the current trace and start a new one."""
        t = self.trace
        self.trace = Trace()
        return t

    def mark(self) -> int:
        """Checkpoint the trace so a failed sub-attempt can be rewound."""
        return len(self.trace.draws)

    def rewind(self, mark: int) -> None:
        """Discard draws logged since ``mark``.

        The duplicate screen retries the BUILD but keeps the pre-selection draws
        (phenomenon, binding subtype, aux site) that precede it. Rewinding to a
        mark taken after pre-selection therefore keeps exactly the draws that
        belong to the surviving derivation.
        """
        del self.trace.draws[mark:]

    def _log(self, label: str, p: float, kind: str,
             n_outcomes: Optional[int] = None) -> None:
        if not self._recording:
            return
        if not (p > _MIN_P):
            raise ValueError(
                f"traced draw {label!r} reported probability {p!r}; a drawn "
                "outcome must have positive probability (ℓ(x) would diverge)"
            )
        self.trace.draws.append(Draw(label, math.log(p), kind, n_outcomes))

    # --- logged primitives -------------------------------------------------

    def bernoulli(self, p: float, label: str) -> bool:
        """``rng.random() < p`` with the decision probability recorded.

        Consumes exactly one variate, like the ``rng.random() < p`` it replaces.
        """
        outcome = super().random() < p
        self._log(label, p if outcome else 1.0 - p, "bernoulli", 2)
        return outcome

    def categorical(self, probs: Dict[str, float], label: str) -> str:
        """Draw a labelled outcome from an explicit categorical.

        Consumes exactly one variate via a cumulative scan, matching the
        ``r = rng.random(); acc += ...`` idiom it replaces. Probabilities are
        used as given (they are the generator's own constants); the final
        outcome absorbs any rounding shortfall, as the original scan did.
        """
        r = super().random()
        acc = 0.0
        names = list(probs)
        for name in names:
            acc += probs[name]
            if r < acc:
                self._log(label, probs[name], "categorical", len(names))
                return name
        last = names[-1]
        self._log(label, probs[last], "categorical", len(names))
        return last

    # --- overridden stdlib draws (call sites unchanged) --------------------

    def choice(self, seq: Sequence, label: str = "choice"):
        out = super().choice(seq)
        n = len(seq)
        # Probability of the VALUE, summing duplicate slots.
        try:
            hits = sum(1 for x in seq if x == out)
        except Exception:                      # unorderable/odd elements
            hits = 1
        self._log(label, hits / n, "choice", n)
        return out

    def choices(self, population, weights=None, *, cum_weights=None, k=1,
                label: str = "choices"):
        out = super().choices(population, weights=weights,
                              cum_weights=cum_weights, k=k)
        if weights is None and cum_weights is None:
            w = [1.0] * len(population)
        elif weights is not None:
            w = list(weights)
        else:                                   # cum_weights -> differences
            cw = list(cum_weights)
            w = [cw[0]] + [cw[i] - cw[i - 1] for i in range(1, len(cw))]
        total = math.fsum(w)
        for value in out:
            try:
                hit = math.fsum(wi for x, wi in zip(population, w) if x == value)
            except Exception:
                hit = total / len(population)
            self._log(label, hit / total, "choices", len(population))
        return out


def bernoulli(rng, p: float, label: str) -> bool:
    """``rng.random() < p``, traced when ``rng`` supports it.

    The generation path is shared with callers that pass a plain
    ``random.Random`` — the separability audit's ``_draw_base``, the test
    harnesses, the probe builders. Those must keep working untraced, so this
    dispatches on capability rather than requiring :class:`TracedRandom`
    everywhere. Both branches consume exactly one variate, so the RNG stream is
    identical either way.
    """
    fn = getattr(rng, "bernoulli", None)
    if fn is not None:
        return fn(p, label)
    return rng.random() < p


def categorical(rng, probs: Dict[str, float], label: str) -> str:
    """Labelled categorical draw, traced when ``rng`` supports it.

    Untraced fallback reproduces the cumulative scan the call sites used, so the
    stream and the outcome are unchanged.
    """
    fn = getattr(rng, "categorical", None)
    if fn is not None:
        return fn(probs, label)
    r = rng.random()
    acc = 0.0
    for name, p in probs.items():
        acc += p
        if r < acc:
            return name
    return list(probs)[-1]


def trace_mark(rng) -> Optional[int]:
    """Checkpoint ``rng``'s trace, or ``None`` when untraced."""
    fn = getattr(rng, "mark", None)
    return fn() if fn is not None else None


def trace_rewind(rng, mark: Optional[int]) -> None:
    """Rewind ``rng``'s trace to ``mark``; no-op when untraced."""
    if mark is None:
        return
    fn = getattr(rng, "rewind", None)
    if fn is not None:
        fn(mark)


def reset_trace(rng) -> None:
    """Begin a fresh derivation record; no-op when untraced."""
    fn = getattr(rng, "reset_trace", None)
    if fn is not None:
        fn()


def derivation_record(rng) -> Optional[Dict[str, object]]:
    """The serializable derivation record for the item just generated.

    ``None`` when ``rng`` is untraced, so the sidecar simply omits the field and
    every existing caller keeps working.
    """
    trace = getattr(rng, "trace", None)
    if trace is None:
        return None
    return {
        "neg_logp": trace.neg_logp,      # ℓ(x), nats, RAW derivation cost
        "n_draws": trace.n_draws,
        "by_label": {k: v for k, v in sorted(trace.by_label().items())},
    }


# ---------------------------------------------------------------------------
# Content-derived item identity (A3, decided 2026-08-06)
# ---------------------------------------------------------------------------
#
# ``item_uid`` must recover the SAME key for the SAME item across regenerations.
# Two rejected alternatives, both recorded because each looks adequate until it
# is not:
#
#   * the POSITIONAL form (``{grammar}-{split}-{index}``, i.e. what ``id`` is) is
#     the very key the field exists to eliminate. It would turn A4's loud
#     "no item_uid" failure into a silent mis-join the first time item order
#     changes.
#   * the BUILD-COUNTER form (what the probe builder uses for ``pair_id``) is
#     stable against write-side reordering but NOT against a change in build
#     order — and build order does move: C7.1 shifted accept order outright
#     (H needed 13,835 attempts for 6,000 items before, 13,210 after).
#
# Only content-derivation survives both. The digest covers (grammar, split,
# surface string, derivation signature): grammar and split are inside the digest
# so cross-split collisions cannot arise, and the derivation signature is
# required because the surface string alone is insufficient — derivational
# unambiguity is FLAGGED on the real corpus, not guaranteed
# (SURPRISAL_REFERENCE_POINTS §6.6), so two genuinely distinct items can share a
# string.
#
# A within-split digest collision therefore means the two items agree on all
# four inputs: a genuine duplicate, not a hash accident (128 bits over ~5·10^5
# items). Duplicates are flagged and counted, never silently merged or dropped.

_UID_BITS = 16          # blake2b digest_size in bytes -> 32 hex chars


def derivation_signature(derivation: Optional[Dict[str, object]]) -> tuple:
    """A comparable fingerprint of one derivation record.

    Rounded to 9 decimals so the value survives a JSON round trip unchanged;
    ``by_label`` is included because two derivations can reach the same total
    cost by spending it at different sites.
    """
    if derivation is None:
        return ()
    by_label = derivation.get("by_label") or {}
    return (
        round(float(derivation["neg_logp"]), 9),
        int(derivation["n_draws"]),
        tuple(sorted((str(k), round(float(v), 9)) for k, v in by_label.items())),
    )


def item_uid(grammar: str, split: str, sentence: str,
             derivation: Optional[Dict[str, object]]) -> str:
    """The content-derived, position-independent item key (32 hex chars).

    Deterministic across processes and machines: ``blake2b`` is seeded by the
    input only (unlike ``hash()``, which is salted per interpreter), and the
    payload is canonical JSON with fixed separators.
    """
    import hashlib
    import json

    payload = json.dumps(
        [grammar, split, sentence, list(derivation_signature(derivation))],
        separators=(",", ":"), ensure_ascii=False,
    )
    return hashlib.blake2b(payload.encode("utf-8"),
                           digest_size=_UID_BITS).hexdigest()


class AcceptanceCounter:
    """Per-(grammar, construction) accept/attempt tallies from generation.

    ``ℓ(x)`` is the raw derivation cost. The corpus is rejection-sampled, so the
    distribution the model actually trains on is

        p_corpus(x) = (quota_c / n) · p_raw(x) / P_raw(accepted, construction=c)

    and therefore

        ℓ_corpus(x) = ℓ(x) + log P_raw(accepted, c) − log(quota_c / n).

    Both correction terms are constants per (grammar, construction): the first is
    estimated here by counting attempts during the very generation run that emits
    the corpus, the second is exact from the quota. Recording them keeps BOTH
    normalizations available downstream instead of baking one in.
    """

    def __init__(self) -> None:
        self.attempts: int = 0
        self.accepted: Dict[str, int] = {}
        self.drawn: Dict[str, int] = {}
        self.eligible: Dict[str, int] = {}

    def note_attempt(self, construction: Optional[str],
                     eligible: bool = False) -> None:
        """Record one draw.

        ``eligible`` means the draw is a VALID label that also passes the length
        window — i.e. everything the corpus requires EXCEPT the quota cap. This
        distinction is the whole point of the counter: ``accepted[c]`` is capped
        at ``quota_c`` by construction, so ``accepted/attempts`` measures the
        quota, not the grammar. Only ``eligible/attempts`` estimates
        ``P_raw(c, len ok)``, which is the factor the correction needs.
        """
        self.attempts += 1
        if construction:
            self.drawn[construction] = self.drawn.get(construction, 0) + 1
            if eligible:
                self.eligible[construction] = self.eligible.get(construction, 0) + 1

    def note_accept(self, construction: str) -> None:
        self.accepted[construction] = self.accepted.get(construction, 0) + 1

    def summary(self) -> Dict[str, object]:
        n = sum(self.accepted.values())
        return {
            "attempts": self.attempts,
            "accepted_total": n,
            "accepted_by_construction": dict(self.accepted),
            "drawn_by_construction": dict(self.drawn),
            "eligible_by_construction": dict(self.eligible),
            # P_raw(c, len ok): the natural mass of eligible draws of each label.
            # Estimated over a sequentially stopped sample (generation halts when
            # every quota is filled), so it carries an O(1/attempts) bias —
            # negligible at corpus sizes, but it is an estimate, not an identity.
            "p_eligible_by_construction": {
                c: self.eligible[c] / self.attempts for c in self.eligible
            } if self.attempts else {},
            # quota_c / n: the exact realized corpus label proportion.
            "corpus_proportion": {
                c: self.accepted[c] / n for c in self.accepted
            } if n else {},
        }
