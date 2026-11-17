"""Scoring a derivation under π_corpus, including one the corpus did not produce (A2.1).

Why this module exists
----------------------
``tracing.TracedRandom`` gives ``ℓ(x) = −log π_corpus(d)`` for a derivation the
corpus generator *made*: the probability is logged at the moment of the draw.
The probe items are built by a different program (``corpusgen/structural_probes``
via ``grammars/violations``), which forces construction sites, filters candidate
lists to the grammatical ones, and rejects on uniqueness. Its own sampling
log-probability ``−log π_probe(d)`` is therefore NOT the quantity we want: the
model was trained on the corpus distribution, so ``neg_logq − (−log π_probe)``
would subtract two quantities defined over different measures.

What we want is the *same derivation* ``d`` scored under the *corpus* measure:
forced sites get the probability π_corpus would have assigned to that choice.
That requires evaluating the corpus generator's choice model at outcomes it did
not itself draw — hence "factoring" it.

The factoring
-------------
The choice model is already written down: every decision site hands its own
distribution to the traced primitive (the ``p`` of a ``bernoulli``, the ``probs``
of a ``categorical``, the population and weights of a ``choice``/``choices``).
The model is not a table to be extracted — it is a function of the derivation
state, and the code that computes it is the generator body itself.

So the factoring is an inversion of control rather than a rewrite:
:class:`ScoringRandom` presents the same primitive surface as ``TracedRandom``
but, instead of drawing, it *consumes a prescribed outcome* and logs −log π of
that outcome under whatever distribution the call site offered. Driving the
corpus generator's own code path with a :class:`ScoringRandom` therefore
evaluates π_corpus at a foreign derivation, site by site, with no duplicated
probability logic that could drift out of sync with the generator.

    recorder = DerivationRecorder(seed)     # corpus side: draw and record
    ...                                     # (or: trace the probe builder)
    d = recorder.derivation()

    result = score_derivation(d, lambda rng: _one_item(rng, lex))
    result.neg_logp                         # ℓ(x) under π_corpus

Representation of an outcome
----------------------------
``Choice.value`` is what makes a derivation replayable, so it is stored in the
form the *replaying* call site can consume, not in the form the original call
site returned:

  bernoulli    bool
  categorical  the outcome name (str)
  choice       the population INDEX (int)
  choices      list of population indices

Indices, not values, for the population draws. The population at a ``choice``
site is frequently a list of freshly built ``Node`` objects or ``(node, parent,
idx)`` tuples; on replay the tree is reconstructed, so those objects are new
instances and value equality cannot recover the slot. An index does, because a
faithful replay reconstructs the *same population in the same order* — and if it
does not, the mismatch is exactly the structural divergence we want to detect.
The logged probability still marginalizes over duplicate slots, as
``TracedRandom`` does: two slots holding the same form are one surface outcome.

Failure modes are distinguished, never absorbed
-----------------------------------------------
Two things can go wrong, and A2.3 needs to tell them apart:

  :class:`StructuralDivergence`  the replay asked for a decision the script does
      not have, or has under a different label/kind, or the script has leftover
      entries. The derivation is not a derivation *of this generator* — the
      code path itself diverged.

  :class:`OutOfSupport`  the site was reached and the label matched, but π_corpus
      assigns the scripted outcome probability zero. The derivation is
      well-formed and simply lies outside the corpus generator's support.

Neither is ever silently repaired and neither returns infinity. Callers turn
them into an explicit null with a reason field (A2.3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from grammars.tracing import Draw, Trace, TracedRandom, _MIN_P

# Outcome kinds, matching TracedRandom's primitives.
KIND_BERNOULLI = "bernoulli"
KIND_CATEGORICAL = "categorical"
KIND_CHOICE = "choice"
KIND_CHOICES = "choices"


# ─────────────────────────────────────────────────────────────────────────────
# Errors
# ─────────────────────────────────────────────────────────────────────────────

class DerivationScoringError(Exception):
    """Base class: this derivation has no ℓ(x) under π_corpus."""


class StructuralDivergence(DerivationScoringError):
    """The replay diverged from the script: wrong label/kind, or wrong length.

    Means the derivation is not a derivation of the corpus generator at all —
    the two programs took different code paths. Distinct from a well-formed
    derivation that merely has zero probability.
    """


class OutOfSupport(DerivationScoringError):
    """π_corpus assigns the scripted outcome probability zero.

    The site was reached and matched; the corpus generator simply would never
    make this choice. ``ℓ(x)`` is ``+∞``, which we report as undefined rather
    than as a number, because an infinity would silently dominate any
    downstream mean.
    """


# ─────────────────────────────────────────────────────────────────────────────
# Derivation representation
# ─────────────────────────────────────────────────────────────────────────────

def outcome_key(value: Any) -> Any:
    """Replay-stable identity of a drawn outcome, or ``None`` when it has none.

    An index alone is not enough to replay a population draw across a
    population CHANGE, and those changes are exactly what scoring a foreign
    derivation runs into. ``build_type2`` under ``force_transitive`` passes
    ``cat3_items(lex, transitivity="transitive")`` — a strictly smaller pool —
    to the very same ``pick`` call, at the same label and the same position. The
    recorded index would then silently select a DIFFERENT verb and return a
    perfectly finite, perfectly wrong ℓ(x). Matching on the key instead makes
    that case what it actually is: either the outcome is in the corpus's pool
    (score it there) or it is not (out of support).

    Keys are tagged by kind so the caller can tell a lexical miss (the corpus
    would never draw this form here — out of support) from a structural miss
    (the populations are built from different trees — divergence).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        form = value.get("form")
        return ("form", form) if form is not None else None
    if isinstance(value, (tuple, list)):
        keys = [outcome_key(v) for v in value]
        return tuple(keys) if all(k is not None for k in keys) else None
    node_id = getattr(value, "node_id", None)
    if node_id is not None:
        return ("node_id", node_id)
    return None


@dataclass(frozen=True)
class Choice:
    """One decision, in replayable form.

    ``label`` and ``kind`` identify the site; ``value`` is the outcome encoded
    as described in the module docstring. ``key`` is :func:`outcome_key` of the
    drawn value — the replay matches on it when it exists and falls back to the
    index when it does not. ``note`` is a human-readable rendering kept for
    debugging only; it is never used for matching, because a reconstructed
    object need not render identically.
    """

    label: str
    kind: str
    value: Any
    key: Any = None
    note: Optional[str] = None


@dataclass(frozen=True)
class Derivation:
    """An ordered script of decisions: everything stochastic about one item."""

    choices: Tuple[Choice, ...] = ()

    def __len__(self) -> int:
        return len(self.choices)

    @property
    def labels(self) -> Tuple[str, ...]:
        return tuple(c.label for c in self.choices)

    def to_json(self) -> List[Dict[str, Any]]:
        return [{"label": c.label, "kind": c.kind, "value": c.value,
                 "key": _key_to_json(c.key)}
                for c in self.choices]

    @classmethod
    def from_json(cls, rows: Sequence[Dict[str, Any]]) -> "Derivation":
        return cls(tuple(
            Choice(label=r["label"], kind=r["kind"], value=r["value"],
                   key=_key_from_json(r.get("key")))
            for r in rows))


@dataclass
class ScoreResult:
    """ℓ(x) under π_corpus, plus what it was computed from."""

    neg_logp: float
    n_choices: int
    by_label: Dict[str, float] = field(default_factory=dict)

    @property
    def logp(self) -> float:
        return -self.neg_logp


# ─────────────────────────────────────────────────────────────────────────────
# Recording side
# ─────────────────────────────────────────────────────────────────────────────

class DerivationRecorder(TracedRandom):
    """``TracedRandom`` that also records each outcome in replayable form.

    Consumes no extra variates: every override delegates to its parent and only
    annotates the ``Draw`` the parent already appended, so a recorder is exactly
    as stream-identical as ``TracedRandom`` is (gate A2.0(a)).

    The value list is needed because −log p alone does not identify a
    derivation: two lexemes of equal Zipf weight log the same probability.
    """

    def __init__(self, seed=None) -> None:
        super().__init__(seed)
        self._choices: List[Choice] = []

    # --- outcome capture --------------------------------------------------

    @staticmethod
    def _index_of(population: Sequence, value: Any) -> int:
        """Slot the drawn value came from, located by IDENTITY.

        ``stdlib`` returns an element of the population, so the element is in
        there by identity — no ``==`` needed, which matters because ``Node`` and
        the tuples built around it have no meaningful equality. When several
        slots hold the identical object (interned strings) the first is
        returned; that is harmless, since equal slots are the same outcome and
        the logged probability already marginalizes over them.
        """
        for i, x in enumerate(population):
            if x is value:
                return i
        # Fall back to equality: a population built by comprehension can hold a
        # value equal to but not identical with what a stdlib method returned.
        for i, x in enumerate(population):
            try:
                if x == value:
                    return i
            except Exception:
                continue
        raise StructuralDivergence(
            "drawn value is not present in its own population; the derivation "
            "cannot be encoded as an index")

    def bernoulli(self, p: float, label: str) -> bool:
        out = super().bernoulli(p, label)
        self._choices.append(Choice(label, KIND_BERNOULLI, bool(out)))
        return out

    def categorical(self, probs: Dict[str, float], label: str) -> str:
        out = super().categorical(probs, label)
        self._choices.append(Choice(label, KIND_CATEGORICAL, out))
        return out

    def choice(self, seq: Sequence, label: str = "choice"):
        out = super().choice(seq, label)
        self._choices.append(
            Choice(label, KIND_CHOICE, self._index_of(seq, out),
                   outcome_key(out), _note(out)))
        return out

    def choices(self, population, weights=None, *, cum_weights=None, k=1,
                label: str = "choices"):
        out = super().choices(population, weights=weights,
                              cum_weights=cum_weights, k=k, label=label)
        idx = [self._index_of(population, v) for v in out]
        keys = [outcome_key(v) for v in out]
        self._choices.append(
            Choice(label, KIND_CHOICES, idx,
                   keys if all(k is not None for k in keys) else None,
                   _note(out[0]) if out else None))
        return out

    # --- trace control, kept in lockstep ----------------------------------

    def reset_trace(self) -> None:
        super().reset_trace()
        self._choices = []

    def rewind(self, mark: int) -> None:
        """Truncate the choice script in lockstep with the trace.

        LOAD-BEARING. The duplicate screen rewinds the trace on every failed
        build retry (``grammar_H/generate.py``, ``grammar_P/generate.py``).
        Without the matching truncation here the script keeps the discarded
        attempt's entries while ``trace.draws`` drops them, and every index
        after the rewind point refers to the wrong site. That desync is what
        made the first A2.0(b) pass report phantom collisions.
        """
        super().rewind(mark)
        del self._choices[mark:]

    def derivation(self) -> Derivation:
        """The replayable script for the item just generated."""
        n_draws, n_choices = len(self.trace.draws), len(self._choices)
        # Every logged primitive appends exactly one entry to each list, so a
        # mismatch means a desync and any script built from it would be wrong.
        if n_draws != n_choices:
            raise AssertionError(
                f"trace/choice desync: {n_draws} draws vs {n_choices} choices")
        return Derivation(tuple(self._choices))


def _key_to_json(key: Any) -> Any:
    """JSON has no tuples, and a key that survives a round trip as a list would
    stop comparing equal to a freshly computed one."""
    if isinstance(key, tuple):
        return {"_tuple": [_key_to_json(k) for k in key]}
    return key


def _key_from_json(raw: Any) -> Any:
    if isinstance(raw, dict) and "_tuple" in raw:
        return tuple(_key_from_json(k) for k in raw["_tuple"])
    return raw


def _note(value: Any) -> Optional[str]:
    """Short debug rendering of a drawn value; never used for matching."""
    try:
        if isinstance(value, dict) and "form" in value:
            return str(value["form"])
        text = repr(value)
    except Exception:
        return None
    return text[:80]


# ─────────────────────────────────────────────────────────────────────────────
# Scoring side
# ─────────────────────────────────────────────────────────────────────────────

class ScoringRandom:
    """RNG-shaped consumer that scores a prescribed derivation under π_corpus.

    Drop this in where the corpus generator expects its ``rng`` and every draw
    is answered from the script instead of from a random stream, while the
    distribution the call site offers is used to log −log π. The result is
    π_corpus evaluated at a derivation this generator did not produce.

    Deliberately NOT a ``random.Random`` subclass. Any un-scripted primitive
    (``random()``, ``randint``, ``shuffle``, …) must fail loudly rather than
    quietly inject entropy the script cannot account for, which would make the
    returned ℓ(x) a number with no meaning. ``grammars/`` currently contains no
    such call — this keeps it that way.
    """

    def __init__(self, derivation: Derivation) -> None:
        self._script = derivation.choices
        self._cursor = 0
        self.trace = Trace()
        # One consumed entry appends exactly one Draw, so a single counter
        # serves as both script cursor and trace length. mark/rewind rely on it.
        self._recording = True

    # --- script plumbing ---------------------------------------------------

    def _next(self, label: str, kind: str) -> Choice:
        if self._cursor >= len(self._script):
            raise StructuralDivergence(
                f"script exhausted at draw {self._cursor}: the generator asked "
                f"for {kind} {label!r} but the derivation has only "
                f"{len(self._script)} choices")
        c = self._script[self._cursor]
        if c.label != label or c.kind != kind:
            raise StructuralDivergence(
                f"divergence at draw {self._cursor}: generator asked for "
                f"{kind} {label!r}, script has {c.kind} {c.label!r}")
        self._cursor += 1
        return c

    def _log(self, label: str, p: float, kind: str,
             n_outcomes: Optional[int] = None) -> None:
        if not self._recording:
            return
        if not (p > _MIN_P):
            raise OutOfSupport(
                f"π_corpus assigns probability {p!r} to the scripted outcome at "
                f"{kind} {label!r}; ℓ(x) is undefined here, not infinite")
        self.trace.draws.append(Draw(label, math.log(p), kind, n_outcomes))

    # --- scored primitives -------------------------------------------------

    def bernoulli(self, p: float, label: str) -> bool:
        outcome = bool(self._next(label, KIND_BERNOULLI).value)
        self._log(label, p if outcome else 1.0 - p, KIND_BERNOULLI, 2)
        return outcome

    def categorical(self, probs: Dict[str, float], label: str) -> str:
        name = self._next(label, KIND_CATEGORICAL).value
        if name not in probs:
            raise OutOfSupport(
                f"scripted outcome {name!r} at categorical {label!r} is not an "
                f"outcome π_corpus offers here ({sorted(probs)})")
        self._log(label, probs[name], KIND_CATEGORICAL, len(probs))
        return name

    def choice(self, seq: Sequence, label: str = "choice"):
        c = self._next(label, KIND_CHOICE)
        value = self._resolve(seq, c.value, c.key, label, KIND_CHOICE)
        self._log(label, _slot_mass(seq, value) / len(seq), KIND_CHOICE, len(seq))
        return value

    def choices(self, population, weights=None, *, cum_weights=None, k=1,
                label: str = "choices"):
        c = self._next(label, KIND_CHOICES)
        idx = c.value
        if len(idx) != k:
            raise StructuralDivergence(
                f"scripted {label!r} has {len(idx)} outcomes but the generator "
                f"asked for k={k}")
        keys = c.key if c.key is not None else [None] * len(idx)
        w = _resolve_weights(population, weights, cum_weights)
        total = math.fsum(w)
        out = []
        for i, key in zip(idx, keys):
            value = self._resolve(population, i, key, label, KIND_CHOICES)
            self._log(label, _weighted_mass(population, w, value) / total,
                      KIND_CHOICES, len(population))
            out.append(value)
        return out

    def _resolve(self, population: Sequence, idx: Any, key: Any,
                 label: str, kind: str):
        """The slot the scripted outcome names, matched on key where possible.

        The index is the fast path and is authoritative when the population is
        unchanged. When the corpus offers a DIFFERENT population at this site —
        the forced-transitive verb pool being the case that matters — the index
        is meaningless and only the key can say whether the corpus could have
        drawn this outcome at all.
        """
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise StructuralDivergence(
                f"scripted {kind} {label!r} carries {idx!r}, expected an index")
        if key is None:
            # Nothing to match on: the index is all we have, so a population of
            # a different size is an unresolvable divergence rather than a
            # zero-probability outcome.
            if not (0 <= idx < len(population)):
                raise StructuralDivergence(
                    f"scripted index {idx} at {kind} {label!r} is outside the "
                    f"reconstructed population of size {len(population)}")
            return population[idx]

        # Key present: trust it over the index.
        if 0 <= idx < len(population) and outcome_key(population[idx]) == key:
            return population[idx]
        for value in population:
            if outcome_key(value) == key:
                return value
        raise _missing_outcome(key, label, kind, len(population))

    @staticmethod
    def _at(population: Sequence, idx: int, label: str, kind: str):
        """Deprecated shim kept for direct callers; see :meth:`_resolve`."""
        if not isinstance(idx, int) or isinstance(idx, bool):
            raise StructuralDivergence(
                f"scripted {kind} {label!r} carries {idx!r}, expected an index")
        if not (0 <= idx < len(population)):
            raise StructuralDivergence(
                f"scripted index {idx} at {kind} {label!r} is outside the "
                f"reconstructed population of size {len(population)}")
        return population[idx]

    # --- trace control, mirroring TracedRandom ----------------------------

    def reset_trace(self) -> None:
        self.trace = Trace()
        self._cursor = 0

    def take_trace(self) -> Trace:
        t = self.trace
        self.trace = Trace()
        return t

    def mark(self) -> int:
        return len(self.trace.draws)

    def rewind(self, mark: int) -> None:
        """Rewind trace AND script cursor together.

        A retry loop that rewinds must also re-read the script from the same
        point, or the second pass would be scored against the first pass's
        remaining entries.
        """
        del self.trace.draws[mark:]
        self._cursor = mark

    # --- unscriptable primitives ------------------------------------------

    def _forbidden(self, name: str):
        raise StructuralDivergence(
            f"{name}() is not a traced primitive, so no scripted outcome exists "
            f"for it; a derivation containing it cannot be scored under "
            f"π_corpus. Convert the call site to bernoulli/categorical/choice.")

    def random(self, *a, **k):        return self._forbidden("random")
    def randint(self, *a, **k):       return self._forbidden("randint")
    def randrange(self, *a, **k):     return self._forbidden("randrange")
    def sample(self, *a, **k):        return self._forbidden("sample")
    def shuffle(self, *a, **k):       return self._forbidden("shuffle")
    def uniform(self, *a, **k):       return self._forbidden("uniform")
    def gauss(self, *a, **k):         return self._forbidden("gauss")
    def getrandbits(self, *a, **k):   return self._forbidden("getrandbits")

    # --- result ------------------------------------------------------------

    @property
    def exhausted(self) -> bool:
        return self._cursor == len(self._script)

    @property
    def remaining(self) -> int:
        return len(self._script) - self._cursor


def _missing_outcome(key: Any, label: str, kind: str,
                     n: int) -> DerivationScoringError:
    """Classify a key that no slot in the offered population carries.

    A LEXICAL miss is a statement about the corpus distribution: the pool at
    this site does not contain this form, so π_corpus assigns it zero — out of
    support. A STRUCTURAL miss (node identity) says the populations were built
    from different trees, so the code paths diverged and no probability
    statement is available at all.
    """
    tag = key[0] if isinstance(key, tuple) and key else None
    if tag == "node_id":
        return StructuralDivergence(
            f"no node matching {key!r} in the reconstructed population of size "
            f"{n} at {kind} {label!r}; the replay built a different tree")
    return OutOfSupport(
        f"outcome {key!r} is not in the population π_corpus offers at {kind} "
        f"{label!r} ({n} slots), so it has probability zero here")


def _slot_mass(population: Sequence, value: Any) -> float:
    """How many slots hold this outcome (duplicates marginalized)."""
    try:
        return float(sum(1 for x in population if x == value))
    except Exception:
        return 1.0


def _weighted_mass(population: Sequence, w: Sequence[float], value: Any) -> float:
    try:
        return math.fsum(wi for x, wi in zip(population, w) if x == value)
    except Exception:
        return math.fsum(w) / len(population)


def _resolve_weights(population, weights, cum_weights) -> List[float]:
    """Same weight resolution as ``TracedRandom.choices``, kept identical."""
    if weights is None and cum_weights is None:
        return [1.0] * len(population)
    if weights is not None:
        return list(weights)
    cw = list(cum_weights)
    return [cw[0]] + [cw[i] - cw[i - 1] for i in range(1, len(cw))]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def score_derivation(derivation: Derivation,
                     replay: Callable[[ScoringRandom], Any],
                     *, require_exhausted: bool = True) -> ScoreResult:
    """Score ``derivation`` under π_corpus by replaying it through ``replay``.

    ``replay`` receives a :class:`ScoringRandom` and must run the CORPUS
    generator's code path — that path is what supplies the distribution at each
    site, and is therefore the definition of π_corpus. Anything else scores the
    derivation under the wrong measure, silently.

    Raises :class:`StructuralDivergence` or :class:`OutOfSupport` rather than
    returning a sentinel: the caller decides how an undefined ℓ(x) is
    represented, and A2.3 requires that decision to record a reason.
    """
    rng = ScoringRandom(derivation)
    replay(rng)
    if require_exhausted and not rng.exhausted:
        raise StructuralDivergence(
            f"replay consumed {rng._cursor} of {len(derivation)} scripted "
            f"choices; {rng.remaining} left over, so the derivation does not "
            f"correspond to this code path")
    return ScoreResult(neg_logp=rng.trace.neg_logp,
                       n_choices=rng.trace.n_draws,
                       by_label={k: v for k, v in
                                 sorted(rng.trace.by_label().items())})
