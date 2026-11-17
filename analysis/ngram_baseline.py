"""Token-level n-gram baseline ``b_tok`` — the axis-mate for the grammar floor.

See ``SURPRISAL_REFERENCE_POINTS.md`` §3 for why this must be TOKEN level. The
cross-entropy decomposition

    E_p[−log b(x)]  =  H(p)  +  D_KL(p ‖ b)

only holds when ``b`` is a proper normalized distribution over the SAME strings
as ``p``. The category-level observer of the separability audit is not: it
predicts an 8-way coarsening, which is a strictly easier target on a different
scale, so its surprisal cannot be subtracted from ``H(p)``. That observer keeps a
separate positional job; this module is what shares an axis with ``ℓ(x)``.

Model
-----
Interpolated **Witten–Bell**, recursively backing off to a uniform distribution
over the target support:

    P(w | h)  =  λ_h · P_ML(w | h)  +  (1 − λ_h) · P(w | h′)
    λ_h       =  c(h) / (c(h) + T_h)          T_h = distinct types after h
    P(w | ε)  =  λ_ε · P_ML(w)  +  (1 − λ_ε) · 1/V

Chosen over Kneser–Ney because normalization here is **structural** rather than a
property to be tuned: each level is a convex combination of two distributions
that each sum to 1 over the support, so every level sums to 1 by construction.
With an estimated baseline, a silent normalization bug would corrupt the KL
without any visible symptom, so a model whose correctness is by construction is
worth more than a point of perplexity. When ``c(h) = 0`` we have ``λ_h = 0`` and
the level degenerates to pure backoff, which is exactly right for unseen contexts.

Support and alignment
---------------------
Tokenization is ``models.tokenize.tokenizer`` — the *same* tokenizer the model
uses; commensurability depends on it. A sentence is ``[BOS] m₁ … m_T [EOS]``. The
predicted positions are ``m₁ … m_T`` and ``[EOS]``: ``[BOS]`` is context only,
never a target, matching the causal-LM convention where logits at ``t`` score the
token at ``t+1``. So the support is *morphemes + ``[EOS]``*, excluding ``[PAD]``,
``[UNK]`` and ``[BOS]``, and ``−log b(x)`` is directly comparable to the model's
summed per-token surprisal.

Contexts are left-padded with ``[BOS]`` to ``order − 1``, so ``b`` is a proper
distribution over variable-length strings terminated by ``[EOS]``.

Order
-----
``DEFAULT_ORDER = 4``, selected on held-out **validation** NLL per token, fit on
each grammar's full 100k train split (2026-07-31). Order 4 is the minimum for all
four grammars, and 5–6 are worse — Witten–Bell's type-based λ stops paying for
itself once contexts get sparse:

    NLL/token (validation)   order 3    order 4    order 5
    H                         2.4499     2.2438     2.2942
    P                         2.4703     2.2663     2.3255
    H′                        2.4676     2.2601     2.3159
    P′                        2.4593     2.2542     2.3074

Re-select if the corpora change. Models are refit on demand (~17 s/grammar)
rather than cached: the count tables run to ~175k contexts at level 3 and
serializing them costs more than rebuilding them.

**Order 1 is kept as a second, separate baseline.** It is the pure lexical-
frequency reader: it knows the (Zipfian) unigram distribution over morphemes and
nothing else. Order 4 is *not* a frequency control — it absorbs local bigram/
trigram structure too — so the two answer different questions, and the pair gives
a ladder (unigram -> 4-gram -> grammar floor) rather than one yardstick. Note
that the frequency confound itself is already removed exactly by ``ell(x)``: open-
class draws go through ``rng.choices(weights=zipf)`` and ``TracedRandom`` logs the
exact normalized weight, so order 1 is a reference point, not a correction. Each
order writes its own parquet.

Usage
-----
    python -m analysis.ngram_baseline --grammar all
    python -m analysis.ngram_baseline --grammar all --order 1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.tokenize import tokenizer  # noqa: E402

# Resolved through `models.dataload`, the single place that knows about corpus
# SETS (corpora/set_N/...). Hardcoding PROJECT_ROOT / "corpora" here would point
# at the directory that CONTAINS the sets, so every split lookup would miss.
from models.dataload import CORPORA_ROOT as CORPORA_DIR  # noqa: E402
DATA_DIR = Path(__file__).resolve().parent / "data"
GRAMMARS = ("H", "P", "Hprime", "Lprime")

# Held-out-selected; see the module docstring for the sweep.
DEFAULT_ORDER = 4

PAD_ID = tokenizer.convert_tokens_to_ids("[PAD]")
UNK_ID = tokenizer.convert_tokens_to_ids("[UNK]")
BOS_ID = tokenizer.convert_tokens_to_ids("[BOS]")
EOS_ID = tokenizer.convert_tokens_to_ids("[EOS]")

# Targets are morphemes and [EOS]. [BOS] is context-only; [PAD]/[UNK] never occur
# (models/dataload.py's alignment check asserts no [UNK] over the corpora).
TARGET_SUPPORT: Tuple[int, ...] = tuple(
    i for i in range(len(tokenizer.get_vocab()))
    if i not in (PAD_ID, UNK_ID, BOS_ID)
)


class InterpolatedWittenBell:
    """Interpolated Witten–Bell n-gram over token ids. Natural log throughout."""

    def __init__(self, order: int = DEFAULT_ORDER, support: Sequence[int] = TARGET_SUPPORT):
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.support = tuple(support)
        self.V = len(self.support)
        self._support_set = set(self.support)
        # counts[k][ctx][w] for context length k = 0 .. order-1
        self.counts: List[Dict[Tuple[int, ...], Dict[int, int]]] = [
            defaultdict(lambda: defaultdict(int)) for _ in range(order)
        ]
        self.totals: List[Dict[Tuple[int, ...], int]] = [
            defaultdict(int) for _ in range(order)
        ]
        self.n_sequences = 0
        self.n_targets = 0

    # --- fitting -----------------------------------------------------------

    def _padded(self, ids: Sequence[int]) -> List[int]:
        """``[BOS]×(order−1)`` ++ ids, so every target has a full context."""
        body = list(ids)
        if body and body[0] == BOS_ID:
            body = body[1:]
        return [BOS_ID] * (self.order - 1) + [BOS_ID] + body

    def fit(self, sequences: Iterable[Sequence[int]]) -> "InterpolatedWittenBell":
        for ids in sequences:
            seq = self._padded(ids)
            self.n_sequences += 1
            # Targets start after the padding + the real [BOS].
            for i in range(self.order, len(seq)):
                w = seq[i]
                if w not in self._support_set:
                    raise ValueError(f"token id {w} outside the target support")
                self.n_targets += 1
                for k in range(self.order):
                    ctx = tuple(seq[i - k:i]) if k else ()
                    self.counts[k][ctx][w] += 1
                    self.totals[k][ctx] += 1
        return self

    # --- scoring -----------------------------------------------------------

    def prob(self, context: Sequence[int], w: int) -> float:
        """P(w | context); context is truncated to the model's order."""
        ctx = tuple(context)[-(self.order - 1):] if self.order > 1 else ()
        return self._p(ctx, w)

    def _p(self, ctx: Tuple[int, ...], w: int) -> float:
        k = len(ctx)
        table = self.counts[k].get(ctx) if k else self.counts[0].get(())
        total = self.totals[k].get(ctx, 0) if k else self.totals[0].get((), 0)
        if k == 0:
            types = len(table) if table else 0
            lam = total / (total + types) if (total + types) else 0.0
            ml = (table.get(w, 0) / total) if total else 0.0
            return lam * ml + (1.0 - lam) / self.V
        if not table or not total:
            return self._p(ctx[1:], w)          # λ_h = 0: pure backoff
        lam = total / (total + len(table))
        return lam * (table.get(w, 0) / total) + (1.0 - lam) * self._p(ctx[1:], w)

    def logprob(self, context: Sequence[int], w: int) -> float:
        p = self.prob(context, w)
        if p <= 0.0:
            raise ValueError(f"zero probability for token {w}; support is broken")
        return math.log(p)

    def sequence_neg_logprob(self, ids: Sequence[int]) -> float:
        """−log b(x) in nats for one tokenized sentence (incl. its [EOS])."""
        seq = self._padded(ids)
        total = 0.0
        for i in range(self.order, len(seq)):
            total -= self.logprob(seq[i - (self.order - 1):i], seq[i])
        return total

    def per_token_neg_logprob(self, ids: Sequence[int]) -> List[float]:
        """Per-target surprisals, aligned to ``m₁ … m_T [EOS]``."""
        seq = self._padded(ids)
        return [-self.logprob(seq[i - (self.order - 1):i], seq[i])
                for i in range(self.order, len(seq))]

    # --- correctness -------------------------------------------------------

    def normalization_error(self, context: Sequence[int]) -> float:
        """|Σ_w P(w | context) − 1|. Structurally ~0; asserted in tests."""
        return abs(math.fsum(self.prob(context, w) for w in self.support) - 1.0)

    def summary(self) -> Dict[str, object]:
        return {
            "order": self.order,
            "support_size": self.V,
            "n_sequences": self.n_sequences,
            "n_targets": self.n_targets,
            "n_contexts_by_level": [len(c) for c in self.counts],
        }


# ---------------------------------------------------------------------------
# Corpus plumbing
# ---------------------------------------------------------------------------

def load_ids(grammar: str, split: str = "train",
             corpora_dir: Path = CORPORA_DIR) -> Tuple[List[str], List[List[int]]]:
    """(ids, tokenized sentences) for one grammar/split, in file order."""
    path = Path(corpora_dir) / split / f"{grammar}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    item_ids, sentences = [], []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            item_ids.append(rec["id"])
            sentences.append(rec["sentence"])
    encoded = tokenizer(sentences)["input_ids"]
    return item_ids, encoded


def fit_grammar(grammar: str, order: int = DEFAULT_ORDER, split: str = "train",
                corpora_dir: Path = CORPORA_DIR) -> InterpolatedWittenBell:
    _, seqs = load_ids(grammar, split, corpora_dir)
    return InterpolatedWittenBell(order=order).fit(seqs)


def score_split(lm: InterpolatedWittenBell, grammar: str, split: str,
                corpora_dir: Path = CORPORA_DIR) -> "pd.DataFrame":
    """Per-string ``−log b_tok(x)`` for one split, keyed by the stable item ``id``.

    Join key is ``id``, never ``pair_id`` — the latter repeats across the IOD and
    OOD probe splits (840 pair_ids, each appearing 4×), so joining on it silently
    merges unrelated items.
    """
    import pandas as pd

    item_ids, seqs = load_ids(grammar, split, corpora_dir)
    rows = []
    for item_id, ids in zip(item_ids, seqs):
        nll = lm.sequence_neg_logprob(ids)
        n_targets = len(ids) - 1          # every token except the leading [BOS]
        rows.append({
            "id": item_id,
            "grammar": grammar,
            "split": split,
            "order": lm.order,
            "n_target_tokens": n_targets,
            "neg_logb": nll,
            "neg_logb_per_token": nll / n_targets if n_targets else float("nan"),
        })
    return pd.DataFrame(rows)


def emit(grammars: Sequence[str] = GRAMMARS, order: int = DEFAULT_ORDER,
         fit_split: str = "train",
         score_splits: Sequence[str] = ("validation", "test_indistribution"),
         corpora_dir: Path = CORPORA_DIR,
         out_dir: Path = DATA_DIR) -> "pd.DataFrame":
    """Fit on ``fit_split``, score ``score_splits``, write one parquet.

    ``b_tok`` is ESTIMATED, so it is never scored on its own fit set: doing so
    would understate ``D_KL(p‖b)`` and inflate ``R`` (SURPRISAL_REFERENCE_POINTS
    §7 B3). ``fit_split`` is excluded from ``score_splits`` if passed by mistake.

    Non-default orders write to their own file. Order is a property of the
    baseline, not of the run, so overwriting the order-4 scores with an order-1
    fit would silently swap the yardstick underneath any downstream join.
    """
    import pandas as pd

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    held_out = [s for s in score_splits if s != fit_split]
    if len(held_out) != len(score_splits):
        print(f"  (dropped {fit_split!r} from scoring: b_tok is estimated and "
              "must not be scored on its fit set)", flush=True)

    frames = []
    for g in grammars:
        lm = fit_grammar(g, order, fit_split, corpora_dir)
        s = lm.summary()
        err = max(lm.normalization_error(c) for c in ((), (BOS_ID,)))
        print(f"[{g}] fit on {fit_split}: order={s['order']} "
              f"support={s['support_size']} seqs={s['n_sequences']} "
              f"targets={s['n_targets']} contexts={s['n_contexts_by_level']} "
              f"max|Σp−1|={err:.2e}", flush=True)
        for split in held_out:
            df = score_split(lm, g, split, corpora_dir)
            frames.append(df)
            print(f"    {split}: n={len(df)} "
                  f"mean −log b={df.neg_logb.mean():.3f} nats  "
                  f"per-token={df.neg_logb_per_token.mean():.4f}", flush=True)
        del lm
    out = pd.concat(frames, ignore_index=True)
    name = ("ngram_baseline_scores.parquet" if order == DEFAULT_ORDER
            else f"ngram_baseline_scores_order{order}.parquet")
    path = out_dir / name
    out.to_parquet(path, index=False)
    print(f"\nwrote {path}  ({len(out)} rows)")
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grammar", choices=[*GRAMMARS, "all"], default="all")
    p.add_argument("--order", type=int, default=DEFAULT_ORDER)
    p.add_argument("--fit-split", default="train", help="split to FIT on")
    p.add_argument("--score-splits", nargs="+",
                   default=["validation", "test_indistribution"],
                   help="held-out splits to score (fit split is excluded)")
    p.add_argument("--fit-only", action="store_true",
                   help="fit and report normalization; do not score or write")
    p.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()

    targets = list(GRAMMARS) if args.grammar == "all" else [args.grammar]
    if args.fit_only:
        for g in targets:
            lm = fit_grammar(g, args.order, args.fit_split, args.corpora_dir)
            s = lm.summary()
            err = max(lm.normalization_error(c) for c in ((), (BOS_ID,)))
            print(f"[{g}] order={s['order']} support={s['support_size']} "
                  f"seqs={s['n_sequences']} targets={s['n_targets']} "
                  f"contexts={s['n_contexts_by_level']} max|Σp−1|={err:.2e}",
                  flush=True)
        return
    emit(targets, args.order, args.fit_split, args.score_splits,
         args.corpora_dir, args.out_dir)


if __name__ == "__main__":
    main()
