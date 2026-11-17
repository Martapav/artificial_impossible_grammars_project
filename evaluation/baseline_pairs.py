"""N-gram margins on the minimal pairs — the frequency and length control.

Computes, for every pair, the same margin the model is scored on but under the
n-gram baselines of `analysis/ngram_baseline.py`:

    margin_b1 = −log b₁(ungrammatical) − −log b₁(grammatical)      order 1
    margin_b4 = −log b₄(ungrammatical) − −log b₄(grammatical)      order 4

`margin_b1` is the **pure lexical-frequency reader**: it knows the Zipfian
unigram distribution over morphemes and nothing else. It is **subtracted** —
`evaluation.aggregate` ranks on `margin_corrected = margin − margin_b1`, which is
zero exactly when the model has captured nothing beyond frequency and token
count (2026-08-05; it was previously reported beside the margin and never
subtracted, which left the two `wrong_site` terms exposed to a length confound
with only a human comparison guarding the ranking). Both the raw margin and this
term stay in the reported table as diagnostics.

Because `−log b₁(x)` is a plain sum over the tokens of `x`, it absorbs *both*
effects a raw whole-string margin is exposed to:

* **which tokens** — the `wrong_item` violations swap one morpheme for another
  of different frequency;
* **how many** — the `wrong_site` violations relocate material, and their members
  end up at different lengths because a relocated substitution site carries a
  different number of CAT4/CAT2 modifiers.

So `margin_b1` is the length control as well as the frequency control, which is
why pairs are not dropped for length mismatch: dropping them would cut the two
`wrong_site` terms to a handful of pairs per grammar, and the correction recovers
the same information without discarding data.

⚠ **Known limitation, and an outstanding measurement.** A unigram term controls
frequency and count. It does **not** control any correlation between block length
and structural position — if licensed and unlicensed sites differ systematically
in block length, length is confounded with the licensing contrast itself and no
unigram subtraction removes that. The rate of length mismatch among `wrong_site`
pairs was quoted as "11–12 of 15", which is n=15 and not a usable estimate; both
the rate and the correlation need measuring at a sample size that supports them.

For `aux`, `wh_i`, `wh_ii` and `neutral_positional` the two members are exact
token permutations, so `margin_b1` is identically 0 by construction. That is
asserted rather than assumed: it is a free, sharp check that the pairing and the
tokenization agree with each other.

`margin_b4` is included because it is nearly free once the fit is done, and it
says something different — order 4 absorbs local bigram/trigram structure too,
so it is the "bounded local reader" rather than the frequency reader. Neither is
a correction; both are reference points (`analysis/ngram_baseline.py`).

Span-restricted margins
-----------------------
Each order also emits `margin_b{k}_span`: the same difference restricted to the
item's critical span, using `targets[].model_token_span` and the same `j−1`
alignment `evaluation/score_pairs.py` uses (surprisal of input position `j` lives
at index `j−1` of the per-target array). It is NaN where an item carries no
target.

This exists because the whole-string/span distinction is *not* free above order
1. For a unigram the two coincide on a length-matched pair — every token outside
the span contributes the same term to both members and cancels — which is what
licenses subtracting a whole-string `margin_b1` from a span margin at all, and
`check_span_equals_string` now verifies that identity instead of leaving it
asserted. For order 4 it fails: changing the token under the span also changes
the *context* of the next `order − 1` positions, so the whole-string `margin_b4`
carries surprisal from outside the span. A span-criterion term must therefore be
compared against `margin_b4_span`, not against `margin_b4`, or the two sides of
the subtraction are sums over different numbers of tokens.

This output is **model independent**: it depends only on the corpora, so it is
computed once and joined onto any number of scored runs.

Usage
-----
    python -m evaluation.baseline_pairs --split structural_probe_IOD \\
        --out results/eval/ngram_margins_IOD.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from analysis.ngram_baseline import fit_grammar
from evaluation.provenance import describe, write_parquet
from evaluation.score_pairs import CORPORA_DIR, GRAMMARS, load_probe_items
from models.dataload import corpus_provenance, require_set_scoped, set_scoped
from models.tokenize import tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ORDERS = (1, 4)


def pair_margins(
    grammars: Sequence[str] = GRAMMARS,
    split: str = "structural_probe_IOD",
    orders: Sequence[int] = ORDERS,
    fit_split: str = "train",
    corpora_dir: Path = CORPORA_DIR,
) -> "pd.DataFrame":
    """Per-pair n-gram margins for one probe split, all grammars.

    The n-grams are fit on `fit_split` (train) and scored on the probe items,
    which are held out from it — the same discipline `ngram_baseline.emit()`
    applies, for the same reason: an estimated baseline scored on its own fit set
    understates its own divergence.
    """
    import pandas as pd

    rows: list[dict] = []
    for grammar in grammars:
        items = load_probe_items(grammar, split, corpora_dir)
        encoded = tokenizer([it["sentence"] for it in items])["input_ids"]
        spans = [[tuple(t["model_token_span"]) for t in (it.get("targets") or [])]
                 for it in items]

        scores: dict[int, list[float]] = {}
        crit: dict[int, list[float]] = {}
        for order in orders:
            lm = fit_grammar(grammar, order=order, split=fit_split,
                             corpora_dir=corpora_dir)
            per_token = [lm.per_token_neg_logprob(ids) for ids in encoded]
            scores[order] = [sum(s) for s in per_token]
            crit[order] = [_span_sum(s, sp) for s, sp in zip(per_token, spans)]
            del lm

        by_pair: dict = {}
        for index, item in enumerate(items):
            by_pair.setdefault(item["pair_id"], {})[item["variant"]] = index

        for pair_id, members in by_pair.items():
            if len(members) != 2:
                raise ValueError(f"{grammar}/{split}: {pair_id} has no partner")
            g, u = members["grammatical"], members["ungrammatical"]
            row = {"grammar": grammar, "split": split, "pair_id": pair_id,
                   "subcase": items[g]["subcase"],
                   "violation": items[u]["violation"],
                   # Tokenized lengths, not morpheme counts: the cancellation
                   # argument behind the b1 correction is about tokens.
                   "n_tokens_g": len(encoded[g]), "n_tokens_u": len(encoded[u]),
                   "same_outside_span": _same_outside_span(
                       encoded[g], encoded[u], spans[g], spans[u])}
            for order in orders:
                row[f"margin_b{order}"] = scores[order][u] - scores[order][g]
                row[f"margin_b{order}_span"] = crit[order][u] - crit[order][g]
            rows.append(row)

    return pd.DataFrame(rows)


def _span_sum(per_token: Sequence[float], spans: Sequence[Sequence[int]]) -> float:
    """Σ surprisal over `spans`, or NaN where the item names no critical position.

    `per_token` is aligned to ``m₁ … m_T [EOS]``, so input position `j` is at
    index `j − 1` — the convention `evaluation/score_pairs.py` scores the model
    under, and the two must agree for the subtraction downstream to be in units.
    """
    if not spans:
        return float("nan")
    total = 0.0
    for start, end in spans:
        if not (1 <= start < end <= len(per_token) + 1):
            raise IndexError(
                f"critical span [{start}, {end}) outside the "
                f"{len(per_token) + 1}-token sequence")
        total += sum(per_token[start - 1:end - 1])
    return total


def _same_outside_span(g_ids: Sequence[int], u_ids: Sequence[int],
                       g_spans: Sequence[Sequence[int]],
                       u_spans: Sequence[Sequence[int]]) -> bool:
    """True when the two members are token-identical everywhere but the span.

    This is the exact precondition for the unigram cancellation argument, and it
    is a structural test rather than a lookup on `violation`: an in-place
    substitution satisfies it, a relocation (`wrong_site`) and a permutation
    (`unlicensed_site`, `positional_*`) do not, because those differ outside the
    span as well even when the lengths match.
    """
    if not g_spans or list(g_spans) != list(u_spans) or len(g_ids) != len(u_ids):
        return False
    masked = set()
    for start, end in g_spans:
        masked.update(range(start, end))
    return all(a == b for i, (a, b) in enumerate(zip(g_ids, u_ids))
               if i not in masked)


def check_permutation_terms(margins: "pd.DataFrame", tol: float = 1e-9) -> dict:
    """`margin_b1` must be exactly 0 wherever the members are permutations.

    A unigram model is order blind, so any non-zero value on a permutation pair
    means the pairing or the tokenization is wrong — a sharp, free check on both.
    """
    permutation = margins.violation.isin(("unlicensed_site",)) | \
        margins.violation.fillna("").str.startswith("positional_")
    subset = margins[permutation]
    worst = float(subset.margin_b1.abs().max()) if len(subset) else 0.0
    return {"n_permutation_pairs": int(len(subset)), "max_abs_margin_b1": worst,
            "ok": len(subset) > 0 and worst <= tol}


def check_span_equals_string(margins: "pd.DataFrame", tol: float = 1e-6) -> dict:
    """On order 1, the span margin must equal the whole-string margin.

    Restricted to `same_outside_span` pairs — the in-place substitutions. There
    the tokens outside the span contribute identically to both members of a
    *unigram* sum and cancel, so the whole-string margin equals the span one.
    That identity is what lets `aggregate.py` subtract a whole-string `margin_b1`
    from a span margin, so it is measured rather than assumed.

    Relocations and permutations are excluded because they differ outside the
    span too, so nothing is expected to cancel — for those `aggregate.py` scores
    the whole string on both sides and the identity is not needed. Order 4 is
    deliberately not checked at all: it is expected to differ even on in-place
    substitutions, which is the reason `margin_b4_span` exists.
    """
    subset = margins[margins["same_outside_span"]]
    if not len(subset):
        return {"n_span_pairs": 0, "max_abs_gap": 0.0, "ok": False}
    gap = (subset["margin_b1_span"] - subset["margin_b1"]).abs()
    worst = float(gap.max())
    return {"n_span_pairs": int(len(subset)), "max_abs_gap": worst,
            "ok": worst <= tol}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grammar", nargs="+", default=["all"])
    parser.add_argument("--split", nargs="+", default=["structural_probe_IOD"])
    parser.add_argument("--orders", nargs="+", type=int, default=list(ORDERS))
    parser.add_argument("--fit-split", default="train")
    parser.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    parser.add_argument("--out", type=Path, default=None,
                        help="output parquet. Default: "
                             "results/eval/<corpus set>/ngram_margins.parquet")
    parser.add_argument("--allow-unscoped-out", action="store_true",
                        help="permit an --out path that does not name the "
                             "corpus set (it will not be distinguishable from "
                             "another generation's)")
    args = parser.parse_args()

    default_out = set_scoped(PROJECT_ROOT / "results" / "eval",
                             args.corpora_dir) / "ngram_margins.parquet"
    args.out = require_set_scoped(args.out or default_out, args.corpora_dir,
                                  override=args.allow_unscoped_out)

    import pandas as pd

    grammars = GRAMMARS if args.grammar == ["all"] else tuple(args.grammar)
    frames = []
    for split in args.split:
        print(f"fitting + scoring {split} ...", flush=True)
        df = pair_margins(grammars, split, args.orders, args.fit_split,
                          args.corpora_dir)
        report = check_permutation_terms(df)
        print(f"  permutation pairs: {report['n_permutation_pairs']}  "
              f"max |margin_b1| = {report['max_abs_margin_b1']:.2e}  "
              f"{'OK' if report['ok'] else 'FAIL'}")
        if not report["ok"]:
            raise SystemExit(
                "margin_b1 is non-zero on permutation pairs; the pairing or the "
                "tokenization disagrees with the probe metadata."
            )

        if 1 in args.orders:
            span_report = check_span_equals_string(df)
            print(f"  in-place-substitution pairs: {span_report['n_span_pairs']}  "
                  f"max |b1_span − b1| = {span_report['max_abs_gap']:.2e}  "
                  f"{'OK' if span_report['ok'] else 'FAIL'}")
            if not span_report["ok"]:
                raise SystemExit(
                    "margin_b1_span disagrees with margin_b1 on length-matched "
                    "span pairs, where the out-of-span tokens must cancel. The "
                    "critical spans or the tokenization are wrong, and the b1 "
                    "correction aggregate.py applies is not in units."
                )
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    # This file is model independent but NOT corpus independent: it is fit on
    # `--fit-split` and scored on the probe sentences of one specific corpus
    # generation. `evaluation.aggregate` refuses to join it onto scores from a
    # different one — see evaluation/provenance.py for what that prevents.
    provenance = corpus_provenance(
        grammars, tuple(args.split) + (args.fit_split,), root=args.corpora_dir)
    write_parquet(out, args.out, provenance)
    print(f"\nwrote {args.out}  ({len(out)} rows)")
    print(f"  corpus: {describe(provenance)}")
    cols = [c for c in out.columns if c.startswith("margin_b")]
    print(out.groupby(["split", "subcase", "violation"])[cols].mean().to_string())


if __name__ == "__main__":
    main()
