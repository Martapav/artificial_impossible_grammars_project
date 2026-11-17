"""The token n-gram baseline must be a proper distribution over strings.

``b_tok`` is only a valid reference point if it normalizes: the decomposition
``E_p[−log b] = H(p) + D_KL(p‖b)`` and the guarantee ``E[−log b] ≥ H(p)`` both
fail silently if it does not. Nothing downstream would look wrong — the KL would
just be a number that is not a KL — so normalization is pinned here rather than
spot-checked. See ``docs/SURPRISAL_REFERENCE_POINTS.md`` §3, §6.3.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from analysis.ngram_baseline import (
    BOS_ID,
    EOS_ID,
    InterpolatedWittenBell,
    TARGET_SUPPORT,
)
from models.tokenize import tokenizer

TOL = 1e-9


def _toy():
    """A small closed corpus over a tiny support, for exact assertions."""
    return [[BOS_ID, 10, 11, 12, EOS_ID],
            [BOS_ID, 10, 11, EOS_ID],
            [BOS_ID, 10, 13, 12, 12, EOS_ID],
            [BOS_ID, 11, 10, EOS_ID]]


def _lm(order, support=(10, 11, 12, 13, EOS_ID)):
    return InterpolatedWittenBell(order=order, support=support).fit(_toy())


# ---------------------------------------------------------------------------
# Normalization — the load-bearing property
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_normalizes_on_seen_contexts(order):
    lm = _lm(order)
    for ctx in ((), (BOS_ID,), (10,), (10, 11), (BOS_ID, 10, 11)):
        assert lm.normalization_error(ctx) < TOL, (order, ctx)


@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_normalizes_on_unseen_contexts(order):
    """An unseen context must back off to something that still sums to 1."""
    lm = _lm(order)
    for ctx in ((99,), (12, 12, 12), (EOS_ID, EOS_ID)):
        assert lm.normalization_error(ctx) < TOL, (order, ctx)


def test_normalizes_on_real_corpus_contexts():
    ids = tokenizer(["aa#4#2 v hhh#4#2", "n#4#2#7 r d#4#1 v"])["input_ids"]
    lm = InterpolatedWittenBell(order=3, support=TARGET_SUPPORT).fit(ids)
    for ctx in ((), (BOS_ID,), tuple(ids[0][1:3]), (EOS_ID,)):
        assert lm.normalization_error(ctx) < 1e-8, ctx


# ---------------------------------------------------------------------------
# Support and alignment
# ---------------------------------------------------------------------------

def test_bos_is_never_a_target():
    assert BOS_ID not in TARGET_SUPPORT
    assert EOS_ID in TARGET_SUPPORT


def test_target_count_excludes_bos():
    """A sentence of T morphemes yields T+1 targets: the morphemes and [EOS]."""
    lm = _lm(3)
    ids = [BOS_ID, 10, 11, 12, EOS_ID]        # 3 morphemes + EOS
    assert len(lm.per_token_neg_logprob(ids)) == 4


def test_sequence_equals_sum_of_per_token():
    lm = _lm(3)
    for ids in _toy():
        assert math.isclose(lm.sequence_neg_logprob(ids),
                            math.fsum(lm.per_token_neg_logprob(ids)),
                            rel_tol=1e-12)


def test_leading_bos_is_optional_and_idempotent():
    """Passing ids with or without the tokenizer's [BOS] scores identically."""
    lm = _lm(3)
    with_bos = [BOS_ID, 10, 11, EOS_ID]
    assert math.isclose(lm.sequence_neg_logprob(with_bos),
                        lm.sequence_neg_logprob([10, 11, EOS_ID]),
                        rel_tol=1e-12)


def test_out_of_support_token_is_rejected():
    lm = InterpolatedWittenBell(order=2, support=(10, 11, EOS_ID))
    with pytest.raises(ValueError, match="outside the target support"):
        lm.fit([[BOS_ID, 10, 99, EOS_ID]])


# ---------------------------------------------------------------------------
# Distributional sanity
# ---------------------------------------------------------------------------

def test_every_support_token_has_positive_probability():
    """Uniform backoff guarantees no zero-probability target, so −log b is finite."""
    lm = _lm(3)
    for w in lm.support:
        assert lm.prob((10, 11), w) > 0.0


def test_neg_logprob_is_positive_and_finite():
    lm = _lm(3)
    for ids in _toy():
        v = lm.sequence_neg_logprob(ids)
        assert 0.0 < v < math.inf


def test_higher_order_fits_training_data_at_least_as_well():
    """On the fitted corpus itself, more context cannot hurt."""
    toy = _toy()
    nll = [sum(InterpolatedWittenBell(order=o, support=(10, 11, 12, 13, EOS_ID))
               .fit(toy).sequence_neg_logprob(s) for s in toy)
           for o in (1, 2, 3)]
    assert nll[0] >= nll[1] >= nll[2]


def test_unigram_order_one_ignores_context():
    lm = _lm(1)
    assert math.isclose(lm.prob((10,), 11), lm.prob((12, 13), 11), rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Held-out emission (B3)
# ---------------------------------------------------------------------------

def test_emit_never_scores_the_fit_split(tmp_path, monkeypatch):
    """b_tok is estimated; scoring it on its own fit set understates D_KL(p‖b)."""
    from analysis import ngram_baseline as nb

    seen = []

    def fake_fit(grammar, order, split, corpora_dir):
        return _lm(2)

    def fake_score(lm, grammar, split, corpora_dir):
        import pandas as pd
        seen.append(split)
        return pd.DataFrame([{"id": f"{grammar}-{split}-0", "grammar": grammar,
                              "split": split, "n_target_tokens": 3,
                              "neg_logb": 1.0, "neg_logb_per_token": 1 / 3}])

    monkeypatch.setattr(nb, "fit_grammar", fake_fit)
    monkeypatch.setattr(nb, "score_split", fake_score)
    nb.emit(grammars=("H",), fit_split="train",
            score_splits=("train", "validation"), out_dir=tmp_path)
    assert "train" not in seen and "validation" in seen


def test_scores_are_finite_positive_and_id_keyed():
    """The emitted table must join on `id` (unique), not `pair_id`."""
    import pandas as pd
    path = Path("analysis/data/ngram_baseline_scores.parquet")
    if not path.exists():
        pytest.skip("run `python -m analysis.ngram_baseline` first")
    df = pd.read_parquet(path)
    assert df.id.is_unique
    assert (df.neg_logb > 0).all()
    assert df.neg_logb.notna().all() and (df.neg_logb != float("inf")).all()
    # per-token NLL is the string NLL divided by the target count
    approx = (df.neg_logb / df.n_target_tokens - df.neg_logb_per_token).abs().max()
    assert approx < 1e-9
