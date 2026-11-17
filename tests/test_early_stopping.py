"""The pilot's stopping machinery: the pair split, the metric, the patience.

These cover the parts that fail *silently*. A pair split that drifts between
processes, or a stopping metric that reads the same items selection reads, would
still produce a sweep that finishes and a parquet that opens — the damage only
shows up as a recipe chosen on a biased number.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pandas as pd
import pytest

from evaluation.aggregate import (
    SCORE_HALF,
    STOP_HALF,
    pair_half,
    select_half,
    to_pairs,
)
from models.grid_search import _Patience, STOP_MIN_DELTA, stopping_metric

PAIR_IDS = [f"H-sp-{i:05d}" for i in range(700)]


# --- the split -------------------------------------------------------------

def test_pair_half_is_deterministic_within_a_process():
    assert [pair_half(p) for p in PAIR_IDS] == [pair_half(p) for p in PAIR_IDS]


def test_pair_half_survives_a_different_hash_seed():
    """The builtin `hash()` is salted per process; blake2b is not.

    If this ever fails, a pair is in the stopping half while training and in the
    scoring half while analysing, and the two halves stop being disjoint.
    """
    script = (
        "from evaluation.aggregate import pair_half;"
        "print(''.join(str(pair_half(f'H-sp-{i:05d}')) for i in range(700)))"
    )
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": here}
        proc = subprocess.run([sys.executable, "-c", script], env=env, cwd=here,
                              capture_output=True, text=True, check=True)
        out.append(proc.stdout.strip())
    assert out[0] == out[1]
    assert out[0] == "".join(str(pair_half(p)) for p in PAIR_IDS)


def test_pair_half_splits_roughly_evenly():
    n_score = sum(pair_half(p) == SCORE_HALF for p in PAIR_IDS)
    assert 0.4 * len(PAIR_IDS) < n_score < 0.6 * len(PAIR_IDS)


def test_select_half_partitions_the_pairs():
    pairs = pd.DataFrame({"pair_id": PAIR_IDS})
    stop = set(select_half(pairs, "stop")["pair_id"])
    score = set(select_half(pairs, "score")["pair_id"])
    assert stop | score == set(PAIR_IDS)
    assert stop & score == set()
    assert len(select_half(pairs, "all")) == len(PAIR_IDS)


def test_select_half_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="half must be"):
        select_half(pd.DataFrame({"pair_id": PAIR_IDS}), "training")


# --- the stopping metric ---------------------------------------------------

def _items(spec) -> pd.DataFrame:
    """A per-item frame in the shape `score_split` returns.

    `spec` is a list of (pair_id, subcase, violation, correct): `correct` puts
    the ungrammatical member above the grammatical one on both statistics, so
    the pair scores 1.0, and reverses it otherwise.
    """
    rows = []
    for pair_id, subcase, violation, correct in spec:
        for variant in ("grammatical", "ungrammatical"):
            ungram = variant == "ungrammatical"
            # Surprisal is higher on the member that should be dispreferred.
            high = ungram if correct else not ungram
            rows.append({
                "id": f"{pair_id}-{variant}",
                "grammar": "H",
                "split": "structural_probe_SELECT",
                "run_id": "r0",
                "epoch": 1.0,
                "pair_id": pair_id,
                "variant": variant,
                "violation": violation if ungram else None,
                "subcase": subcase,
                "n_morphemes": 10,
                "neg_logq": 20.0 + (1.0 if high else 0.0),
                "neg_logq_crit": 2.0 + (1.0 if high else 0.0),
            })
    return pd.DataFrame(rows)


def _by_half(n: int, want: int) -> list[str]:
    return [p for p in PAIR_IDS if pair_half(p) == want][:n]


def test_stopping_metric_reads_the_stop_half_only():
    """Right on the stopping half, wrong on the scoring half -> 1.0."""
    spec = [(p, "aux", "unlicensed_site", True) for p in _by_half(10, STOP_HALF)]
    spec += [(p, "aux", "unlicensed_site", False) for p in _by_half(10, SCORE_HALF)]
    assert stopping_metric(_items(spec)) == pytest.approx(1.0)

    # ... and the mirror image, so the test cannot pass by reading everything.
    flipped = [(p, s, v, not c) for p, s, v, c in spec]
    assert stopping_metric(_items(flipped)) == pytest.approx(0.0)


def test_stopping_metric_ignores_the_gate_terms():
    """A failed neutral control is a gate, not an objective term."""
    spec = [(p, "aux", "unlicensed_site", True) for p in _by_half(10, STOP_HALF)]
    with_gate = spec + [(p, "neutral_agreement", "agreement", False)
                        for p in _by_half(20, STOP_HALF)[10:]]
    assert stopping_metric(_items(with_gate)) == pytest.approx(1.0)


def test_stopping_metric_weights_terms_equally_not_pairs():
    """One term right, one wrong -> 0.5, whatever the pair counts."""
    stop_ids = _by_half(30, STOP_HALF)
    spec = [(p, "aux", "unlicensed_site", True) for p in stop_ids[:25]]
    spec += [(p, "refl", "wrong_item", False) for p in stop_ids[25:30]]
    assert stopping_metric(_items(spec)) == pytest.approx(0.5)


def test_stopping_metric_is_nan_without_objective_pairs():
    spec = [(p, "neutral_agreement", "agreement", True)
            for p in _by_half(6, STOP_HALF)]
    assert stopping_metric(_items(spec)) != stopping_metric(_items(spec))  # NaN


# --- the patience tracker --------------------------------------------------

def test_patience_fires_after_flat_epochs():
    tracker = _Patience(patience=3)
    assert tracker.update(0.50) is False          # first epoch sets the best
    assert tracker.update(0.50) is False
    assert tracker.update(0.50) is False
    assert tracker.update(0.50) is True


def test_patience_resets_on_a_real_improvement():
    tracker = _Patience(patience=2)
    tracker.update(0.50)
    assert tracker.update(0.50) is False
    assert tracker.update(0.60) is False          # improvement: counter resets
    assert tracker.update(0.60) is False
    assert tracker.update(0.60) is True


def test_improvement_below_min_delta_does_not_reset():
    tracker = _Patience(patience=2, min_delta=0.01)
    tracker.update(0.50)
    assert tracker.update(0.505) is False         # real, but under the threshold
    assert tracker.update(0.509) is True


def test_nan_never_improves():
    tracker = _Patience(patience=2)
    assert tracker.update(float("nan")) is False
    assert tracker.update(float("nan")) is True


def test_a_falling_metric_still_stops():
    tracker = _Patience(patience=2)
    tracker.update(0.60)
    assert tracker.update(0.50) is False
    assert tracker.update(0.40) is True


def test_min_delta_default_is_the_module_constant():
    assert _Patience(patience=1).min_delta == STOP_MIN_DELTA
