"""Per-epoch minimal-pair scoring during training, as a `TrainerCallback`.

Lives here rather than in `models/grid_search.py` because both training entry
points need it and for different reasons:

* the **grid search** scores the SELECT probe to decide when a run has
  converged — the stopping rule (`stopping_metric` + `_Patience`);
* the **production run** (`models/train.py`) scores it to keep the *learning
  curve*, with early stopping switched off (`patience=0`): every kept model
  trains the same fixed number of epochs, so the curve is a record, not a
  decision. It answers the question validation loss cannot — validation loss
  bottoms around epoch 6-7 while the probe margins keep climbing, and only a
  per-phenomenon curve shows which rules are still being learned at the cap.

Per-ITEM rows, not a summary. `S[r]` z-scores each term across the recipes being
compared, so it cannot be computed inside a single run anyway; keeping the raw
rows means `evaluation/aggregate.py` can be re-run, re-weighted or re-scoped
without retraining anything.
"""
from __future__ import annotations

from transformers import TrainerCallback

# The probe the recipe is SELECTED on — never the reporting probes, and never
# OOD (docs/RECIPE_SELECTION.md §1.7). It is also the right split for
# the production learning curve: it has already done its selection work, so
# scoring it 30x per run cannot contaminate the reported IOD/OOD numbers.
SELECT_SPLIT = "structural_probe_SELECT"

# An epoch counts as an improvement only if it beats the best by this much, in
# accuracy points. Without it the tracker's patience is reset by pure noise and
# nothing ever stops.
STOP_MIN_DELTA = 0.001


def stopping_metric(frame) -> float:
    """Mean over the objective terms of their accuracy on the STOP half.

    Accuracy rather than mean margin because the seven terms are not in the same
    units — whole-string margins are nats over ~24 tokens, span margins nats over
    one — so averaging raw margins would let the movement terms decide that a run
    has converged while every other rule is still improving. Accuracy is unit
    free, so each rule gets an equal say.

    Deliberately NOT `S[r]`: that z-scores each term across the recipes being
    compared, so it does not exist inside a single run. This is only a
    convergence detector; the ranking still happens in `evaluation.aggregate`.

    Returns NaN when the frame carries no objective pairs, which the caller
    treats as "no improvement" rather than as an error.
    """
    from evaluation.aggregate import GATE_TERMS, select_half, to_pairs

    # to_pairs on the FULL frame, then halve: term names are derived from the
    # pairs it is given, so naming them on half the data could name them
    # differently from the analysis side.
    objective = select_half(to_pairs(frame), "stop")
    objective = objective[~objective["term"].isin(GATE_TERMS)]
    if objective.empty:
        return float("nan")
    return float(objective.groupby("term")["acc"].mean().mean())


class _Patience:
    """Tracks the best stopping metric and how many epochs ago it was beaten.

    Separate from the callback so the stopping decision can be tested without a
    Trainer, a model or a corpus.
    """

    def __init__(self, patience: int, min_delta: float = STOP_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("-inf")
        self.since_best = 0

    def update(self, value: float) -> bool:
        """Records one epoch's value; True means training should stop.

        A NaN never improves on the best (`value == value` is False for NaN), so
        a run whose probe scoring degenerates stops on patience instead of
        training to the cap on a metric that means nothing.
        """
        if value == value and value > self.best + self.min_delta:
            self.best = value
            self.since_best = 0
        else:
            self.since_best += 1
        return self.since_best >= self.patience


class ProbeScoringCallback(TrainerCallback):
    """Scores a minimal-pair probe at every epoch end and keeps the raw rows.

    `patience=0` disables early stopping entirely: the callback then only
    records, and the run trains to its epoch count whatever the curve does. That
    is the production setting — all 20 kept models must see the same number of
    epochs, or the H-vs-P contrast is confounded by each grammar stopping at its
    own best moment.
    """

    def __init__(self, grammar: str, run_id: str, *, split: str = SELECT_SPLIT,
                 device: str = "cpu", batch_size: int = 128,
                 patience: int = 0):
        self.grammar = grammar
        self.run_id = run_id
        self.split = split
        self.device = device
        self.batch_size = batch_size
        self.frames: list = []
        # patience=0 disables early stopping: every run trains to the cap.
        self.patience = _Patience(patience) if patience else None

    def on_epoch_end(self, args, state, control, **kwargs):
        from evaluation.score_pairs import score_split

        model = kwargs["model"]
        frame = score_split(
            model, self.grammar, self.split,
            device=self.device, batch_size=self.batch_size,
            run_id=self.run_id, epoch=round(state.epoch, 4),
        )
        self.frames.append(frame)
        if self.patience is not None and self.patience.update(stopping_metric(frame)):
            control.should_training_stop = True
        return control

    def scores(self):
        import pandas as pd
        return pd.concat(self.frames, ignore_index=True) if self.frames else None
