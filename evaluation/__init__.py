"""Model-side evaluation: minimal-pair scoring and recipe selection.

`score_pairs`    — per-item model surprisal (whole-string and critical-span).
`aggregate`      — pairs items, applies the per-violation criterion, builds S[r].
`probe_callback` — the same scoring, at every epoch end, as a TrainerCallback.
`baseline_pairs` — the n-gram margins the objective is corrected against.
`test_eval`      — held-out language-modelling numbers on `test_indistribution`.
"""
