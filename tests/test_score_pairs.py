"""Correctness of the model-side minimal-pair scorer.

The failure modes here are all *silent*: an off-by-one in the t/t+1 offset scores
a neighbouring morpheme, a padding leak makes a score depend on batch
composition, a log-base slip rescales every number. All of them leave plausible
output. So each is asserted rather than eyeballed.

Uses an untrained model throughout — every property tested is a property of the
scoring arithmetic, not of what the model learned.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from evaluation.score_pairs import (
    GRAMMARS,
    PROBE_SPLITS,
    assert_nats_convention,
    check_span_convention,
    load_probe_items,
    score_sentences,
    score_split,
)
from models.model import build_model
from models.tokenize import tokenizer


@pytest.fixture(scope="module")
def model():
    # Smallest thing that runs; dropout off so scoring is deterministic.
    m = build_model(n_layer=2, n_embd=128, n_head=4, dropout=0.0, seed=0)
    m.eval()
    return m


@pytest.fixture(scope="module")
def sentences():
    items = load_probe_items("H", "structural_probe_IOD")
    return [it["sentence"] for it in items[:24]]


# --- conventions -----------------------------------------------------------

def test_log_base_is_nats():
    assert_nats_convention()


def test_span_convention_holds_on_every_grammar_and_split():
    """`model_token_span` indexes the BOS-inclusive sequence.

    Checked where it is checkable: on the index-aligned violations the span must
    cover exactly the tokens that differ between the two members. Under an
    off-by-one the span would land on a shared token and this would fail.
    """
    for grammar in GRAMMARS:
        for split in PROBE_SPLITS:
            report = check_span_convention(grammar, split)
            assert report["aligned_pairs_checked"] > 0
            assert report["span_tokens_differ"] == report["aligned_pairs_checked"]
            assert report["spans_out_of_range"] == 0


# --- the arithmetic --------------------------------------------------------

def test_neg_logq_equals_hand_computed_sum(model):
    """Plan step C2: recompute one string's surprisal by hand, token by token.

    Deliberately written the long way — an explicit python loop over positions
    with its own softmax — so it shares no code path with the batched
    implementation it is checking.
    """
    sentence = load_probe_items("H", "structural_probe_IOD")[0]["sentence"]
    ids = tokenizer(sentence)["input_ids"]

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]

    manual = 0.0
    for position in range(1, len(ids)):          # [BOS] at 0 is never a target
        distribution = F.log_softmax(logits[position - 1].float(), dim=-1)
        manual -= distribution[ids[position]].item()

    (row,) = score_sentences(model, [sentence], [[]])
    assert row["n_target_tokens"] == len(ids) - 1
    assert row["neg_logq"] == pytest.approx(manual, abs=1e-4)


def test_critical_span_is_the_surprisal_of_that_position(model):
    """`neg_logq_crit` must equal the hand-computed surprisal over the span."""
    item = next(it for it in load_probe_items("H", "structural_probe_IOD")
                if it["targets"])
    ids = tokenizer(item["sentence"])["input_ids"]
    start, end = item["targets"][0]["model_token_span"]

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]

    manual = 0.0
    for position in range(start, end):
        distribution = F.log_softmax(logits[position - 1].float(), dim=-1)
        manual -= distribution[ids[position]].item()

    (row,) = score_sentences(model, [item["sentence"]], [[[start, end]]])
    assert row["neg_logq_crit"] == pytest.approx(manual, abs=1e-4)
    assert row["n_crit_tokens"] == end - start


def test_offset_by_one_would_be_caught(model):
    """The t/t+1 offset is not accidentally symmetric.

    Guards the previous two tests: if scoring at `position` and at `position−1`
    gave the same answer, they would pass under an off-by-one too.
    """
    item = next(it for it in load_probe_items("H", "structural_probe_IOD")
                if it["targets"])
    ids = tokenizer(item["sentence"])["input_ids"]
    start, end = item["targets"][0]["model_token_span"]

    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits[0]

    correct = -F.log_softmax(logits[start - 1].float(), dim=-1)[ids[start]].item()
    shifted = -F.log_softmax(logits[start].float(), dim=-1)[ids[start]].item()
    assert abs(correct - shifted) > 1e-3


def test_whole_string_equals_sum_over_all_positions(model, sentences):
    """`neg_logq` is the sum of its parts: scoring each position as its own span
    and adding must reproduce the whole-string total."""
    for sentence in sentences[:4]:
        n = len(tokenizer(sentence)["input_ids"])
        (whole,) = score_sentences(model, [sentence], [[]])
        (parts,) = score_sentences(model, [sentence], [[[j, j + 1] for j in range(1, n)]])
        assert parts["neg_logq_crit"] == pytest.approx(whole["neg_logq"], abs=1e-4)


# --- batching --------------------------------------------------------------

def test_padding_does_not_leak(model, sentences):
    """A sentence's score must not depend on what it was batched with.

    GPT-2 assigns position ids by `arange`, so right-padding is safe only if the
    attention mask is applied — this is what would catch a missing mask.
    """
    ordered = sorted(sentences, key=lambda s: len(s))
    alone = score_sentences(model, ordered, [[]] * len(ordered), batch_size=1)
    mixed = score_sentences(model, ordered, [[]] * len(ordered), batch_size=len(ordered))
    for a, m in zip(alone, mixed):
        assert a["neg_logq"] == pytest.approx(m["neg_logq"], abs=1e-3)
        assert a["n_target_tokens"] == m["n_target_tokens"]


def test_batch_size_does_not_change_scores(model, sentences):
    small = score_sentences(model, sentences, [[]] * len(sentences), batch_size=3)
    large = score_sentences(model, sentences, [[]] * len(sentences), batch_size=64)
    for s, l in zip(small, large):
        assert s["neg_logq"] == pytest.approx(l["neg_logq"], abs=1e-3)


# --- sanity of the emitted table -------------------------------------------

def test_scores_are_finite_and_positive(model):
    df = score_split(model, "H", "structural_probe_IOD")
    # Size is a build parameter (--probe-n-per-subcase), not a constant: pin the
    # scorer against the corpus on disk rather than a number that goes stale.
    assert len(df) == len(load_probe_items("H", "structural_probe_IOD"))
    assert len(df) % 14 == 0        # 7 subcases x 2 members
    assert df.neg_logq.notna().all() and (df.neg_logq > 0).all()
    assert df.neg_logq_crit.notna().all()        # every probe item has a target
    assert (df.neg_logq_crit > 0).all()
    assert (df.neg_logq_crit <= df.neg_logq + 1e-6).all()


def test_untrained_model_is_at_chance(model):
    """An untrained model must not prefer grammatical members.

    A scorer that leaked the label — via item order, or by scoring the two
    members differently — would show up here as accuracy away from 0.5.
    """
    df = score_split(model, "H", "structural_probe_IOD")
    pivot = df.pivot(index="pair_id", columns="variant", values="neg_logq")
    accuracy = (pivot["grammatical"] < pivot["ungrammatical"]).mean()
    assert 0.30 < accuracy < 0.70


def test_untrained_surprisal_near_uniform(model):
    """Sanity on the scale: an untrained model is near log(V) nats per token."""
    df = score_split(model, "H", "structural_probe_IOD")
    assert df.neg_logq_per_token.mean() == pytest.approx(math.log(len(tokenizer)), abs=0.6)


def test_passthrough_columns_survive(model):
    df = score_split(model, "H", "structural_probe_IOD", run_id="unit-test")
    for column in ("id", "pair_id", "variant", "violation", "subcase", "run_id"):
        assert column in df.columns
    assert df["id"].is_unique                    # `id` is the join key
    assert set(df.variant) == {"grammatical", "ungrammatical"}
    assert (df.run_id == "unit-test").all()


def test_sidecar_drift_is_detected(tmp_path):
    """A sidecar describing a different sentence must raise, not mis-score."""
    import json

    split_dir = tmp_path / "fake_split"
    split_dir.mkdir()
    (split_dir / "H.jsonl").write_text(
        json.dumps({"id": "x-0", "sentence": "a#4#1 v"}) + "\n", encoding="utf-8")
    (split_dir / "H.meta.jsonl").write_text(
        json.dumps({"id": "x-0", "sentence": "b#4#1 v", "targets": []}) + "\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="sidecar drift"):
        load_probe_items("H", "fake_split", corpora_dir=tmp_path)


# --- grammaticality direction ----------------------------------------------

def test_lower_neg_logq_is_the_preferred_member():
    """A margin is positive when the model assigns the grammatical member the
    LOWER `neg_logq`.

    `neg_logq` is a surprisal, `−log q`, so "the model prefers x" means "smaller
    number". `aggregate.to_pairs` therefore orients the margin as
    ``ungrammatical − grammatical``. Getting that subtraction backwards is the
    most consequential silent error in the whole pipeline: every accuracy would
    read as 1 − acc, a model that had learned the rule perfectly would be
    reported at 0%, and nothing else in the output would look wrong. Pinned on a
    hand-built frame so it tests the orientation and nothing else.
    """
    import pandas as pd
    from evaluation.aggregate import to_pairs

    # Two pairs, one per criterion geometry. In both, the grammatical member is
    # the CHEAPER one, so a correct implementation reports margin > 0 and acc 1.
    rows = [
        # whole-string criterion (relocation)
        dict(pair_id="p-string", variant="grammatical", violation=None,
             construction="wh_movement", subcase="wh_i",
             neg_logq=10.0, neg_logq_crit=1.0, n_morphemes=8),
        dict(pair_id="p-string", variant="ungrammatical", violation="unlicensed_site",
             construction="wh_movement", subcase="wh_i",
             neg_logq=14.0, neg_logq_crit=3.0, n_morphemes=8),
        # critical-span criterion (in-place substitution)
        dict(pair_id="p-span", variant="grammatical", violation=None,
             construction="anaphoric_binding_refl", subcase="refl",
             neg_logq=20.0, neg_logq_crit=2.0, n_morphemes=9),
        dict(pair_id="p-span", variant="ungrammatical", violation="wrong_item",
             construction="anaphoric_binding_refl", subcase="refl",
             neg_logq=20.5, neg_logq_crit=5.0, n_morphemes=9),
    ]
    for row in rows:
        row.update(grammar="H", split="structural_probe_IOD")
    pairs = to_pairs(pd.DataFrame(rows)).set_index("pair_id")

    # Orientation: ungrammatical − grammatical, for both statistics.
    assert pairs.loc["p-string", "margin_string"] == pytest.approx(4.0)
    assert pairs.loc["p-string", "margin_span"] == pytest.approx(2.0)
    assert pairs.loc["p-span", "margin_span"] == pytest.approx(3.0)

    # The criterion assignment picks the right one of the two per geometry.
    assert pairs.loc["p-string", "criterion"] == "string"
    assert pairs.loc["p-span", "criterion"] == "span"
    assert pairs.loc["p-string", "margin"] == pytest.approx(4.0)
    assert pairs.loc["p-span", "margin"] == pytest.approx(3.0)

    # Preferring the grammatical member scores as correct, on both criteria.
    assert (pairs["acc"] == 1.0).all()


def test_preferring_the_ungrammatical_member_scores_zero():
    """The converse of the above — the direction is pinned from both sides.

    Without this, a sign flip that made every margin negative would still pass
    the test above if that test only checked magnitudes.
    """
    import pandas as pd
    from evaluation.aggregate import to_pairs

    rows = [
        dict(pair_id="p", variant="grammatical", violation=None,
             construction="wh_movement", subcase="wh_i",
             neg_logq=14.0, neg_logq_crit=3.0, n_morphemes=8),
        dict(pair_id="p", variant="ungrammatical", violation="unlicensed_site",
             construction="wh_movement", subcase="wh_i",
             neg_logq=10.0, neg_logq_crit=1.0, n_morphemes=8),
    ]
    for row in rows:
        row.update(grammar="H", split="structural_probe_IOD")
    pairs = to_pairs(pd.DataFrame(rows))

    assert pairs["margin"].iloc[0] == pytest.approx(-4.0)
    assert pairs["acc"].iloc[0] == 0.0


# --- label columns ---------------------------------------------------------

def test_label_columns_name_a_phenomenon_and_a_structural_type():
    """`family` and `grammar_type` carry the two axes, not constants.

    Before 2026-08-04 `family` was the walk tag (a per-grammar constant) and
    `grammar_type` duplicated `grammar`, so neither column could be grouped on.
    """
    from grammars.metadata import GRAMMAR_STRUCTURAL_TYPE

    for grammar in GRAMMARS:
        items = load_probe_items(grammar, "structural_probe_SELECT")
        families = {it["family"] for it in items}
        assert families == {"anaphoric_binding", "wh_movement",
                            "auxiliary_movement", "agreement"}, grammar
        assert {it["grammar_type"] for it in items} == {
            GRAMMAR_STRUCTURAL_TYPE[grammar]}, grammar
        # `grammar_type` must no longer restate the grammar name.
        assert grammar not in {it["grammar_type"] for it in items}


def test_rule_type_partitions_divergence_from_controls():
    """H'/L' wh items separate by which clause licenses them; H/P have no axis."""
    from grammars.metadata import (
        RULE_TYPE_CLAUSE_I_DIVERGENT,
        RULE_TYPE_CLAUSE_I_SHARED,
        RULE_TYPE_CLAUSE_II,
    )

    for grammar in ("Hprime", "Lprime"):
        items = load_probe_items(grammar, "structural_probe_SELECT")
        wh = [it for it in items if it["construction"] == "wh_movement"]
        other = [it for it in items if it["construction"] != "wh_movement"]

        assert all(it["rule_type"] is None for it in other), grammar
        assert {it["rule_type"] for it in wh} == {
            RULE_TYPE_CLAUSE_I_DIVERGENT,
            RULE_TYPE_CLAUSE_I_SHARED,
            RULE_TYPE_CLAUSE_II,
        }, grammar

        # clause (ii) is exactly the wh_ii subcase; clause (i) is exactly wh_i.
        by_subcase = {"wh_i": set(), "wh_ii": set()}
        for it in wh:
            by_subcase[it["subcase"]].add(it["rule_type"])
        assert by_subcase["wh_ii"] == {RULE_TYPE_CLAUSE_II}, grammar
        assert by_subcase["wh_i"] == {RULE_TYPE_CLAUSE_I_DIVERGENT,
                                      RULE_TYPE_CLAUSE_I_SHARED}, grammar

        # Both members of a pair land in the same bucket, or the pair is split
        # across rule types and the crosstab counts it twice.
        by_pair = {}
        for it in wh:
            by_pair.setdefault(it["pair_id"], set()).add(it["rule_type"])
        assert all(len(v) == 1 for v in by_pair.values()), grammar

    for grammar in ("H", "P"):
        items = load_probe_items(grammar, "structural_probe_SELECT")
        assert all(it["rule_type"] is None for it in items), grammar


def test_embedding_depth_is_populated_for_both_families(model):
    """The positional family's depth is no longer an all-null column."""
    for grammar in GRAMMARS:
        df = score_split(model, grammar, "structural_probe_SELECT")
        assert str(df["embedding_depth"].dtype) == "Int64", grammar
        assert df["embedding_depth"].notna().all(), grammar
        assert df["embedding_depth"].min() >= 0, grammar


def test_item_uid_is_a_stable_join_key():
    """`id` indexes the file; `item_uid` identifies the item."""
    for grammar in GRAMMARS:
        items = load_probe_items(grammar, "structural_probe_SELECT")
        uids = [it["item_uid"] for it in items]
        assert all(u is not None for u in uids), grammar
        assert len(set(uids)) == len(uids), grammar
        # Independent of position: reordering the file must not change it.
        assert uids[0] == f"{items[0]['pair_id']}/{items[0]['variant']}"


# --- write-time validation -------------------------------------------------

def _valid_frame():
    import pandas as pd

    rows = []
    for epoch in (1.0, 2.0):
        for i, variant in enumerate(("grammatical", "ungrammatical")):
            rows.append(dict(
                run_id="r", grammar="H", split="structural_probe_SELECT",
                id=f"H-structural_probe_SELECT-{i:05d}", epoch=epoch,
                pair_id="H-sp-00000", variant=variant,
                violation=None if i == 0 else "wrong_item",
                n_target_tokens=12, neg_logq=20.0 + i, neg_logq_crit=2.0 + i,
            ))
    return pd.DataFrame(rows)


def test_validate_accepts_a_well_formed_frame():
    from evaluation.score_pairs import validate_scores

    validate_scores(_valid_frame())


@pytest.mark.parametrize("break_it, message", [
    (lambda d: d.assign(neg_logq=-1.0), "neg_logq <= 0"),
    (lambda d: d.assign(neg_logq_crit=d.neg_logq + 1), "neg_logq_crit > neg_logq"),
    (lambda d: __import__("pandas").concat([d, d.iloc[[0]]]), "not unique"),
    (lambda d: d.iloc[1:], "item set changes between epochs"),
])
def test_validate_rejects(break_it, message):
    from evaluation.score_pairs import validate_scores

    with pytest.raises(ValueError, match=message):
        validate_scores(break_it(_valid_frame()))


def test_validate_rejects_unequal_lengths_on_index_aligned_pairs():
    """An in-place substitution that changed the token count is a real bug."""
    from evaluation.score_pairs import validate_scores

    frame = _valid_frame()
    frame.loc[frame.variant == "ungrammatical", "n_target_tokens"] = 13
    with pytest.raises(ValueError, match="n_target_tokens"):
        validate_scores(frame)


def test_validate_allows_unequal_lengths_on_relocation_pairs():
    """`wrong_site` moves a word to a site with different inflection, so the
    two members legitimately differ in length — 169 of the 2100 SELECT pairs do.
    An unconditional equality check would abort every run on correct data."""
    from evaluation.score_pairs import validate_scores

    frame = _valid_frame()
    frame.loc[frame.variant == "ungrammatical", "violation"] = "wrong_site"
    frame.loc[frame.variant == "ungrammatical", "n_target_tokens"] = 13
    validate_scores(frame)


# --- warmup migration ------------------------------------------------------

def test_resolved_warmup_matches_what_warmup_ratio_computed():
    """The migration must not move the schedule by a single step.

    `TrainingArguments.get_warmup_steps` is `ceil(total_steps * ratio)`; the
    resolved value has to equal it for every cell of the grid.
    """
    from transformers import TrainingArguments

    from models.train import planned_total_steps, resolve_warmup_steps

    for n_examples in (10_000, 100_000):
        for batch_size in (32, 64):
            for epochs in (2, 60):
                total = planned_total_steps(n_examples, batch_size, epochs)
                args = TrainingArguments(output_dir="/tmp", warmup_ratio=0.05)
                assert resolve_warmup_steps(0.05, total) == args.get_warmup_steps(total)


def test_pinned_warmup_is_cap_independent_and_matches_the_grid():
    """The production pin decouples warmup length from the epoch cap.

    `constant_with_warmup` does not anneal, so the cap reshapes the trajectory
    only through the warmup length. A `warmup_ratio` ties that length to the cap;
    the absolute pin holds it fixed, so training to any cap reproduces the first
    E epochs of the grid-selected schedule. `DEFAULT_WARMUP_STEPS` is exactly the
    grid's cap-60 warmup (= 3.0 epochs at the 100k/64 production corpus).
    """
    from models.train import (DEFAULT_WARMUP_STEPS, planned_total_steps,
                              resolve_warmup_steps)

    per_epoch = planned_total_steps(100_000, 64, 1)
    assert DEFAULT_WARMUP_STEPS == 3 * per_epoch                    # 3.0 epochs
    # The grid's cap-60 warmup under the 0.05 ratio is that same pin.
    grid_total = planned_total_steps(100_000, 64, 60)
    assert resolve_warmup_steps(0.05, grid_total) == DEFAULT_WARMUP_STEPS
    # The ratio path moves with the cap; the pin does not.
    assert resolve_warmup_steps(0.05, planned_total_steps(100_000, 64, 12)) \
        != resolve_warmup_steps(0.05, planned_total_steps(100_000, 64, 30))


def test_build_recipe_warmup_sentinel():
    """`--warmup-steps 0` opts back into the ratio; any positive value pins it."""
    import argparse

    from models import train as T

    base = dict(
        n_layer=T.DEFAULT_N_LAYER, n_embd=T.DEFAULT_N_EMBD, n_head=T.DEFAULT_N_HEAD,
        n_positions=T.DEFAULT_N_POSITIONS, dropout=T.DEFAULT_DROPOUT,
        learning_rate=T.DEFAULT_LEARNING_RATE, batch_size=T.DEFAULT_BATCH_SIZE,
        epochs=T.DEFAULT_EPOCHS, max_steps=None, warmup_ratio=T.DEFAULT_WARMUP_RATIO,
        weight_decay=T.DEFAULT_WEIGHT_DECAY, lr_scheduler=T.DEFAULT_LR_SCHEDULER,
        subset_size=None)
    assert T.build_recipe(argparse.Namespace(**base, warmup_steps=4689)).warmup_steps == 4689
    assert T.build_recipe(argparse.Namespace(**base, warmup_steps=0)).warmup_steps is None


def test_frozen_recipe_is_the_corrected_schedule():
    """The default CLI recipe is on-recipe: 12 epochs, warmup pinned, no deviations."""
    import argparse

    from models import train as T

    ns = argparse.Namespace(
        n_layer=T.DEFAULT_N_LAYER, n_embd=T.DEFAULT_N_EMBD, n_head=T.DEFAULT_N_HEAD,
        n_positions=T.DEFAULT_N_POSITIONS, dropout=T.DEFAULT_DROPOUT,
        learning_rate=T.DEFAULT_LEARNING_RATE, batch_size=T.DEFAULT_BATCH_SIZE,
        epochs=T.DEFAULT_EPOCHS, max_steps=None, warmup_ratio=T.DEFAULT_WARMUP_RATIO,
        warmup_steps=T.DEFAULT_WARMUP_STEPS, weight_decay=T.DEFAULT_WEIGHT_DECAY,
        lr_scheduler=T.DEFAULT_LR_SCHEDULER, subset_size=None)
    assert T.DEFAULT_EPOCHS == 12
    assert T.FROZEN_RECIPE["warmup_steps"] == T.DEFAULT_WARMUP_STEPS
    assert T.recipe_deviations(T.build_recipe(ns)) == {}


def test_planned_total_steps_matches_trainer_arithmetic():
    from models.train import planned_total_steps

    # 100k examples at bs=64 -> ceil(100000/64) = 1563 batches/epoch.
    assert planned_total_steps(100_000, 64, 1) == 1563
    assert planned_total_steps(100_000, 64, 60) == 1563 * 60
    # `max_steps` wins outright, as it does in Trainer.
    assert planned_total_steps(100_000, 64, 60, max_steps=500) == 500
    # A batch larger than the split still runs one step per epoch.
    assert planned_total_steps(10, 64, 3) == 3
