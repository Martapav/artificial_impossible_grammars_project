"""Per-item model surprisal on the structural probes (and on corpus splits).

Emits one row per *item*, never per pair: pairing is `aggregate.py`'s job, so
the raw scores stay reusable for the string-level `R` analysis, which has no
pairs at all. Two quantities per item:

* ``neg_logq``      — whole-string ``−log q(x)`` in nats, summed over the targets
  ``m₁ … m_T [EOS]``.
* ``neg_logq_crit`` — the same sum restricted to the item's critical span, the
  one position the construction's rule targets.

Which of the two is the valid read depends on the *geometry of the violation*,
not on which looks better; `aggregate.py` assigns it from the recorded
``violation`` field. See `docs/RECIPE_SELECTION.md` §Step 3.

Conventions (these are what make the numbers commensurable with `b_tok`)
-----------------------------------------------------------------------
A sentence is ``[BOS] m₁ … m_T [EOS]``. Logits at position ``t`` score the token
at ``t+1``, so the targets are ``m₁ … m_T`` and ``[EOS]``; ``[BOS]`` is context
only and never a target. This is exactly `analysis/ngram_baseline.py`'s
convention, and the two are only subtractable because of it — so
``n_target_tokens`` is computed identically here (``len(ids) − 1``).

Natural log throughout (`F.log_softmax` is nats, `ℓ(x)` is `math.log`); asserted
by `assert_nats_convention`. `SURPRISAL_REFERENCE_POINTS.md` §6.5.

Joining scores back to generation records
-----------------------------------------
On ``item_uid``, never on ``id``. ``id`` is ``{grammar}-{split}-{index}`` — a
dense index into the split file as it was written, so it identifies a LINE, not
an item; rebuild the corpus and every previously written score joins onto the
wrong record with no error anywhere. ``item_uid`` is assigned by the probe
builder from the pair it belongs to and does not move when the file is rebuilt.
`models/grid_search.py` records each split file's SHA-256 in the run manifest,
which is what makes a rebuild between two invocations detectable at all.

Label columns
-------------
``family`` is the PHENOMENON family (anaphoric_binding / wh_movement /
auxiliary_movement / agreement) and ``grammar_type`` is the STRUCTURAL type
(hierarchical / positional). Both used to carry something else — the sidecar's
walk tag and the grammar name respectively — so neither could be grouped on.
``rule_type`` names which of H′/L′'s two wh clauses licenses an item and whether
the two grammars disagree about it; it is null for H, P and every non-wh
construction. `_derive_labels` recovers all three from sidecars that predate
them, so none of this needs a corpus rebuild to become readable.

Critical-span indexing
----------------------
``targets[].model_token_span`` is ``[start, end)`` indexed into the tokenized
sequence **including the leading ``[BOS]``**, so ``start >= 1`` always and the
span indexes the same array as ``input_ids``. Verified against the corpora: on
every ``wrong_item`` / ``agreement`` pair — the ones whose members are index
aligned — the tokens under the span are exactly the tokens that differ between
the grammatical and ungrammatical member. `check_span_convention` re-runs that
check on demand; `tests/test_score_pairs.py` pins it.

The surprisal of input position ``j`` lives at index ``j−1`` of the per-target
array (the ``t``/``t+1`` offset). Getting this off by one is silent — it would
score a neighbouring morpheme and still produce plausible numbers — so it is
asserted rather than commented.

Usage
-----
    # from a saved checkpoint bundle (models/train.py writes final_model/)
    python -m evaluation.score_pairs --model <run_dir>/final_model \\
        --grammar H --split structural_probe_IOD --out results/eval/H_iod.parquet

    # every grammar, both probe splits, one file
    python -m evaluation.score_pairs --model <dir> --grammar all \\
        --split structural_probe_IOD structural_probe_OOD --out scores.parquet

    # programmatic: an in-memory model, e.g. from a TrainerCallback mid-run
    from evaluation.score_pairs import score_split
    df = score_split(model, "H", "structural_probe_IOD", device="cuda")
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F

from models.tokenize import tokenizer
from models.dataload import CORPORA_ROOT as CORPORA_DIR
from models.dataload import corpus_provenance, require_set_scoped
from evaluation.provenance import describe, write_parquet

GRAMMARS = ("H", "P", "Hprime", "Lprime")
# All three minimal-pair splits. SELECT belongs here even though it must never
# be REPORTED: leaving it out made a bare invocation silently skip it, which is
# a scoring gap, not a safeguard. What keeps selection out of the reported
# numbers is the corpus-level disjointness of SELECT from IOD/OOD
# (`corpusgen/probe_checks.py`) and what `aggregate.py` chooses to summarize —
# not the absence of scores.
PROBE_SPLITS = ("structural_probe_IOD", "structural_probe_OOD",
                "structural_probe_SELECT")

BOS_ID = tokenizer.convert_tokens_to_ids("[BOS]")
EOS_ID = tokenizer.convert_tokens_to_ids("[EOS]")
PAD_ID = tokenizer.convert_tokens_to_ids("[PAD]")

# Annotation columns carried through from the sidecar so `aggregate.py` never
# has to re-open the corpora. `violation` is load-bearing: it selects the
# criterion. `pair_id` pairs items; `item_uid` is the join key back to the
# generation records (`pair_id` repeats across IOD/OOD and must never be used as
# a join key on its own; `id` is a dense index into the split file and is only
# valid while that file is byte-identical — see `_derive_labels`).
_PASSTHROUGH = (
    "item_uid", "pair_id", "variant", "violation", "construction", "subcase",
    "family", "grammar_type", "rule_type", "length", "n_morphemes",
    "embedding_depth",
)

# Columns that must come out of the parquet as a nullable integer. `embedding_depth`
# is genuinely absent for the pre-2026-08-04 positional corpora, and pandas turns
# an object column of ints-and-Nones into float64 (or, if every value is None,
# into all-object) — either way the depth axis stops comparing equal to an int
# and every `== 2` filter silently returns nothing.
_INT64_COLUMNS = ("length", "n_morphemes", "embedding_depth", "n_target_tokens",
                  "n_crit_tokens")


# --- corpus loading --------------------------------------------------------

def load_probe_items(grammar: str, split: str,
                     corpora_dir: Path = CORPORA_DIR) -> List[dict]:
    """Items for one grammar/split: sentence + sidecar annotations, in file order.

    Reads both ``<grammar>.jsonl`` and ``<grammar>.meta.jsonl`` and asserts they
    agree on ``id`` and ``sentence`` rather than trusting the sidecar alone — a
    silent sidecar drift would attach the wrong critical span to a sentence and
    every downstream number would still look reasonable.
    """
    base = Path(corpora_dir) / split
    text_path, meta_path = base / f"{grammar}.jsonl", base / f"{grammar}.meta.jsonl"
    if not text_path.exists():
        raise FileNotFoundError(text_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)

    with text_path.open(encoding="utf-8") as fh:
        text = [json.loads(line) for line in fh if line.strip()]
    with meta_path.open(encoding="utf-8") as fh:
        meta = [json.loads(line) for line in fh if line.strip()]

    if len(text) != len(meta):
        raise ValueError(
            f"{grammar}/{split}: {len(text)} sentences but {len(meta)} sidecar "
            f"records; the sidecar is out of sync with the corpus."
        )
    items = []
    for t, m in zip(text, meta):
        if t["id"] != m["id"] or t["sentence"] != m["sentence"]:
            raise ValueError(
                f"{grammar}/{split}: sidecar drift at {t['id']!r} — the sidecar "
                f"describes a different item than the corpus line."
            )
        item = {"id": m["id"], "sentence": m["sentence"],
                "targets": m.get("targets") or []}
        for key in _PASSTHROUGH:
            item[key] = m.get(key)
        _derive_labels(item, m, grammar)
        items.append(item)
    _fill_rule_type(items, grammar)
    return items


def _derive_labels(item: dict, meta: dict, grammar: str) -> None:
    """Fill the four label columns from whatever the sidecar actually carries.

    Three of these were being written wrong, and one was being written empty:

    * ``family`` took the sidecar's *walk tag* ("tree"/"tokens"), so the column
      was a per-grammar constant and named no phenomenon.
    * ``grammar_type`` took the grammar label, duplicating the ``grammar``
      column exactly.
    * ``rule_type`` did not exist, so nothing downstream could separate the
      H′/L′ divergence items from their controls.
    * ``embedding_depth`` was null for every positional item.

    Corrected here rather than only in the generator because every value is
    recoverable from sidecars that already exist on disk — ``construction`` gives
    the phenomenon family, the walk tag gives the structural type, the stamped
    ``divergence`` gives the rule type, and ``n_embedded_clauses`` is the
    positional depth exactly (``metadata._token_features``). Fixing only the
    generator would have made every one of these columns require a full corpus
    rebuild to become readable. A sidecar that already carries the corrected
    field wins; this is the fallback, not the definition.
    """
    from grammars import metadata as M

    if meta.get("phenomenon_family") is not None:
        item["family"] = meta["phenomenon_family"]
    else:
        item["family"] = M.phenomenon_family(meta.get("construction"))

    item["grammar_type"] = (meta.get("structural_type")
                            or M.structural_type(meta.get("family") or grammar))

    if item.get("embedding_depth") is None:
        item["embedding_depth"] = meta.get("n_embedded_clauses")

    if item.get("item_uid") is None and meta.get("pair_id") is not None:
        item["item_uid"] = f"{meta['pair_id']}/{meta.get('variant')}"

    # Carried for the rule-type fill below; not a passthrough column of its own.
    item["_divergence"] = meta.get("divergence")


def _fill_rule_type(items: List[dict], grammar: str) -> None:
    """Rule type per PAIR, from the grammatical member's clause-(i) verdicts.

    Pair-level and not item-level on purpose. Both members carry stamped
    verdicts, but only the grammatical member was fronted from a *licensed*
    site, so only its verdicts answer "which clause licenses this configuration".
    The ungrammatical member's own divergence flag describes a site the grammar
    rejects outright, and letting it label its own row would put the two halves
    of a minimal pair in different rule-type buckets — which is precisely the
    grouping the field exists to make possible.
    """
    from grammars import metadata as M

    by_pair: dict = {}
    for item in items:
        if item.get("pair_id") is not None and item.get("variant") == "grammatical":
            by_pair[item["pair_id"]] = item.pop("_divergence", None)
    for item in items:
        item.pop("_divergence", None)
        if item.get("rule_type") is None:
            item["rule_type"] = M.licensing_rule_type(
                grammar, item.get("subcase"), by_pair.get(item.get("pair_id")))


# --- convention checks -----------------------------------------------------

def assert_nats_convention() -> None:
    """`F.log_softmax` is natural log. Cheap, and a base mismatch is silent."""
    logits = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    lp = F.log_softmax(logits, dim=-1)[0, 0].item()
    expected = -0.6931471805599453          # log(1/2) in nats
    # 1e-6 is far below any plausible base slip (log2 vs ln is a factor of 1.44)
    # and far above float round-off, which is what a tighter bound would trip on.
    if abs(lp - expected) > 1e-6:
        raise AssertionError(
            f"log_softmax returned {lp}, expected {expected} nats; the model "
            f"side and ℓ(x)/b_tok are not on the same log base."
        )


def check_span_convention(grammar: str, split: str,
                          corpora_dir: Path = CORPORA_DIR) -> dict:
    """Confirms ``model_token_span`` indexes the BOS-inclusive token sequence.

    Uses the index-aligned violations (``wrong_item`` / ``agreement``), where the
    two members differ at exactly the position the span names. Under the correct
    convention the span tokens differ between members and everything else is
    identical; under an off-by-one they would match. Returns a report rather
    than raising, so the CLI can print it.
    """
    items = load_probe_items(grammar, split, corpora_dir)
    encoded = tokenizer([it["sentence"] for it in items])["input_ids"]
    by_pair: dict = {}
    for item, ids in zip(items, encoded):
        by_pair.setdefault(item["pair_id"], {})[item["variant"]] = (item, ids)

    checked = span_differs = out_of_range = 0
    for members in by_pair.values():
        if len(members) != 2:
            continue
        (g_item, g_ids), (u_item, u_ids) = members["grammatical"], members["ungrammatical"]
        for item, ids in ((g_item, g_ids), (u_item, u_ids)):
            for tgt in item["targets"]:
                start, end = tgt["model_token_span"]
                if not (1 <= start < end <= len(ids)):
                    out_of_range += 1
        if u_item["violation"] not in ("wrong_item", "agreement", "no_substitution"):
            continue
        g_spans = [t["model_token_span"] for t in g_item["targets"]]
        u_spans = [t["model_token_span"] for t in u_item["targets"]]
        if g_spans != u_spans or len(g_ids) != len(u_ids):
            continue
        checked += 1
        if any(g_ids[s:e] != u_ids[s:e] for s, e in g_spans):
            span_differs += 1
    return {
        "grammar": grammar, "split": split,
        "aligned_pairs_checked": checked,
        "span_tokens_differ": span_differs,
        "spans_out_of_range": out_of_range,
        "ok": checked > 0 and span_differs == checked and out_of_range == 0,
    }


# --- scoring ---------------------------------------------------------------

@torch.no_grad()
def score_sentences(
    model,
    sentences: Sequence[str],
    spans: Sequence[Sequence[Sequence[int]]],
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> List[dict]:
    """``−log q`` per sentence, whole-string and restricted to `spans`.

    `spans` is one list of ``[start, end)`` pairs per sentence (empty where the
    item has no critical position, e.g. neutral corpus items).
    """
    assert_nats_convention()
    was_training = model.training
    model.eval()
    model.to(device)

    rows: List[dict] = []
    for lo in range(0, len(sentences), batch_size):
        chunk = list(sentences[lo:lo + batch_size])
        chunk_spans = spans[lo:lo + batch_size]
        enc = tokenizer(chunk, padding=True, return_tensors="pt")
        ids = enc["input_ids"].to(device)
        mask = enc["attention_mask"].to(device)

        logits = model(input_ids=ids, attention_mask=mask).logits
        # .float() so a bf16/fp16 checkpoint does not lose precision in the
        # softmax; the sums below run over ~25 terms and are read to 4 decimals.
        logprobs = F.log_softmax(logits.float(), dim=-1)

        # surprisal[b, t] is the surprisal of input position t+1 (the t/t+1
        # offset); position 0 is [BOS] and is never a target.
        targets = ids[:, 1:]
        gathered = logprobs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        surprisal = -gathered
        valid = mask[:, 1:].bool()
        surprisal = surprisal.masked_fill(~valid, 0.0)

        n_targets = valid.sum(dim=1)
        totals = surprisal.sum(dim=1)

        for i, item_spans in enumerate(chunk_spans):
            n_tok = int(n_targets[i].item())
            crit = 0.0
            crit_tokens = 0
            for start, end in item_spans:
                if not (1 <= start < end <= n_tok + 1):
                    raise IndexError(
                        f"critical span [{start}, {end}) outside the "
                        f"{n_tok + 1}-token sequence {chunk[i]!r}"
                    )
                # input position j -> surprisal index j-1
                crit += float(surprisal[i, start - 1:end - 1].sum().item())
                crit_tokens += end - start
            rows.append({
                "n_target_tokens": n_tok,
                "neg_logq": float(totals[i].item()),
                "neg_logq_crit": crit if item_spans else float("nan"),
                "n_crit_tokens": crit_tokens,
            })

    if was_training:
        model.train()
    return rows


def score_split(
    model,
    grammar: str,
    split: str,
    *,
    device: str = "cpu",
    batch_size: int = 64,
    corpora_dir: Path = CORPORA_DIR,
    run_id: str | None = None,
    epoch: float | None = None,
) -> "pd.DataFrame":
    """Per-item scores for one grammar/split, keyed by the stable item ``id``.

    `run_id` / `epoch` are stamped through so a grid sweep can concatenate every
    cell's scores into one long table and `aggregate.py` can group on them.
    """
    import pandas as pd

    items = load_probe_items(grammar, split, corpora_dir)
    scored = score_sentences(
        model,
        [it["sentence"] for it in items],
        [[t["model_token_span"] for t in it["targets"]] for it in items],
        device=device, batch_size=batch_size,
    )

    rows = []
    for item, s in zip(items, scored):
        row = {"id": item["id"], "grammar": grammar, "split": split}
        if run_id is not None:
            row["run_id"] = run_id
        if epoch is not None:
            row["epoch"] = epoch
        for key in _PASSTHROUGH:
            row[key] = item[key]
        row.update(s)
        row["neg_logq_per_token"] = (
            s["neg_logq"] / s["n_target_tokens"] if s["n_target_tokens"] else float("nan")
        )
        rows.append(row)
    return _typed(pd.DataFrame(rows))


def _typed(frame: "pd.DataFrame") -> "pd.DataFrame":
    """Cast the count columns to pandas' nullable ``Int64``.

    ``astype("Int64")`` refuses a float column with a fractional part, so a
    count that has silently become non-integral raises here instead of being
    truncated into the parquet.
    """
    for column in _INT64_COLUMNS:
        if column in frame.columns:
            frame[column] = frame[column].astype("Int64")
    return frame


# --- write-time validation -------------------------------------------------

# Violations whose two members are index-aligned edits of one string: the
# ungrammatical member substitutes in place, so both members tokenize to the
# same length. `n_target_tokens` MUST match within these pairs, and a mismatch
# means the pair is not the minimal pair it claims to be — which would make
# `aggregate.py`'s critical-span margin a comparison between two different
# positions.
_INDEX_ALIGNED_VIOLATIONS = ("wrong_item", "agreement", "no_substitution")


def validate_scores(frame: "pd.DataFrame", *, context: str = "") -> None:
    """Every invariant the scores must satisfy, checked before they are written.

    Raises ``ValueError`` listing every failure, not just the first: a run that
    has broken one of these has usually broken several, and finding out one at a
    time costs a training run each.

    On ``n_target_tokens`` — the requested invariant was "constant within
    pair_id", unconditionally. It is not, and cannot be. For the RELOCATION
    violations (``wrong_site``, ``unlicensed_site``, ``positional_*``) the
    ungrammatical member moves a word to a different site, where it takes
    different inflection and so a different morpheme count; on the SELECT probe
    that is 169 of 2100 pairs, all of them ``wrong_site``. `aggregate.py` is
    built around this — it assigns those violations the whole-string criterion
    precisely because a span comparison is not well posed across a relocation,
    and it reports `margin_b1` to control for the length difference rather than
    dropping the pairs. An unconditional assertion here would abort every run on
    correct data. The check is therefore scoped to the index-aligned violations,
    where equal length is a real guarantee and its failure is a real bug.
    """
    problems: List[str] = []
    tag = f"{context}: " if context else ""

    n = len(frame)
    if n == 0:
        raise ValueError(f"{tag}no rows to write")

    bad = frame["neg_logq"] <= 0
    if bad.any():
        problems.append(
            f"{int(bad.sum())}/{n} rows have neg_logq <= 0; −log q is a sum of "
            f"positive surprisals and can only be non-positive if the sign or "
            f"the mask is wrong (worst: {frame.loc[bad, 'neg_logq'].min()})"
        )

    crit = frame["neg_logq_crit"]
    over = crit.notna() & (crit > frame["neg_logq"] + 1e-6)
    if over.any():
        problems.append(
            f"{int(over.sum())}/{n} rows have neg_logq_crit > neg_logq; the "
            f"critical span is a SUBSET of the scored targets, so its sum cannot "
            f"exceed the whole-string sum — the span is indexing outside the "
            f"target array"
        )

    key = [c for c in ("run_id", "grammar", "split", "id", "epoch")
           if c in frame.columns]
    dup = frame.duplicated(key, keep=False)
    if dup.any():
        offenders = frame.loc[dup, key].head(3).to_dict("records")
        problems.append(
            f"{int(dup.sum())}/{n} rows are not unique on {key}; an item was "
            f"scored twice in one epoch (e.g. {offenders})"
        )

    if "epoch" in frame.columns and frame["epoch"].notna().any():
        run_key = ["run_id"] if "run_id" in frame.columns else []
        for run, sub in frame.groupby(run_key + ["grammar", "split"], dropna=False):
            per_epoch = {e: frozenset(g["id"]) for e, g in sub.groupby("epoch")}
            sets = set(per_epoch.values())
            if len(sets) > 1:
                sizes = {e: len(s) for e, s in per_epoch.items()}
                problems.append(
                    f"{run}: the scored item set changes between epochs "
                    f"({sizes}); the epochs are not comparable and every "
                    f"per-item learning curve has a hole in it"
                )

    if "pair_id" in frame.columns and frame["pair_id"].notna().any():
        aligned = frame[frame["violation"].isin(_INDEX_ALIGNED_VIOLATIONS)]
        pairs = frame[frame["pair_id"].isin(aligned["pair_id"])]
        group = [c for c in ("run_id", "grammar", "split", "epoch", "pair_id")
                 if c in pairs.columns]
        if group:
            spread = pairs.groupby(group)["n_target_tokens"].nunique()
            broken = spread[spread > 1]
            if len(broken):
                problems.append(
                    f"{len(broken)} index-aligned pair(s) differ in "
                    f"n_target_tokens (e.g. {list(broken.index[:3])}); an "
                    f"in-place substitution changed the token count, so the "
                    f"critical-span margin is comparing two different positions"
                )

    if problems:
        raise ValueError(
            f"{tag}probe scores failed validation:\n  - "
            + "\n  - ".join(problems)
        )


def score_many(
    model,
    grammars: Iterable[str] = GRAMMARS,
    splits: Iterable[str] = PROBE_SPLITS,
    **kwargs,
) -> "pd.DataFrame":
    import pandas as pd
    frames = [score_split(model, g, s, **kwargs) for g in grammars for s in splits]
    return _typed(pd.concat(frames, ignore_index=True))


# --- checkpoint loading ----------------------------------------------------

def load_model(path: Path):
    """Loads a saved checkpoint bundle (`models/train.py` writes `final_model/`).

    Asserts the checkpoint's vocabulary matches the current tokenizer: a lexicon
    change between training and scoring would silently reindex every token.
    """
    from transformers import GPT2LMHeadModel

    model = GPT2LMHeadModel.from_pretrained(str(path))
    if model.config.vocab_size != len(tokenizer):
        raise ValueError(
            f"checkpoint vocab_size {model.config.vocab_size} != current "
            f"tokenizer size {len(tokenizer)}; the lexicon changed since "
            f"training and the token ids no longer mean the same thing."
        )
    return model


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


# --- CLI -------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=None,
                        help="checkpoint directory (e.g. <run_dir>/final_model)")
    parser.add_argument("--grammar", nargs="+", default=["all"])
    parser.add_argument("--split", nargs="+", default=list(PROBE_SPLITS),
                        help="any corpus split; probe splits give pairs, corpus "
                             "splits give the string-level scores for R")
    parser.add_argument("--out", type=Path, default=None, help="output .parquet")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--run-id", default=None, help="stamped into every row")
    parser.add_argument("--corpora-dir", type=Path, default=CORPORA_DIR)
    parser.add_argument("--check-spans", action="store_true",
                        help="verify the span convention on the probe splits and exit")
    parser.add_argument("--allow-unscoped-out", action="store_true",
                        help="permit an --out path that does not name the corpus set")
    args = parser.parse_args()
    if args.out is not None:
        args.out = require_set_scoped(args.out, args.corpora_dir,
                                      override=args.allow_unscoped_out)

    grammars = GRAMMARS if args.grammar == ["all"] else tuple(args.grammar)

    if args.check_spans:
        ok = True
        for g in grammars:
            for s in args.split:
                rep = check_span_convention(g, s, args.corpora_dir)
                ok = ok and rep["ok"]
                print(f"{rep['grammar']:8s} {rep['split']:22s} "
                      f"aligned_pairs={rep['aligned_pairs_checked']:4d} "
                      f"span_tokens_differ={rep['span_tokens_differ']:4d} "
                      f"out_of_range={rep['spans_out_of_range']:3d}  "
                      f"{'OK' if rep['ok'] else 'FAIL'}")
        raise SystemExit(0 if ok else 1)

    if args.model is None or args.out is None:
        parser.error("--model and --out are required unless --check-spans is given")

    device = resolve_device(args.device)
    model = load_model(args.model)
    print(f"model: {args.model}  device: {device}")

    df = score_many(model, grammars, args.split, device=device,
                    batch_size=args.batch_size, corpora_dir=args.corpora_dir,
                    run_id=args.run_id)
    validate_scores(df, context=str(args.out))
    # The scores are only interpretable against the corpus they were computed
    # on, and every downstream join key is positional — see evaluation/
    # provenance.py. Stamp the corpus identity into the file itself.
    provenance = corpus_provenance(grammars, args.split, root=args.corpora_dir)
    write_parquet(df, args.out, provenance)
    print(f"wrote {args.out}  ({len(df)} rows)")
    print(f"  corpus: {describe(provenance)}")
    for (g, s), sub in df.groupby(["grammar", "split"]):
        print(f"  {g:8s} {s:22s} n={len(sub):5d} "
              f"mean −log q={sub.neg_logq.mean():8.3f} "
              f"per-token={sub.neg_logq_per_token.mean():.4f}")


if __name__ == "__main__":
    main()
