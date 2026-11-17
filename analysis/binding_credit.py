"""Binding forced-draw credit — the entropy of draws that binding marginalizes.

On the binding constructions ``ℓ_raw(x)`` (``grammars/tracing.py``) systematically
OVER-charges relative to ``−log p_corpus(x)``, so the differential surprisal
``Δ = neg_logq − ℓ_corpus`` (``analysis/differential_surprisal.py``) comes out
*negative* — an impossible KL — concentrated entirely in
``anaphoric_binding_{pron,refl}``. Two per-``(grammar, subtype)`` constants
account for it, both measured here on the real generator (the shipped corpora are
never rewritten):

1. **Target-block marginalization.** The binding transform coindexes the target
   nominal to its antecedent — ``grammars/grammar_H/transforms.py:_coindex`` (tree
   family H, H′) / ``grammars/grammar_P/transforms.py:_coindex`` (token family
   P, L′) — overwriting its head lexeme + INFL number and stripping its
   determiner/modifiers, then substitutes a *deterministic* CAT1PRON
   (``cat1pron_item`` draws no variate). The target's original block draws — the
   head-lexeme ``pick`` (Zipf), the CAT2 modifier loop, the INFL-number
   ``choice`` — are charged in ``ℓ_raw`` but never reach the surface: a latent
   the transform marginalizes. ``ℓ_raw`` is therefore an *upper bound* on
   ``−log p_corpus`` (``SURPRISAL_REFERENCE_POINTS.md`` §6.6), and the exact
   correction subtracts the entropy of those draws.

2. **Geometry-resample log Z.** Binding generation redraws the base until it
   admits the subtype's forced geometry (refl: an embedded transitive nucleus,
   rule A′; P/L′: the offset + antecedent-ordinal geometry), ``rewind``-ing
   failed passes (``grammars/grammar_*/generate.py:_one_item``). The surviving
   base is drawn from ``p_raw(· | geometry)``, but ``ℓ_raw`` charges the
   *unconditional* cost, over-charging by exactly ``−log P(geometry)``. The
   acceptance correction ``correction_c`` (``analysis/entropy_baseline.py``) does
   NOT capture this — it reweights only at the label+length grain, and the
   geometry resample is a deeper *within-item* rejection
   (``docs/SURPRISAL_REFERENCE_POINTS.md`` §5, "Binding forced-draw credit").

    credit(grammar, subtype) = E[block cost] + (−log P(geometry | screened base))

The block is read from the target node/token *before* ``_coindex`` overwrites it
and priced from the generator's own draw distributions (both families share one
``lexicon.json``, one ZIPF config, ``P_CAT2`` and the binary INFL-number draw, so
the pricing is one formula). The log Z is the acceptance rate of the loop's own
geometry predicate (``has_refl_licensing_nucleus`` / ``binding.has_licensed_pair``),
tallied over a real generation run.

Usage:
    python -m analysis.binding_credit                 # emit for all four grammars
    python -m analysis.binding_credit --grammar H --n 30000
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from grammars.tracing import TracedRandom  # noqa: E402
from lexicon.config import ZIPF_Q, ZIPF_S  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent / "data"
GRAMMARS = ("H", "P", "Hprime", "Lprime")
SUBTYPES = ("anaphoric_binding_refl", "anaphoric_binding_pron")

# Both families share lexicon.json + ZIPF config, and both draw the CAT2 loop at
# P_CAT2 and INFL1_number uniformly over two values, so the block price is one
# formula. Asserted against each grammar's own constants in `_family_constants`.
_LOG2 = math.log(2.0)                       # INFL1_number ∈ {"3","4"}


# ─────────────────────────────────────────────────────────────────────────────
# Block cost: entropy of the marginalized target-block draws
# ─────────────────────────────────────────────────────────────────────────────

def _zipf_neglogp(item: dict, pool: list) -> float:
    """−log p of drawing ``item`` from ``pool`` under ``grammars/*/lexicon.pick``.

    Mirrors ``pick`` exactly: open-class pools (``zipf_rank > 0``) are Zipf–
    Mandelbrot weighted; closed-class pools are uniform. The probability is of
    the drawn VALUE (``pick`` sums duplicate slots, but lexicon items are unique
    here), matching the tracer's ``choices`` convention.
    """
    if pool and pool[0].get("zipf_rank", 0) > 0:
        w = [1.0 / (it["zipf_rank"] + ZIPF_Q) ** ZIPF_S for it in pool]
        z = math.fsum(w)
        wi = 1.0 / (item["zipf_rank"] + ZIPF_Q) ** ZIPF_S
        return -math.log(wi / z)
    return -math.log(1.0 / len(pool))


def _block_cost(head_lex: dict, modifier_lexes: List[dict],
                cat1_pool: list, cat2_pool: list, p_cat2: float) -> float:
    """−log p of the draws ``_coindex`` renders latent on one target nominal.

    Head-lexeme ``pick`` + INFL-number ``choice`` (always), plus the CAT2 loop
    for a countable head: ``k`` continuation Bernoullis, one terminating
    Bernoulli, and ``k`` modifier ``pick`` s (``build_type1`` / ``build_sentence``).
    The determiner (CAT4) is deterministic given number, and the gender is
    lexeme-inherent — neither draws a variate, so neither enters the credit.
    """
    cost = _zipf_neglogp(head_lex, cat1_pool) + _LOG2
    if head_lex.get("countability") == "countable":
        k = len(modifier_lexes)
        cost += k * (-math.log(p_cat2)) + (-math.log(1.0 - p_cat2))
        for m in modifier_lexes:
            cost += _zipf_neglogp(m, cat2_pool)
    return cost


# ─────────────────────────────────────────────────────────────────────────────
# Per-family generation harness (spy on _coindex; tally the geometry predicate)
# ─────────────────────────────────────────────────────────────────────────────

def _tree_family(module_grammar: str):
    """Import hooks for the tree family (H, H′): reader, modules, pools, predicate."""
    from grammars.grammar_H import transforms as tr
    from grammars.grammar_H.lexicon import cat1_items, load
    from grammars.grammar_H import generate as gen_H
    from grammars.grammar_H.rules import P_CAT2
    if module_grammar == "H":
        gen = gen_H
    else:
        from grammars.grammar_H_prime import generate as gen  # type: ignore
    lex = load()
    pools = (cat1_items(lex), lex["cat2"], P_CAT2)

    def read_target(args) -> Tuple[dict, List[dict]]:
        target = args[0]                              # _coindex(target, anchor)
        head = next(c for c in target.children if c.role == "head")
        mods = [c.lex for c in target.children if c.role == "modifier"]
        return head.lex, mods

    # The refl geometry loop lives in grammar_H._one_item (H′ reuses it via
    # `_h_one_item`), so its `has_refl_licensing_nucleus` resolves in gen_H's
    # namespace for BOTH grammars — patch it there, not on the H′ driver.
    return tr, gen, lex, pools, read_target, (gen_H, "has_refl_licensing_nucleus")


def _token_family(module_grammar: str):
    """Import hooks for the token family (P, L′)."""
    from grammars.grammar_P import transforms as tr
    from grammars.grammar_P.lexicon import cat1_items, load
    from grammars.grammar_P import binding
    from grammars.grammar_P.rules import P_CAT2
    if module_grammar == "P":
        from grammars.grammar_P import generate as gen
    else:
        from grammars.grammar_L_prime import generate as gen  # type: ignore
    lex = load()
    pools = (cat1_items(lex), lex["cat2"], P_CAT2)

    def read_target(args) -> Tuple[dict, List[dict]]:
        toks, _i, j = args                            # _coindex(toks, anchor, target)
        head_tok = toks[j]
        mods = []
        k = j + 1
        while k < len(toks) and toks[k].cat in ("CAT4", "CAT2") \
                and toks[k].clause_id == head_tok.clause_id:
            if toks[k].cat == "CAT2":                 # CAT4 is deterministic
                mods.append(toks[k].lex)
            k += 1
        return head_tok.lex, mods

    return tr, gen, lex, pools, read_target, ("binding", binding)


def measure_credit(grammar: str, n: int = 30000, seed: int = 20260912
                   ) -> pd.DataFrame:
    """Estimate ``credit`` for both binding subtypes of ``grammar`` on ``n``
    fresh generated items. Returns rows ``(grammar, subtype, block, block_se,
    logz, credit, n_block, n_screened)``."""
    import random

    is_tree = grammar in ("H", "Hprime")
    tr, gen, lex, pools, read_target, predicate = (
        _tree_family(grammar) if is_tree else _token_family(grammar))
    cat1_pool, cat2_pool, p_cat2 = pools

    # geometry log Z tally: screened bases -> admitted, per subtype
    geo: Dict[str, List[int]] = {s: [0, 0] for s in SUBTYPES}   # [calls, trues]

    # --- spy on _coindex: price the target block before it is overwritten ---
    # _coindex fires exactly once per NON-skipped binding item, in generation
    # order (a *_skipped item returns before it), so the ordered list of block
    # costs aligns 1:1 with the ordered non-skipped binding phenomena.
    blocks_in_order: List[float] = []
    orig_coindex = tr._coindex

    def spy_coindex(*args):
        head_lex, mods = read_target(args)
        blocks_in_order.append(
            _block_cost(head_lex, mods, cat1_pool, cat2_pool, p_cat2))
        return orig_coindex(*args)

    tr._coindex = spy_coindex

    # --- tally the loop's geometry predicate (its acceptance rate is the log Z) ---
    if is_tree:
        # refl-only: has_refl_licensing_nucleus is called once per screened refl
        # base in the loop, and NOT by the transform (which calls the underlying
        # refl_licensing_nuclei directly). Pron has no geometry check -> log Z 0.
        pred_host, pred_name = predicate
        orig_pred = getattr(pred_host, pred_name)

        def spy_pred(tree):
            out = orig_pred(tree)
            geo["anaphoric_binding_refl"][0] += 1
            geo["anaphoric_binding_refl"][1] += 1 if out else 0
            return out
        setattr(pred_host, pred_name, spy_pred)
    else:
        # P/L′: has_licensed_pair(cats, subtype) is called once per screened base,
        # carrying the subtype, and NOT by the transform (which calls
        # licensed_pairs). Both subtypes resample on geometry.
        _, binding_mod = predicate
        orig_pred = binding_mod.has_licensed_pair

        def spy_pred(cats, subtype, *a, **k):
            out = orig_pred(cats, subtype, *a, **k)
            key = ("anaphoric_binding_refl" if subtype == "Refl"
                   else "anaphoric_binding_pron")
            geo[key][0] += 1
            geo[key][1] += 1 if out else 0
            return out
        binding_mod.has_licensed_pair = spy_pred

    # Generate item-by-item (NOT gen.generate_with_metadata, which returns a list
    # of n whole parse trees at once and blows memory at this n); each grammar's
    # `_one_item(rng, lex)` is the per-item core the eager builder loops over, and
    # keeps only one tree alive. We keep the phenomena, discard the trees.
    subtypes_in_order: List[str] = []
    rng = random.Random(seed)
    try:
        for _ in range(n):
            _surface, phenomenon, _tree = gen._one_item(rng, lex)
            if phenomenon in SUBTYPES:
                subtypes_in_order.append(phenomenon)
    finally:
        tr._coindex = orig_coindex
        if is_tree:
            setattr(pred_host, pred_name, orig_pred)
        else:
            binding_mod.has_licensed_pair = orig_pred

    # Bucket the ordered block costs by the ordered non-skipped binding phenomena.
    if len(subtypes_in_order) != len(blocks_in_order):
        raise AssertionError(
            f"{grammar}: {len(blocks_in_order)} _coindex calls but "
            f"{len(subtypes_in_order)} non-skipped binding items — the 1:1 "
            "call/item alignment the credit relies on is broken.")
    block: Dict[str, List[float]] = {s: [] for s in SUBTYPES}
    for s, c in zip(subtypes_in_order, blocks_in_order):
        block[s].append(c)

    import numpy as np
    rows = []
    for s in SUBTYPES:
        vals = np.array(block[s], dtype=float)
        n_block = len(vals)
        blk = float(vals.mean()) if n_block else float("nan")
        blk_se = float(vals.std() / math.sqrt(n_block)) if n_block else float("nan")
        calls, trues = geo[s]
        # No geometry predicate ran for this subtype (tree pron): log Z is 0.
        logz = 0.0 if calls == 0 else -math.log(trues / calls) if trues else float("inf")
        rows.append(dict(grammar=grammar, subtype=s, block=blk, block_se=blk_se,
                         logz=logz, credit=blk + logz, n_block=n_block,
                         n_screened=calls))
    return pd.DataFrame(rows)


def emit(grammars=GRAMMARS, n: int = 30000, seed: int = 20260912,
         out_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Measure each grammar and (re)write ``binding_credit.parquet``.

    The parquet is rewritten AFTER EACH grammar, and rows for grammars not in
    ``grammars`` are carried over from any existing file. Generation is
    RAM-heavy on a small box (the four-grammar single-process run has been
    OOM-killed here), so this makes ``--grammar all`` crash-resilient — a kill
    mid-run keeps the grammars already done — and lets a follow-up
    ``--grammar <the survivor>`` fill the rest in a fresh process.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "binding_credit.parquet"
    kept = pd.DataFrame()
    if path.exists():
        prior = pd.read_parquet(path)
        kept = prior[~prior["grammar"].isin(grammars)]

    done = [kept] if len(kept) else []
    for g in grammars:
        df = measure_credit(g, n=n, seed=seed)
        for _, r in df.iterrows():
            print(f"[{g}/{r.subtype}] block={r.block:.3f}±{r.block_se:.3f} "
                  f"logz={r.logz:.3f} -> credit={r.credit:.3f} nats "
                  f"(n_block={r.n_block}, screened={r.n_screened})", flush=True)
        done.append(df)
        out = pd.concat(done, ignore_index=True)
        order = {gname: i for i, gname in enumerate(GRAMMARS)}
        out = out.sort_values(
            ["grammar", "subtype"],
            key=lambda s: s.map(order) if s.name == "grammar" else s
        ).reset_index(drop=True)
        out.to_parquet(path, index=False)     # rewrite after each grammar
    print(f"\nwrote {path}  ({len(out)} rows)")
    return out


def credit_map(path: Path = DATA_DIR / "binding_credit.parquet"
               ) -> Dict[Tuple[str, str], float]:
    """``{(grammar, subtype): credit}`` from the emitted table."""
    df = pd.read_parquet(path)
    return {(r.grammar, r.subtype): float(r.credit) for _, r in df.iterrows()}


def apply_credit(ell: pd.DataFrame, grammar: str,
                 path: Path = DATA_DIR / "binding_credit.parquet",
                 *, ell_column: str = "ell_corpus",
                 out_column: str = "ell_corrected") -> pd.DataFrame:
    """Add ``out_column`` = ``ell_column`` minus the binding credit.

    The credit is subtracted only on the two binding constructions (their
    ``construction`` value IS the subtype key); every other construction is
    copied through unchanged. Raises if a binding construction present in ``ell``
    has no credit for ``grammar`` — a silent zero there would reinstate the
    negative-KL artefact this correction exists to remove.
    """
    cmap = credit_map(path)
    out = ell.copy()

    def _credit(construction: str) -> float:
        if construction not in SUBTYPES:
            return 0.0
        key = (grammar, construction)
        if key not in cmap:
            raise KeyError(
                f"no binding credit for {key!r}; run "
                "`python -m analysis.binding_credit` for this grammar.")
        return cmap[key]

    out[out_column] = out[ell_column] - out["construction"].map(_credit)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grammar", choices=[*GRAMMARS, "all"], default="all")
    p.add_argument("--n", type=int, default=30000,
                   help="items to generate per grammar (binding is ~5%% of draws)")
    p.add_argument("--seed", type=int, default=20260912)
    p.add_argument("--out-dir", type=Path, default=DATA_DIR)
    args = p.parse_args()
    targets = list(GRAMMARS) if args.grammar == "all" else [args.grammar]
    emit(targets, n=args.n, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
