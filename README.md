# Hierarchical vs. Linear Grammar Encoding in Transformer LMs

Master's thesis project (CIMeC, University of Trento): do transformer LMs encode
hierarchical vs. linear grammatical structure in distinct internal
representations? Small GPT-2 models are trained on four artificial grammars and
evaluated with surprisal: on minimal-pair structural probes (licensed vs.
unlicensed application of the same rule) and on held-out text, measured against
the grammar floor ℓ(x) and an n-gram surface baseline.

- **H** — hierarchical: every rule is stated over constituent structure
- **P** — positional: every rule is a count on the string, anchored at the
  sentence start
- **H′** (`Hprime`) — H, with P's positional wh clause (i) (the third nominal of
  the sentence)
- **P′** (code key `Lprime`) — P, with H's structural wh clause (i) (the matrix
  object that hosts an embedded clause), read off the generation history

Each primed grammar differs from its parent only in wh clause (i), so the two
grammars in each pair can license different wh targets in the same base sentence.
`docs/TRANSFORM_RULES.md` calls P′ by its older name **L′**. The code key stays
`Lprime` (module `grammar_L_prime`), and tables and plots show it as P′.

---

## Prerequisites

- Python 3.10+
- NVIDIA GPU recommended. The pinned `torch==2.5.1+cu121` needs a driver that
  supports CUDA 12.1; with an older driver it runs on CPU only.

## Setup

```bash
cd thesis-project/
python3.10 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, transformers, numpy, scipy, pandas, matplotlib; print('OK')"
```

## Generate the lexicon

One shared lexicon, used by all four grammars:

```bash
python -m lexicon.cli --seed 42 --out lexicon.json
python -m lexicon.cli --verify lexicon.json     # re-hash + re-validate
```

## Build corpora

```bash
python -m scripts.build_all_corpora --grammar all
python -m scripts.build_all_corpora --grammar H --train-n 80000
python -m scripts.build_all_corpora --grammar Hprime
```

Grammars: `H`, `P`, `Hprime`, `Lprime` (or `all`). Output:
`corpora/{set}/{split}/{grammar}.jsonl`, one file per grammar per split.

Each generation writes its own **corpus set** and is never overwritten, so a
trained model's corpus stays on disk unchanged for as long as the model matters.
A new generation needs a **new `--seed`**: the generators are deterministic in
it, so a fresh set at an already-used seed reproduces that set byte for byte and
aborts. Pick a seed no other set records (`--set` defaults to the lowest `set_N`
not already present).

```bash
python -m scripts.build_all_corpora --seed 44                 # a new generation (next free set_N)
python -m scripts.build_all_corpora --set set_5 --seed 45     # a name you choose
python -m scripts.build_all_corpora --grammar P --set set_5   # add P to set_5
```

The production corpus is **`set_3`**: all four grammars, built with `--seed 43` at
commit `141fb23`. Every reported model was trained and scored on it.

The guard is per FILE, so a set can be filled one grammar at a time; rebuilding
a grammar already in that set aborts and names it. `manifest.json` records the
set's files with their sha256 plus one `builds` entry per invocation (seed,
sizes, git SHA, lexicon hash). Everything downstream reads the set named by
`THESIS_CORPUS_SET` (default `set_3`, the production corpus), resolved in one place,
`models/dataload.py`.

| Split                     | N     | Distribution           | Length  |
|---------------------------|-------|------------------------|---------|
| `train`                   | 100k  | unbalanced 70/10/10/10 | [2,25]  |
| `validation`              | 10k   | unbalanced 70/10/10/10 | [2,25]  |
| `test_indistribution`     | 15k   | unbalanced 70/10/10/10 | [2,25]  |
| `structural_probe_IOD`    | 4200  | 2100 minimal pairs     | [2,25]  |
| `structural_probe_OOD`    | 4200  | 2100 minimal pairs     | [26,48] |
| `structural_probe_SELECT` | 4200  | 2100 minimal pairs     | [2,25]  |

Probe sizes are `n_per_subcase x 7 subcases x 2 members`, with `n_per_subcase`
defaulting to **300** for all three probes (`--probe-n-per-subcase` for
IOD/OOD, `--select-n-per-subcase` for SELECT). SELECT is the recipe-selection
probe — a separate draw (seed `+300000`) in the IOD length window, kept disjoint
from the reporting probes so taking an argmax over recipes does not bias the
reported number. There is deliberately no OOD selection probe.

### Rebuilding only the probes

```bash
python -m scripts.build_all_corpora --grammar all --probes-only --set set_5
```

`--probes-only` regenerates IOD, OOD **and** SELECT (there is no per-probe flag)
and leaves `train`, `validation` and `test_indistribution` untouched. The probes
are evaluation-only minimal-pair corpora, so rebuilding them does not invalidate
any trained model — use this after changing pair construction in
`corpusgen/structural_probes.py`, then re-run `evaluation/score_pairs.py`.

It requires an explicit `--set`: auto-naming would build a set holding probes and
no training data, which `models/dataload.py` will not even recognise as a set.
And it will not overwrite probes that already exist in the named set without
`--force` — `pair_id` is positional, so rebuilding in place silently re-points
every probe score already written against that set onto different items.

Every build (both paths) ends with `corpusgen/probe_checks.py`, which re-reads the
written files and **raises** if a grammar's probes are not disjoint: a pair
duplicated within one probe, a pair shared by two probes, or even a shared
grammatical member. Nothing in the builder dedups — the probes are separated
only by seed offset, and collision probability grows with the product of the
sample sizes — so this is what keeps raising `--select-n-per-subcase` honest. It
matters because SELECT chooses the recipe while IOD/OOD report the result; a
shared pair would make the reported number one that selection had already
optimized. Overlap *across grammars* is not checked and is expected (H/H′ and
P/L′ share a base pipeline).

See `docs/TRANSFORM_RULES.md` — the single specification for all four
grammars: rule formalization (§1–§6), generation machinery (§7), the metadata
sidecar (§8), the structural-probe design (§9) and the split architecture (§10).
Probe construction lives in `corpusgen/structural_probes.py`.

## Test

```bash
pytest tests/ -q
```

## Train

Grid search picks the recipe; `models.train` trains the 20 models (4 grammars x
5 seeds, 12-epoch cap) that the evaluation below scores.

```bash
python -m models.train --dry-run                    # list the runs
python -m models.train --grammars H --seeds 0 \
    --subset-size 500 --max-steps 20                # local smoke test
python -m models.train --device cuda --bf16           # full run
```

How the recipe was selected (the grid/pilot sweeps, the objective, the frozen
result) is in **[docs/RECIPE_SELECTION.md](docs/RECIPE_SELECTION.md)**.
Sweep artefacts land in `results/<sweep>/`: `run_manifest.jsonl` (what was asked
for, on which corpora), `results.{jsonl,csv}`, `summary*.csv`, and one probe-score
shard per run under `probe_scores/<run_id>.parquet`.

## Evaluate and analyse

- **Structural probes:** `evaluation/score_pairs.py` computes per-item surprisal
  on the IOD/OOD/SELECT probes, and `evaluation/aggregate.py` pairs the items
  into minimal-pair margins.
- **Held-out text:** `evaluation/test_eval.py` runs the held-out pass on
  `test_indistribution`. The `eval_final.json` that `models.train` writes is the
  validation split only.
- **Reports:** the figures and tables come from scripts in
  `analysis/reports/{corpus,training,heldout,probes,ch2}/`, run from the repo
  root as `python -m analysis.reports.<group>.<script>`. Output goes under
  `results/`, which is git-ignored.

The full command sequence and every output file are described in
[docs/results_documents.md](docs/results_documents.md); what each analysis computes
is in [analysis/README.md](analysis/README.md).

---

## Usage

Each grammar exposes the same three entry points:

```python
from grammars.grammar_H.generate import generate, generate_with_metadata, GrammarHGenerator

sents = generate(100, seed=42)                # List[str]
items = generate_with_metadata(20, seed=42)   # (surface, label, tree/token-list)
gen   = GrammarHGenerator(seed=42)            # batch interface (slot-balanced)
```

Swap `grammar_H` → `grammar_P`, `grammar_H_prime`, or `grammar_L_prime`:
`GrammarHPrimeGenerator(seed=42)`, `GrammarLPrimeGenerator(seed=42)`.

**Surface format** (shared across all grammars, one vocabulary/tokenizer): words
are space-separated, morphemes within a word are `#`-separated, order
`stem#number#gender` (+`#tense` on verbs).

**Phenomenon mix:** each sentence has exactly one pre-selected phenomenon: 70%
neutral, 10% anaphoric binding (split into `refl` and `pron`, 5% each), 10%
auxiliary movement, 10% wh-movement. If the phenomenon can't apply to the drawn
base, the item gets a `_skipped` label and the corpus builder discards it.

For H′ and P′, the train/test distribution comes from the parent grammar's
pipeline (H for H′, P for P′) with the other family's wh clause (i) swapped in.
Binding and aux are the parent's own rules. The **structural probe**
splits (IOD/OOD) are minimal-pair corpora contrasting grammatical vs.
ungrammatical rule application; the OOD split reuses the identical construction
logic at longer lengths ([26,48]) to test length/structural generalization.

Per-grammar notes (surface format, module map, the positional construction
pipeline): `docs/TRANSFORM_RULES.md` §7.

---

## Documentation map

- **[docs/README.md](docs/README.md)** — index of all documentation.
- **[docs/TRANSFORM_RULES.md](docs/TRANSFORM_RULES.md)** — the authoritative
  rule specification for all four grammars.
- **[experiment.md](docs/experiment.md)** — the design brief: lexicon, typology,
  generation pipeline, tokenization.
- **[analysis/README.md](analysis/README.md)** — the corpus/statistical analysis,
  with [SURPRISAL_REFERENCE_POINTS.md](docs/SURPRISAL_REFERENCE_POINTS.md) and
  [SEPARABILITY.md](docs/SEPARABILITY.md).
- **[docs/RECIPE_SELECTION.md](docs/RECIPE_SELECTION.md)** — how the training
  recipe and epoch budget were chosen.
- **[docs/results_documents.md](docs/results_documents.md)** — every file training
  and evaluation write, and the commands that regenerate them.

---
