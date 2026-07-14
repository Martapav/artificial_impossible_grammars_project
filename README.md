# Hierarchical vs. Linear Grammar Encoding in Transformer LMs

Master's thesis project (CIMeC, University of Trento): do transformer LMs encode
hierarchical vs. linear grammatical structure in distinct internal
representations? The method is Distributed Alignment Search (DAS) over models
trained on four artificial grammars:

- **H** — fully hierarchical
- **P** — fully positional (counter-automaton)
- **H′** (`Hprime`) — H, but the wh complex-NP island is a positional surface rule
- **L′** (`Lprime`) — P, but the wh complex-NP island is a structural generation-history rule

H′ and L′ differ from H and L only in that one rule, and make opposite predictions on the same
strings.

---

## Prerequisites

- Python 3.10+
- Git (for the pyvene source install)
- NVIDIA GPU recommended (CUDA 11.7+)

## Setup

```bash
cd thesis-project/
python3.10 -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt          # pyvene is pulled from GitHub
python -c "import torch, numpy, scipy, pandas, matplotlib, pyvene; print('OK')"
```

If the pyvene entry is missing after a `pip freeze`, reinstall it manually:
`pip install git+https://github.com/stanfordnlp/pyvene.git`.

## Generate the lexicon

One shared lexicon, used by all four grammars:

```bash
python -m lexicon.cli --seed 42 --out lexicon.json
python -m lexicon.cli --verify lexicon.json     # re-hash + re-validate
```

## Build corpora

```bash
python scripts/build_all_corpora.py --grammar all
python scripts/build_all_corpora.py --grammar H --train-n 80000
python scripts/build_all_corpora.py --grammar Hprime
```

Grammars: `H`, `P`, `Hprime`, `Lprime` (or `all`). Output:
`corpora/{split}/{grammar}.jsonl`, one file per grammar per split.

| Split                 | N    | Distribution           | Length  |
|-----------------------|------|------------------------|---------|
| `train`               | 80k  | unbalanced 70/10/10/10 | [2,25]  |
| `validation`          | 10k  | unbalanced 70/10/10/10 | [2,25]  |
| `test_representative` | 10k  | unbalanced 70/10/10/10 | [2,25]  |
| `test_indistribution` | 400  | balanced 4×100         | [2,25]  |
| `test_generalization` | 400  | balanced 4×100         | [25,48] |

See `grammars/TRANSFORM_RULES.md` for the rule formalization; probe construction lives in `corpora/probes.py`.

## Test

```bash
pytest tests/ -q
```

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

**Phenomenon mix** (per sentence): 70% neutral, 10% each anaphoric binding /
auxiliary movement / wh-movement. wh and aux are mutually exclusive. Binding
resolves to `_refl` or `_pron` (~50/50). A phenomenon that can't apply to the drawn
base yields a `_skipped` label, dropped by the balanced sampler.

For H′ and L′, the train/test distribution is produced by the base grammar's
pipeline (H for H′, P for L′) with the differing complex-NP rule swapped in; the
two grammars diverge only on the wh complex-NP island. The **generalization**
split combines divergence probes for that island with depth/length probes of the
rules each grammar keeps unchanged.

Per-grammar notes: `grammars/grammar_H/README.md`, `grammars/grammar_P/README.md`.

---
