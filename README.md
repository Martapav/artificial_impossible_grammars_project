# Hierarchical vs. Linear Grammar Encoding in Transformer LMs

Master's thesis project (CIMeC, University of Trento): do transformer LMs encode
hierarchical vs. linear grammatical structure in distinct internal
representations? The method is Distributed Alignment Search (DAS) over models
trained on four artificial grammars:

- **H** — fully hierarchical
- **P** — fully positional (counter-automaton)
- **H′** — H, but the wh complex-NP island is a positional surface rule
- **L′** — P, but the wh complex-NP island is a structural generation-history rule

H′ and L′ differ only in that one rule, and make opposite predictions on the same
strings — the contrast the experiment turns on.

---

## Prerequisites

- Python 3.10+
- Git (for the pyvene source install)
- NVIDIA GPU recommended (CUDA 11.7+); CPU works but is slow

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
python scripts/build_all_corpora.py --grammar H --train-n 100000 --test-n 5000
python scripts/build_all_corpora.py --grammar H_prime --rule-type hierarchical
```

All four grammars run end-to-end. Output: `corpora/{split}/{grammar}[_{rule_type}].jsonl`
for splits `train`, `test_indistribution`, `test_generalization`. Default training
size is 100,000 sentences/grammar. The mixed grammars (H_prime, L_prime) write one
file per `rule_type`; omitting `--rule-type` builds both.

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

Swap `grammar_H` → `grammar_P`. The mixed grammars take a `rule_type`:
`GrammarHPrimeGenerator(rule_type="linear", seed=42)`,
`GrammarLPrimeGenerator(rule_type="hierarchical", seed=42)`.

**Surface format** (shared across all grammars, one vocabulary/tokenizer): words
are space-separated, morphemes within a word are `#`-separated, order
`stem#number#gender` (+`#tense` on verbs). Compound tense surfaces as a bare CAT3
followed by CAT3AUX carrying the inflections:

```
bcc#3#1 u fxf#3#1 j n#3#1#7 c#3#1
└ subject (CAT1+CAT4+CAT2)  └ CAT3 (bare)
                              └ CAT3AUX (INFL1+INFL3)
                                     └ object
```

**Phenomenon mix** (per sentence): 70% neutral, 10% each anaphoric binding /
auxiliary movement / wh-movement. wh and aux are mutually exclusive. Binding
resolves to `_refl` or `_pron` (~50/50). A phenomenon that can't apply to the drawn
base yields a `_skipped` label, dropped by the balanced sampler.

For the mixed grammars, `rule_type` does **not** change the train/test
distribution (sentences are identical across rule types; only the tag differs) —
it selects which rule the **generalization** split probes (H′ → `linear`,
L′ → `hierarchical`; the other type reuses the base grammar's depth/length probes).

Per-grammar notes: `grammars/grammar_H/README.md`, `grammars/grammar_P/README.md`.

---
