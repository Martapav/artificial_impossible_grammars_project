# Grammar H — Fully Hierarchical Generator

Every agreement, binding, and movement rule is defined over constituent
structure. A sentence is built as a tree (Type0/1/2/3/4 + dependent Type1),
transformed on the tree, then linearized.

## Quick start

```python
from grammars.grammar_H.generate import generate, generate_with_metadata, GrammarHGenerator

strings = generate(100, seed=42)               # List[str]
items   = generate_with_metadata(20, seed=42)  # List[(surface, label, Node)]

gen = GrammarHGenerator(seed=42)
batch = gen.generate_batch(100, constructions=[
    "neutral",
    "anaphoric_binding_refl", "anaphoric_binding_pron",
    "auxiliary_movement", "wh_movement",
])
```

## Phenomenon mix

70% `neutral`, 10% each `anaphoric_binding` / `auxiliary_movement` /
`wh_movement`, sampled per sentence. Binding resolves to `_refl` or `_pron`
(~50/50). A phenomenon that can't apply to the drawn base yields a `_skipped`
label, dropped by the balanced sampler.

## Surface format

Words space-separated; morphemes within a word `#`-separated, order
`stem#number#gender` (+`#tense` on verbs). Compound tense: CAT3 is bare and
CAT3AUX carries INFL1+INFL3.

```
a#3#1 j#3#1#7 c#3#1        subject + verb + object (singular, gender 1, present)
a#3#1 j n#3#1#7 c#3#1      compound tense (bare CAT3, inflected CAT3AUX)
```

## Modules

| Module | Role |
|---|---|
| `nodes.py` | Tree node dataclass |
| `lexicon.py` | Loads the shared `lexicon.json`, category lookups |
| `rules.py` | Continuation probabilities + phenomenon proportions |
| `build.py` | Tree construction (Type0/1/2/3/4, T1dep) |
| `features.py` | Inflection assignment + head-driven agreement |
| `transforms.py` | Binding, aux movement, wh-movement + islands |
| `linearize.py` | Tree → surface string |
| `generate.py` | Pipeline orchestration + generalization set |

## Wh-movement islands

`transforms.py:_collect_wh_candidates` enumerates eligible Type1 targets and
rejects those in an island:

- the matrix subject (subject-phrase extraction),
- a Type1 inside a Type4 (adjunct island),
- a Type1 inside a Type3 with `attachment=="np"` (complex-NP island).

PP-under-VP (`attachment=="vp"`) is not an island. Only licit targets are
produced, so H never emits island-violating strings; generating ungrammatical
wh-movement for minimal pairs would need a separate ignore-islands path.

## Generalization set

Varies embedding depth and dependency length beyond the training caps: forces a
complex subject (≥2 CAT2 modifiers + a PP-under-NP) and applies each
transformation.

## Limitations

- Lexical items sampled uniformly (Zipfian frequency not yet applied).
- Attachment rates are length-control knobs in `rules.py`
  (`P_AUX=0.30`, `P_PP_UNDER_NP=0.25`, `P_PP_UNDER_VP=0.20`, `P_TYPE4=0.15`).
