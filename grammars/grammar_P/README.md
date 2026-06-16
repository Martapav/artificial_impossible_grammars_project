# Grammar P — Fully Positional Generator

The null model for hierarchical encoding. Every agreement, binding, and movement
rule is stated over **surface position and lexical category**, computed by a
left-to-right scan. No tree is built: a sentence is a flat `list[Token]` in
surface order (`tokens.py`), and each step operates only on the string state from
prior steps. Feature assignment is inline in `build.py` (a fixed-position copy),
so there is no `features.py`.

## Quick start

```python
from grammars.grammar_P.generate import generate, generate_with_metadata, GrammarPGenerator

strings = generate(100, seed=42)               # List[str]
items   = generate_with_metadata(20, seed=42)  # List[(surface, label, List[Token])]

gen = GrammarPGenerator(seed=42)
batch = gen.generate_batch(100, constructions=[
    "neutral",
    "anaphoric_binding_refl", "anaphoric_binding_pron",
    "auxiliary_movement", "wh_movement",
])
```

## Phenomenon mix

Same as Grammar H: 70% `neutral`, 10% each binding / aux movement / wh-movement.
Binding resolves to `_refl`/`_pron` (~50/50); inapplicable phenomena yield
`_skipped` (dropped by the sampler); wh and aux are mutually exclusive.

## Surface format

Byte-identical to Grammar H (shared vocabulary/tokenizer). The clause domain used
by binding and the wh-scan is `clause_id` — a positional index delimited by CAT9,
set during construction (not a dominance domain).

```
a#3#1 j#3#1#7 c#3#1        subject + verb + object
a#3#1 j n#3#1#7 c#3#1      compound tense (bare CAT3, inflected CAT3AUX)
```

## Construction pipeline (`build.py`, per clause, left-to-right)

1. **Base** — CAT1 (subject) + CAT3 (verb) [+ CAT1 (object) if transitive].
   Binding/wh clauses are forced transitive.
2. **CAT3AUX** (compound tense) — before inflection, so INFL1/INFL3 route onto it.
3. **Inflection** — INFL1 on subject; INFL1 (copied) + INFL3 on verb/aux; INFL1 on object.
4. **CAT4** — one after each countable CAT1, form fixed by its INFL1_number.
5. **First expansion** — CAT2 after CAT1/CAT4, CAT5 after CAT3/CAT3AUX.
6. **Second expansion** — CAT6 PPs (nominal/verbal zone), then CAT9 substrings
   (each recurses into a fresh clause).

## Modules

| Module | Role |
|---|---|
| `tokens.py` | Flat positional token dataclass |
| `lexicon.py` | Loads shared `lexicon.json`, category lookups |
| `rules.py` | Bounds, probabilities, inflectional value pools |
| `build.py` | Positional construction + inline inflection/agreement |
| `transforms.py` | Binding, aux fronting, wh blocking |
| `linearize.py` | Token list → surface string |
| `generate.py` | Orchestration + generalization set |
| `generator.py` | `GrammarPGenerator` shim for the corpus builder |

## Wh-movement blocking (`transforms.py:_wh_candidates`)

A CAT1 is blocked from fronting iff:

1. it is the **second CAT1** in the sentence (object in the SVO base),
2. it is the **first CAT1 after a CAT9** (subordinate subject), or
3. it is **immediately preceded by a CAT6** (PP complement).

The matrix subject (first CAT1) is blocked by none of these and is the canonical
frontable target — the mirror image of Grammar H, where the subject is an island.
The whole nominal block (CAT1 + inflections + CAT4 + CAT2) moves, a null morpheme
marks the gap, and CAT8 is prefixed.

Condition 3 is the surface rule that **Grammar L′** replaces with a
generation-history rule; the CAT6 `attachment` zone and the complement's
`licensor_attachment` are recorded on the tokens for that use.

## Generalization set

Varies **embedding depth** (CAT9 nesting) and **dependency length**
(subject-to-verb CAT2 interveners) beyond the training caps: depths `{2,3}` ×
forced subject-CAT2 `{2,4}`, 5 items per cell, per transformation.
`force_cat2_subject` and `force_cat9_depth` (threaded through `build_sentence`)
control the two axes.

## Conventions & limitations

- Inflection digits follow `lexicon.json`: number `{3,4}`, gender `{1,2}`,
  tense `{7,8,9}`.
- CAT2 restricted to countable CAT1 (proper-like nouns take no modifiers); CAT2
  copies its host CAT1's INFL1. PP complements are minimal (`MAX_CAT2_COMPLEMENT=0`).
- Reflexive copies the subject's number/gender (person free); pronoun substitutes
  the subject CAT1 (free, non-object position).
- CAT9 substrings are appended at clause end.
- Uniform lexical sampling (no Zipfian frequency yet); at most one transformation
  per sentence.
- Rates: `P_AUX=0.30`, `P_CAT6_NP=0.25`, `P_CAT6_VP=0.20`, `P_CAT9=0.15` (in `rules.py`).
