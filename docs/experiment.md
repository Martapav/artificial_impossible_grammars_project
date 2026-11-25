# Grammar design brief — lexicon, typology, generation pipeline

The corpus-generator design: the shared typological backbone, the lexicon, the
syntagm/positional schemas, the generation pipeline, and tokenization. The
transformation rules, the conditioning of the base on each phenomenon and the split
architecture are specified in [`TRANSFORM_RULES.md`](TRANSFORM_RULES.md), which wins
wherever the two documents overlap. Section numbers are kept stable because code
cites them (`experiment.md §1/§2/§2.4/§5/§8.2/§10`).

The project investigates whether transformer LMs encode hierarchical vs. linear
grammatical structure in distinct internal representations, using four formally
defined artificial grammars (H, H′, L′, P) to control the training signal and
support causal claims about what structural properties are learned.

---

## 1. Shared Typological Backbone

Applies to all four grammars without exception.

- Non pro-drop: subject must always be overtly expressed.
- Morphology is agglutinative (lexicon v0.3.0, §2.6): a word is a concatenation of
  discrete, invariant morphemes — no fusion, no alternations. Words are
  space-delimited; morphemes within a word are `#`-delimited, and every morpheme is
  exactly one model token (§10). (Lexicon v0.2.0, used for corpora up to set_3, had
  no derivational morphology: one lexical morpheme per word.)
- Basic linear order: SVO. All syntagms are head-first.
- The language is prepositional (prepositions precede their complements).
- No case morphology, no passivization, no moods, no morphophonological alternations.
- Morpheme order within a word-form: derivational prefix (if any), stem, inner
  suffix (if any), outer suffix (if any), then the inflectional morpheme(s):
  `PRE < STEM < S_inner < S_outer < INFL1 < INFL3`. Derivation is always inside
  inflection.
- Inflectional order within a word: for INFL1-bearing items, number precedes gender;
  for INFL3-bearing items, tense follows the lexical item (and CAT3AUX if present).
- Sentence production capacity: unbounded. Length is controlled externally at
  corpus-construction time, not by the grammar rules.

---

## 2. Lexicon

### 2.1 Alphabet

36-symbol alphanumeric alphabet: a–z (26) plus 0–9 (10). Letters are reserved for
lexical morphemes; digits for inflectional morphemes. The two sets never overlap.

**Letters vs morphemes.** A *letter* is an alphabet symbol; a *morpheme* is a
lexicon entry (a stem or an affix) spelled with letters. The distinction is formal:
a string is a morpheme because the lexicon lists it, not because of its spelling. A
single letter may be a morpheme (the stem `a`), but no single letter is an affix
(§2.6). There are no phonotactic constraints and no pre-existing semantics, so
letters combine freely and may repeat within a morpheme. The model never sees
letters: each morpheme is one atomic token (§10).

### 2.2 Lexical pools by category

Each category has a non-overlapping dedicated letter set; its morphemes are strings
over that set. Inter-categorical homonymy is excluded by construction.

For the open classes (CAT1, CAT2, CAT3, CAT5), lexicon v0.3.0 **samples** stems once
rather than enumerating the combinatorial space (§3). Stem length is **not** bounded
by the pool size: stems may be longer than the category's pool. Strings reserved as
affixes (§2.6) are never stems. The closed classes keep their fixed inventories.

| Category | Letter pool | Notes |
|----------|-------------|-------|
| CAT1 | {a, b, c, d, e} | |
| CAT1PRON | {m} = Pron, {mm} = Refl | Exactly two items; both inflect for gender and number. No person distinction. |
| CAT2 | {f, g, h, x} | |
| CAT3 | {i, j, k, l} | |
| CAT3AUX | {n} | Exactly one item |
| CAT4 | {u} = singular, {v} = plural | Number-marked intrinsically; no inflectional morphemes |
| CAT5 | {o, p, q} | |
| CAT6 | {r, y, z} | |
| CAT8 | {w} | Exactly one item |
| CAT9 | {s, t} | Exactly two items |

### 2.3 Inflectional pools

| Category | Parameters | Values | Morpheme pool |
|----------|------------|--------|---------------|
| INFL1 | number (2), gender (2) | ±sg, ±m | {1, 2, 3, 4} — one digit per value-combination |
| INFL3 | tense (3) | present, past, future | {5, 6, 7} |

INFL1 is two separate morpheme tokens (number, then gender) immediately following
the lexical morpheme of any INFL1-bearing item (CAT1, CAT2, CAT3 in agreement
position). INFL3 is one morpheme token immediately following CAT3 or CAT3AUX. The
null morpheme represents gaps (trace of a moved element); not lexicalized at
generation, used in annotation/evaluation only.

### 2.4 Lexical features

- **CAT1 [±cnt]**: countable ([+cnt]) items require a CAT4 determiner and admit CAT2
  modifiers; non-countable ([-cnt], proper-noun-type) items take no CAT4 and no CAT2.
- **CAT3 [±tr]**: transitive ([+tr]) items require an object Type1 syntagm;
  intransitive ([-tr]) items take no object.
- **CAT6 attachment partition**: the three CAT6 items are partitioned into two
  non-overlapping pools at lexicon-generation time — two belong to the
  CAT1-selecting pool (PP-under-NP) and one to the CAT3-selecting pool (PP-under-VP).
  Fixed for the entire paradigm, not drawn randomly.

### 2.5 Frequency distribution

Open-class lexicon entries (stems and derived words, §2.6) receive Zipfian frequency
ranks, one ranking per category; the generator samples occurrences with probability
∝ 1/rank (Zipf s=1). Closed-class items are selected by grammatical rule, not
frequency.

Morphological complexity is **partially** orthogonal to frequency (lexicon v0.3.0).
Each entry gets a score and entries are ranked by ascending score:

    score = N(0,1) + α · max(0, stem_length − pool_size) + β · n_affixes,   α = β = 0.5

- Stems no longer than their category's pool size get no length effect: their
  position in the ranking is random.
- Stems longer than the pool size drift toward the rare end, by α per extra letter.
  This is a brevity effect only for long stems.
- Each affix moves an entry toward the rare end by β, so derived words are on
  average rarer than their bare stem.

α and β are design parameters, not estimates from natural language. They were chosen
from a pre-generation simulation of the usage they induce ("mild": about a quarter of
NOM occurrences bare, mean stem length in use ≈5.2). Under this weighting the corpus
is less affixed than the lexicon, and both figures are reported. (v0.2.0 ranked stems
purely by length, shortest = most frequent.)

### 2.6 Derivational morphology (lexicon v0.3.0)

The semantically loaded categories — CAT1 (NOM), CAT2 (ADJ), CAT3 (VERB), CAT5
(ADV) — have order-bound **derivational affixes**. Affixes are *feature-free*. They
carry only their position in the word and their lexical selection; no syntactic rule,
transformation, licensing condition or probe reads them. They therefore leave the
H/P/H′/L′ contrast untouched and are identical across all four grammars.

**Template.** `(PRE#)STEM(#S_inner)(#S_outer)#INFL…`: at most one prefix and at most
two suffixes, in two position classes. The inner class sits next to the stem and
precedes the outer class. Each affix belongs to exactly one class, so no affix
occurs twice in a word. A word has at most 4 lexical morphemes.

**Suffixing preference.** Suffixes are three times as common as prefixes, both in
the inventory and in usage. The direction follows the cross-linguistic suffixing
preference (Greenberg, J. H. (1957). Order of affixing: A study in general
linguistics. In *Essays in Linguistics*, pp. 86–94. University of Chicago Press).
The 3:1 magnitude is a design choice, not an empirical estimate: these languages are
artificial and differ from natural ones on many levels.

**Inventories** (affix forms are 2–3 letters over the category's own pool, drawn by
a seeded RNG and recorded in the lexicon; reserved, so never a stem):

| Category | Prefixes | Inner suffixes | Outer suffixes | Total |
|---|---|---|---|---|
| CAT1 (NOM) | 6 | 6 | 13 | 25 |
| CAT3 (VERB) | 6 | 6 | 13 | 25 |
| CAT2 (ADJ) | 4 | 4 | 9 | 17 |
| CAT5 (ADV) | 4 | 4 | 9 | 17 |

**Lexicon magnitude.** About 100,000 open-class entries (inflection excluded)
built on about 5,000 stems, 20 entries per stem. The stems are split across
categories in proportion to WordNet 3.0 unique strings (nouns 117,798; adjectives
21,479 incl. satellites; verbs 11,529; adverbs 4,481; total 155,287 → 75.9 / 13.8 /
7.4 / 2.9%; Princeton University, *WordNet 3.0 database statistics*, wnstats(7WN)).

**Derived words are fixed lexicon entries**, generated once together with the
lexicon, not composed at sentence time. Each stem has exactly one bare entry and 19
derived entries. Across the lexicon, the entries' lexical-morpheme counts hit
these shares exactly:

| Lexical morphemes | 1 (bare) | 2 | 3 | 4 |
|---|---|---|---|---|
| Share of entries | 5% | 30% | 45% | 20% |

The mean is 2.8, an agglutinative profile. The 5% bare share follows from the
stem target (one bare entry per stem, 20 entries per stem).

**Composition.** Stem first, then affixes. Each of a derived entry's affixes goes to
the suffix side with probability ¾, subject to the slot limits. This gives:
- 2 morphemes: one suffix (.75) or one prefix (.25);
- 3 morphemes: two suffixes (.6) or prefix + suffix (.4), since two prefixes are
  impossible;
- 4 morphemes: always prefix + two suffixes.

A lone suffix takes the inner or outer class in proportion to the class sizes.
Affixes are drawn uniformly within their class, and a stem never has two identical
entries.

**Lexical identity.** The identity of a word is the whole derived entry (prefix +
stem + suffixes). Binding coreference and every same-word duplicate screen compare
whole derived forms (TRANSFORM_RULES §3). A pronominalized word carries no
derivational affixes.

---

## 3. Lexical Generation Algorithm

Lexicon v0.3.0. The lexicon is generated **once** and does not try to cover the
combinatorial space. Each phase draws from its own seeded RNG stream, so changing one
phase's parameters never shifts another phase's draws.

A. **Affixes.** Per open category, sample the prefix, inner-suffix and outer-suffix
   forms (§2.6 sizes) without replacement from the 2–3-letter strings over the pool.
B. **Stems.** Per open category, the stem count is 5,000 × the WordNet share
   (§2.6). Lengths follow a log-normal shape peaking near 5 letters, with a long
   tail. Short lengths have few possible strings (a pool of n letters has only n^L
   strings of length L). So each length is filled to at most 50% of its free strings
   (those not reserved as affixes), and the excess moves to longer lengths. Strings
   are sampled with repetition of letters; duplicates and affix strings are rejected.
   A capacity guard stops the build if a length cannot be filled.
C. **Features.** [±cnt] and cat6-compatibility for CAT1, [±tr] for CAT3, by seeded
   draw, per stem. Every entry of a stem inherits the stem's features.
D. **Derived entries.** Per stem, one bare entry plus 19 derived entries, with
   morpheme-count quotas and composition as in §2.6.
E. **Frequency ranks.** Per category, rank the entries by the §2.5 score.
F. **Closed classes.** CAT6 pools (two CAT1-selecting, one CAT3-selecting — a fixed
   partition) and the other fixed inventories, as before.
G. **Export.** A structured JSON object with all features, the affix inventory, and
   for every entry its stem, affixes and derived surface form. The lexicon is
   grammar-agnostic; all four grammars import from it. Each corpus set records which
   lexicon file built it.

(Lexicon v0.2.0 enumerated every string up to length = pool size, ranked stems by
length, and had no affixes. Its generator is kept, frozen and self-contained, in
`lexicon/legacy_v020.py` —
`python -m lexicon.cli --lexicon-version 0.2.0 --seed 42` reproduces `lexicon.json`
exactly, and `tests/test_legacy_lexicon.py` guards it. The two generators share only
the data model and the hashing.)

---

## 4. Syntactic Categories and Syntagm Types (hierarchical grammars: H, H′, L′ base)

### 4.1 Syntagm types

**Type0 (Sentence)**: top-level node. Expands as Type1 + Type2 (the S node).

**Type1 (Nominal projection)**: headed by CAT1.
- [+cnt]: `CAT1 INFL1n INFL1g CAT4 (CAT2 INFL1n INFL1g)*`
- [-cnt]: `CAT1 INFL1n INFL1g`

After the core nominal, Type1 may take zero or more Type3 dependents (PP
adjuncts/arguments) and zero or more Type4 dependent (relative clause, positioned
finally). **Dependent Type1 (T1dep)** appears in non-subject positions (objects of
Type2, complements of Type6) with the same internal structure.

**Type2 (Verbal projection)**: headed by CAT3.
- Simple: `CAT3 INFL1n INFL1g INFL3 (CAT5)* (Type3)* (T1dep)*`
- Compound: `CAT3 CAT3AUX INFL1n INFL1g INFL3 (CAT5)* (Type3)* (T1dep)*`

CAT3AUX immediately follows CAT3; inflectional morphemes follow CAT3AUX; CAT5
adverbs follow them. The object T1dep is obligatory iff CAT3 is [+tr]. The
Type3 adjunct here is a PP-under-VP (CAT6 from the CAT3-selecting pool), optional
regardless of transitivity. Agreement: INFL1n/INFL1g on CAT3 (or CAT3AUX) must match
the subject Type1's CAT1 head.

**Type3 (Prepositional phrase)**: headed by CAT6, non-inflecting. `CAT6 T1dep`.
Attaches either as a dependent of a CAT1 head (PP-under-NP, CAT1-selecting CAT6) or
as an adjunct of a CAT3 head (PP-under-VP, CAT3-selecting CAT6).

**Type4 (Subordinate clause)**: headed by CAT9. `CAT9 Type0`. Attaches exclusively
as a dependent of a CAT1 head (relative clause). Type4 introduces a complete Type0,
which may itself contain further Type4 dependents; depth is not hard-capped by the
rule, only by the length sampler.

### 4.2 Linear surface order

```
[CAT1 INFL1n INFL1g (CAT4) (CAT2 INFL1n INFL1g)* (Type3)* (Type4)?]_subj
[CAT3 (CAT3AUX) INFL1n INFL1g INFL3 (CAT5)* (Type3)?]_verb
[CAT1 INFL1n INFL1g (CAT4) (CAT2 INFL1n INFL1g)* (Type3)* (Type4)?]_obj
```

---

## 5. Positional Grammar Base (Grammar P, and the positional backbone of L′)

In positional grammars there are no syntagms, heads, c-command or dominance. All
well-formedness conditions are stated in terms of category identity and ordinal
position from a left-to-right scan. The positional rules produce surface strings
maximally similar to hierarchical output except where transformations necessarily
diverge.

### 5.1 Positional construction rules (applied in order, left-to-right scan)

1. **Base order**: CAT1 + CAT3 + CAT1 (subject, verb, object). Intransitive CAT3:
   omit object.
2. **CAT4 insertion**: for each [+cnt] CAT1 in scan order, insert CAT4 immediately
   after its inflectional morphemes (form set by the CAT1's INFL1n).
3. **CAT3AUX insertion**: if compound tense, insert CAT3AUX immediately after CAT3,
   before its inflectional morphemes.
4. **CAT2 insertion**: immediately after CAT1 (or CAT4 if present).
5. **CAT5 insertion**: immediately after CAT3 (or CAT3AUX if present).
6. **CAT6 insertion**: a nominal-zone PP lands at the end of its host's
   `CAT1 [CAT4] [CAT2]*` block (as in H, where Type3 follows all CAT2s); a
   verbal-zone PP lands after the last CAT5, else after CAT3 (or CAT3AUX). Each CAT6
   must be immediately followed by its own CAT1.
7. **CAT9 insertion**: initializes a new substring following the same rules from
   step 1. The host is the object with probability `Q_EMB_OBJECT` when the clause is
   transitive, else the subject; the substring goes at the end of the host's nominal
   zone (before the verb for a subject host, clause-final for an object host), the
   same placement Grammar H uses for Type4.
8. **Inflection**: INFL1n/INFL1g for CAT3 are set by counting to the subject CAT1
   (position 1) and copying its INFL1. The object CAT1 takes independently drawn
   INFL1.

**Constraint on scan-based generation**: no step may depend on prediction of items
not yet encountered. Each step operates only on already-processed positions.

---

## 6. Grammar definitions

See [`TRANSFORM_RULES.md`](TRANSFORM_RULES.md) §1–§2 and §6. In brief: **H** states
every rule over constituent structure; **P** states every rule as a count on the
string, anchored at the sentence start; **H′** = H with P's positional wh clause (i);
**L′** (displayed as P′) = P with H's structural wh clause (i). Each prime differs
from its parent only in that clause.

## 7. Syntactic transformations

Binding, auxiliary movement and wh-movement are specified in
[`TRANSFORM_RULES.md`](TRANSFORM_RULES.md) §3 (binding), §4 (aux) and §6 (wh).

---

## 8. Generation Procedure

### 8.1 Hierarchical generation (H, H′, and the hierarchical sub-system of L′)

Per sentence, in order:
1. **Transformation pre-selection** — draw which phenomenon applies; the base is
   then conditioned on that phenomenon's licensing predicate and nothing else
   (`TRANSFORM_RULES.md` §1c, §7).
2. **Base sentence generation** — choose a CAT1 subject ([±cnt] → CAT4 need) and a
   CAT3 ([±tr] → object need).
3. **Inflection** — subject CAT1 freely drawn; CAT3/CAT3AUX match the subject
   (INFL3 free); object T1dep independently drawn; CAT2 modifiers agree with head.
4. **Obligatory non-inflecting insertion** — CAT4 by [±cnt]/INFL1n; CAT3AUX for
   compound tense (reposition inflectional morphemes).
5. **First expansion (obligatory constituents)** — per head, draw whether/what to
   expand; insert head-first then inflectional morphemes.
6. **Second expansion (adjuncts)** — rate-bounded Type3 PP adjuncts and Type4
   relative clauses at admissible sites.
7. **Syntactic transformation** — apply the pre-selected function to the completed
   string.

### 8.2 Positional generation (P, and the positional sub-system of H′)

1. **Transformation pre-selection** — as §8.1.
2. **Base sentence generation** — SVO base (CAT1 CAT3 CAT1 transitive; CAT1 CAT3
   intransitive). No inflection yet.
3. **Inflection** — scan L→R: INFL1n/INFL1g after the first CAT1 (subject);
   INFL1n/INFL1g (copied from subject) + INFL3 after CAT3 (or CAT3AUX); INFL1n/INFL1g
   for the second CAT1 (object) after its lexical morpheme.
4. **Obligatory non-inflecting insertion** — CAT4 where required; CAT3AUX for
   compound tense (reposition inflectional morphemes).
5. **First expansion** — CAT2 modifiers and CAT5 adverbs per §5.
6. **Second expansion** — CAT6 and CAT9 per §5; when CAT9 introduces a clause, treat
   the new substring as a fresh sentence and apply steps 2–5 recursively.
7. **Syntactic transformation** — apply the pre-selected transformation.

### 8.3 Generation-history tracking (required for L′, recommended for all)

The generator maintains a per-sentence record: for every non-base item added, its
category, the item that licensed its insertion, and the role it was inserted in
(e.g. "T3 argument of CAT1_head at position i", "T4 relative clause of CAT1 at k").
L′ reads its structural wh clause (i) off this record (`wh_structural.py`), and the
structural probes use it to build divergence items. It must be complete before the transformation step.

---

## 9. Data splits

The splits (train / validation / test_indistribution plus the three minimal-pair
structural probes IOD / OOD / SELECT, with the metadata sidecar) are specified in
[`TRANSFORM_RULES.md`](TRANSFORM_RULES.md) §8–§10. Each split is drawn on its own
RNG trajectory (a distinct seed offset from one base seed).

---

## 10. Tokenization

Tokenize by splitting on the separator (whitespace, then `#`). Each token is exactly
one morpheme: a stem, a derivational affix (§2.6), or an inflectional morpheme. Stems
are atomic — the model never sees their letters, so stem length in letters is a
property of the lexicon, not of the token stream. There are no multi-morpheme tokens
and no subword tokenization. The vocabulary is closed and known in advance from the
lexicon the corpus set was built with: stems, affixes, inflectional digits, closed
classes. Reserved affix strings are never stems, so every token is unambiguously a
stem or an affix.

---

## 11. Key implementation notes

- All random draws use a seeded, recorded RNG.
- The lexicon is grammar-agnostic and generated once; all four grammars import it.
- The positional generator must never make forward-looking decisions: each step
  operates only on the string state after all prior insertions.
- The hierarchical generator operates on a tree; transformations are applied on the
  tree and the surface string is read off at the end.
- For L′, the generation-history record must be complete before the transformation.
- Transformation step for all grammars: applied after the string is fully generated.
  Pre-selection (step 1) only conditions the base; it does not pre-build the
  transformation.
