# Transformation rules — authoritative specification (all four grammars)

The single specification for Grammar H (hierarchical), Grammar P (positional),
and the mixed pair H′/L′ (L′ is displayed as P′ in tables and plots; its code key
is `Lprime`). It supersedes `experiment.md` §6–§7 and states the current rules
only.

**Code.** `grammar_H/{build,features,transforms,linearize,generate}.py`,
`grammar_P/{build,binding,wh,transforms,linearize,generate}.py`,
`grammar_H_prime/`, `grammar_L_prime/`, `wh_structural.py`, `verdicts.py`,
`licensing.py`, `violations.py`, `metadata.py`, `corpusgen/structural_probes.py`.

**Tests.** `test_reformalized_rules.py`, `test_refl_embedded_domain.py`,
`test_embedded_aux.py`, `test_verdicts.py`, `test_metadata_sidecar.py`,
`test_structural_probes.py`, `test_generation.py`, `test_determinism.py`.

Rules are stated once and read everywhere: `grammar_P/binding.py` (P binding),
`grammar_P/wh.py` (P-family wh clauses), `wh_structural.py` (structural wh clause
(i), shared by H and L′). Generation, the surfacing screens, the annotation
predicates and the probe site tables all import from those modules.

---

## 1. Design principles

Every transformation must satisfy, per grammar pair:

- **(a) Blockability.** There exist drawn structures where the phenomenon cannot
  apply. Because the corpus builder discards `*_skipped` items, blockability is
  realized **distributionally**: the model's evidence is the contrast between
  configurations where the phenomenon systematically occurs and matched ones where
  it never does. Blockability is a property of the RULE SYSTEM, not of the data.
- **(b) Non-reducible divergence.** For matched base structures, H's rule and P's
  rule select **different** targets, in both directions, and neither selection
  function is a surface variant of the other.
- **(c) Minimal conditioning.** Phenomenon-conditional generation conditions the
  base on the phenomenon's **licensing predicate and nothing else**:

      p(base | the phenomenon's licensing conditions are satisfiable in this base)

  The canonical implementation is **rejection sampling on the licensing predicate
  itself** — draw naturally, keep if licensed — minimal by construction because the
  acceptance test *is* the rule. Construction-time forcing is admissible **only
  where it forces exactly the predicate**; forcing beyond it makes something the
  grammar never named correlate with the phenomenon, an invisible confound in the
  corpus. A residual surface regularity is measured on the correctly conditioned
  distribution and handled in probe design/analysis, never cancelled by distorting
  the generative distribution.

  *One documented exception* (§6): where a base must serve all four grammars and
  their licensing predicates are not co-satisfiable at a common minimum, the
  **union** of the four predicates is forced, and the over-conditioned grammars are
  named at the point of forcing.

  *Lesson (§6):* forcing a conjunct makes it vacuous unless it still varies across
  candidates within one sentence.

Two framings apply throughout:

- **Licensing, not blocking.** Rules state where a phenomenon CAN occur. The model
  gets direct positive evidence of the licensed set; the unlicensed complement
  never fronts or substitutes.
- **Derivational rules, surface composition.** Each rule is evaluated at its own
  derivational step — binding and wh on the **base string/tree**, aux on the built
  clause. Transformations compose afterwards, so a derived surface may obscure the
  base configuration (intended: it mirrors derivational opacity). *Consequence:*
  any code holding a possibly-transformed structure must recover base order before
  applying a licensing predicate. For a **structural** rule it is enough that the
  candidate itself did not move; for a **positional** rule the rest of the sentence
  moving is equally disqualifying. `metadata.base_order_tokens` is the one place
  that does this recovery for the P family.

One phenomenon is pre-selected per sentence (70% neutral, 10% per transformation;
binding refines to refl/pron at 5%/5%). Pronouns come only from the one binding
phenomenon (§5).

## 2. Definitions

- **Nucleus (H).** For any clause (Type0, matrix or embedded): the subject head
  CAT1, the verb, and the object head CAT1 — nothing else. PP complements, CAT2
  hosts, and all material inside Type3/Type4 are outside every nucleus.
- **Item / distance (P).** One item = one `Token` = one whitespace-delimited
  surface word = one lexeme with its `#`-attached inflections. Inflection values are
  morphemes, not items. Distances are counted on the **base string**. The one
  surface word that is not an item is the `GAP` (`#NULL#`), removed by base recovery.
- **CAT1-position (P).** Any CAT1 **or CAT1PRON** token: the ordinal scans do not
  distinguish nouns from pronouns.
- **Coreferential duplicate.** The derivational intermediate of binding: the same
  CAT1 lexeme with the same INFL1_number (gender is inherent) in two positions.
- **Block.** The unit all four grammars move and substitute: head CAT1/CAT1PRON
  plus its CAT4 determiner and contiguous CAT2 modifiers. Any PP-under-NP (Type3)
  and embedded clause (Type4) hosted by the target is **stranded** in situ,
  immediately after the gap or the pronoun. The transformation's surface effect is
  identical across the family; the grammars differ **only** in licensing.

---

## 3. Anaphoric binding

Binding is a transformation on a *licensing base* that already contains the
coreferential duplicate. Generation builds that duplicate for real: a shared
`_coindex` step overwrites the **target** with the **anchor**'s lexeme +
INFL1_number + INFL1_gender and strips the target's direct modifiers, *before* the
substitution spells it out as the pronoun. The anchor is never touched (it fixes
verb agreement); a subject target's clause verb is realigned at coindexation.
`violations.apply_binding` runs the identical `_coindex`, so generation,
annotation and the probes share one notion of eligibility.

### 3.1 Grammar H (hierarchical)

**Refl — rule A′.** A CAT1PRON-Refl must be bound within its nucleus, **and that
nucleus must be EMBEDDED** (its Type0 dominated by ≥1 Type4, at **any depth**,
`_embedded_type0s`). The **matrix nucleus never licenses**. Enforced conjuncts, in
`transforms._refl`:

1. the nucleus has both a subject and an object;
2. both are full `Type1` (no pron antecedent, no double substitution);
3. the **object** (the second occurrence) is obligatorily substituted, copying the
   subject head's number/gender;
4. no verb realignment (the object controls no verb).

*Conditioning: rejection sampling, nothing built (§1c).* A′ requires an embedded
nucleus with both a subject and an object, both full `Type1` — that, and only that,
is the licensing predicate. Refl items are conditioned by rejection sampling on the
predicate (`transforms.has_refl_licensing_nucleus`); nothing is constructed.
`transforms.refl_licensing_nuclei` states the predicate once and all three
consumers read it: the transform, generation's resampler, and the probes'
`base_conformity`. The matrix clause is drawn naturally — A′ names no matrix
property, so an intransitive matrix still licenses refl in its embedded nucleus.

*Residual counting signature.* With a subject host the embedded clause closes
before the matrix verb, so a licensed object has one verb to its left and an
unlicensed one has two; whole-side category counts separate them. An object host
reverses the ordering, so the signatures cancel to the extent the host mix is
balanced. Under natural matrix transitivity the mix is subject-weighted, so
cancellation is partial and a residual count cue is expected — measured on the
correctly conditioned distribution and handled in probe design/analysis, not
cancelled by forcing.

- *Surfacing constraint:* no final string may contain the same CAT1 stem as
  subject and object of one **embedded** nucleus (generation screens with
  `has_nucleus_duplicate`). A **matrix** same-nucleus duplicate is unlicensed by
  both sub-rules and is left to surface unsubstituted — the H family's blocking
  evidence.
- *Blocking:* under phenomenon-driven generation the geometry is forced, so refl
  skips are 0%.

**Pron — Principle B analog.** A CAT1PRON-Pron must be free in its nucleus: the
coreferential pair spans two positions **not** forming the subject–object pair of
one nucleus. Pairs where one nominal dominates the other are excluded. No
transitivity is required anywhere.

- The licensed set is **order-symmetric and purely structural** — both members of
  every licensed pair are licensed. `P_PRON_FIRST = 0.20` then decides which member
  is spelled out as the pronoun (a **realization** choice, the only linear notion
  in H's rules).
- *Optionality:* substitution of a cross-nucleus duplicate is optional at the
  grammar level. Phenomenon-labeled pron items always substitute; unsubstituted
  cross-nucleus duplicates surface in neutral items via chance draws — H's
  blocking-contrast evidence.
- *Blocking:* no second nominal outside the nucleus → `_skipped`. Pron is
  **unforced** (measured skip rate ~55%).
- Pron–pron chains are excluded (antecedents and targets must be full CAT1).

**Agreement fixup.** Substituting a clause **subject** realigns that clause's
verb-carrier INFL1 features with the pron.

### 3.2 Grammar P (positional)

No domain notion. Both sub-rules are **counts on the base string**, deterministic
in every respect: the antecedent is the **first** occurrence, the substituted item
is always the **second**, and substitution in a licensed configuration is
**obligatory**. Each rule reads two reference points: one **absolute** (a
CAT1-position ordinal from the sentence beginning) and one **relative** to it (the
surface distance to the target). No landmark anchors.

The geometry lives in `grammar_P/binding.py`. A pair `(antecedent i, target j)` is
licensed for a subtype iff **all** of:

1. `cats[i] == "CAT1"` — a CAT1PRON is counted by the ordinal scan but can never be
   an antecedent (no pron-pron chains);
2. `cats[j] == "CAT1"` and `j == i + off`, with `off ∈ {3}` for **Refl (rule B)**
   and `{2, 4}` for **Pron (N9)**;
3. the antecedent's 1-based CAT1-position ordinal is `3` for Refl, `2 or 3` for
   Pron — **the first nominal never licenses**.

`binding.licensed_pairs(cats, subtype, forms)` takes `forms=None` to ask the
geometry question alone (generation, probe-site enumeration) and a form list to
enforce coreference (annotation).

- *Surfacing constraint:* a same-stem CAT1 pair in a **licensed configuration** (a
  licensed distance **and** a licensed antecedent ordinal) must never surface
  unsubstituted (`has_licensed_duplicate`).
- *Blocking:* duplicates at any other distance, **and** duplicates at a licensed
  distance whose antecedent sits at an unlicensed ordinal, surface unsubstituted —
  P's positional blocking evidence.
- *Forced geometry.* Both P binding rules are purely positional (an offset and an
  antecedent ordinal) and name no clause structure. Conditioning is **rejection
  sampling on the geometry itself** — resample the base until it hosts the subtype's
  geometry — never by repairing a base (the scan-based builder has no look-ahead; a
  spliced item would stamp a regularity on every binding item). **No embedding is
  forced.**
- Distance-disjointness — refl {3} vs pron {2,4} — preserves determinism now that
  both rules admit antecedent ordinal 3. `binding.check_disjoint()` asserts it at
  import.

### 3.3 Why this opposition is non-trivial

`ₓ` marks the coreferential pair; "ant ord" is the antecedent's CAT1-position
ordinal. Every row was checked against the shipped predicates.

| Base configuration | ant ord | distance | P | H |
|---|---|---|---|---|
| `CAT1 CAT4 CAT6 CAT1 CAT9 CAT1ₓ CAT4 CAT3 CAT1ₓ` — **embedded** nucleus, ant = 3rd nominal | 3 | 3 | refl | refl (**the agreeing anchor**) |
| **embedded** nucleus, ant at any other ordinal | ≠3 | 3 | **unlicensed** | refl |
| **embedded** nucleus, any other spacing | 3 | ≠3 | unlicensed | refl |
| **matrix** `CAT1ₓ CAT4 CAT3 CAT1ₓ` | 1 | 3 | **unlicensed** | neither |
| **matrix**, proper-like subject `CAT1ₓ CAT3 CAT1ₓ` | 1 | 2 | **unlicensed** | neither |
| **matrix**, one CAT2 or an aux intervening | 1 | 4 | **unlicensed** | neither |
| `…CAT1ₓ CAT6 CAT3 CAT1ₓ` — cross-nucleus pair, ant = 2nd nominal | 2 | 2 | pron | pron (agree) |
| `…CAT1ₓ CAT4 CAT9 CAT1ₓ…` (duplicate = embedded subject) | 3 | 3 | refl | pron |
| `…CAT1ₓ CAT4 CAT6 CAT1ₓ…` (duplicate = PP complement) | 3 | 3 | refl | pron |
| any pair anchored on the **first** nominal | 1 | any | **unlicensed** | per H's rules |

P's rule is genuinely positional (any intervening word flips the distance, and
moving the antecedent one nominal flips the ordinal); H's rule is genuinely
structural (indifferent to both, sensitive to dominance). Neither reduces to the
other. The **matrix** rows are the class both families leave unlicensed: for H
because A′ excludes matrix nuclei and Pron excludes same-nucleus pairs; for P
because a matrix subject is the first nominal in a head-initial string and ordinal
1 licenses nothing — so **P refl's antecedent is never the matrix subject**.

### 3.4 The coverage asymmetry (kept by design)

H's two sub-rules are no longer complementary over duplicate configurations: A′
excludes matrix nuclei and Pron excludes same-nucleus pairs, so a **matrix
subject–object pair falls through both**. This is deliberate — widening Pron to
absorb those pairs would change its measured separability profile; leaving it alone
buys a genuinely blocked configuration (the distributional evidence §1(a) asks
for). Both families now have duplicate configurations that never substitute. What
must be controlled is the **aggregate** duplicate-bearing-but-unsubstituted rate
across the two corpora.

### 3.5 Structural parity for refl is unattainable

Once §1c is applied, the refl slices of H and P **cannot** be structurally matched,
and no knob is to be moved to make them so. A′ *requires* an embedded nucleus (H
refl embedding 100%); rule B names no clause structure (P refl embedding ~52%,
above its ~15% neutral rate because the offset-3/ordinal-3 geometry correlates with
embedding). Matrix transitivity diverges independently on the same slice (H ~49% at
natural rate vs P ~78%). Both are genuine consequences of the rules — **accept,
document, carry as covariates, do not recalibrate.** Neither embedding rate nor
matrix transitivity is a matched constant across the four corpora; both must enter
downstream analysis as covariates. The covariate attaches to the
**phenomenon label**, so downstream models control for it by conditioning on
**slice**, not grammar.

---

## 4. Auxiliary movement

- **H:** front the CAT3AUX of the **matrix Type2**. A matrix Type2 holds at most
  one CAT3AUX by construction, so no selection among auxes arises and the rule
  consults no linear position. An aux only inside an embedded clause is a structural
  blocking case (`_skipped`).
- **P:** front the **first CAT3AUX in left-to-right scan**, regardless of clause.
  Never blocked when an aux exists anywhere. This scan is the entire divergence.

Both fronting operations leave a `#NULL#` **trace** at the vacated slot, mirroring
wh so §8.2 target localization is uniform. Compound tense is verb-then-aux, so the
trace sits immediately after the main verb; the surprisal target is the next
contentful word, or `[EOS]` when the aux was clause-final. The trace strips to
**zero model tokens** (the model's token sequence is unchanged; only the surface
string and metadata gain it, and `length` gains 1). Because fronting removes the
aux from index *k*, leaves a GAP there, and prepends it at 0, **every token shifts
by exactly +1**, so item distances and CAT1-position ordinals are preserved — a
fact worth knowing, not a licence to skip base recovery.

When the aux phenomenon is drawn, an aux is forced to exist but its site is drawn:
**matrix** with probability `Q_AUX_MATRIX`, otherwise **embedded-only** (an
embedded clause is forced, one embedded verb realized as compound tense, matrix
clause simple). Neither aux rule refers to the embedded clause's attachment, so on
the embedded-only branch the **host is drawn subject-or-object at `Q_EMB_OBJECT`**,
not pinned (§1c). The rules decide the outcome: P fronts the embedded aux (the
divergence direction H never produces); H skips the embedded-only plan (~25%).

**Embedded-aux density — `P_AUX_EMBEDDED = 0.60`.** In-situ compound tense inside
an **embedded** clause is drawn at 0.60, not the matrix `P_AUX = 0.30`. One knob,
identical across all four grammars, read only at `grammar_H/build.build_type4` and
`grammar_P/build._maybe_attach_cat9`; the matrix draw is unaffected. It lifts ≥2-aux
sentences to ~5–6.4% of the corpus so the aux structural probe tests a
configuration the model actually sees. The four grammars do **not** match on ≥2-aux
exposure (~1.4pp spread, family-wise) — this is an **exposure covariate** for any
cross-grammar aux probe comparison, deliberately not equalised (per-grammar knob
values would violate the matched-procedure constraint). Known cost, not offset:
π_corpus scoring coverage on P/L′ aux probe items falls (the corpus replay asserts
`aux_site = matrix`; a more-often-embedded first aux builds a different sentence).
The **at-most-one-matrix-aux invariant** is unaffected (0 violations in 8,000
draws per grammar).

## 5. Pronoun realization

CAT1PRON pronouns arise from exactly one mechanism: the anaphoric-binding
phenomenon (§3). Each pronoun is a substituted coreferential duplicate from the
closed-class `cat1pron` pool; the base grammars never emit a CAT1PRON. One
phenomenon applies per sentence, so a sentence's only pronoun (if any) is the one
produced by its binding label, and only binding-labeled items (~5%/5%) contain one.
Neutral, wh and aux sentences are pronoun-free; a pronoun is never a wh target in a
naturally generated sentence.

## 6. Wh-movement

The particle CAT8 is prefixed and the target block moves to string-initial
position, leaving a `#NULL#` gap. Licensing is evaluated on the **base**
string/tree, before fronting. Two clauses per grammar:

| Grammar | clause (i) — the swap axis | clause (ii) |
|---|---|---|
| H | **matrix object that HOSTS an embedded clause** *(structural)* | **inside a Type3 dominated by a Type4** *(structural; any attachment, any nesting depth)* |
| **H′** | **third CAT1-position** *(positional, = P's)* | = H |
| P | **third CAT1-position** of the sentence | **second CAT1-position after the second CAT6** *(positional)* |
| **L′** | **matrix object that HOSTS an embedded clause**, from generation history *(structural, = H's)* | = P |

Each prime differs from its parent in exactly clause (i), and the swap sits on the
one licensing notion with **no local surface shadow**. H reads the structural
clause off the constituent tree, L′ off the flat token list (`wh_structural.py`,
two readers). The positional clauses live in `grammar_P/wh.py`.

**Targets.** CAT1PRON counts as a CAT1-position for the ordinal scans; as a
*target*, CAT1 and CAT1PRON-**Pron** are licit, CAT1PRON-**Refl** never fronts.

### Clause (i)

> A nominal is licensed iff it is the **object of the matrix clause** **and it
> hosts an embedded clause.**

A conjunctive **narrowing**: nothing becomes newly licensed. The licensor stays at
the extraction site — fronting moves only the block and strands the hosted clause
at the gap, so the *gap* carries the evidence for licensing, not the filler
(`verdicts.recover_tree`, `metadata._wh_facts_tree` read the gap's following
siblings via `_stranded_has_type4`).

**Forcing: the union case (the documented §1c exception).** The four grammars'
clause-(i) predicates are not co-satisfiable at a common minimum: H and L′ require a
matrix object hosting an embedded clause (entailing matrix transitivity and
embedding); H′ and P require only a third CAT1-position. Because the matched
procedure requires **one base plan for all four**, and matrix-transitive-plus-
embedding is strictly the stronger predicate, the wh plan forces the **union** (a
matrix-transitive clause + one embedded clause). This is the only place forcing
exceeds a grammar's own predicate.

**Named consequence.** For **H′ and P**, wh items are conditioned on matrix
transitivity and embedding, which their clause (i) does not require. Any analysis of
H′/P wh items must treat matrix transitivity as a constant of the slice and check
whether it correlates with anything the positional clause can read.

**Why the host must be MIXED.** The host is drawn subject-or-object, **not pinned**:
pinning makes every matrix object a host, so the second conjunct becomes vacuous
inside the wh slice and the rule collapses to plain "matrix object". Mixing restores
the contrast, at the cost of clause (i) being available in ~53% of wh items.

> General lesson: forcing a conjunct makes it vacuous unless it still varies across
> candidates within one sentence.

*Known and accepted:* E3's difficulty is **directional** — the licensing clause
sits to the target's right, so it is hard for a left-to-right observer, not hard in
both directions. Defensible for a causal LM, but a weaker claim than the positional
arm supports, and stated as such.

### P-family clause (ii)

> The **second** CAT1-position after the **second CAT6** of the sentence.

At most one licensed position per sentence: the anchor ordinal picks exactly one
CAT6 and the nominal ordinal exactly one position after it. No fallback to an
earlier nominal or the first CAT6. Three features are load-bearing: an **ordinal
over the anchors** (the second CAT6 must be counted from the sentence beginning,
keeping the clause inside P's one-reference-point-at-sentence-start constraint);
**CAT6 rather than CAT5** (CAT5s are appended contiguously, so "the second CAT5"
collapses into a `CAT5 CAT5` bigram; CAT6s each head their own PP and are never
contiguous); and **the second nominal after it, not the first** (a CAT6 is
immediately followed by its complement, so n=1 would reinstate adjacency). 100% of
its licensed sites are positions clause (i) does not already license.

*Accepted cost.* Any "second nominal after X" clause implies ≥3 nominals (clause
(i)'s own condition), so clause (ii) can only add sites **inside** bases clause (i)
already covers, never rescue one it misses.

### What no longer licenses anywhere

The matrix subject; embedded subjects and objects outside a Type4-dominated Type3;
Type3s attached in the matrix clause. **CAT5 appears in no licensing clause**, and
CAT9 never did (only a clause boundary). Embedded subjects were cut (surface pattern
perfectly regular: first CAT1 after CAT9); matrix Type3s were cut (a `CAT1 after
CAT6` bigram). H's clause (ii) has no local regularity: the Type4's right edge is
unmarked, so deciding whether a `CAT6 CAT1` lies inside or after the embedded clause
requires verb-counting across the boundary — the same epistemic grade as clause (i).

### Blockability

| grammar | wh skip | aux skip |
|---|---|---|
| H | ~22% | ~25% |
| H′ | 0% | ~24% |
| P | 0% | 0% |
| L′ | ~32% | 0% |

The forced base always supplies three CAT1-positions, so the positional-clause
grammars (H′, P) can no longer block wh at all, while the structural-clause ones
(H, L′) still can — their clause (i) also needs the embedding to have landed on the
matrix object, which the mixed host leaves to a draw. That asymmetry is a
consequence of the forcing, not the rules — the same trade refl made: blockability
traded for a phenomenon slice that is actually populated. The H/P divergence
remains bidirectional.

---

## 7. Generation machinery

### Per item

1. **Phenomenon draw** (70/10/10/10; binding sub-draw refl/pron 50/50).
2. **Site draws** — aux site (`Q_AUX_MATRIX`). The refl nucleus is *not* drawn
   (A′ fixes it as embedded).
3. **Build**, forcing exactly each phenomenon's licensing predicate and no more
   (§1c). Everything not listed is drawn at its natural rate:
   - *refl and pron, BOTH families*: **nothing is forced at build time** — all four
     binding rules are conditioned by resampling alone (step 5).
   - *wh, both families*: matrix-transitive clause + one embedded clause, host
     **mixed**, inner transitivity natural — the **union** predicate, over-
     conditioning H′ and P (§6). The only build-time forcing that exceeds a
     grammar's own predicate.
   - *embedded-only aux, both families*: one embedded clause with compound tense,
     **host drawn at `Q_EMB_OBJECT`**, nesting probabilistic. Independent of the
     binding rules.
4. **Duplicate screen** — resample (max 20) while a forbidden chance duplicate
   exists (H: same-stem **embedded** nucleus subject/object; P: same-stem pair in a
   licensed configuration).
5. **Binding-geometry resample** — resample the base until it hosts the rule (max
   500, both families): P onto the offset + antecedent ordinal, H's refl onto A′'s
   embedded-nucleus predicate. If the cap is ever approached it is raised, never met
   by reintroducing build-time forcing.
6. **The transformation**, returning the refined label or `*_skipped`.
7. **Linearization.**

Skipped items are discarded by the corpus builder's slot quota, so the labeled
corpus is conditioned on applicability. Transformed strings may incidentally
exhibit surface patterns the base rules exclude (derivational opacity, not a screen
failure).

### Family parity

Both families draw embedding **once per clause** at 0.15, one embedded clause per
draw, nesting via the embedded clause's own recursive draw. The **host** is shared
and explicit (`Q_EMB_OBJECT = 0.5`). PP expansion is likewise aligned (single PP
draws per host, complement nominals nesting at a shallow rate, no lexical
`cat6_compatible` gate); the nominal-zone PP lands at the **end** of the host's
`CAT1 [CAT4] [CAT2]*` block in both families.

All four grammars match on every measured base-structure row
(embedding ~14.7%, PP ~51%, PP-under-Type4 ~7.6%, ≥3 CAT1-positions ~40.5%,
transitive ~48.5%) and on neutral length (mean 8.9–9.0, p99 28–29). **One row does
NOT match: ≥2-aux exposure** — an exposure covariate, deliberately not equalised
(§4). These rows are measured on **neutral** bases; within a phenomenon slice the
forcing of §1c can shift them (§3.5).

### Probability knobs (`rules.py`)

| Knob | Value | Role |
|---|---|---|
| `PHENOMENON_PROBS` | .70/.10/.10/.10 | neutral / binding / aux / wh |
| refl vs pron sub-draw | 0.50 | refines binding to 5%/5% |
| `Q_EMB_OBJECT` | 0.50 | Type4/CAT9 host = object vs subject; also the refl and wh plans' forced embedding host |
| `P_PRON_FIRST` (H) | 0.20 | pron substitutes the first occurrence |
| `Q_AUX_MATRIX` | 0.75 | aux-item aux sits in the matrix clause |
| `REFL_OFFSET` / `PRON_OFFSETS` (P) | 3 / (2, 4) | licensed binding distances |
| `REFL_ANT_ORDINALS` / `PRON_ANT_ORDINALS` (P) | (3,) / (2, 3) | licensed antecedent CAT1-position ordinals |
| `WH_I_NOMINAL_ORDINAL` (P) | 3 | clause (i) |
| `WH_II_ANCHOR_CAT` / `_ANCHOR_ORDINAL` / `_NOMINAL_ORDINAL` (P) | CAT6 / 2 / 2 | clause (ii) |
| `P_AUX` | 0.30 | in-situ compound tense in the **matrix** clause |
| `P_AUX_EMBEDDED` | 0.60 | in-situ compound tense in an **embedded** clause (§4); read at `build_type4` and `_maybe_attach_cat9` only |
| `P_TYPE4`/`P_CAT9`, `P_CAT2`, `P_CAT5`, `P_CAT6_*`, `P_PP_*` | — | structural expansion rates |

### Surface format and module map

Words are space-separated; morphemes within a word `#`-separated, order
`stem#number#gender` (+`#tense` on verbs). Compound tense: CAT3 is bare and CAT3AUX
carries INFL1+INFL3. The two families are byte-identical in format and share one
vocabulary and tokenizer.

```
a#3#1 j#3#1#7 c#3#1        subject + verb + object (singular, gender 1, present)
a#3#1 j n#3#1#7 c#3#1      compound tense (bare CAT3, inflected CAT3AUX)
```

| Grammar H | Grammar P | Role |
|---|---|---|
| `nodes.py` | `tokens.py` | node / flat-token dataclass |
| `lexicon.py` | `lexicon.py` | loads the shared `lexicon.json`, category lookups |
| `rules.py` | `rules.py` | probabilities, bounds, inflectional pools |
| `build.py` | `build.py` | structure construction (P also does inline inflection) |
| `features.py` | — | inflection assignment + head-driven agreement |
| `transforms.py` | `transforms.py` | binding, aux movement, wh-movement |
| — | `binding.py`, `wh.py` | the positional licensing rules, one copy each |
| `linearize.py` | `linearize.py` | structure → surface string (+ `to_tokens_with_refs`) |
| `generate.py` | `generate.py` | pipeline orchestration |

Grammar P builds each clause left-to-right, no tree, no look-ahead: base (subject +
verb [+ object]) → CAT3AUX → inflection → CAT4 → first expansion (CAT2, CAT5) →
second expansion (CAT6 PPs, then CAT9 substrings, each recursing). The clause domain
used by binding and the wh scans is `clause_id`, a positional index delimited by
CAT9 — not a dominance domain.

---

## 8. Corpus metadata sidecar

**Goal.** Persist, at generation time, an exact per-sentence record carrying (a) the
structural description and (b) the localization of the **target token** for
surprisal, plus **control tokens** for differential-surprisal analysis. Nothing is
parsed from the surface or recovered by replay — every field is read from the actual
`Node` tree / `Token` list that produced the emitted string.

Output is a sidecar `corpora/{set}/{split}/{grammar}.meta.jsonl`, line-aligned to the
corpus file, with an explicit stable `id` on both records. The training loader reads
only `sentence`, so the sidecar leaves training untouched.

### 8.1 Target-token definition

- **Binding** → the substituted CAT1PRON; target morpheme = its **stem**.
- **Aux & wh movement** → the **stem morpheme of the first contentful word after the
  gap** (the gap renders as the null morpheme and strips to zero model tokens).
- **Sentence-final-gap fallback** → the `[EOS]` model token.

### 8.2 Localization

Targets and controls are localized as **model-token indices** (morpheme level), not
word ordinals: each is a `model_token_span = [start, end)` into the tokenized
sequence, accounting for the `[BOS]` offset, `#`-morpheme expansion and null-morpheme
removal. Every span round-trips against the real tokenizer.

### 8.3 Annotated-token schema

```
{
  "role": "target" | "licensed_unrealized" | "unlicensed" | "category_baseline",
  "category": "CAT1" | "CAT1PRON" | "CAT3AUX" | ...,
  "licensing_status":  { "H": "licensed|unlicensed|na", "P": ..., "Hprime": ..., "Lprime": ... },
  "off_frame_status":  { "<phenomenon>": { "H": ..., ... }, ... },
  "model_token_span": [start, end],
  "word_index": <int>
}
```

- **`role`** — *target* = licensed *and* selected; *licensed_unrealized* = licensed
  but not the fired instance; *unlicensed* = a category-eligible candidate at a
  position the grammar does not license; *category_baseline* = same category, no
  active dependency.
- **`licensing_status`** is grammar-relative, recovered on the **base** and mapped to
  the derived surface span. Own family exact; cross-family only where a clean verdict
  exists — aux → all four keys exact (preserving the H↔P divergence); binding and wh
  → own + same-family prime exact, cross-family `na`. **Family collapse:** H′
  delegates binding and aux to H, L′ to P, so those two phenomena have only two
  distinct verdicts; only wh has four.
- **`off_frame_status`** — a candidate's own-rule verdict for phenomena *other* than
  the sentence's active one (keyed by phenomenon name, excluding the active one).
  It makes naturally occurring blocking/licensing configurations for non-active
  phenomena visible. Computed with the same site/licensing primitives the
  on-frame annotators and probes use. Two guards make it sound: `role == "fronted"`
  is excluded from candidacy (a fronted node's fresh GAP would otherwise double-count
  it), and base order is recovered before any positional predicate
  (`metadata.base_order_tokens`).

### 8.4 Structural fields

`length`, `morphemes_mean`, `n_clauses`, `embedding_depth` (tree families only;
`null` for P/L′ — positional grammars have no constituent tree, and their depth axis
is `n_embedded_clauses`, exact because positional embedding is a non-branching
chain), `n_embedded_clauses`, `n_modifiers`, `n_verbs`, `expansion_sites`,
`has_aux`, `has_gap`, `has_pron`, plus the site primitives `bind_site`, `aux_site`,
and the wh extraction site/clause. Cross-family comparison uses `length`,
`n_clauses`, `expansion_sites`. Site primitives are **threaded** from the
`_one_item` draws, never inferred; what is not drawn is filled from the struct's
unique GAP/CAT1PRON clause.

### 8.5 Emission path

`generate_string` attaches a transient `_payload` (struct + site draws);
`generate_batch` calls `metadata.finalize_item` on **accepted** items only —
annotation consumes no RNG, so the trajectory is exact and rejected draws cost
nothing. The record raises if the identity hook does not reproduce the emitted
surface. `CorpusBuilder._write` splits each item into corpus record + sidecar
record, stamps the shared `id` (`{grammar}-{split}-{index:05d}`) on both, and writes
the line-aligned sidecar; a partially annotated batch raises.

---

## 9. Structural probes

Two small minimal-pair corpora that investigate the **hypothesis space the model
has inferred**, by contrasting items licensed vs unlicensed by the grammar's own
rules. The logic is identical for all four grammars; only licensing decides what
counts as a violation.

- The **IOD** probe tests surprisal at crucial positions on configurations that
  match the model's training exposure (the conformity-matched stratum); the
  unmatched stratum tests generalization and is read separately.
- The **OOD** probe is structured identically but contains long, complex
  grammatical sentences. IOD and OOD share the **same generation rules with the same
  probabilities** — OOD strings are simply longer.

**Pair shape.** Both members are transformed, on one shared base: the grammatical
member applies the rule at a *licensed* site, the ungrammatical member applies the
**same** rule at an *unlicensed* site. Both members are always well-formed strings;
the causal contrast is licensed vs unlicensed application. Surprisal is recorded for
all tokens and the analysis asks whether it peaks at the ungrammatical member's
target. A ±transformation design would only show that a substitution/movement was
*registered*, not *where the rule licenses it*.

### Subcases

Seven subcases, two members per pair. `n_per_subcase` is a build parameter
(`--probe-n-per-subcase` / `--select-n-per-subcase`), **300** by default, i.e. 2,100
pairs / 4,200 items per grammar per probe. Binding subcases split their quota over
`BINDING_MODES`: `refl` three ways (wrong_item / wrong_site / no_substitution),
`pron` two (wrong_item / wrong_site).

| Subcase | Base requirement | Grammatical | Ungrammatical |
|---|---|---|---|
| `wh_i`, `wh_ii` | ≥1 nominal licensed *exclusively* by that clause, and ≥1 eligible nominal licensed by neither | front the licensed block | front an unlicensed block |
| `aux` | ≥2 CAT3AUX (one licensed, one not) | front the licensed aux | front an unlicensed aux |
| `refl`, `pron` | ≥1 licensed site and ≥1 unlicensed site, each with a feats source | substitute at the licensed site | **wrong_item** (correct site, wrong pron subtype — site must be subtype-exclusive), **wrong_site** (correct subtype, unlicensed site), and — `refl` only — **no_substitution** (licensed configuration left bare) |
| `neutral_agreement` | an agreement dependency | the base | flip **one** inflection value on **one** agreeing token |
| `neutral_positional` | any neutral base | the base | move **one** token to an illegal position |

Both members of a binding pair are **genuine coreferential duplicates**: each site
is coindexed to its anchor by the same `_coindex` step generation uses (§3) — the
anchor's full CAT1 lexeme and INFL1 feats, with a subject target's clause verb
realigned in both members — before it is spelled out as the pronoun. Coreference and
agreement are therefore identical across the pair; the members differ **only** in the
licensing boundary under test (`wrong_site`: site licensed vs. not; `wrong_item`:
correct vs. swapped subtype at the *same* subtype-exclusive licensed site;
`no_substitution`: pronoun written vs. left bare). The anchor is the rule's own
**licensing antecedent** wherever the site is licensed (`wrong_item`,
`no_substitution`, and the grammatical member of `wrong_site`); only the unlicensed
member of `wrong_site` has no licenser and borrows the nearest preceding full CAT1
purely to keep agreement coherent.

### Mechanism

Bases are drawn from the grammar's **natural generation**, then kept only if they
admit the pair. Under §1c this is the *same* conditioning the corpus applies for
every phenomenon whose forcing is exactly its licensing predicate. Two residuals
remain, both recorded per item:

- **wh — reproduced by forcing the union in the probe too**
  (`structural_probes.UNION_FORCED_SUBCASES`). The corpus forces the §6 union for
  every wh item; a naturally-drawn probe base could license wh without that
  geometry and lie off the corpus's path, so the probe reproduces the corpus's
  conditioning rather than stratifying around it.
- **aux.** The corpus conditions on **one** aux; the probe pair requires **two**
  (one licensed, one not). Two-aux bases are in support but rare, so aux pairs sit
  in a low-mass region of the training distribution. **The exposure is thin, and
  this bounds what an aux result can say:** a weak aux result is ambiguous between
  failure to learn the rule and thin exposure to two-aux sentences, and must be
  reported as such. `base_conformity.density` carries `n_aux` and `n_matrix_aux` per
  item so the stratum is identifiable.

**Scope statement.** Because the pair shape requires a **licensed** site, every
H-family aux base carries a matrix aux, so the probe cannot represent the
"aux exists only in an embedded clause and H licenses nothing" case — that is a
**no-application** case, excluded by the licensed-vs-unlicensed pair shape by
construction. H's aux blocking evidence is **distributional** (the training corpus,
where H skips the embedded-only branch), not a probe target.

**Conformity field.** Every probe meta record carries `base_conformity`, computed at
generation time by the **same forcing predicate the corpus generator uses**: whether
this base lies on the corpus's conditioning path for its phenomenon, and if not,
which conjunct fails. It makes the two residuals a **stratification** (matched vs
unmatched) rather than a silent mixture. `matched` is a **support** predicate (can
the corpus produce a base like this at all) — the right question for the wh residual,
the wrong one for aux (whose residual is low mass, not zero mass, and reads 100%
matched); aux is separated by the `density` block instead.

Per pair: draw one base, compute a grammar-relative **site table**
(`violations.wh_sites` / `aux_sites` / `binding_sites`), deep-copy the base twice,
apply the **same** site-directed operator to both copies. Neutral pairs apply a
single edit to one copy. Rejection sampling enforces the length window (on the
grammatical member), EOS exclusion for movement targets (both members), and, for
neutrals, that the edit changed the surface. Both members get the standard §8 meta
record, built on each member's own struct — exact by construction. Wh pairs are
stamped with both clause-(i) verdicts, so divergence-relevant pairs are identifiable.

**The block-granularity prerequisite.** All four grammars move and substitute
whole **blocks** (§2): stranding rather than deleting means a violating application
always leaves contentful post-gap material (which the EOS-free requirement needs),
and the grammars differ only in licensing.

---

## 10. Splits and build

```
corpora/set_N/
  train/{g}.jsonl                  100,000   lengths [2,25]   70/5/5/10/10 mix
  validation/{g}.jsonl              10,000   lengths [2,25]   same
  test_indistribution/{g}.jsonl     15,000   lengths [2,25]   same
  structural_probe_IOD/{g}.jsonl     4,200   lengths [2,25]   2,100 pairs
  structural_probe_OOD/{g}.jsonl     4,200   lengths [26,48]  2,100 pairs
  structural_probe_SELECT/{g}.jsonl  4,200   lengths [2,25]   2,100 pairs
```

Probe sizes are `n_per_subcase` × 7 subcases × 2 members, at the default
`n_per_subcase=300`. Each with a line-aligned `.meta.jsonl`. Train/validation/test
are independent, non-overlapping RNG trajectories (distinct seed offsets from one
base seed); the probes use their own `random.Random` at seed +200k / +250k / +300k,
independent of build order and of each other. Probe records additionally carry
`pair_id`, `variant` (`grammatical`/`ungrammatical`), `subcase`, `violation`, and
the two clause-(i) verdict stamps.

Build: `python -m scripts.build_all_corpora --grammar all --seed <seed>` (the
production set, `set_3`, used seed 43). Drift check
(opt-in, after regeneration): `CORPUS_DETERMINISM=1 pytest tests/test_metadata_sidecar.py`
rebuilds every split at the on-disk sizes and byte-compares.

### Item identity — `item_uid`

Every corpus record and sidecar record carries `item_uid`: for the three corpus
splits a **content-derived** 32-hex `blake2b` digest over (grammar, split, surface
string, derivation signature), assigned by `CorpusBuilder._write` via
`grammars.tracing.item_uid`; for the three probe splits the builder's own
`{pair_id}/{variant}`.

It is the **only** admissible join key from model scores back to generation records.
`id` (`{grammar}-{split}-{index}`) names a LINE, not an item: a rebuild in a
different order re-points it silently. Only content-derivation recovers the same key
for the same item across regenerations.

