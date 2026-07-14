# Transformation rules — official formalization (all four grammars)

**Status: implemented (2026-07-13; wh system revised 2026-07-14 per audit
§F).** This document is the authoritative description of the transformation
rules of Grammar H (hierarchical), Grammar P (positional), and — for the wh
system, §6 — the mixed pair H′/L′, after the reformalization decided in
`LINEAR_HIERARCHICAL_TRIVIALITY_AUDIT.md` (per-rule audit + assessment +
D1–D10 decisions + the final §F wh design). It supersedes the corresponding
sections of the original spec (`experiment.md` §7) and the retired
EMBEDDED_ENRICHMENT notes (background binding is §5 below).
H′/L′ share their parents' generation plan **by construction**: each parent's
`_one_item` accepts the transform dispatcher as a parameter, and each prime
passes its own transform module while delegating every non-wh phenomenon back
to the parent's dispatcher — identical draws, forced structures, screens, and
background binding, differing only in the wh clause-(i) swap specified in §6.

Code: `grammars/grammar_H/{transforms,build,generate,rules}.py`,
`grammars/grammar_P/{transforms,build,generate,rules}.py`,
`grammars/grammar_H_prime/`, `grammars/grammar_L_prime/`,
`grammars/verdicts.py`.
Regression tests: `tests/test_reformalized_rules.py`,
`tests/test_embedded_enrichment.py`, `tests/test_verdicts.py`,
`tests/test_probes.py`.

---

## 1. Design principles

Every transformation must satisfy two criteria, evaluated per grammar pair:

- **(a) Blockability.** There exist drawn structures where the phenomenon
  cannot apply. Because the corpus builder discards `*_skipped` items,
  blockability is realized **distributionally**: the model's evidence is the
  contrast between configurations where the phenomenon systematically occurs
  and structurally/positionally matched configurations where it never does
  (e.g. unsubstituted duplicates at unlicensed positions; in-situ embedded
  auxiliaries that never front in H).
  *Blockability is a property of the RULE SYSTEM, not of the data*
  (decision, 2026-07-14): exposing the model to blocked configurations is a
  training-distribution device for slimming the hypothesis space, not part
  of the rules' definition. In particular, blocked configurations may be
  unreachable in some regions of the length spectrum (e.g. every P-family
  string of ≥ 25 tokens has a third CAT1-position, so wh is never blocked
  there) — this is a fact about licensing-at-length, not a defect, and
  evaluation splits must not require blocked items where the rules cannot
  produce them.
- **(b) Non-reducible divergence.** For matched base structures, H's rule and
  P's rule select **different** targets (in both directions), and neither
  selection function is statable as a surface variant of the other.

Two framing choices apply across all rules:

- **Licensing, not blocking.** Rules are stated as the positions where a
  phenomenon CAN occur. The model receives direct positive evidence of the
  licensed set; the unlicensed complement simply never fronts/substitutes.
- **Derivational rules, surface composition.** Each rule is evaluated at its
  own derivational step (binding and wh licensing on the **base string/tree**;
  aux on the built clause). Transformations compose afterwards, so a derived
  surface string may obscure the base configuration (a fronted block shifts
  positional distances; movement separates a reflexive from its antecedent).
  This is intended: it mirrors the derivational opacity of natural language
  and prevents the licensed sets from collapsing into exceptionless surface
  n-gram cues.

One phenomenon is pre-selected per sentence (70% neutral, 10% per
transformation; binding refines to refl/pron at 5%/5%). Background binding
(§5) is the only sanctioned second substitution.

## 2. Definitions

- **Nucleus (H).** For any clause (Type0, matrix or embedded): the subject
  head CAT1, the verb, and the object head CAT1 — nothing else. PP
  complements, CAT2 hosts, and all material inside Type3/Type4 are outside
  every nucleus (decision D2). "Minimal complete transitive clause" always
  means this nucleus.
- **Item / distance (P).** One item = one whitespace token of the surface
  string (CAT4, CAT2, CAT5, CAT6, CAT9, CAT3AUX, CAT1PRON all count). The
  `#`-attached inflection values are morphemes of their word, not items
  (decision D4). Distances are counted on the **base string** (before any
  transformation).
- **CAT1-position (P).** Any CAT1 **or CAT1PRON** token: the ordinal scans of
  the wh rule do not distinguish nouns from pronouns (decision D7).
- **Coreferential duplicate.** The derivational intermediate of binding: the
  same CAT1 lexeme with the **same INFL1_number** (gender is inherent to the
  lexeme) in two positions (decision D3). Duplicates are derivational only —
  see the per-rule surfacing constraints.

## 3. Anaphoric binding

### 3.1 Grammar H (hierarchical)

**Refl (Principle A analog).** A CAT1PRON-Refl must be bound within its
nucleus: the derivation places the same CAT1 as subject and object of one
nucleus — **matrix or embedded** (embedded with probability `P_BIND_EMB`) —
and obligatorily substitutes the **second** occurrence (the object) with
Refl, which copies the subject head's number/gender.

- *Surfacing constraint:* no final string may contain the same CAT1 stem as
  subject and object of one nucleus (that configuration exists only as the
  refl intermediate). Generation screens chance duplicates of this shape in
  **all** items and resamples (`has_nucleus_duplicate`).
- *Blocking:* a sentence with no transitive nucleus hosts no refl. Under
  phenomenon-driven generation the licensing nucleus is forced (matrix
  transitivity, or a forced embedded transitive Type4), so refl skips are
  rare; the distributional evidence is the screen (same-nucleus duplicates
  never surface) plus refl's strict nucleus locality.

**Pron (Principle B analog).** A CAT1PRON-Pron must be free in its nucleus:
the coreferential pair spans two positions **not** forming the subject–object
pair of one nucleus (cross-clause pairs, pairs involving PP complements,
etc.). Either occurrence may be substituted — the second in the majority of
cases (`1 − P_PRON_FIRST`) — copying the surviving antecedent's features. No
transitivity is required anywhere. Dominance pairs (one nominal containing
the other) are excluded.

- *Optionality:* substitution of a cross-nucleus duplicate is optional at the
  grammar level. Phenomenon-labeled pron items always substitute (that is
  what the label means); unsubstituted cross-nucleus duplicates surface in
  neutral/background items via chance draws (~2% per nominal pair under the
  Zipf lexicon) — this is H's blocking-contrast evidence.
- *Blocking:* no second nominal outside the nucleus → `_skipped`.
- Pron–pron coreference chains are excluded (antecedents and substitution
  targets must be full CAT1 nominals).

**Agreement fixup.** Substituting a clause **subject** realigns that clause's
verb-carrier INFL1 features with the pron (the copy happened at assignment
time); with D3 feature identity this is only observable when rules compose.

### 3.2 Grammar P (positional)

No domain notion. Both sub-rules are pure distance licensing on the base
string, deterministic in every respect (decision D5): the antecedent is the
**first** occurrence, the substituted item is always the **second**, and
substitution at a licensed distance is **obligatory**.

**Refl.** Licensed iff the coreferential CAT1 sits exactly **3 items** after
its antecedent CAT1. The second occurrence is substituted by CAT1PRON-Refl,
copying the antecedent's number/gender.

**Pron.** Licensed iff the coreferential CAT1 sits exactly **2 or 4 items**
after its antecedent. Same substitution with CAT1PRON-Pron.

- *Surfacing constraint:* a same-stem CAT1 pair at distance 2, 3, or 4 in the
  base string must never surface unsubstituted. Generation screens all items
  (`has_licensed_duplicate`) and resamples.
- *Blocking:* duplicates at any **other** distance are unlicensed and surface
  unsubstituted (chance draws) — P's positional blocking evidence. A drawn
  string with no CAT1 pair at a licensed offset skips the phenomenon.
- The same agreement fixup applies when the substituted token is a subject
  (its clause's CAT3/CAT3AUX realign).
- CAT1PRON tokens count as intervening items but are never antecedents or
  targets.

### 3.3 Why this opposition is non-trivial

The two rules **coincide** on exactly one anchor configuration — the minimal
transitive clause with a countable subject (`CAT1 CAT4 CAT3 CAT1`, object at
distance 3) — and come apart in both directions everywhere else:

| Base configuration | distance | P | H |
|---|---|---|---|
| `CAT1ₓ CAT4 CAT3 CAT1ₓ` | 3 | refl | refl (anchor case — agree) |
| proper-like subject: `CAT1ₓ CAT3 CAT1ₓ` | 2 | pron | refl |
| one CAT2 or an aux intervening | 4 | pron | refl |
| `CAT1ₓ CAT4 CAT9 CAT1ₓ…` (duplicate = embedded subject) | 3 | refl | pron |
| `CAT1ₓ CAT4 CAT6 CAT1ₓ…` (duplicate = PP complement) | 3 | refl | pron |

P's rule is genuinely positional (any intervening word flips it); H's rule is
genuinely structural (indifferent to distance). Neither reduces to the other.
Measured on 6k items/grammar (seed 11): 13% of P refl items cross a clause
boundary (P-refl/H-pron direction); 26% of P pron items are same-nucleus
subject–object pairs (P-pron/H-refl direction); 83% of H refl items sit at a
surface distance ≠ 3 (H-refl/P-unlicensed direction).

### 3.4 The coverage asymmetry (kept by design)

H's two sub-rules are complementary over duplicate configurations (every pair
is nucleus-internal or not); P's licensed distances {2,3,4} are a strict
subset of possible distances, so P has duplicates that can never substitute.
This asymmetry is **constitutive of the opposition**: a positional rule
system can only achieve complementary coverage by licensing every position,
i.e. by becoming vacuous. What is controlled instead is the **aggregate**
statistic: the rate of duplicate-bearing-but-unsubstituted sentences should
be comparable across the two corpora (both currently ride on the same ~2%
chance-duplicate rate filtered by each grammar's own screen; verify at
corpus-rebuild time, D10).

## 4. Auxiliary movement

Rule definitions are unchanged; the **generation plan** changed.

- **H:** front the **matrix** CAT3AUX to sentence-initial position. An aux
  only inside an embedded clause is a structural blocking case (`_skipped`).
- **P:** front the **first CAT3AUX in left-to-right scan**, regardless of
  clause. Never blocked when an aux exists anywhere.

When the aux phenomenon is drawn, an aux is forced to exist, but its site is
drawn: **matrix** with probability `Q_AUX_MATRIX`, otherwise **embedded-only**
(an embedded clause is forced and one embedded verb is realized as compound
tense; the matrix clause stays simple). Both grammars run the same plan; the
rules decide the outcome:

- P fronts the embedded aux (measured: 24% of P aux items front a
  `clause_id ≠ 0` aux) — the divergence direction H never produces;
- H skips the embedded-only plan (measured: 22% of H aux items skip —
  blockability finally exercised); its corpus evidence is that fronted auxes
  are always the matrix one while embedded auxes appear in situ (background
  `P_AUX` embedding).
- The pre-existing thin divergence (subject-attached embedded aux linearly
  precedes the matrix aux → P fronts the embedded one even when a matrix aux
  exists) remains on top of this.

## 5. Background binding (successor of the embedded enrichment)

With probability `P_BG_BIND`, a sentence whose own phenomenon is **not**
binding receives one additional pronominal substitution, applied by the
grammar's own binding rule restricted to embedded material (H: refl in an
embedded nucleus, or a pron pair with one member inside a Type4; P: a
licensed-distance pair whose target token is embedded). A no-op when the
geometry is absent. The construction label is unchanged.

Purpose: (i) embedded domains keep exhibiting the binding phenomena
(the original enrichment's motivation); (ii) CAT1PRONs occur inside wh/aux
sentences, so pronoun fronting (D7) is actually exercised. It applies
**before** the phenomenon transform, so a background Pron can be a wh target.

## 6. Wh-movement — licensing formalization (§F, all four grammars)

The blocking islands are replaced by licensed-position sets, two clauses per
grammar (audit §F: the old clause (ii) — embedded subject / second-after-CAT9
— was removed everywhere, and H's Type3 clause was restricted to Type4
territory). The particle CAT8 is prefixed and the target moves to
string-initial position, leaving a `#NULL#` gap; licensing is evaluated on
the **base** string/tree, before fronting.

| Grammar | clause (i) | clause (ii) |
|---|---|---|
| H | **matrix object** *(structural)* | **inside a Type3 dominated by a Type4** *(structural; any attachment, any nesting depth)* |
| **H′** | **third CAT1-position** *(positional, = P's (i))* | = H |
| P | **third CAT1-position** of the sentence | **first CAT1-position after a CAT5** (per CAT5) |
| **L′** | **matrix object**, from generation history (`role == "object"`, clause 0) *(structural, = H's (i))* | = P |

Each prime differs from its parent in exactly clause (i) — the swap axis —
and the swap sits on the one licensing notion with **no local surface
shadow**: the matrix object's ordinal and context shift with subject-internal
complements, embedding, adverbs, and compound tense. H fronts the full
constituent; P fronts the nominal block; each prime uses its parent's
granularity and information channel (H′ scans terminals; L′ reads history).

**What no longer licenses anywhere:** the matrix subject, embedded subjects
and objects outside a Type4-dominated Type3, and Type3s attached in the
matrix clause. CAT9 appears in no licensing clause — it is only a clause
boundary. Embedded subjects were cut because their surface pattern was perfectly regular,
(first CAT1 after CAT9, measured 100%); matrix Type3s because theirs was a
bigram (CAT1 after CAT6). H's remaining clause (ii) has no such local regularity:
the Type4's right edge is unmarked, so deciding whether a `CAT6 CAT1` lies
inside or after the embedded clause requires verb-counting/argument tracking
across the boundary — the same epistemic grade as clause (i). 

**Pronouns (all grammars, D7):** CAT1PRON counts as a CAT1-position for
ordinal scans (P and H′); as a *target*, CAT1 and CAT1PRON-**Pron** are
licit, CAT1PRON-**Refl** never fronts (an anaphor stays in its licensing
configuration).

**Blockability (measured on 6k draws, seed 13, 2026-07-14):** a grammar
skips when no licensed position exists. H 42.7% of wh draws (intransitive
matrix + no Type4-dominated Type3), P 51.3% (fewer than three CAT1-positions
and no post-CAT5 nominal — the bare transitive is the mass case), H′ 56.9%,
L′ 40.4%.

**H-vs-P divergence (measured per targetable nominal over 6k H-pipeline
neutral trees, seed 14):** 1,189 licensed by both (matrix object at ordinal
3 or after a CAT5 — the agreement anchor), **3,157 H-only** (bare-transitive
matrix objects; Type4-internal Type3 comps at unlicensed ordinals), **1,746
P-only** (embedded subjects/objects and PP comps landing on ordinal 3;
post-CAT5 nominals in any clause — P's conditions cross clause boundaries
blindly), 8,566 neither. 2,123 of the neutral trees contain at least one
divergent position. Bidirectional, in-distribution, with the licensed-set
imbalance much reduced from the three-clause design (was 5,852 vs 536).

**H′-vs-L′ clause-(i) verdicts (natural fronted wh items, 6k draws each,
seed 15):** H′ items — 61% divergent (H′-licenses/L′-not: third-position
nominals that are not the matrix object), 39% coherent (matrix object at
ordinal 3, or clause-(ii) frontings licensed by neither verdict). L′ items —
64% divergent L′-licenses/H′-not (bare-transitive matrix objects), 2.5%
divergent H′-licenses/L′-not (post-CAT5 targets at ordinal 3), 34% coherent.
Both divergence directions occur in natural draws; verdicts are recovered
from sampled structure by `grammars/verdicts.py` (base-string reconstruction
— no hand-built probes, per the boundedness decisions).

## 7. Generation machinery (both grammars)

**Embedding calibration (2026-07-14).** Both families draw embedding **once
per clause** at 0.15, one embedded clause per draw, nesting via the embedded
clause's own recursive draw. The **host** is shared and explicit
(`Q_EMB_OBJECT = 0.5`): the object when the clause is transitive with that
probability, else the subject; the embedded clause attaches at the end of
the host's nominal zone (H: Type4 appended finally within the host Type1;
P: CAT9 clause inserted before the clause's verb for a subject host, at the
clause's token end for an object host). Forced embedding (H
`attach_forced_type4`, P `force_cat9_depth`) stays subject-anchored in both.
This replaced two asymmetric legacy behaviors: H's per-Type1-host draw
(which compounded with the nominal count — embedding 27% vs P's 15%,
≥2-clause tail 13% vs 2%, length p99 73 vs 28) and P's
rightmost-CAT2/CAT4 insertion scan (an artifact that scattered CAT9 across
pre-verbal/mid/post-object positions at ~56/11/33%). Post-calibration the
families match on embedding rate (~14.5%), placement (~80% pre-verbal /
20% post-object in both), and neutral length (mean ~9.0, p99 ~30 both).
H/H' generalization items force Type4 depth ∈ {2,3}, mirroring P's
`force_cat9_depth` grid.

**PP-machinery calibration (2026-07-14, same day).** Three residual H/P
asymmetries in PP expansion were removed, aligning the two families' routes
to sentence length (H formerly reached ≥25 tokens without embedding 8× as
often as P, inverting structure-at-length at OOD): (1) PP draws are now
**single** per host in H (one PP-under-NP per nominal, one PP-under-VP per
clause — P already was); (2) complement nominals nest at the explicit
shallow rate `P_PP_UNDER_NP_COMP = 0.15` (mirrors P's `P_CAT6_NP_COMP`)
instead of implicitly re-entering the 0.25 loop; (3) the lexical
`cat6_compatible` gate on PP hosts is **dropped** (decision: the feature
serves no purpose; it remains in the lexicon but is consulted nowhere).
The change touches expansion probabilities only — no transform, verdict, or
probe module consults PP counts, nesting depth, or `cat6_compatible`
(verified statically and by re-measured divergence: H-vs-P per-nominal
1,374 H-only / 1,279 P-only / 1,186 both; H′/L′ verdict directions intact).
Post-fix, all four grammars match on every measured base-structure row
(embedding ~14.7%, PP ~51%, PP-under-Type4 ~7.6%, ≥3 CAT1-positions ~40.5%,
transitive ~48.5%) and on neutral length (mean 8.9–9.0, p99 28–29).

Per item: (1) phenomenon draw (70/10/10/10; binding sub-draw refl/pron
50/50); (2) site draws — refl nucleus (`P_BIND_EMB`), aux site
(`Q_AUX_MATRIX`); (3) build, with forced geometry where the plan requires it
(matrix transitivity or a forced embedded transitive Type4 for H refl; a
forced CAT9 clause with compound tense for embedded-only aux); (4) duplicate
screen — resample (max 20 attempts) while a forbidden chance duplicate exists
(H: same-stem nucleus subject/object; P: same-stem pair at distance 2/3/4);
(5) background binding for non-binding items (`P_BG_BIND`); (6) the
transformation, which returns the refined label or `*_skipped`; (7)
linearization. Skipped items are discarded by the corpus builder's slot
quota (the labeled corpus is conditioned on applicability); generalization
item loops likewise resample on `_skipped` and on screen hits.

Transformed strings may incidentally exhibit surface patterns the base rules
exclude (e.g. a duplicate pair pulled to distance 3 by a later substitution
or fronting): measured 5/6,000 P items, all in transformed sentences, none in
untransformed ones. This is derivational opacity, not a screen failure (§1).

### Probability knobs (rules.py, both grammars unless noted)

| Knob | Value | Role |
|---|---|---|
| `PHENOMENON_PROBS` | .70/.10/.10/.10 | neutral / binding / aux / wh |
| refl vs pron sub-draw | 0.50 | refines binding to 5%/5% |
| `P_BIND_EMB` (H) | 0.25 | refl nucleus embedded rather than matrix |
| `P_PRON_FIRST` (H) | 0.20 | pron substitutes the first occurrence |
| `Q_AUX_MATRIX` | 0.75 | aux-item aux sits in the matrix clause |
| `P_BG_BIND` | 0.15 | background binding attempt on non-binding items |
| `REFL_OFFSET` / `PRON_OFFSETS` (P) | 3 / (2, 4) | licensed binding distances |
| `P_AUX`, `P_TYPE4`/`P_CAT9`, `P_CAT2`, `P_CAT5`, `P_CAT6_*`, `P_PP_*` | unchanged | structural expansion rates |

### Measured rates (6k items/grammar, seed 11 — re-verify at rebuild, D10)

| Quantity | H | P |
|---|---|---|
| wh skip rate | 30% | 52% |
| binding skip rate | 34% | 58% |
| aux skip rate | 22% (embedded-only plans) | 0% (by design) |
| aux divergence direction | — (skips) | 24% of aux items front an embedded aux |
| refl divergence direction | 83% of refl at distance ≠ 3 | 13% of refl cross-clause |
| pron divergence direction | — | 26% of pron same-nucleus SVO |
| background pron per non-binding item | ~2% | ~1.3% |

## 8. Deferred / follow-ups

1. **H′/L′ redesign (D9).** Their wh pair still implements the old blocking
   formulation over `licensor_attachment`; it must be restated inside the
   licensing frame (one clause of H's/P's licensed set replaced by a
   surface / generation-history approximation), together with the
   attachment-ambiguity reframing and the intervening-particle decision.
   Until then H′/L′ inherit the new binding/aux via shared imports but keep
   old wh — do not regenerate their corpora.
2. **Probes.** `corpora/probes.py` and `tests/test_probes.py` encode the old
   blocking design (blocked-phenomenon buckets, complex-NP verdicts) and must
   be rebuilt after (1).
3. **Corpus rebuild checklist (D10).** Re-run the rate table above on fresh
   seeds; verify per-construction length distributions across grammars;
   verify the aggregate unsubstituted-duplicate rates match across H and P
   (add a boost knob if chance rates drift apart); consider raising
   `P_BG_BIND` or relaxing its embedded-only scope if pronoun fronting is too
   rare (currently a handful of items per 6k).
