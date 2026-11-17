"""Constants and bounds for Grammar P (positional). Mirrors grammar_H/rules.py."""

P_CAT2: float = 0.40   # continuation prob. of attaching another CAT2 modifier to a CAT1
P_CAT5: float = 0.30   # continuation prob. of attaching another CAT5 adverb to a CAT3
P_AUX: float = 0.30
# In-situ compound tense inside an EMBEDDED clause — the same knob and the same
# value as grammar_H/rules.py, read at exactly one site here
# (build._maybe_attach_cat9). Rationale and the 0.60 calibration are recorded
# there and in TRANSFORM_RULES §4.
P_AUX_EMBEDDED: float = 0.60
P_CAT6_NP: float = 0.25
P_CAT6_NP_COMP: float = 0.15   # PP nesting inside a complement NP (lower → shallower)
P_CAT6_VP: float = 0.20
P_CAT9: float = 0.15
# Host of the clause-level embedded clause: the object (when the clause is
# transitive) with this probability, else the subject. Shared with Grammar H
# (embedding-placement parity, 2026-07-14).
Q_EMB_OBJECT: float = 0.50
# Aux items: probability that the forced auxiliary sits in the matrix clause.
# Otherwise it is embedded-only: P fronts it (first in scan), H must skip.
Q_AUX_MATRIX: float = 0.75

# Positional binding offsets (distance in surface items from the antecedent
# CAT1 to the licensed CAT1 position; every whitespace token counts, the
# #-attached inflection values do not exist as separate items).
REFL_OFFSET: int = 3
PRON_OFFSETS: tuple = (2, 4)
# Licensed antecedent CAT1-POSITION ORDINALS, 1-based from the sentence start,
# counting CAT1 and CAT1PRON alike (D7). The 2026-07-30 narrowing: refl "rule B"
# (antecedent is the third CAT1-position) and pron "N9" (second or third — the
# first nominal never licenses pron). This conjunct is what makes each rule a
# count from the sentence beginning instead of a fixed-window distance; see
# TRANSFORM_RULES §3.2. The licensing logic lives in binding.py —
# these are its constants, and the offset sets must stay disjoint (D5).
REFL_ANT_ORDINALS: tuple = (3,)
PRON_ANT_ORDINALS: tuple = (2, 3)

# Positional wh licensing (P family: P and L′). Clause (i) is an ordinal over
# CAT1-positions; clause (ii) is an ordinal over ANCHOR-category occurrences plus
# an ordinal over the CAT1-positions following the chosen anchor. Both are counts
# from the sentence beginning. The licensing logic lives in wh.py.
#
# Clause (ii) changed 2026-07-30 from "first CAT1-position after a CAT5, per CAT5"
# — a CAT5→nominal bigram, window-recoverable at U 0.981 → 0.947. The anchor is now
# CAT6 because CAT6s are never contiguous (each heads its own PP) whereas a clause's
# CAT5s are inserted adjacently, which made "the 2nd CAT5" a `CAT5 CAT5` bigram.
# The nominal ordinal is 2 because a CAT6 is immediately followed by its own
# complement. See TRANSFORM_RULES §6.
WH_I_NOMINAL_ORDINAL: int = 3
WH_II_ANCHOR_CAT: str = "CAT6"
WH_II_ANCHOR_ORDINAL: int = 2
WH_II_NOMINAL_ORDINAL: int = 2

# 70% neutral (no transformation); 10% per transformation. Identical to H.
PHENOMENA: list = [
    "neutral",
    "anaphoric_binding",
    "auxiliary_movement",
    "wh_movement",
]
PHENOMENON_PROBS: dict = {
    "neutral":             0.70,
    "anaphoric_binding":   0.10,
    "auxiliary_movement":  0.10,
    "wh_movement":         0.10,
}

# Refined labels emitted after pronominal substitution (Refl vs Pron), used for
# optional slot-controlled balancing in BaseGrammarGenerator.generate_batch.
REFINED_PHENOMENA: list = [
    "neutral",
    "anaphoric_binding_refl",
    "anaphoric_binding_pron",
    "auxiliary_movement",
    "wh_movement",
]

# ── Inflectional value pools (digit assignment lives in lexicon.json) ─────────

INFL1_NUMBER_VALUES: list = ["3", "4"]
INFL1_GENDER_VALUES: list = ["1", "2"]      # gender comes from item inherent_gender
INFL3_TENSE_VALUES: list = ["7", "8", "9"]

# Maps INFL1_number digit values to CAT4 number string labels.
INFL1_NUMBER_TO_CAT4: dict = {"3": "singular", "4": "plural"}

SEPARATOR: str = "#"
NULL_MORPHEME: str = "#NULL#"

