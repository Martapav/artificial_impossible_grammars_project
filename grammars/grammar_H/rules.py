"""Constants and bounds for Grammar H (spec §4, §8, §9.1)."""

# Geometric continuation probabilities (subcritical → finite trees almost surely).
# Provisional length-control knobs; a distributional sampler will replace them.
P_CAT2: float = 0.40   # prob. of attaching another CAT2 modifier to a CAT1
P_CAT5: float = 0.30   # continuation prob. of attaching another CAT5 adverb to a CAT3

P_AUX: float = 0.30
P_PP_UNDER_NP: float = 0.25
# PP nesting inside a complement NP (lower → shallower); mirrors Grammar P's
# P_CAT6_NP_COMP (PP-machinery parity, 2026-07-14).
P_PP_UNDER_NP_COMP: float = 0.15
P_PP_UNDER_VP: float = 0.20
P_TYPE4: float = 0.15
# Host of the clause-level embedded clause: the object (when the clause is
# transitive) with this probability, else the subject. Shared with Grammar P
# (embedding-placement parity, 2026-07-14).
Q_EMB_OBJECT: float = 0.50
# Background binding (successor of the embedded enrichment, see
# grammars/TRANSFORM_RULES.md): probability that a non-binding sentence carries
# one additional embedded-scoped pronominal substitution as background variation.
P_BG_BIND: float = 0.15
# Reflexive nucleus placement: probability that a refl item's licensing nucleus
# is an embedded clause rather than the matrix clause.
P_BIND_EMB: float = 0.25
# Pron substitution: probability that the FIRST occurrence of the coreferential
# pair is the one substituted (majority: the second).
P_PRON_FIRST: float = 0.20
# Aux items: probability that the forced auxiliary sits in the matrix clause.
# Otherwise it is embedded-only: H must skip (structural blocking), P fronts it.
Q_AUX_MATRIX: float = 0.75
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

# Refined labels emitted after pronominal substitution (used for slot-controlled
# balancing in BaseGrammarGenerator.generate_batch).
REFINED_PHENOMENA: list = [
    "neutral",
    "anaphoric_binding_refl",
    "anaphoric_binding_pron",
    "auxiliary_movement",
    "wh_movement",
]

INFL1_NUMBER_TO_CAT4: dict = {"3": "singular", "4": "plural"}

SEPARATOR: str = "#"
NULL_MORPHEME: str = "#NULL#"
