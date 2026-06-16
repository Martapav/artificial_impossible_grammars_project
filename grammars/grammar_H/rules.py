"""Constants and bounds for Grammar H (spec §4, §8, §9.1)."""

# Geometric continuation probabilities (subcritical → finite trees almost surely).
# Provisional length-control knobs; a distributional sampler will replace them.
P_CAT2: float = 0.40   # prob. of attaching another CAT2 modifier to a CAT1
P_CAT5: float = 0.30   # continuation prob. of attaching another CAT5 adverb to a CAT3

P_AUX: float = 0.30
P_PP_UNDER_NP: float = 0.25
P_PP_UNDER_VP: float = 0.20
P_TYPE4: float = 0.15
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
