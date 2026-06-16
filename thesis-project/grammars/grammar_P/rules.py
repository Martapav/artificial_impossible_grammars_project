"""Constants and bounds for Grammar P (positional). Mirrors grammar_H/rules.py."""

P_CAT2: float = 0.40   # continuation prob. of attaching another CAT2 modifier to a CAT1
P_CAT5: float = 0.30   # continuation prob. of attaching another CAT5 adverb to a CAT3
P_AUX: float = 0.30
P_CAT6_NP: float = 0.25
P_CAT6_NP_COMP: float = 0.15   # PP nesting inside a complement NP (lower → shallower)
P_CAT6_VP: float = 0.20
P_CAT9: float = 0.15

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

