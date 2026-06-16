"""All numeric and structural parameters for lexicon generation.
"""

VERSION: str = "0.2.0"

# Global fallback upper bound on combinatorial stem length, used only for a
# combinatorial category somehow absent from CATEGORY_MAX_STEM_LENGTH (defined
# below, after LETTER_PARTITIONS) and as the CLI --max-stem-length default.
MAX_STEM_LENGTH: int = 5
P_PROPER_LIKE: float = 0.1
P_CAT6_COMPATIBLE: float = 0.5
NULL_MORPHEME: str = "#NULL#"
SEPARATOR: str = "#"
CATEGORY_SIZE_CAPS: dict[str, int] = {}

# Zipf-Mandelbrot frequency parameters for open-class lexical sampling.
# s=1.0 is canonical Zipf (frequency ∝ 1/rank); q=0.0 means no Mandelbrot offset.
# Only combinatorial categories get frequency ranks; closed-class items (CAT4,
# CAT6, CAT8, CAT9, CAT1PRON, CAT3AUX) are selected by grammatical rule, not
# frequency, and keep zipf_rank=0 as a sentinel.
ZIPF_S: float = 1.0
ZIPF_Q: float = 0.0
ZIPF_CATEGORIES: tuple[str, ...] = ("CAT1", "CAT2", "CAT3", "CAT5")

# Letters reserved for lexical stems, partitioned by category.
# CAT1PRON, CAT4 use fixed forms (not combinatorial generation).
# CAT3AUX, CAT6, CAT8, CAT9 also have a hardcoded small inventory.
LETTER_PARTITIONS: dict[str, tuple[str, ...]] = {
    "CAT1":     ("a", "b", "c", "d", "e"),
    "CAT2":     ("f", "g", "h", "x"),
    "CAT3":     ("i", "j", "k", "l", "m"),
    "CAT3AUX":  ("n",),
    "CAT4":     ("u", "v"),
    "CAT5":     ("o", "p", "q"),
    "CAT6":     ("r", "y", "z"),
    "CAT8":     ("w",),
    "CAT9":     ("s", "t"),
    "CAT1PRON": (),
}

# Categories whose stems are produced by permutation-with-repetition over their
# letter pool (everything else uses a fixed/hardcoded inventory).
COMBINATORIAL_CATEGORIES: tuple[str, ...] = ("CAT1", "CAT2", "CAT3", "CAT5")

# Per-category maximum STEM length (in letters; inflectional morphemes are not
# involved). Rule: max length == the size of the category's letter pool, so an
# "open" category with more member letters admits longer stems and thus
# exponentially more stems. The stem count is sum_{k=1..n} n^k for pool size n.
# Current pools → counts:
#   CAT1 (n=5) -> 3905   CAT3 (n=5) -> 3905   CAT2 (n=4) -> 340   CAT5 (n=3) -> 39
CATEGORY_MAX_STEM_LENGTH: dict[str, int] = {
    cat: len(LETTER_PARTITIONS[cat]) for cat in COMBINATORIAL_CATEGORIES
}

# CAT6 attachment partition (CLAUDE.md §2.4). Two items are PP-under-NP
# (CAT1-selecting); one is PP-under-VP (CAT3-selecting). This is a fixed,
# deterministic assignment that all four grammars import.
CAT6_PP_UNDER_NP_LETTERS: tuple[str, ...] = ("r", "y")
CAT6_PP_UNDER_VP_LETTERS: tuple[str, ...] = ("z",)

# Digits reserved for inflectional morphemes, partitioned by parameter.
DIGIT_PARTITIONS: dict[str, tuple[str, ...]] = {
    "INFL1_gender":      ("1", "2"),
    "INFL1_number":      ("3", "4"),
    "INFL3_tense":       ("7", "8", "9"),
}
