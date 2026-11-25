"""Lexicon generation module for the artificial grammar research project.

Public API:
    Lexicon             — top-level artifact dataclass
    assemble_lexicon    — generate a lexicon v0.3.0 from a seed
                          (frozen v0.2.0 generator: lexicon.legacy_v020.assemble_lexicon)
    validate            — run all integrity checks
    serialize           — write a Lexicon to JSON
    deserialize         — load and validate a Lexicon from JSON
    LexiconValidationError
    LexiconIntegrityError
"""

from .generation import assemble_lexicon
from .schema import Lexicon
from .serialization import deserialize, serialize
from .validation import LexiconIntegrityError, LexiconValidationError, validate

__all__ = [
    "Lexicon",
    "assemble_lexicon",
    "validate",
    "serialize",
    "deserialize",
    "LexiconValidationError",
    "LexiconIntegrityError",
]
