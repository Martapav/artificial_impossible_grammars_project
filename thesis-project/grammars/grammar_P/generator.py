"""Re-exports GrammarPGenerator for backward compatibility with build_all_corpora.py."""

from __future__ import annotations

from .generate import (
    GrammarPGenerator,
    generate,
    generate_with_metadata,
)

CONSTRUCTIONS = [
    "neutral",
    "anaphoric_binding",
    "auxiliary_movement",
    "wh_movement",
]

__all__ = [
    "GrammarPGenerator",
    "generate",
    "generate_with_metadata",
    "CONSTRUCTIONS",
]
