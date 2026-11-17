"""Display names for the lexical categories and inflection parameters.

Presentation layer only. The data keys in ``lexicon.json``, the corpora and the
generation code all keep ``CAT1``/``INFL1``/… ; these linguistic names are
substituted into produced plots and tables so the readable outputs use them.

A token is replaced only when it stands alone — not when it is part of a larger
identifier such as ``P_CAT6_COMPATIBLE`` or ``grammar_P/rules.py:P_CAT9``. Those
name real code constants, and the descriptive tables cite them so the numbers can
be spot-checked, so they are left verbatim.
"""

import re

# Longer/compound tokens (CAT1PRON, CAT3AUX) are listed first so they match
# before their shorter prefixes (CAT1, CAT3).
CATEGORY_DISPLAY = {
    "CAT1PRON": "PRON",
    "CAT3AUX": "AUX",
    "CAT1": "NOM",
    "CAT2": "ADJ",
    "CAT3": "VERB",
    "CAT4": "DET",
    "CAT5": "ADV",
    "CAT6": "PREP",
    "CAT8": "INT",
    "CAT9": "SUB",
    "INFL1": "INFL-NOM",
    "INFL3": "INFL-VERB",
}

# Standalone tokens only: not immediately preceded by an identifier character or
# underscore (excludes P_CAT2, rules.py:P_CAT9), and not followed by an
# alphanumeric (an underscore after is allowed, so cat6_compatible still matches).
_TOKENS = sorted(CATEGORY_DISPLAY, key=len, reverse=True)
_PATTERN = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(_TOKENS) + r")(?![A-Za-z0-9])",
                      re.IGNORECASE)


def _sub(m: "re.Match") -> str:
    name = CATEGORY_DISPLAY[m.group(0).upper()]
    return name if m.group(0).isupper() else name.lower()


def relabel(text) -> str:
    """Substitute standalone category / inflection tokens with display names.

    Case is preserved: ``CAT1`` -> ``NOM``, ``cat6_compatible`` -> ``prep_compatible``.
    """
    return _PATTERN.sub(_sub, str(text))
