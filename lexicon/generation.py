"""Lexicon v0.3.0 generation: sampled stems, derived entries, Zipf ranks.

Spec: experiment.md §2.2, §2.5, §2.6, §3. (The v0.2.0 generator that produced
``lexicon.json`` lives, frozen, in ``lexicon/legacy_v020.py``.)

The lexicon is generated ONCE and does not try to cover the combinatorial
space. Every phase draws from its OWN seeded stream, ``random.Random(f"{seed}:<phase>")``,
so changing one phase's parameters never shifts another phase's draws:

  A  "affixes"  — per open category (in OPEN_CATEGORIES order), per slot (in
                  AFFIX_SLOTS order): sample the affix forms without replacement
                  from the 2-3-letter strings over the category's pool.
  B  "stems"    — per open category: per-length stem counts from the truncated
                  log-normal (deterministic, no draws), then per length (ascending)
                  sample that many distinct free strings.
  C  "features" — CAT1 stems in stem-id order: gender, countability,
                  cat6_compatible (same draws, same order as v0.2.0); then CAT3
                  stems in stem-id order: transitivity.
  D  "derive"   — per open category: shuffle the global morpheme-count quota, deal
                  19 derived counts to each stem, then compose each derived entry
                  (side draw, class draw, form draw; redraw a duplicate).
  E  "zipf"     — per open category: one N(0,1) draw per entry in entry-id order;
                  rank by score = noise + ZIPF_ALPHA*max(0, len-pool) + ZIPF_BETA*n_affixes.
"""

import itertools
import math
import random
from datetime import datetime, timezone

from .config import (
    AFFIX_INVENTORY,
    AFFIX_LENGTHS,
    AFFIX_SLOTS,
    CAT6_PP_UNDER_NP_LETTERS,
    CAT6_PP_UNDER_VP_LETTERS,
    ENTRIES_PER_STEM,
    LETTER_PARTITIONS,
    MAX_PREFIXES,
    MAX_SUFFIXES,
    MORPH_SHARES,
    OPEN_CATEGORIES,
    P_CAT6_COMPATIBLE,
    P_PROPER_LIKE,
    P_SUFFIX_SIDE,
    SEPARATOR,
    SHORT_LEN_CAP,
    STEM_LEN_LOGNORM_MEDIAN,
    STEM_LEN_LOGNORM_SIGMA,
    STEM_LEN_MAX,
    STEM_TARGET,
    VERSION,
    WORDNET_30_UNIQUE_STRINGS,
    ZIPF_ALPHA,
    ZIPF_BETA,
)
from .schema import (
    AffixItem,
    CAT1Item,
    CAT1PRONItem,
    CAT2Item,
    CAT3AUXItem,
    CAT3Item,
    CAT4Item,
    CAT5Item,
    CAT6Item,
    CAT8Item,
    CAT9Item,
    InflectionalInventory,
    InflectionalParameter,
    Lexicon,
    LexiconMetadata,
)


class LexiconCapacityError(ValueError):
    """A requested stem/affix count does not fit the category's string space."""


def _phase_rng(seed: int, phase: str) -> random.Random:
    """Independent, reproducible stream per generation phase."""
    return random.Random(f"{seed}:{phase}")


def partition_alphabet() -> dict[str, tuple[str, ...]]:
    """Return the letter partition mapping from config, validated for disjointness."""
    seen: dict[str, str] = {}
    for cat, letters in LETTER_PARTITIONS.items():
        for letter in letters:
            if not (letter.islower() and letter.isalpha()):
                raise ValueError(
                    f"Partition entry {letter!r} for {cat} is not a lowercase Latin letter.")
            if letter in seen:
                raise ValueError(f"Letter {letter!r} appears in more than one category partition.")
            seen[letter] = cat
    return LETTER_PARTITIONS


def generate_combinatorial_stems(alphabet: tuple[str, ...], max_length: int) -> list[str]:
    """All strings of lengths 1..max_length over alphabet, sorted (len, lex).

    Utility (v0.2.0 enumerated its stems this way); v0.3.0 samples instead.
    """
    stems = ["".join(c) for L in range(1, max_length + 1)
             for c in itertools.product(alphabet, repeat=L)]
    stems.sort(key=lambda s: (len(s), s))
    return stems


def _strings_of_length(alphabet: tuple[str, ...], length: int) -> list[str]:
    return ["".join(c) for c in itertools.product(alphabet, repeat=length)]


# ── Category sizes ───────────────────────────────────────────────────────────

def stem_counts(total: int = STEM_TARGET) -> dict[str, int]:
    """Stems per open category: ``total`` split by WordNet 3.0 shares
    (largest-remainder rounding, so the counts sum exactly to ``total``)."""
    weights = WORDNET_30_UNIQUE_STRINGS
    z = sum(weights[c] for c in OPEN_CATEGORIES)
    raw = {c: total * weights[c] / z for c in OPEN_CATEGORIES}
    out = {c: math.floor(v) for c, v in raw.items()}
    rest = total - sum(out.values())
    for c in sorted(OPEN_CATEGORIES, key=lambda c: (-(raw[c] - out[c]), c))[:rest]:
        out[c] += 1
    return out


# ── Phase A: affixes ─────────────────────────────────────────────────────────

def sample_affixes(seed: int) -> list[AffixItem]:
    """Affix inventory for every open category (experiment.md §2.6)."""
    rng = _phase_rng(seed, "affixes")
    out: list[AffixItem] = []
    for cat in OPEN_CATEGORIES:
        pool = LETTER_PARTITIONS[cat]
        space = sorted(s for L in AFFIX_LENGTHS for s in _strings_of_length(pool, L))
        need = sum(AFFIX_INVENTORY[cat][slot] for slot in AFFIX_SLOTS)
        if need > len(space):
            raise LexiconCapacityError(
                f"{cat}: {need} affixes requested but only {len(space)} strings of "
                f"length {AFFIX_LENGTHS} exist over pool {pool}.")
        drawn = rng.sample(space, need)
        i = 0
        for slot in AFFIX_SLOTS:
            k = AFFIX_INVENTORY[cat][slot]
            for j, form in enumerate(sorted(drawn[i:i + k], key=lambda s: (len(s), s))):
                out.append(AffixItem(id=f"{cat.lower()}_{slot}_{j:02d}", form=form,
                                     category=cat, slot=slot))
            i += k
    return out


# ── Phase B: stems ───────────────────────────────────────────────────────────

def _lognorm_mass(length: int) -> float:
    """Probability mass of a discretized log-normal at integer ``length``."""
    mu, s = math.log(STEM_LEN_LOGNORM_MEDIAN), STEM_LEN_LOGNORM_SIGMA
    cdf = lambda x: 0.5 * (1 + math.erf((math.log(x) - mu) / (s * math.sqrt(2))))
    return cdf(length + 0.5) - cdf(max(length - 0.5, 1e-9))


def stem_length_counts(cat: str, n_stems: int, reserved: set[str]) -> dict[int, int]:
    """Stems per length: log-normal shape, each length capped at SHORT_LEN_CAP of
    its free strings, the excess redistributed (by shape) over uncapped lengths.
    Deterministic (no RNG). Raises LexiconCapacityError if the stems don't fit."""
    n = len(LETTER_PARTITIONS[cat])
    lengths = range(1, STEM_LEN_MAX + 1)
    reserved_by_len: dict[int, int] = {}
    for r in reserved:
        reserved_by_len[len(r)] = reserved_by_len.get(len(r), 0) + 1
    cap = {L: int(SHORT_LEN_CAP * (n ** L - reserved_by_len.get(L, 0))) for L in lengths}
    w = {L: _lognorm_mass(L) for L in lengths}
    alloc = {L: 0.0 for L in lengths}
    remaining, free = float(n_stems), set(lengths)
    while remaining > 1e-9:
        if not free:
            raise LexiconCapacityError(
                f"{cat}: {n_stems} stems do not fit under the length caps "
                f"(SHORT_LEN_CAP={SHORT_LEN_CAP}, STEM_LEN_MAX={STEM_LEN_MAX}).")
        z = sum(w[L] for L in free)
        share = {L: remaining * w[L] / z for L in free}
        over = [L for L in free if alloc[L] + share[L] > cap[L]]
        if not over:
            for L in free:
                alloc[L] += share[L]
            remaining = 0.0
        else:
            for L in over:
                remaining -= cap[L] - alloc[L]
                alloc[L] = cap[L]
                free.discard(L)
    # integer counts summing exactly to n_stems (largest remainder, within caps)
    counts = {L: math.floor(alloc[L]) for L in lengths}
    rest = n_stems - sum(counts.values())
    order = sorted(lengths, key=lambda L: (-(alloc[L] - counts[L]), L))
    for L in order:
        if rest == 0:
            break
        if counts[L] < cap[L]:
            counts[L] += 1
            rest -= 1
    if rest:
        raise LexiconCapacityError(f"{cat}: could not place {rest} stems within caps.")
    return {L: k for L, k in counts.items() if k}


def sample_stems(seed: int, counts_by_cat: dict[str, int],
                 affixes: list[AffixItem]) -> dict[str, list[str]]:
    """Distinct stems per open category, sorted (len, lex). Never an affix string."""
    rng = _phase_rng(seed, "stems")
    out: dict[str, list[str]] = {}
    for cat in OPEN_CATEGORIES:
        pool = LETTER_PARTITIONS[cat]
        reserved = {a.form for a in affixes if a.category == cat}
        stems: list[str] = []
        for L, k in sorted(stem_length_counts(cat, counts_by_cat[cat], reserved).items()):
            if len(pool) ** L <= 200_000:   # small space: exact sample of the free strings
                free = [s for s in _strings_of_length(pool, L) if s not in reserved]
                stems.extend(rng.sample(free, k))
            else:                            # large space: rejection sampling
                chosen: set[str] = set()
                while len(chosen) < k:
                    s = "".join(rng.choice(pool) for _ in range(L))
                    if s not in reserved:
                        chosen.add(s)
                stems.extend(sorted(chosen))
        stems.sort(key=lambda s: (len(s), s))
        out[cat] = stems
    return out


# ── Phase C: stem features ───────────────────────────────────────────────────

def stem_features(seed: int, stems: dict[str, list[str]]) -> dict[str, list[dict]]:
    """Inherent features per stem (in stem-id order), shared by all its entries."""
    rng = _phase_rng(seed, "features")
    feats: dict[str, list[dict]] = {c: [{} for _ in stems[c]] for c in OPEN_CATEGORIES}
    for f in feats["CAT1"]:
        gender = rng.choice(("1", "2"))
        countability = "proper_like" if rng.random() < P_PROPER_LIKE else "countable"
        cat6_compat = rng.random() < P_CAT6_COMPATIBLE
        f.update(inherent_gender=gender, countability=countability,
                 cat4_required=(countability == "countable"), cat6_compatible=cat6_compat)
    for f in feats["CAT3"]:
        f.update(transitivity=rng.choice(("transitive", "intransitive")))
    return feats


# ── Phase D: derived entries ─────────────────────────────────────────────────

def morph_quota(n_stems: int) -> dict[int, int]:
    """Entries per lexical-morpheme count for one category (exact quotas).

    Count 1 (bare) = one per stem; counts 2-4 = MORPH_SHARES of all entries,
    largest-remainder rounded so the derived counts sum to n_stems*(EPS-1)."""
    total = n_stems * ENTRIES_PER_STEM
    derived = total - n_stems
    ks = [k for k in sorted(MORPH_SHARES) if k > 1]
    raw = {k: total * MORPH_SHARES[k] for k in ks}
    z = sum(raw.values())
    raw = {k: derived * v / z for k, v in raw.items()}
    out = {k: math.floor(v) for k, v in raw.items()}
    for k in sorted(ks, key=lambda k: (-(raw[k] - out[k]), k))[: derived - sum(out.values())]:
        out[k] += 1
    return {1: n_stems, **out}


def _draw_sides(rng: random.Random, n_affixes: int) -> tuple[int, int]:
    """(n_prefixes, n_suffixes): each affix -> suffix side w.p. P_SUFFIX_SIDE,
    conditioned on the slot limits (rejection = renormalization)."""
    while True:
        n_suf = sum(rng.random() < P_SUFFIX_SIDE for _ in range(n_affixes))
        n_pre = n_affixes - n_suf
        if n_pre <= MAX_PREFIXES and n_suf <= MAX_SUFFIXES:
            return n_pre, n_suf


def _compose(rng: random.Random, n_affixes: int, inv: dict[str, list[str]]) -> dict[str, str]:
    """One derived entry's affixes (experiment.md §2.6 composition)."""
    n_pre, n_suf = _draw_sides(rng, n_affixes)
    slots = {"prefix": "", "suffix_inner": "", "suffix_outer": ""}
    if n_pre:
        slots["prefix"] = rng.choice(inv["prefix"])
    if n_suf == 2:
        slots["suffix_inner"] = rng.choice(inv["suffix_inner"])
        slots["suffix_outer"] = rng.choice(inv["suffix_outer"])
    elif n_suf == 1:
        ni, no = len(inv["suffix_inner"]), len(inv["suffix_outer"])
        cls = "suffix_inner" if rng.random() < ni / (ni + no) else "suffix_outer"
        slots[cls] = rng.choice(inv[cls])
    return slots


def derived_form(prefix: str, stem: str, suffix_inner: str, suffix_outer: str) -> str:
    """The entry's surface form: its lexical morphemes joined by SEPARATOR."""
    return SEPARATOR.join(m for m in (prefix, stem, suffix_inner, suffix_outer) if m)


def derive_entries(seed: int, stems: dict[str, list[str]],
                   affixes: list[AffixItem]) -> dict[str, list[dict]]:
    """Per open category: per stem, 1 bare + (ENTRIES_PER_STEM-1) derived entries.

    Returns plain dicts {stem_idx, prefix, suffix_inner, suffix_outer, n_morphemes}
    in entry-id order (stem order, bare first)."""
    rng = _phase_rng(seed, "derive")
    out: dict[str, list[dict]] = {}
    per_stem = ENTRIES_PER_STEM - 1
    for cat in OPEN_CATEGORIES:
        inv = {slot: [a.form for a in affixes if a.category == cat and a.slot == slot]
               for slot in AFFIX_SLOTS}
        quota = morph_quota(len(stems[cat]))
        deck = [k for k in (2, 3, 4) for _ in range(quota[k])]
        rng.shuffle(deck)
        entries: list[dict] = []
        for s_idx in range(len(stems[cat])):
            entries.append(dict(stem_idx=s_idx, prefix="", suffix_inner="",
                                suffix_outer="", n_morphemes=1))
            seen: set[tuple] = set()
            for k in sorted(deck[s_idx * per_stem:(s_idx + 1) * per_stem]):
                for _attempt in range(1000):
                    slots = _compose(rng, k - 1, inv)
                    key = (slots["prefix"], slots["suffix_inner"], slots["suffix_outer"])
                    if key not in seen:
                        break
                else:
                    raise LexiconCapacityError(
                        f"{cat}: cannot find a new {k}-morpheme entry for stem #{s_idx}.")
                seen.add(key)
                entries.append(dict(stem_idx=s_idx, n_morphemes=k, **slots))
        out[cat] = entries
    return out


# ── Phase E: frequency ranks ─────────────────────────────────────────────────

def zipf_ranks(seed: int, stems: dict[str, list[str]],
               entries: dict[str, list[dict]]) -> dict[str, list[int]]:
    """1-based rank per entry (entry-id order), per category (experiment.md §2.5)."""
    rng = _phase_rng(seed, "zipf")
    out: dict[str, list[int]] = {}
    for cat in OPEN_CATEGORIES:
        pool = len(LETTER_PARTITIONS[cat])
        scores = []
        for e in entries[cat]:
            L = len(stems[cat][e["stem_idx"]])
            scores.append(rng.gauss(0.0, 1.0) + ZIPF_ALPHA * max(0, L - pool)
                          + ZIPF_BETA * (e["n_morphemes"] - 1))
        order = sorted(range(len(scores)), key=lambda i: (scores[i], i))
        ranks = [0] * len(scores)
        for r, i in enumerate(order, start=1):
            ranks[i] = r
        out[cat] = ranks
    return out


# ── Item construction ────────────────────────────────────────────────────────

_ITEM_CLASS = {"CAT1": CAT1Item, "CAT2": CAT2Item, "CAT3": CAT3Item, "CAT5": CAT5Item}


def build_open_items(stems, feats, entries, ranks) -> dict[str, list]:
    """Assemble the frozen entry items per open category."""
    out: dict[str, list] = {}
    for cat in OPEN_CATEGORIES:
        low = cat.lower()
        items, j_by_stem = [], {}
        for e, rank in zip(entries[cat], ranks[cat]):
            s_idx = e["stem_idx"]
            j = j_by_stem.get(s_idx, 0)
            j_by_stem[s_idx] = j + 1
            stem = stems[cat][s_idx]
            items.append(_ITEM_CLASS[cat](
                id=f"{low}_{s_idx:04d}_{j:02d}",
                form=derived_form(e["prefix"], stem, e["suffix_inner"], e["suffix_outer"]),
                zipf_rank=rank,
                stem=stem,
                stem_id=f"{low}_{s_idx:04d}",
                prefix=e["prefix"],
                suffix_inner=e["suffix_inner"],
                suffix_outer=e["suffix_outer"],
                n_morphemes=e["n_morphemes"],
                **feats[cat][s_idx],
            ))
        out[cat] = items
    return out


# ── Closed classes (unchanged from v0.2.0) ───────────────────────────────────

def build_cat4() -> list[CAT4Item]:
    """Build the two CAT4 determiner items.

    Spec §2.2: {u}=singular, {v}=plural.
    """
    return [
        CAT4Item(id="cat4_u", form="u", number="singular"),
        CAT4Item(id="cat4_v", form="v", number="plural"),
    ]


def build_cat1pron() -> list[CAT1PRONItem]:
    """Build the two hardcoded CAT1PRON pronominal items.

    Spec §2.2: {m}=Pron, {mm}=Refl.
    """
    return [
        CAT1PRONItem(id="cat1pron_pron", form="m", subclass="Pron"),
        CAT1PRONItem(id="cat1pron_refl", form="mm", subclass="Refl"),
    ]


def build_cat9() -> list[CAT9Item]:
    """Build the two hardcoded CAT9 subordinator items.

    Spec §2.2: {s, t}. Both head Type4 (relative clauses).
    """
    return [
        CAT9Item(id="cat9_s", form="s"),
        CAT9Item(id="cat9_t", form="t"),
    ]


def build_cat8() -> list[CAT8Item]:
    """Build the singleton CAT8 wh particle.

    Spec §2.2: {w}. Single item used only by wh-movement.
    """
    return [CAT8Item(id="cat8_w", form="w")]


def build_cat3aux() -> list[CAT3AUXItem]:
    """Build the singleton CAT3AUX auxiliary item.

    Spec §2.2: {n}.
    """
    return [CAT3AUXItem(id="cat3aux_n")]


def build_cat6() -> list[CAT6Item]:
    """Build the three CAT6 prepositional items with attachment partition.

    Spec §2.4: three items partitioned 2:1 — two CAT1-selecting (PP-under-NP),
    one CAT3-selecting (PP-under-VP). 
    """
    items: list[CAT6Item] = []
    for letter in CAT6_PP_UNDER_NP_LETTERS:
        items.append(CAT6Item(id=f"cat6_{letter}", form=letter, attachment="cat1_selecting"))
    for letter in CAT6_PP_UNDER_VP_LETTERS:
        items.append(CAT6Item(id=f"cat6_{letter}", form=letter, attachment="cat3_selecting"))
    return items


def build_inflectional_inventory() -> InflectionalInventory:
    """Build the complete inflectional inventory from the canonical parameter definitions."""
    parameters = (
        InflectionalParameter(
            name="INFL1_number",
            values=("3", "4"),
            dimension_label="number-like",
            applies_to=("CAT1", "CAT1PRON", "CAT2", "CAT3", "CAT3AUX"),
        ),
        InflectionalParameter(
            name="INFL1_gender",
            values=("1", "2"),
            dimension_label="gender-like",
            applies_to=("CAT1", "CAT1PRON", "CAT2", "CAT3", "CAT3AUX"),
        ),
        InflectionalParameter(
            name="INFL3_tense",
            values=("7", "8", "9"),
            dimension_label="tense",
            applies_to=("CAT3", "CAT3AUX"),
        ),
    )
    return InflectionalInventory(parameters=parameters)




# ── Assembly ─────────────────────────────────────────────────────────────────

def items_dict(lexicon_parts: dict) -> dict:
    """The hashed content (all items, not metadata). ``affixes`` is included only
    when present, so a v0.2.0 lexicon hashes exactly as it always did."""
    from .serialization import _dataclass_to_dict  # noqa: PLC0415

    keys = ("cat1", "cat1pron", "cat2", "cat3", "cat3aux", "cat4", "cat5",
            "cat6", "cat8", "cat9")
    out = {k: [_dataclass_to_dict(x) for x in lexicon_parts[k]] for k in keys}
    out["inflectional"] = _dataclass_to_dict(lexicon_parts["inflectional"])
    if lexicon_parts.get("affixes"):
        out["affixes"] = [_dataclass_to_dict(x) for x in lexicon_parts["affixes"]]
    return out


def assemble_lexicon(seed: int) -> Lexicon:
    """Generate lexicon v0.3.0 from ``seed`` (phases A-E, module docstring).

    Two calls with the same seed produce an identical Lexicon (up to the
    generation timestamp in the metadata).
    """
    import importlib
    import json as _json

    from .serialization import compute_content_hash  # noqa: PLC0415

    partition_alphabet()
    affixes = sample_affixes(seed)                                   # A
    n_stems = stem_counts()
    stems = sample_stems(seed, n_stems, affixes)                     # B
    feats = stem_features(seed, stems)                               # C
    entries = derive_entries(seed, stems, affixes)                   # D
    ranks = zipf_ranks(seed, stems, entries)                         # E
    open_items = build_open_items(stems, feats, entries, ranks)

    parts = {
        "cat1": tuple(open_items["CAT1"]),
        "cat1pron": tuple(build_cat1pron()),
        "cat2": tuple(open_items["CAT2"]),
        "cat3": tuple(open_items["CAT3"]),
        "cat3aux": tuple(build_cat3aux()),
        "cat4": tuple(build_cat4()),
        "cat5": tuple(open_items["CAT5"]),
        "cat6": tuple(build_cat6()),
        "cat8": tuple(build_cat8()),
        "cat9": tuple(build_cat9()),
        "inflectional": build_inflectional_inventory(),
        "affixes": tuple(affixes),
    }
    content_hash = compute_content_hash(items_dict(parts))

    config_mod = importlib.import_module(".config", package=__name__.rsplit(".", 1)[0])
    raw_snapshot = {
        k: v for k, v in vars(config_mod).items()
        if not k.startswith("_") and isinstance(v, (str, int, float, dict, list, tuple, bool))
    }
    config_snapshot = _json.loads(_json.dumps(raw_snapshot, default=list))

    category_sizes = {
        "CAT1": len(parts["cat1"]), "CAT1PRON": len(parts["cat1pron"]),
        "CAT2": len(parts["cat2"]), "CAT3": len(parts["cat3"]),
        "CAT3AUX": len(parts["cat3aux"]), "CAT4": len(parts["cat4"]),
        "CAT5": len(parts["cat5"]), "CAT6": len(parts["cat6"]),
        "CAT8": len(parts["cat8"]), "CAT9": len(parts["cat9"]),
    }
    metadata = LexiconMetadata(
        version=VERSION,
        seed=seed,
        generation_timestamp=datetime.now(timezone.utc).isoformat(),
        max_stem_length={c: max(len(s) for s in stems[c]) for c in OPEN_CATEGORIES},
        category_sizes=category_sizes,
        category_alphabets={c: list(v) for c, v in LETTER_PARTITIONS.items()},
        config_snapshot=config_snapshot,
        content_hash=content_hash,
        stem_counts={c: len(stems[c]) for c in OPEN_CATEGORIES},
    )
    return Lexicon(metadata=metadata, **parts)
