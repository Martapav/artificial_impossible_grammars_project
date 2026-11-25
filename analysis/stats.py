"""Descriptive-statistics and plotting library for the corpus analysis.

Pure functions over the per-sentence feature tables produced by
``analysis.extract_features``. Table functions return tidy pandas DataFrames;
plot functions return a Matplotlib Figure (and optionally save a PNG). Styling
follows the project's data-viz palette: a fixed, CVD-separated categorical hue
per grammar, recessive gridlines, thin marks, legends with a secondary
marker/linestyle channel, single-hue sequential ramps for magnitude heatmaps and
a blue-gray-red diverging ramp for correlation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from analysis.category_labels import relabel as _cat  # display names for CAT/INFL

DATA_DIR = Path(__file__).resolve().parent / "data"
FIG_DIR = Path(__file__).resolve().parent / "figures"

# ---------------------------------------------------------------------------
# Palette (validated reference categorical hues; fixed order, never cycled)
# ---------------------------------------------------------------------------
GRAMMAR_ORDER: List[str] = ["H", "P", "Hprime", "Lprime"]
GRAMMAR_COLORS: Dict[str, str] = {
    "H": "#2a78d6",       # blue
    "P": "#eb6834",       # orange
    "Hprime": "#1baf7a",  # aqua
    "Lprime": "#4a3aa7",  # violet
}
GRAMMAR_MARKERS: Dict[str, str] = {"H": "o", "P": "s", "Hprime": "^", "Lprime": "D"}
GRAMMAR_LINES: Dict[str, str] = {"H": "-", "P": "--", "Hprime": "-.", "Lprime": ":"}
GRAMMAR_LABEL: Dict[str, str] = {
    "H": "H (hierarchical)", "P": "P (positional)",
    "Hprime": "H′", "Lprime": "P′",
}

# Chart chrome / ink
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE = "#fcfcfb"

# Single-hue sequential blue ramp (100->700) for magnitude heatmaps.
_BLUE_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
               "#256abf", "#184f95", "#0d366b"]
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", _BLUE_STEPS)
# Diverging blue<->red with neutral gray midpoint for correlation matrices.
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#184f95", "#3987e5", "#f0efec", "#e34948", "#b02020"])

# Structural numeric features analysed throughout.
NUMERIC_FEATURES: List[str] = [
    "length", "morphemes_mean", "n_clauses",
    "embedding_depth", "n_embedded_clauses", "n_modifiers", "n_verbs",
    "expansion_sites",
]
PHENOMENON_ORDER = [
    "neutral", "auxiliary_movement", "wh_movement",
    "anaphoric_binding_refl", "anaphoric_binding_pron",
]
# Structural co-occurrence flags (presence of a background/optional structure).
COOCC_FLAGS = {
    "has_aux": "aux present",
    "has_gap": "movement gap",
    "multi_clause": "≥2 clauses",
    "has_modifier": "≥1 modifier",
}


def apply_style() -> None:
    """Set Matplotlib rcParams to the project chart style (call once)."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Arial"],
        "font.size": 11,
        "axes.edgecolor": BASELINE,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "text.color": INK,
        "legend.frameon": False,
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_all(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load and concatenate every ``*_train_features.parquet`` table."""
    frames = []
    for g in GRAMMAR_ORDER:
        p = Path(data_dir) / f"{g}_train_features.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(
            f"No feature tables in {data_dir}. Run analysis.extract_features first.")
    df = pd.concat(frames, ignore_index=True)
    # derived flags used in co-occurrence analysis
    df["multi_clause"] = df["n_clauses"] >= 2
    df["has_modifier"] = df["n_modifiers"] >= 1
    df["grammar"] = pd.Categorical(df["grammar"], categories=GRAMMAR_ORDER, ordered=True)
    return df


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def length_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-grammar length distribution summary."""
    g = df.groupby("grammar", observed=True)["length"]
    out = pd.DataFrame({
        "n": g.size(),
        "mean": g.mean(),
        "std": g.std(),
        "min": g.min(),
        "p25": g.quantile(0.25),
        "median": g.median(),
        "p75": g.quantile(0.75),
        "p95": g.quantile(0.95),
        "max": g.max(),
    })
    return out.round(2)


def feature_summary(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Per-grammar mean/median/max for one numeric feature."""
    g = df.groupby("grammar", observed=True)[feature]
    return pd.DataFrame({
        "mean": g.mean(), "median": g.median(),
        "std": g.std(), "max": g.max(),
    }).round(3)


def construction_mix(df: pd.DataFrame) -> pd.DataFrame:
    """Proportion of each construction within each grammar (row-normalised)."""
    tab = pd.crosstab(df["grammar"], df["construction"], normalize="index")
    cols = [c for c in PHENOMENON_ORDER if c in tab.columns]
    return tab[cols].round(4)


def length_by_construction(df: pd.DataFrame) -> pd.DataFrame:
    """Mean length per grammar x construction."""
    tab = df.pivot_table(index="grammar", columns="construction",
                         values="length", aggfunc="mean", observed=True)
    cols = [c for c in PHENOMENON_ORDER if c in tab.columns]
    return tab[cols].round(2)


def length_correlations(df: pd.DataFrame,
                        features: Optional[List[str]] = None) -> pd.DataFrame:
    """Pearson correlation of length with each structural feature, per grammar."""
    features = features or [f for f in NUMERIC_FEATURES if f != "length"]
    rows = {}
    for g, sub in df.groupby("grammar", observed=True):
        rows[g] = {
            # a constant feature (zero variance) has undefined correlation
            f: (sub["length"].corr(sub[f]) if sub[f].std() > 0 else np.nan)
            for f in features
        }
    return pd.DataFrame(rows).T.reindex(GRAMMAR_ORDER).round(3)


def cooccurrence_rates(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Per grammar: P(structural flag | construction) as a matrix."""
    out = {}
    flags = list(COOCC_FLAGS)
    for g, sub in df.groupby("grammar", observed=True):
        mat = sub.groupby("construction", observed=True)[flags].mean()
        mat = mat.reindex([c for c in PHENOMENON_ORDER if c in mat.index])
        mat.columns = [COOCC_FLAGS[c] for c in mat.columns]
        out[str(g)] = mat.round(3)
    return out


def expansion_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Per-grammar expansion-site distribution summary."""
    g = df.groupby("grammar", observed=True)["expansion_sites"]
    return pd.DataFrame({
        "mean": g.mean(), "median": g.median(), "p95": g.quantile(0.95),
        "max": g.max(), "%_with_embedding": df.groupby("grammar", observed=True)["multi_clause"].mean(),
    }).round(3)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, name: Optional[str]):
    if name:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIG_DIR / f"{name}.png")
    return fig


def _style_axis(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_length_distributions(df: pd.DataFrame, save: Optional[str] = "length_distributions"):
    """Small-multiple length histograms, one panel per grammar, shared axes."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True, sharey=True)
    bins = np.arange(df["length"].min(), df["length"].max() + 2) - 0.5
    for ax, g in zip(axes.ravel(), GRAMMAR_ORDER):
        sub = df[df["grammar"] == g]["length"]
        ax.hist(sub, bins=bins, color=GRAMMAR_COLORS[g], edgecolor=SURFACE, linewidth=0.3)
        m = sub.mean()
        ax.axvline(m, color=INK, linewidth=1.2, linestyle="--")
        ax.text(m + 0.5, ax.get_ylim()[1] * 0.9, f"mean {m:.1f}",
                color=INK, fontsize=9, va="top")
        ax.set_title(GRAMMAR_LABEL[g], color=GRAMMAR_COLORS[g], loc="left")
        _style_axis(ax)
    for ax in axes[-1]:
        ax.set_xlabel("sentence length (tokens)")
    for ax in axes[:, 0]:
        ax.set_ylabel("count")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, save)


def plot_length_by_construction(df: pd.DataFrame, save: Optional[str] = "length_by_construction"):
    """Grouped box plot: length distribution per construction, split by grammar."""
    phen = [c for c in PHENOMENON_ORDER if c in df["construction"].unique()]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    n_g = len(GRAMMAR_ORDER)
    width = 0.8 / n_g
    for gi, g in enumerate(GRAMMAR_ORDER):
        data = [df[(df["grammar"] == g) & (df["construction"] == c)]["length"].values
                for c in phen]
        positions = np.arange(len(phen)) + (gi - (n_g - 1) / 2) * width
        bp = ax.boxplot(data, positions=positions, widths=width * 0.9,
                        patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.2),
                        whiskerprops=dict(color=BASELINE), capprops=dict(color=BASELINE))
        for box in bp["boxes"]:
            box.set(facecolor=GRAMMAR_COLORS[g], edgecolor=SURFACE, linewidth=0.5, alpha=0.9)
    ax.set_xticks(range(len(phen)))
    ax.set_xticklabels([c.replace("_", "\n") for c in phen], fontsize=9)
    ax.set_ylabel("sentence length (tokens)")
    ax.set_title("Sentence length by construction and grammar", loc="left")
    handles = [mpl.patches.Patch(facecolor=GRAMMAR_COLORS[g], label=GRAMMAR_LABEL[g])
               for g in GRAMMAR_ORDER]
    ax.legend(handles=handles, ncol=n_g, loc="upper center", bbox_to_anchor=(0.5, -0.13))
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, save)


def plot_construction_mix(df: pd.DataFrame, save: Optional[str] = "construction_mix"):
    """Stacked horizontal bars of the construction mix per grammar."""
    mix = construction_mix(df)
    fig, ax = plt.subplots(figsize=(10, 3.6))
    left = np.zeros(len(mix))
    # sequential-ish distinct hues per phenomenon (categorical, fixed order)
    phen_colors = ["#c3c2b7", "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7"]
    for j, c in enumerate(mix.columns):
        vals = mix[c].values
        ax.barh(range(len(mix)), vals, left=left, color=phen_colors[j % len(phen_colors)],
                edgecolor=SURFACE, linewidth=1.5, label=c)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.03:
                ax.text(l + v / 2, i, f"{v*100:.0f}%", ha="center", va="center",
                        color=INK if j == 0 else SURFACE, fontsize=9)
        left += vals
    ax.set_yticks(range(len(mix)))
    ax.set_yticklabels([GRAMMAR_LABEL[g] for g in mix.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("proportion of corpus")
    ax.set_title("Construction mix by grammar", loc="left")
    ax.grid(False)
    ax.legend(ncol=len(mix.columns), loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, save)


def plot_length_vs_feature(df: pd.DataFrame, feature: str = "expansion_sites",
                           save: Optional[str] = None):
    """Binned mean length vs a structural feature, one line per grammar."""
    save = save or f"length_vs_{feature}"
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for g in GRAMMAR_ORDER:
        sub = df[df["grammar"] == g]
        grp = sub.groupby(feature, observed=True)["length"]
        x = grp.mean().index.values
        y = grp.mean().values
        n = grp.size().values
        keep = n >= 30  # suppress noisy sparse bins
        ax.plot(x[keep], y[keep], color=GRAMMAR_COLORS[g], linestyle=GRAMMAR_LINES[g],
                marker=GRAMMAR_MARKERS[g], markersize=6, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=0.6, label=GRAMMAR_LABEL[g])
    ax.set_xlabel(feature.replace("_", " "))
    ax.set_ylabel("mean sentence length (tokens)")
    ax.set_title(f"Length vs {feature.replace('_', ' ')}", loc="left")
    ax.legend(ncol=2)
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, save)


def plot_feature_correlation(df: pd.DataFrame, grammar: str = "H",
                             save: Optional[str] = None):
    """Correlation heatmap among numeric features for one grammar (diverging)."""
    save = save or f"feature_correlation_{grammar}"
    sub = df[df["grammar"] == grammar]
    feats = [f for f in NUMERIC_FEATURES if sub[f].std() > 0]
    corr = sub[feats].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=9)
    for i in range(len(feats)):
        for j in range(len(feats)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=INK if abs(v) < 0.6 else SURFACE)
    ax.set_title(f"Feature correlations — {GRAMMAR_LABEL[grammar]}", loc="left")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, save)


def plot_cooccurrence_heatmap(df: pd.DataFrame, grammar: str = "H",
                              save: Optional[str] = None):
    """Heatmap of P(structural flag | construction) for one grammar (sequential)."""
    save = save or f"cooccurrence_{grammar}"
    mat = cooccurrence_rates(df)[grammar]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    im = ax.imshow(mat.values, cmap=SEQ_BLUE, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(mat.columns, rotation=25, ha="right", fontsize=9)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(mat.index, fontsize=9)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color=INK if v < 0.55 else SURFACE)
    ax.set_title(f"Phenomenon × structure co-occurrence — {GRAMMAR_LABEL[grammar]}",
                 loc="left", fontsize=12)
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).outline.set_visible(False)
    fig.tight_layout()
    return _save(fig, save)


def plot_expansion_distribution(df: pd.DataFrame, save: Optional[str] = "expansion_distribution"):
    """Distribution of expansion-site counts per grammar (grouped bars)."""
    fig, ax = plt.subplots(figsize=(10, 5))
    maxv = int(df["expansion_sites"].quantile(0.99))
    xs = np.arange(0, maxv + 1)
    n_g = len(GRAMMAR_ORDER)
    width = 0.8 / n_g
    for gi, g in enumerate(GRAMMAR_ORDER):
        sub = df[df["grammar"] == g]["expansion_sites"]
        counts = np.array([(sub == k).mean() for k in xs])
        ax.bar(xs + (gi - (n_g - 1) / 2) * width, counts, width=width * 0.95,
               color=GRAMMAR_COLORS[g], edgecolor=SURFACE, linewidth=0.4,
               label=GRAMMAR_LABEL[g])
    ax.set_xticks(xs)
    ax.set_xlabel("expansion sites (embedded clauses + modifiers)")
    ax.set_ylabel("proportion of sentences")
    ax.set_title("Expansion-site distribution by grammar", loc="left")
    ax.legend(ncol=n_g)
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, save)


# ---------------------------------------------------------------------------
# Lexicon frequency design (the hard-coded Zipf–Mandelbrot sampling law)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
# Category ordering for the frequency plot: largest rank range first so the
# 2x2 needle grid reads left-to-right, top-to-bottom by inventory size.
_ZIPF_PANEL_ORDER = ["CAT1", "CAT2", "CAT3", "CAT5"]
# Monochrome-blue channel for the log–log companion (needles themselves stay
# black). CAT2 and CAT3 have the same rank count, so they share a hue and CAT3
# is dashed to show the two distributions coincide exactly.
_ZIPF_CAT_STYLE = {
    "CAT1": dict(color="#0d366b", linestyle="-", linewidth=1.6),
    "CAT2": dict(color="#256abf", linestyle="-", linewidth=1.6),
    "CAT3": dict(color="#256abf", linestyle="--", linewidth=1.6),
    "CAT5": dict(color="#6da7ec", linestyle="-", linewidth=2.4),
}
# Qualitative hue per category for the single-panel overlay (one colour per
# category's needles). CVD-separated; distinct from the grammar palette.
_ZIPF_CAT_HUE = {
    "CAT1": "#2f6fd0",  # blue
    "CAT2": "#d9642a",  # orange
    "CAT3": "#17a07a",  # green
    "CAT5": "#8a4fbf",  # purple
}


def zipf_design(lexicon_path: Optional[Path] = None) -> Dict[str, dict]:
    """Read the frozen lexicon and return the per-category sampling law.

    For each open-class (combinatorial) category the sampling probability of the
    item at rank *r* is ``p(r) ∝ 1/(r + q)^s``, normalised within the category.
    ``s`` and ``q`` come from ``lexicon.config`` (the single source of truth);
    closed-class categories carry ``zipf_rank == 0`` and are excluded.

    Returns a dict keyed by category name with ``ranks`` (1..N), ``probs``
    (normalised), and the scalar ``s``, ``q``, ``n``.
    """
    import json  # noqa: PLC0415
    from lexicon import config as lexcfg  # noqa: PLC0415

    from lexicon.paths import resolve_lexicon_path  # noqa: PLC0415
    lexicon_path = Path(lexicon_path) if lexicon_path else resolve_lexicon_path()
    lex = json.loads(lexicon_path.read_text())
    s, q = lexcfg.ZIPF_S, lexcfg.ZIPF_Q

    out: Dict[str, dict] = {}
    for cat in lexcfg.ZIPF_CATEGORIES:
        items = lex[cat.lower()]
        ranks = np.array(sorted(it["zipf_rank"] for it in items), dtype=float)
        if (ranks == 0).any():
            raise ValueError(f"{cat} contains a closed-class sentinel (zipf_rank=0).")
        weights = 1.0 / (ranks + q) ** s
        out[cat] = {"ranks": ranks, "probs": weights / weights.sum(),
                    "s": s, "q": q, "n": len(ranks)}
    return out


def plot_zipf_distribution(lexicon_path: Optional[Path] = None,
                           save: Optional[str] = "zipf_distribution"):
    """Hard-coded Zipf–Mandelbrot sampling distribution of the open-class lexicon.

    Left 2x2 block — one panel per combinatorial category — draws the per-rank
    sampling probability as thin black needles (spikes) on linear axes: the
    literal 'black lines'. On a linear rank axis the 1/r decay is steep, so only
    the head is visible; that is expected and the reason for the companion.

    Right panel — the same four distributions on log–log axes, where canonical
    Zipf (s = 1) is a straight line of slope −1. The vertical offset between the
    lines is the within-category normaliser: a smaller inventory concentrates
    more mass on its top ranks. CAT2 and CAT3 have identical rank counts, so
    their laws coincide (CAT3 drawn dashed on top of CAT2).
    """
    design = zipf_design(lexicon_path)
    s = design["CAT1"]["s"]
    q = design["CAT1"]["q"]

    fig = plt.figure(figsize=(13.5, 6.3), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.45])
    needle_axes = {
        "CAT1": fig.add_subplot(gs[0, 0]), "CAT2": fig.add_subplot(gs[0, 1]),
        "CAT3": fig.add_subplot(gs[1, 0]), "CAT5": fig.add_subplot(gs[1, 1]),
    }
    ax_ll = fig.add_subplot(gs[:, 2])

    # --- linear needle panels (black spikes) ---
    # On a full linear rank axis the 1/r head collapses to one invisible spike
    # (CAT1 spans 3,905 ranks), so each panel zooms to the same head window; the
    # full range lives in the log–log companion. HEAD_K ≈ where p(r) has fallen
    # to ~1/HEAD_K of p(rank 1).
    HEAD_K = 60
    for i, cat in enumerate(_ZIPF_PANEL_ORDER):
        ax = needle_axes[cat]
        d = design[cat]
        n = d["n"]
        xmax = min(n, HEAD_K)
        ax.vlines(d["ranks"], 0.0, d["probs"], color=INK, linewidth=1.1)
        ax.set_xlim(0.5, xmax + 0.5)
        ax.set_ylim(0.0, d["probs"][0] * 1.12)
        title = f"{_cat(cat)}  ·  N = {n:,} ranks"
        if n > xmax:
            title += f"  (head 1–{xmax})"
        ax.set_title(title, loc="left", fontsize=11)
        # head probability, annotated in the free upper-right corner
        ax.annotate(f"p(rank 1) = {d['probs'][0]:.3f}", xy=(0.97, 0.9),
                    xycoords="axes fraction", color=INK_2, fontsize=8.5,
                    ha="right", va="top")
        if i % 2 == 0:
            ax.set_ylabel("sampling prob.")
        if i >= 2:
            ax.set_xlabel("frequency rank")
        _style_axis(ax)

    # --- log–log companion, all four categories ---
    for cat in _ZIPF_PANEL_ORDER:
        d = design[cat]
        ax_ll.plot(d["ranks"], d["probs"], label=f"{_cat(cat)}  (N = {d['n']:,})",
                   solid_capstyle="round", **_ZIPF_CAT_STYLE[cat])
    # reference slope −1 guide, anchored to the CAT1 head
    rmax = max(design[c]["n"] for c in _ZIPF_PANEL_ORDER)
    r_ref = np.array([1.0, rmax], dtype=float)
    c0 = design["CAT1"]["probs"][0]
    ax_ll.plot(r_ref, c0 / r_ref, color=MUTED, linestyle=(0, (1, 2)),
               linewidth=1.0, zorder=0)
    ax_ll.annotate("slope −1", xy=(rmax * 0.55, c0 / (rmax * 0.55)),
                   color=MUTED, fontsize=9, ha="right", va="bottom", rotation=-33)
    ax_ll.set_xscale("log")
    ax_ll.set_yscale("log")
    ax_ll.grid(True, which="both", axis="both", color=GRID, linewidth=0.6)
    ax_ll.set_xlabel("frequency rank (log)")
    ax_ll.set_ylabel("sampling probability (log)")
    ax_ll.set_title("Rank–frequency, log–log", loc="left")
    ax_ll.legend(title="within-category law", loc="lower left", fontsize=9,
                 title_fontsize=9)
    _style_axis(ax_ll)

    return _save(fig, save)


def plot_zipf_overlay(lexicon_path: Optional[Path] = None, head_k: int = 60,
                      save: Optional[str] = "zipf_distribution_overlay"):
    """The four categories' Zipf laws overlaid in one linear needle panel.

    Same head window as :func:`plot_zipf_distribution` (ranks 1..``head_k``), but
    all four categories share one axis with a distinct needle colour each. Needles
    are dodged by a small fraction of a rank so same-rank spikes sit side by side
    instead of stacking. Sharing the y-axis makes the head-mass ordering directly
    visible: the smaller the inventory, the taller its rank-1 needle (CAT5 > CAT2
    = CAT3 > CAT1). CAT2 and CAT3 have identical rank counts, so their needles are
    the same height at every rank.
    """
    design = zipf_design(lexicon_path)

    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    offsets = np.linspace(-0.3, 0.3, len(_ZIPF_PANEL_ORDER))
    ymax = max(design[c]["probs"][0] for c in _ZIPF_PANEL_ORDER)
    for cat, dx in zip(_ZIPF_PANEL_ORDER, offsets):
        d = design[cat]
        keep = d["ranks"] <= head_k
        ax.vlines(d["ranks"][keep] + dx, 0.0, d["probs"][keep],
                  color=_ZIPF_CAT_HUE[cat], linewidth=1.5,
                  label=f"{_cat(cat)}  (N = {d['n']:,})")
    ax.set_xlim(0.3, head_k + 0.7)
    ax.set_ylim(0.0, ymax * 1.06)
    ax.set_xlabel("frequency rank")
    ax.set_ylabel("sampling probability")
    ax.legend(title="category", ncol=2, loc="upper right")
    _style_axis(ax)
    fig.tight_layout()
    return _save(fig, save)
