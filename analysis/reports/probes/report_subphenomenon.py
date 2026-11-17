"""Fine-grained per-sub-phenomenon probe figures -> results/analysis/evaluation/
structural_probes/per_subphenomenon/.

One figure per probe term (10 of them, including the two neutral gate controls).
Each figure stacks three panels that share an x-axis of grammar x {IOD, OOD}, and
every minimal pair is drawn as a dot (item-level, jittered, low alpha):

    margin           per-pair within-pair margin (nats). The WHOLE-STRING margin
                     is shown for every term (marker o); the CRITICAL-SPAN margin
                     is shown ONLY for the in-place-substitution terms (marker D)
                     -- span is not a minimal contrast for the relocation terms,
                     so it is neither computed nor drawn there
                     (evaluation/aggregate.py, CRITERION_BY_VIOLATION).
    accuracy         per-pair 0/1 dots: 1 where the model preferred the
                     grammatical member (margin > 0), 0 otherwise, in the term's
                     native criterion. The group's mean accuracy is annotated.
    margin over b4   per-pair margin minus the order-4 n-gram's margin under the
                     term's native criterion (0 = matches a 4-gram on that pair).

Whole-string margin for all terms + span margin for the in-place-substitution
terms is exactly the split evaluation.aggregate already draws; this module just
exposes both, per pair, at the finest grain.

Backing data: per_subphenomenon/pairs_long.csv (every pair drawn, with
margin_string, margin_span [substitution terms only], acc, margin_over_b4).

Run:  python -m analysis.reports.probes.report_subphenomenon
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.category_labels import relabel
from analysis.report_common import (
    GRAMMARS, GRAMMAR_COLOR, GRAMMAR_DISPLAY, SPLIT_DISPLAY,
    TERM_ORDER, attach_pair_baselines, load_probe_pairs, outdir, savefig,
)

OUT = outdir("evaluation", "structural_probes", "per_subphenomenon")
SPLITS = ["structural_probe_IOD", "structural_probe_OOD"]
GATE_TERMS = {"neutral_agreement", "neutral_positional"}
JIT = np.random.default_rng(0)


def _darken(hex_color: str, factor: float = 0.5) -> tuple:
    """Darken a hex colour toward black by `factor` (span vs whole-string cue)."""
    import matplotlib.colors as mc
    r, g, b = mc.to_rgb(hex_color)
    return (r * factor, g * factor, b * factor)


def attach_margin_over_b4(pairs: pd.DataFrame) -> pd.DataFrame:
    """Per-pair native-criterion margin over the order-4 n-gram reference.

    Delegates the (span-aware) baseline join to
    :func:`analysis.report_common.attach_pair_baselines`; this wrapper only adds
    ``margin_span_kept`` -- the critical-span margin retained for the
    in-place-substitution terms, dropped for the relocation terms.
    """
    m = attach_pair_baselines(pairs)
    # Per task: span margin is kept ONLY for the in-place-substitution terms.
    m["margin_span_kept"] = m["margin_span"].where(m["criterion"].eq("span"), np.nan)
    return m


def _xpos():
    """(grammar, split) -> x position; IOD/OOD adjacent, gap between grammars."""
    pos, ticks, labels = {}, [], []
    for gi, g in enumerate(GRAMMARS):
        for si, sp in enumerate(SPLITS):
            x = gi * 3 + si
            pos[(g, sp)] = x
            ticks.append(x)
            labels.append(f"{GRAMMAR_DISPLAY[g]}\n{SPLIT_DISPLAY[sp]}")
    return pos, ticks, labels


def _dots(ax, x, values, color, *, filled, marker="o", width=0.32, size=5):
    """Jittered dot column at x; mean drawn as a short black bar."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return
    jit = (JIT.random(values.size) - 0.5) * width
    ax.scatter(np.full(values.size, x) + jit, values, s=size,
               facecolors=color if filled else "none", edgecolors=color,
               linewidths=0.5, alpha=0.18)
    mean = float(values.mean())
    ax.plot([x - width / 2, x + width / 2], [mean, mean], color="black", lw=1.6)
    return mean


def make_figure(term: str, sub: pd.DataFrame, out_png):
    is_span_term = sub["criterion"].iloc[0] == "span"
    pos, ticks, labels = _xpos()
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.4, 3]})

    # Panel 1: margin (whole-string always; critical-span for substitution terms).
    ax = axes[0]
    for g in GRAMMARS:
        for sp in SPLITS:
            d = sub[(sub.grammar == g) & (sub.split == sp)]
            filled = sp == "structural_probe_IOD"
            xc = pos[(g, sp)]
            _dots(ax, xc - (0.2 if is_span_term else 0.0), d["margin_string"],
                  GRAMMAR_COLOR[g], filled=filled, marker="o")
            if is_span_term:
                _dots(ax, xc + 0.2, d["margin_span_kept"],
                      _darken(GRAMMAR_COLOR[g]), filled=filled, marker="D")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("within-pair margin (nats)")
    ax.set_title("margin", fontsize=11)

    # Panel 2: accuracy as per-pair 0/1 dots (native criterion) + mean annotation.
    ax = axes[1]
    for g in GRAMMARS:
        for sp in SPLITS:
            d = sub[(sub.grammar == g) & (sub.split == sp)]
            filled = sp == "structural_probe_IOD"
            xc = pos[(g, sp)]
            vals = d["acc"].to_numpy(dtype=float)
            yj = vals + (JIT.random(vals.size) - 0.5) * 0.16
            xj = np.full(vals.size, xc) + (JIT.random(vals.size) - 0.5) * 0.36
            ax.scatter(xj, yj, s=5, facecolors=GRAMMAR_COLOR[g] if filled else "none",
                       edgecolors=GRAMMAR_COLOR[g], linewidths=0.5, alpha=0.10)
            if vals.size:
                ax.text(xc, 1.14, f"{vals.mean():.2f}", ha="center", va="bottom",
                        fontsize=7, color="black")
    ax.set_ylim(-0.25, 1.28)
    ax.set_yticks([0, 1], ["0 (wrong)", "1 (correct)"])
    ax.set_ylabel("per-pair accuracy")
    ax.set_title("accuracy  (dot per pair; number = group mean)", fontsize=11)

    # Panel 3: margin over the order-4 baseline (native criterion).
    ax = axes[2]
    for g in GRAMMARS:
        for sp in SPLITS:
            d = sub[(sub.grammar == g) & (sub.split == sp)]
            filled = sp == "structural_probe_IOD"
            _dots(ax, pos[(g, sp)], d["margin_over_b4"], GRAMMAR_COLOR[g],
                  filled=filled)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("margin over b4 (nats)")
    ax.set_title("margin over b4  (0 = matches a 4-gram)", fontsize=11)

    axes[-1].set_xticks(ticks, labels, fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.2, axis="y")

    # Legend: fill = split; marker = criterion (only where span is shown).
    handles = [plt.Line2D([], [], marker="o", ls="", mfc="0.3", mec="0.3",
                          label="IOD (filled)"),
               plt.Line2D([], [], marker="o", ls="", mfc="none", mec="0.3",
                          label="OOD (open)"),
               plt.Line2D([], [], color="black", lw=1.6, label="group mean")]
    if is_span_term:
        handles += [plt.Line2D([], [], marker="o", ls="", mfc="0.55", mec="0.55",
                               label="whole-string margin (left, light)"),
                    plt.Line2D([], [], marker="D", ls="", mfc="0.15", mec="0.15",
                               label="critical-span margin (right, dark)")]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left", ncol=1,
                   framealpha=0.9)

    gate = "  (gate control)" if term in GATE_TERMS else ""
    crit = "span" if is_span_term else "whole-string"
    axes[0].annotate(f"term: {relabel(term)}{gate}   native criterion: {crit}",
                     xy=(0, 1), xycoords="axes fraction", xytext=(0, 14),
                     textcoords="offset points", fontsize=10, fontweight="bold")
    fig.tight_layout()
    savefig(fig, out_png)


def main() -> None:
    pairs = load_probe_pairs()
    pairs = attach_margin_over_b4(pairs)

    long_cols = ["term", "criterion", "grammar", "split", "seed", "pair_id",
                 "margin_string", "margin_span_kept", "acc", "margin_over_b4"]
    (pairs[long_cols].rename(columns={"margin_span_kept": "margin_span"})
     .to_csv(OUT / "pairs_long.csv", index=False))

    present = [t for t in TERM_ORDER if t in set(pairs["term"])]
    for term in present:
        sub = pairs[pairs["term"] == term]
        safe = term.replace(":", "__")
        make_figure(term, sub, OUT / f"{safe}.png")

    print(f"[per_subphenomenon] wrote {len(present)} figures + pairs_long.csv -> {OUT}")


if __name__ == "__main__":
    main()
