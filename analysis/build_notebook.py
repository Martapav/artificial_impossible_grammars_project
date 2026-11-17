"""Assemble analysis/corpus_analysis.ipynb from the stats library.

Run after the feature tables exist (analysis/extract_features). Produces a
runnable report notebook: markdown narrative + code cells that render the tables
and figures defined in analysis.stats. Execute/refresh it with:

    jupyter nbconvert --to notebook --execute --inplace analysis/corpus_analysis.ipynb
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parent / "corpus_analysis.ipynb"


def md(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(text.strip("\n"))


CELLS = [
    md(r"""
# Corpus descriptive statistics

Descriptive analysis of the **training corpora** for the four artificial grammars
of the thesis — **H** (fully hierarchical), **P** (fully positional), **H′**
(H with a positional complex-NP island rule) and **P′** (P with a structural
complex-NP island rule).

Each grammar's training split is 100 000 sentences drawn with seed 42 under the
canonical construction quota (70 % neutral / 10 % auxiliary movement / 10 %
wh-movement / 5 % reflexive binding / 5 % pronominal binding).

**What this report covers**

1. Corpus overview and construction mix
2. Sentence-length distributions
3. Length ↔ construction / structural-complexity relationships
4. Phenomenon occurrences and co-occurrences with background structure
5. Expansion sites and feature correlations

> **Methodology note — generation-time metadata sidecar.** The structural
> signals used below (embedding depth, clause count, expansion sites,
> aux/gap presence) are computed *inside* the generators at
> emission and persisted line-aligned to each corpus file as
> `corpora/train/<grammar>.meta.jsonl` (TRANSFORM_RULES §8; shared stable `id`).
> `analysis/extract_features.py` reads that sidecar directly — no replay — so
> every row describes the exact on-disk corpus sentence. Positional grammars
> (P, P′) have no constituent tree, so their `embedding_depth` IS
> `n_embedded_clauses` — equal to the positional embedding depth exactly
> (non-branching chain), and populated as of 2026-08-04 rather than left `NaN`.
> Comparable across families via `n_clauses` and `expansion_sites`.
"""),
    code(r"""
import sys
from pathlib import Path

# make the project root importable so `import analysis.stats` works from the notebook
ROOT = Path.cwd()
while not (ROOT / "analysis").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from IPython.display import display

import analysis.stats as st

pd.set_option("display.max_columns", 30)
st.apply_style()

df = st.load_all()
print(f"Loaded {len(df):,} sentences across {df['grammar'].nunique()} grammars.")
df.head(3)
"""),
    md("## 1. Corpus overview and construction mix"),
    code(r"""
# Corpus size and headline structural averages per grammar.
overview = (df.groupby("grammar", observed=True)
              .agg(sentences=("length", "size"),
                   mean_length=("length", "mean"),
                   mean_clauses=("n_clauses", "mean"),
                   mean_expansion_sites=("expansion_sites", "mean")))
display(overview.round(2).rename(index=st.GRAMMAR_LABEL))

print("Construction mix (row-normalised proportions):")
display(st.construction_mix(df).rename(index=st.GRAMMAR_LABEL))
st.plot_construction_mix(df);
"""),
    md("""
## 2. Sentence-length distributions

Lengths are token counts (space-separated words). All splits analysed here use
the training window **[2, 25]**.
"""),
    code(r"""
display(st.length_summary(df).rename(index=st.GRAMMAR_LABEL))
st.plot_length_distributions(df);
"""),
    md("""
## 3. Length ↔ construction and structural complexity

Do particular constructions or structural features drive sentence length? The
table below gives mean length per construction; the correlation table reports the
Pearson correlation of length with each structural feature, per grammar.
"""),
    code(r"""
print("Mean length by construction:")
display(st.length_by_construction(df).rename(index=st.GRAMMAR_LABEL))
st.plot_length_by_construction(df);
"""),
    code(r"""
print("Correlation of sentence length with structural features (Pearson):")
display(st.length_correlations(df).rename(index=st.GRAMMAR_LABEL))
# Length is driven most strongly by expansion sites — show the relationship.
st.plot_length_vs_feature(df, "expansion_sites");
st.plot_length_vs_feature(df, "n_clauses");
"""),
    md("""
## 4. Phenomenon occurrences and co-occurrences

Each sentence carries exactly one *labelled* construction (they are mutually
exclusive by design). "Co-occurrence" here means how often a labelled
construction coincides with **background structure** that can appear
independently: an auxiliary, a movement gap, multiple clauses, or a
modifier. The diagonal sanity checks hold —
`auxiliary_movement` always carries an aux, and both movement constructions
always carry a gap (aux movement now leaves a trace just as wh does, so
`movement gap` is 1.0 for `auxiliary_movement` and `wh_movement` alike; it is
not a wh-specific flag).
"""),
    code(r"""
cooc = st.cooccurrence_rates(df)
for g in st.GRAMMAR_ORDER:
    print(f"P(structure | construction) — {st.GRAMMAR_LABEL[g]}")
    display(cooc[g])
"""),
    code(r"""
for g in st.GRAMMAR_ORDER:
    st.plot_cooccurrence_heatmap(df, g);
"""),
    md("""
## 5. Expansion sites and feature correlations

**Expansion sites** = embedded clauses + modifiers — the count of optional,
recursive structure a sentence realises. It is the dominant driver of length
(section 3). The correlation heatmaps show how the structural features relate
within each grammar family.
"""),
    code(r"""
display(st.expansion_summary(df).rename(index=st.GRAMMAR_LABEL))
st.plot_expansion_distribution(df);
"""),
    code(r"""
# Feature-correlation heatmaps: one hierarchical grammar and one positional.
st.plot_feature_correlation(df, "H");
st.plot_feature_correlation(df, "P");
"""),
    md("""
---
### Reproducing this report

```bash
# 1. build the cached feature tables (~tens of minutes; one-time)
python -m analysis.extract_features --grammar all --n 100000 --seed 42

# 2. execute the notebook end-to-end
jupyter nbconvert --to notebook --execute --inplace analysis/corpus_analysis.ipynb
```

All figures are also written to `analysis/figures/*.png` as they render.
"""),
]


def build() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = CELLS
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbf.write(nb, NB_PATH)
    print(f"wrote {NB_PATH}")


if __name__ == "__main__":
    build()
