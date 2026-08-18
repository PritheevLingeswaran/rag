"""Exploratory data analysis of the grounding dataset.

The dataset modelled in Stage 12: one row per response sentence, twelve
engineered lexical features, and a binary target -- is this sentence
supported by the documents it was generated from?

Built from the committed RAGBench test sample so the analysis is
reproducible from the repository alone, with no network access.

    python eval/eda_grounding.py

Writes six figures to eval/figures/ and prints every table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from app.core.grounding_features import (  # noqa: E402
    FEATURE_NAMES,
    extract_response,
)

SAMPLE = REPO_ROOT / "eval" / "ragbench_sample_v1.jsonl"
FIGDIR = REPO_ROOT / "eval" / "figures"

INK = "#1f2126"
ACCENT = "#a06c10"
GREY = "#9a958c"


def build_frame() -> pd.DataFrame:
    rows = []
    for line in SAMPLE.open(encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        pairs = [p for p in (r.get("response_sentences") or [])
                 if isinstance(p, (list, tuple)) and len(p) >= 2]
        if not pairs:
            continue
        context = "\n\n".join(r.get("documents") or [])
        bad = set(r.get("unsupported_response_sentence_keys") or [])
        vectors = extract_response([p[1] for p in pairs], context)
        for (key, text), vec in zip(pairs, vectors):
            rows.append({
                **dict(zip(FEATURE_NAMES, vec)),
                "unsupported": int(key in bad),
                "domain": (r.get("dataset_name") or "?").replace("_test", ""),
                "n_documents": len(r.get("documents") or []),
                "sentence_chars": len(text),
            })
    return pd.DataFrame(rows)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    if not SAMPLE.exists():
        print(f"missing {SAMPLE.name}", file=sys.stderr)
        return 1
    FIGDIR.mkdir(exist_ok=True)
    df = build_frame()

    # ---- 1. overview -----------------------------------------------
    section("1. DATASET OVERVIEW")
    print(f"shape          : {df.shape[0]} rows x {df.shape[1]} columns")
    print("unit of a row  : one sentence of a generated answer")
    print("target         : unsupported (1 = not backed by its documents)")
    print(f"missing values : {int(df.isna().sum().sum())}")
    print(f"duplicate rows : {int(df.duplicated().sum())}")
    print("\ncolumn types")
    print(df.dtypes.to_string())

    # ---- 2. target balance -----------------------------------------
    section("2. TARGET BALANCE")
    counts = df["unsupported"].value_counts().sort_index()
    print(f"supported   (0): {counts.get(0, 0):5d}  "
          f"{counts.get(0, 0) / len(df):6.1%}")
    print(f"unsupported (1): {counts.get(1, 0):5d}  "
          f"{counts.get(1, 0) / len(df):6.1%}")
    print(f"\nimbalance ratio : 1 : {counts.get(0, 0) / max(counts.get(1, 1), 1):.1f}")
    print("=> a model predicting 'always supported' would score "
          f"{counts.get(0, 0) / len(df):.1%} accuracy while catching nothing.")
    print("   Accuracy is therefore the wrong metric; this project reports")
    print("   precision/recall/F1 on the positive class, and AUC.")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["supported", "unsupported"], counts.values,
           color=[GREY, ACCENT], width=.6)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 15, f"{v}\n{v / len(df):.1%}", ha="center", color=INK)
    ax.set_ylabel("sentences")
    ax.set_title("Class balance of the target")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_1_class_balance.png", dpi=200)

    # ---- 3. descriptive statistics ---------------------------------
    section("3. DESCRIPTIVE STATISTICS (features)")
    desc = df[list(FEATURE_NAMES)].describe().T[
        ["mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(desc.round(3).to_string())

    # ---- 4. distributions by class ---------------------------------
    section("4. FEATURE DISTRIBUTIONS BY CLASS")
    fig, axes = plt.subplots(3, 4, figsize=(16, 9))
    for ax, name in zip(axes.ravel(), FEATURE_NAMES):
        for label, colour, tag in ((0, GREY, "supported"),
                                   (1, ACCENT, "unsupported")):
            ax.hist(df.loc[df.unsupported == label, name], bins=20,
                    alpha=.65, color=colour, label=tag, density=True)
        ax.set_title(name, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=8)
    axes.ravel()[0].legend(fontsize=8)
    fig.suptitle("Feature distributions, supported vs unsupported sentences")
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_2_distributions.png", dpi=200)

    # ---- 5. separation ----------------------------------------------
    section("5. CLASS SEPARATION PER FEATURE")
    print(f"{'feature':<22}{'supported':>11}{'unsupported':>13}"
          f"{'diff':>9}{'point-biserial r':>19}")
    seps = []
    for name in FEATURE_NAMES:
        m0 = df.loc[df.unsupported == 0, name].mean()
        m1 = df.loc[df.unsupported == 1, name].mean()
        r = float(np.corrcoef(df[name], df["unsupported"])[0, 1])
        seps.append((name, m0, m1, m1 - m0, r))
        print(f"{name:<22}{m0:>11.3f}{m1:>13.3f}{m1 - m0:>+9.3f}{r:>+19.3f}")
    seps.sort(key=lambda t: -abs(t[4]))
    print(f"\nstrongest signal : {seps[0][0]} (r = {seps[0][4]:+.3f})")
    print(f"weakest signal   : {seps[-1][0]} (r = {seps[-1][4]:+.3f})")
    print("=> no single feature separates the classes; the largest |r| is "
          f"{abs(seps[0][4]):.2f}.")
    print("   That is the argument for a MODEL over a one-feature threshold.")

    fig, ax = plt.subplots(figsize=(8, 5))
    names = [s[0] for s in seps]
    vals = [s[4] for s in seps]
    ax.barh(names[::-1], vals[::-1],
            color=[ACCENT if v > 0 else GREY for v in vals[::-1]])
    ax.axvline(0, color=INK, lw=.8)
    ax.set_xlabel("point-biserial correlation with 'unsupported'")
    ax.set_title("Which features carry signal")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_3_feature_signal.png", dpi=200)

    # ---- 6. correlation ---------------------------------------------
    section("6. FEATURE CORRELATION (multicollinearity check)")
    corr = df[list(FEATURE_NAMES)].corr()
    print(corr.round(2).to_string())
    pairs = [(a, b, corr.loc[a, b])
             for i, a in enumerate(FEATURE_NAMES)
             for b in FEATURE_NAMES[i + 1:]
             if abs(corr.loc[a, b]) > 0.6]
    print("\nhighly correlated pairs (|r| > 0.6):")
    for a, b, v in sorted(pairs, key=lambda t: -abs(t[2])):
        print(f"  {a:<22} {b:<22} {v:+.3f}")
    print("=> these overlap features measure the same thing three ways.")
    print("   Consequence, documented in Stage 12: individual logistic")
    print("   coefficients are NOT separately interpretable.")

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(FEATURE_NAMES)))
    ax.set_xticklabels(FEATURE_NAMES, rotation=90, fontsize=8)
    ax.set_yticks(range(len(FEATURE_NAMES)))
    ax.set_yticklabels(FEATURE_NAMES, fontsize=8)
    for i in range(len(FEATURE_NAMES)):
        for j in range(len(FEATURE_NAMES)):
            ax.text(j, i, f"{corr.values[i, j]:.1f}", ha="center",
                    va="center", fontsize=6,
                    color="white" if abs(corr.values[i, j]) > .6 else INK)
    fig.colorbar(im, shrink=.8)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_4_correlation.png", dpi=200)

    # ---- 7. per domain -----------------------------------------------
    section("7. BREAKDOWN BY DOMAIN")
    g = df.groupby("domain").agg(
        sentences=("unsupported", "size"),
        unsupported=("unsupported", "sum"),
        rate=("unsupported", "mean"),
        mean_coverage=("coverage", "mean"),
        mean_docs=("n_documents", "mean"),
    ).sort_values("rate", ascending=False)
    print(g.round(3).to_string())
    print("\n=> the unsupported rate ranges "
          f"{g['rate'].min():.1%} to {g['rate'].max():.1%} across domains.")
    print("   A single global threshold cannot be right for all of them,")
    print("   which is what the Stage 10 per-domain results confirmed.")

    top = g["unsupported"].idxmax()
    share = g.loc[top, "unsupported"] / df["unsupported"].sum()
    print(f"\n!! {share:.1%} of ALL unsupported sentences come from one "
          f"domain ({top}).")
    print("   Any headline metric on this dataset is therefore largely a")
    print(f"   {top} metric. This is why the project reports per-domain")
    print("   results rather than a single number.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].bar(g.index, g["rate"], color=ACCENT, width=.6)
    axes[0].set_ylabel("unsupported rate")
    axes[0].set_title("Label rate by domain")
    axes[1].bar(g.index, g["mean_coverage"], color=GREY, width=.6)
    axes[1].set_ylabel("mean lexical coverage")
    axes[1].set_title("Mean coverage by domain")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_5_by_domain.png", dpi=200)

    # ---- 8. the key bivariate view -----------------------------------
    section("8. THE FEATURE THE SHIPPED RULE USES")
    thr = 0.7
    below = df[df.coverage < thr]
    print(f"shipped rule: flag when coverage < {thr}")
    print(f"  sentences flagged      : {len(below)} ({len(below)/len(df):.1%})")
    print(f"  of those, truly bad    : {int(below.unsupported.sum())} "
          f"({below.unsupported.mean():.1%})")
    missed = df[(df.coverage >= thr) & (df.unsupported == 1)]
    print(f"  unsupported but missed : {len(missed)} "
          f"({len(missed)/int(df.unsupported.sum()):.1%} of all unsupported)")
    print("\n=> the rule over-flags heavily and still misses many.")
    print("   Both error directions are visible in the overlap below.")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [df.loc[df.unsupported == 0, "coverage"],
            df.loc[df.unsupported == 1, "coverage"]]
    bp = ax.boxplot(data, tick_labels=["supported", "unsupported"],
                    patch_artist=True, widths=.5, orientation="horizontal")
    for patch, colour in zip(bp["boxes"], [GREY, ACCENT]):
        patch.set_facecolor(colour)
        patch.set_alpha(.75)
    for m in bp["medians"]:
        m.set_color(INK)
    ax.axvline(thr, color=ACCENT, ls="--", lw=1.4)
    ax.text(thr - .01, 2.42, f"shipped threshold {thr}", ha="right",
            color=ACCENT, fontsize=9)
    ax.set_xlabel("lexical coverage")
    ax.set_title("Why one threshold cannot separate the classes")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGDIR / "eda_6_coverage_overlap.png", dpi=200)

    section("FIGURES WRITTEN")
    for f in sorted(FIGDIR.glob("eda_*.png")):
        print(f"  {f.relative_to(REPO_ROOT)}  ({f.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
