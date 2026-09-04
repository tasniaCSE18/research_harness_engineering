"""Generate result-analysis figures for harness_paper.tex from results.json.

Figures (saved to ../fig/):
  fig_harness_pipeline.png   - variant A/B/C architecture diagram (methodology)
  fig_metric_comparison.png  - grouped bar: field-exact / field-F1 / doc-full per variant
  fig_calibration.png        - reliability diagram (confidence vs accuracy) for B/C + ECE gap
  fig_cost_accuracy.png      - calls/doc vs field-exact accuracy trade-off
  fig_error_by_field.png     - per-field error breakdown for variant C

Run: venv/bin/python research/make_figures.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "src"))
import config  # noqa: E402
from evaluate import exact, field_f1  # noqa: E402

FIG_DIR = BASE.parent / "fig"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

COLORS = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868"}


def load():
    res = json.loads((BASE / "outputs" / "results.json").read_text())
    return res["records"]


def compute_table(records, n_samples):
    """Per-variant metrics. Call cost is variant-specific (not the shared
    per-doc total in results.json['n_calls']): A issues one call, B issues
    n_samples calls, C issues n_samples plus one correction call per doc
    that has at least one flagged field."""
    gts = [r["gt"] for r in records]
    n = len(records)
    n_flagged_docs = sum(1 for r in records if r["flagged"])
    calls_per_doc = {
        "A": 1.0,
        "B": float(n_samples),
        "C": n_samples + n_flagged_docs / n,
    }
    table = {}
    for variant in ["A", "B", "C"]:
        preds = [r[variant] for r in records]
        fex = sum(int(exact(p.get(f, ""), g.get(f, ""))) for p, g in zip(preds, gts) for f in config.SCHEMA_FIELDS)
        ff1 = sum(field_f1(p.get(f, ""), g.get(f, "")) for p, g in zip(preds, gts) for f in config.SCHEMA_FIELDS)
        full = sum(1 for p, g in zip(preds, gts) if all(exact(p.get(f, ""), g.get(f, "")) for f in config.SCHEMA_FIELDS))
        nf = n * len(config.SCHEMA_FIELDS)
        table[variant] = {
            "field_exact": fex / nf,
            "field_f1": ff1 / nf,
            "doc_full": full / n,
            "calls_per_doc": calls_per_doc[variant],
        }
    return table


def compute_calibration(records):
    pairs = []
    for r in records:
        for f in config.SCHEMA_FIELDS:
            pairs.append((r["conf"].get(f, 0.0), int(exact(r["C"].get(f, ""), r["gt"].get(f, "")))))
    bins = 10
    bn, ba, bc = [0] * bins, [0.0] * bins, [0.0] * bins
    for c, ok in pairs:
        b = min(int(c * bins), bins - 1)
        bn[b] += 1
        ba[b] += ok
        bc[b] += c
    return bn, ba, bc, bins


# ---------------------------------------------------------------- fig 1 ---
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.2)

    def box(x, y, w, h, text, color, fontsize=8.3):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                            linewidth=1.1, edgecolor="#333333", facecolor=color, alpha=0.92)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, wrap=True)

    def arrow(x0, y0, x1, y1, style="-|>"):
        a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=12,
                             linewidth=1.1, color="#333333")
        ax.add_patch(a)

    # image input
    box(0.1, 1.6, 1.3, 0.9, "Receipt\nImage $I$", "#EAEAEA")
    arrow(1.4, 2.05, 2.0, 2.05)

    # Variant A
    box(2.0, 3.1, 2.7, 0.9, "Variant A\nSchema call\n$T{=}0$", COLORS["A"])
    arrow(4.7, 3.55, 5.5, 3.55)
    box(5.5, 3.1, 2.2, 0.9, "$\\hat{y}_A$", "#EAEAEA")

    # Variant B
    box(2.0, 1.6, 2.7, 0.9, "Variant B\n$N$ samples $T{=}0.5$\n+ majority vote", COLORS["B"])
    arrow(4.7, 2.05, 5.5, 2.05)
    box(5.5, 1.6, 2.2, 0.9, "agg $\\hat{y}_B$\n+ conf$(f)$", "#EAEAEA")

    # Variant C
    box(2.0, 0.1, 2.7, 0.9, "Variant C\nB + targeted\nre-read if conf$(f){<}\\tau$", COLORS["C"])
    arrow(4.7, 0.55, 5.5, 0.55)
    box(5.5, 0.1, 2.2, 0.9, "$\\hat{y}_C$", "#EAEAEA")

    # confidence routing arrow B -> C
    arrow(6.6, 1.6, 6.6, 1.0, style="-|>")
    ax.text(6.85, 1.3, "flagged\nfields", fontsize=7.3, va="center")

    # evaluation block
    arrow(7.7, 3.55, 8.7, 2.1)
    arrow(7.7, 2.05, 8.7, 2.05)
    arrow(7.7, 0.55, 8.7, 2.0)
    box(8.7, 1.55, 3.1, 1.0, "Evaluation\nFieldExact / FieldF1\nDocFull / ECE", "#C9C9E8")

    ax.set_title("Harness pipeline: three variants around a frozen VLM", fontsize=10.5)
    fig.savefig(FIG_DIR / "fig_harness_pipeline.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 2 ---
def fig_metric_comparison(table):
    variants = ["A", "B", "C"]
    metrics = ["field_exact", "field_f1", "doc_full"]
    labels = ["Field Exact", "Field F1", "Doc Full"]
    x = np.arange(len(metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    for i, v in enumerate(variants):
        vals = [table[v][m] for m in metrics]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=f"Variant {v}", color=COLORS[v])
        for b, val in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, val + 0.015, f"{val:.3f}",
                    ha="center", va="bottom", fontsize=7.5, rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.15)
    ax.set_title("Reliability metrics across harness variants (pilot)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(FIG_DIR / "fig_metric_comparison.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3 ---
def fig_calibration(bn, ba, bc, bins):
    centers = [(i + 0.5) / bins for i in range(bins)]
    acc = [ba[i] / bn[i] if bn[i] else np.nan for i in range(bins)]
    conf = [bc[i] / bn[i] if bn[i] else np.nan for i in range(bins)]

    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    mask = [n > 0 for n in bn]
    xs = [c for c, m in zip(centers, mask) if m]
    ys_acc = [a for a, m in zip(acc, mask) if m]
    ys_conf = [c for c, m in zip(conf, mask) if m]
    sizes = [n * 40 for n, m in zip(bn, mask) if m]
    ax.scatter(ys_conf, ys_acc, s=sizes, color=COLORS["C"], alpha=0.85, edgecolor="black",
               linewidth=0.6, label="Observed bins (size = #predictions)")
    for cx, cy, n in zip(ys_conf, ys_acc, [n for n, m in zip(bn, mask) if m]):
        ax.annotate(f"n={n}", (cx, cy), textcoords="offset points", xytext=(6, -3), fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence (self-consistency agreement)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("Reliability diagram (Variant C) — ECE = 0.277")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig_calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 4 ---
def fig_cost_accuracy(table):
    variants = ["A", "B", "C"]
    calls = [table[v]["calls_per_doc"] for v in variants]
    acc = [table[v]["field_exact"] for v in variants]

    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    y_offsets = {"A": (0, 18), "B": (0, 18), "C": (0, -22)}
    for v, c, a in zip(variants, calls, acc):
        ax.scatter(c, a, s=220, color=COLORS[v], edgecolor="black", zorder=3)
        ax.annotate(f"{v} ({c:.1f} calls/doc)", (c, a), textcoords="offset points",
                    xytext=y_offsets[v], fontsize=8.5, fontweight="bold", ha="center")
    ax.plot(calls, acc, linestyle=":", color="gray", zorder=1)
    ax.set_xlabel("Model calls per document")
    ax.set_ylabel("Field-level exact match")
    ax.set_title("Cost vs. accuracy across harness variants")
    ax.set_xlim(0, max(calls) + 2.0)
    ax.set_ylim(0.78, 0.92)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig_cost_accuracy.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5 ---
def fig_error_by_field(records):
    err = Counter()
    for r in records:
        for f in config.SCHEMA_FIELDS:
            if not exact(r["C"].get(f, ""), r["gt"].get(f, "")):
                err[f] += 1
    fields = config.SCHEMA_FIELDS
    counts = [err.get(f, 0) for f in fields]

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    bars = ax.bar(fields, counts, color=["#4C72B0", "#DD8452", "#C44E52", "#55A868"])
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, c + 0.02, str(c), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Documents with a field error (Variant C)")
    ax.set_title("Per-field error breakdown (pilot, n=3 docs)")
    ax.set_ylim(0, max(counts + [1]) + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(FIG_DIR / "fig_error_by_field.png")
    plt.close(fig)


def main():
    res = json.loads((BASE / "outputs" / "results.json").read_text())
    records = res["records"]
    n_samples = res["config"].get("n_samples", 3)
    table = compute_table(records, n_samples)
    bn, ba, bc, bins = compute_calibration(records)

    fig_pipeline()
    fig_metric_comparison(table)
    fig_calibration(bn, ba, bc, bins)
    fig_cost_accuracy(table)
    fig_error_by_field(records)

    print("Wrote figures to", FIG_DIR)
    for p in sorted(FIG_DIR.glob("fig_*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
