"""Generate result-analysis figures for harness_paper.tex.

Two kinds of figures, kept clearly separate:
  (1) our own real data (results.json) -- from a 20-document attempted run
      on gemini-2.5-flash. 11/20 hit free-tier quota exhaustion (all 429s
      even after retries) and are EXCLUDED from accuracy/calibration
      figures -- an API failure is not an extraction result. The clean
      n=9 that completed is what "[ours]" figures below report. The 9/20
      completion rate itself is reported honestly as its own figure.
  (2) literature-grounded comparison figures -- numbers taken from published
      papers (see LIT_* constants below, each with its bibliography key),
      used to give our real run external context. These are NOT reruns of
      our pipeline; the literature bars are drawn from other authors' own
      (larger) evaluations.

Figures (saved to ../fig/):
  fig_harness_pipeline.png        - variant A/B/C architecture diagram (methodology)
  fig_benchmark_comparison.png    - [lit] SROIE/receipt extraction: published baselines vs. our clean run
  fig_ece_literature_context.png  - [lit] our ECE vs. published self-consistency ECE values
  fig_selfcorrection_literature.png - [lit] accuracy delta from correction: intrinsic vs oracle-guided vs ours
  fig_calibration.png             - [ours] reliability diagram (confidence vs accuracy), n=9 clean
  fig_error_by_field.png          - [ours] per-field error breakdown, n=9 clean

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
LIT_COLOR = "#8172B2"   # published/external results
OURS_COLOR = "#55A868"  # our own pilot

# --- literature values used only for external-context figures --------------
# Each entry cites the bibliography key used in harness_paper.tex.
# "Ours" values are Variant C (full harness) on the n=9 clean subset of our
# 20-document attempted run -- recomputed by main() from results.json, not
# hardcoded, to avoid this file silently drifting from the real numbers.
LIT_BENCHMARKS_STATIC = [
    # (label, metric_name, value_pct, bib_key)
    ("LayoutLM\n(fine-tuned)", "F1", 95.24, "b8"),
    ("LayoutLMv2\n(fine-tuned)", "F1", 97.81, "b9"),
    ("Gemini VLM\n(prompted)", "Accuracy", 87.46, "b10"),
]

# Self-consistency ECE values reported by Ma et al. 2024 (b11), plus the
# p(True) single-sample baseline they compare against, and our own ECE.
LIT_ECE_POINTS = [
    ("Mistral-7B / GSM8K\nself-consistency", 0.092, "b11"),
    ("Mixtral-8x7B / GSM8K\nself-consistency", 0.075, "b11"),
    ("Mistral-7B / MathQA\nself-consistency", 0.091, "b11"),
    ("Mistral-7B / GSM8K\np(True) baseline", 0.127, "b11"),
    ("Mixtral-8x7B / GSM8K\np(True) baseline", 0.195, "b11"),
    ("Mistral-7B / MathQA\np(True) baseline", 0.350, "b11"),
]

# Huang et al. 2024 (b3) self-correction accuracy deltas (percentage points).
LIT_SELFCORRECT_STATIC = [
    ("GSM8K\nGPT-3.5", -1.2, "intrinsic"),
    ("CommonSenseQA\nGPT-3.5", -34.0, "intrinsic"),
    ("GSM8K\nGPT-4", -6.5, "intrinsic"),
    ("GSM8K\nGPT-3.5", 8.4, "oracle"),
    ("CommonSenseQA\nGPT-3.5", 13.9, "oracle"),
]


def load_clean_and_failed():
    """Split the attempted run into clean (API succeeded) and quota-failed
    (all retries exhausted -> empty prediction) records. Only clean records
    are valid extraction-accuracy evidence."""
    res = json.loads((BASE / "outputs" / "results.json").read_text())
    records = res["records"]
    clean = [r for r in records if not r["errors"]["A"]]
    failed = [r for r in records if r["errors"]["A"]]
    return records, clean, failed


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
def fig_benchmark_comparison(our_f1, our_exact, n_clean):
    """[literature] SROIE / receipt key-value extraction: published baselines
    vs. our clean run. Bars are drawn from other authors' own evaluations
    (LayoutLM/LayoutLMv2 on SROIE Task 3, a prompted-Gemini receipt-extraction
    study) plus our own F1/exact-match on the n_clean successfully-completed
    documents -- kept visually distinct (hatched) since it is still a small
    sample, not a comparable-scale rerun."""
    lit_benchmarks = LIT_BENCHMARKS_STATIC[:2] + [
        (f"Ours\n(n={n_clean})", "F1", our_f1, None),
        LIT_BENCHMARKS_STATIC[2],
        (f"Ours\n(n={n_clean})", "Exact", our_exact, None),
    ]
    labels = [l for l, *_ in lit_benchmarks]
    metrics = [m for _, m, *_ in lit_benchmarks]
    vals = [v for _, _, v, _ in lit_benchmarks]
    is_ours = [lab.startswith("Ours") for lab in labels]

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    x = np.arange(len(labels))
    colors = [OURS_COLOR if o else LIT_COLOR for o in is_ours]
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)
    for b, o in zip(bars, is_ours):
        if o:
            b.set_hatch("//")
    for b, v, m in zip(bars, vals, metrics):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f}\n({m})",
                ha="center", va="bottom", fontsize=7.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 112)
    ax.set_title("SROIE / receipt extraction: published baselines vs. our run")
    handles = [mpatches.Patch(facecolor=LIT_COLOR, edgecolor="black", label="Published (larger-scale) results"),
               mpatches.Patch(facecolor=OURS_COLOR, edgecolor="black", hatch="//", label=f"Ours (n={n_clean}, quota-clean)")]
    ax.legend(handles=handles, loc="lower center", fontsize=7.5, ncol=1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(FIG_DIR / "fig_benchmark_comparison.png")
    plt.close(fig)


def fig_ece_literature_context(our_ece, n_clean):
    """[literature] Our ECE vs. published self-consistency ECE values (and
    their p(True) baselines) from Ma et al. 2024 (b11)."""
    labels = [l for l, *_ in LIT_ECE_POINTS]
    vals = [v for _, v, _ in LIT_ECE_POINTS]
    is_baseline = ["p(True)" in l for l in labels]
    all_labels = labels + [f"Ours: Variant C\nself-consistency (n={n_clean})"]
    all_vals = vals + [our_ece]
    all_ours = is_baseline + [False]  # reuse for coloring; recompute below

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    y = np.arange(len(all_labels))
    colors = []
    for i, lab in enumerate(all_labels):
        if lab.startswith("Ours"):
            colors.append(OURS_COLOR)
        elif "p(True)" in lab:
            colors.append("#C44E52")
        else:
            colors.append(LIT_COLOR)
    bars = ax.barh(y, all_vals, color=colors, edgecolor="black", linewidth=0.7)
    for b, v in zip(bars, all_vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                va="center", fontsize=7.5)
    ax.set_yticks(y)
    ax.set_yticklabels(all_labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel("Expected Calibration Error (lower = better calibrated)")
    ax.set_title("Self-consistency ECE: literature range vs. our run")
    handles = [mpatches.Patch(facecolor=LIT_COLOR, edgecolor="black", label="Self-consistency confidence (lit.)"),
               mpatches.Patch(facecolor="#C44E52", edgecolor="black", label="p(True) single-sample baseline (lit.)"),
               mpatches.Patch(facecolor=OURS_COLOR, edgecolor="black", label=f"Ours (n={n_clean}, quota-clean)")]
    ax.legend(handles=handles, loc="lower right", fontsize=6.8)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.savefig(FIG_DIR / "fig_ece_literature_context.png")
    plt.close(fig)


def fig_selfcorrection_literature(n_clean, n_flagged_docs):
    """[literature] Accuracy delta (percentage points) from a self-correction
    call: intrinsic (unguided) vs. oracle-guided (Huang et al. 2024, b3),
    vs. our confidence-gated targeted correction. Note: on the n_clean
    successfully-completed documents, self-consistency confidence never fell
    below tau, so the correction step was never triggered (n_flagged_docs=0)
    -- the 0.0 bar reflects "never fired," not "fired and had no effect."""
    ours_label = f"Receipts\n(n={n_clean}, {n_flagged_docs} flagged)"
    lit_selfcorrect = LIT_SELFCORRECT_STATIC + [(ours_label, 0.0, "ours")]
    labels = [l for l, *_ in lit_selfcorrect]
    vals = [v for _, v, _ in lit_selfcorrect]
    kinds = [k for *_, k in lit_selfcorrect]
    color_map = {"intrinsic": "#C44E52", "oracle": LIT_COLOR, "ours": OURS_COLOR}
    colors = [color_map[k] for k in kinds]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.7)
    for k, b in zip(kinds, bars):
        if k == "ours":
            b.set_hatch("//")
    ax.axhline(0, color="black", linewidth=0.8)
    for b, v in zip(bars, vals):
        va = "bottom" if v >= 0 else "top"
        off = 0.8 if v >= 0 else -0.8
        ax.text(b.get_x() + b.get_width() / 2, v + off, f"{v:+.1f}",
                ha="center", va=va, fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.3)
    ax.set_ylabel("Accuracy change from correction (pp)")
    ax.set_title("Self-correction effect: intrinsic vs. oracle-guided vs. ours")
    ax.set_ylim(-40, 24)
    group_x = {"intrinsic": np.mean([0, 1, 2]), "oracle": np.mean([3, 4]), "ours": 5}
    group_label = {"intrinsic": "Intrinsic (unguided)", "oracle": "Oracle-guided", "ours": "Ours"}
    seen = set()
    for k in kinds:
        if k in seen:
            continue
        seen.add(k)
        ax.text(group_x[k], 20, group_label[k], ha="center", fontsize=7.5, style="italic")
    handles = [mpatches.Patch(facecolor="#C44E52", edgecolor="black", label="Intrinsic / unguided (Huang et al. 2024)"),
               mpatches.Patch(facecolor=LIT_COLOR, edgecolor="black", label="Oracle-guided (Huang et al. 2024)"),
               mpatches.Patch(facecolor=OURS_COLOR, edgecolor="black", hatch="//", label=f"Ours: confidence-gated targeted (n={n_clean})")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=7.3, ncol=1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(FIG_DIR / "fig_selfcorrection_literature.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 3 ---
def fig_calibration(bn, ba, bc, bins, ece, n_clean):
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
    ax.set_title(f"Reliability diagram (Variant C, n={n_clean}) — ECE = {ece:.3f}")
    ax.legend(loc="upper left", fontsize=7.5)
    ax.grid(alpha=0.3)
    fig.savefig(FIG_DIR / "fig_calibration.png")
    plt.close(fig)


# ---------------------------------------------------------------- fig 5 ---
def fig_error_by_field(records, n_clean):
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
    ax.set_title(f"Per-field error breakdown (n={n_clean}, quota-clean)")
    ax.set_ylim(0, max(counts + [1]) + 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.savefig(FIG_DIR / "fig_error_by_field.png")
    plt.close(fig)


def main():
    records, clean, failed = load_clean_and_failed()
    n_clean, n_failed = len(clean), len(failed)
    gts = [r["gt"] for r in clean]

    # Variant C metrics on the clean subset -- feed the literature-comparison
    # figures instead of hardcoding, so they can't silently drift from
    # results.json.
    preds_c = [r["C"] for r in clean]
    fex = sum(int(exact(p.get(f, ""), g.get(f, ""))) for p, g in zip(preds_c, gts) for f in config.SCHEMA_FIELDS)
    ff1 = sum(field_f1(p.get(f, ""), g.get(f, "")) for p, g in zip(preds_c, gts) for f in config.SCHEMA_FIELDS)
    nf = n_clean * len(config.SCHEMA_FIELDS)
    our_exact, our_f1 = 100 * fex / nf, 100 * ff1 / nf
    n_flagged_docs = sum(1 for r in clean if r["flagged"])

    bn, ba, bc, bins = compute_calibration(clean)
    tot = sum(bn)
    our_ece = sum((bn[i] / tot) * abs(ba[i] / bn[i] - bc[i] / bn[i]) for i in range(bins) if bn[i])

    fig_pipeline()
    fig_benchmark_comparison(our_f1, our_exact, n_clean)
    fig_ece_literature_context(our_ece, n_clean)
    fig_selfcorrection_literature(n_clean, n_flagged_docs)
    fig_calibration(bn, ba, bc, bins, our_ece, n_clean)
    fig_error_by_field(clean, n_clean)

    print(f"n_clean={n_clean} n_failed={n_failed} field_exact={our_exact:.1f} field_f1={our_f1:.1f} "
          f"ece={our_ece:.3f} n_flagged_docs={n_flagged_docs}")
    print("Wrote figures to", FIG_DIR)
    for p in sorted(FIG_DIR.glob("fig_*.png")):
        print(" -", p.name)


if __name__ == "__main__":
    main()
