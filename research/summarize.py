"""Summarize results.json into a paper-ready findings report (markdown + console).

Computes the comparison table across pipeline variants and prints concrete
research insights (which fields fail, does self-correction help, calibration).
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import config  # noqa: E402
from evaluate import exact, field_f1, Metrics  # noqa: E402


def compute(records):
    gts = [r["gt"] for r in records]
    table = {}
    for variant in ["A", "B", "C"]:
        preds = [r[variant] for r in records]
        confs = [r["conf"] for r in records]
        n_calls = sum(r["n_calls"] for r in records)
        fex = sum(int(exact(p.get(f, ""), g.get(f, ""))) for p, g in zip(preds, gts) for f in config.SCHEMA_FIELDS)
        ff1 = sum(field_f1(p.get(f, ""), g.get(f, "")) for p, g in zip(preds, gts) for f in config.SCHEMA_FIELDS)
        full = sum(1 for p, g in zip(preds, gts) if all(exact(p.get(f, ""), g.get(f, "")) for f in config.SCHEMA_FIELDS))
        n = len(records)
        nf = n * len(config.SCHEMA_FIELDS)
        table[variant] = {
            "field_exact": fex / nf,
            "field_f1": ff1 / nf,
            "doc_full": full / n,
            "n_calls": n_calls,
            "calls_per_doc": n_calls / n,
        }
    # calibration for B/C
    eces = {}
    for variant in ["B", "C"]:
        pairs = []
        for r in records:
            for f in config.SCHEMA_FIELDS:
                pairs.append((r["conf"].get(f, 0.0), int(exact(r[variant].get(f, ""), r["gt"].get(f, "")))))
        bins = 10
        bn, ba, bc = [0] * bins, [0.0] * bins, [0.0] * bins
        for c, ok in pairs:
            b = min(int(c * bins), bins - 1)
            bn[b] += 1; ba[b] += ok; bc[b] += c
        tot = max(1, sum(bn))
        ece = sum((bn[b] / tot) * abs(ba[b] / bn[b] - bc[b] / bn[b]) for b in range(bins) if bn[b])
        eces[variant] = ece
    # error analysis
    err_by_field = Counter()
    for r in records:
        for f in config.SCHEMA_FIELDS:
            if not exact(r["C"].get(f, ""), r["gt"].get(f, "")):
                err_by_field[f] += 1
    corr_improve = sum(1 for r in records if any(
        not exact(r["B"].get(f, ""), r["gt"].get(f, "")) and exact(r["C"].get(f, ""), r["gt"].get(f, ""))
        for f in config.SCHEMA_FIELDS))
    corr_regress = sum(1 for r in records if any(
        exact(r["B"].get(f, ""), r["gt"].get(f, "")) and not exact(r["C"].get(f, ""), r["gt"].get(f, ""))
        for f in config.SCHEMA_FIELDS))
    flagged_help = sum(1 for r in records if r["flagged"] and any(
        not exact(r["B"].get(f, ""), r["gt"].get(f, "")) and exact(r["C"].get(f, ""), r["gt"].get(f, ""))
        for f in config.SCHEMA_FIELDS))
    return table, eces, err_by_field, {"improved": corr_improve, "regressed": corr_regress, "flagged_docs_helped": flagged_help}


def to_md(table, eces, err_by_field, corr) -> str:
    L = []
    L.append("# Harness Comparison — SROIE pilot\n")
    L.append("| variant | field exact | field F1 | doc full | calls/doc | ECE |")
    L.append("|---|---|---|---|---|---|")
    for v in ["A", "B", "C"]:
        t = table[v]
        L.append(f"| {v} | {t['field_exact']:.3f} | {t['field_f1']:.3f} | {t['doc_full']:.3f} | {t['calls_per_doc']:.1f} | {eces.get(v, float('nan')):.3f} |")
    L.append("")
    L.append("## Error analysis (variant C)") 
    for f, c in err_by_field.most_common():
        L.append(f"- `{f}`: {c} docs wrong")
    L.append("")
    L.append("## Self-correction effect")
    L.append(f"- docs where correction fixed a field: **{corr['improved']}**")
    L.append(f"- docs where correction hurt a field: **{corr['regressed']}**")
    L.append(f"- flagged docs where correction helped: **{corr['flagged_docs_helped']}**")
    L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    res = json.load(open(config.OUTPUTS / "results.json"))
    records = res.get("records", [])
    if not records:
        print("No records — did the run complete without quota errors?")
        sys.exit(1)
    table, eces, err_by_field, corr = compute(records)
    for v in ["A", "B", "C"]:
        t = table[v]
        print(f"{v}: field_exact={t['field_exact']:.3f} field_f1={t['field_f1']:.3f} "
              f"doc_full={t['doc_full']:.3f} calls/doc={t['calls_per_doc']:.1f} ECE={eces.get(v, float('nan')):.3f}")
    print("\nError by field (C):", dict(err_by_field))
    print("Self-correct improve/regress:", corr)
    md = to_md(table, eces, err_by_field, corr)
    (config.OUTPUTS / "FINDINGS.md").write_text(md)
    print("\nWrote ->", config.OUTPUTS / "FINDINGS.md")
