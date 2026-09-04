"""Evaluation harness: score extraction pipelines against SROIE ground truth.

Implements the reliability metrics needed for the paper:
  - field-level exact-match accuracy
  - document-level full-match (all fields correct)
  - normalized field F1 (char/word overlap for near-miss credit)
  - confidence calibration (ECE) for the confidence-aware pipeline
  - cost proxy (number of model calls per document)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from config import SCHEMA_FIELDS


# --- string similarity helpers --------------------------------------------

def _norm(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def exact(a: str, b: str) -> bool:
    return _norm(a) == _norm(b) and bool(_norm(a))


def f1_words(a: str, b: str) -> float:
    A, B = set(_norm(a).split()), set(_norm(b).split())
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    inter = len(A & B)
    p = inter / len(A)
    r = inter / len(B)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def field_f1(a: str, b: str) -> float:
    """Char-level F1 over normalized strings — robust for dates/addresses."""
    A, B = _norm(a), _norm(b)
    if not A and not B:
        return 1.0
    if not A or not B:
        return 0.0
    from collections import Counter

    ca, cb = Counter(A), Counter(B)
    inter = sum((ca & cb).values())
    if inter == 0:
        return 0.0
    p = inter / len(A)
    r = inter / len(B)
    return 2 * p * r / (p + r)


# --- metrics ---------------------------------------------------------------

@dataclass
class Metrics:
    field_exact: float
    doc_full: float
    field_f1: float
    n_calls_total: int
    n_docs: int
    # calibration (only meaningful for confidence-aware runs)
    ece: float | None = None
    low_conf_frac: float | None = None

    def summary(self) -> str:
        s = (
            f"docs={self.n_docs}  calls={self.n_calls_total}  "
            f"field_exact={self.field_exact:.3f}  doc_full={self.doc_full:.3f}  "
            f"field_f1={self.field_f1:.3f}"
        )
        if self.ece is not None:
            s += f"  ECE={self.ece:.3f}  low_conf_frac={self.low_conf_frac:.3f}"
        return s


def evaluate(
    preds: list[dict[str, str]],
    gts: list[dict[str, str]],
    n_calls_total: int,
    confidences: list[dict[str, float]] | None = None,
) -> Metrics:
    """preds/gts: list of dicts over SCHEMA_FIELDS (aligned)."""
    n = len(preds)
    fields = SCHEMA_FIELDS
    fex = 0
    ff1 = 0.0
    ffull = 0
    for p, g in zip(preds, gts):
        ok = True
        for f in fields:
            e = exact(p.get(f, ""), g.get(f, ""))
            fex += int(e)
            ff1 += field_f1(p.get(f, ""), g.get(f, ""))
            ok = ok and e
        ffull += int(ok)
    ece = low_frac = None
    if confidences is not None and confidences:
        # expected calibration error over per-field (conf, correct) pairs
        pairs = []
        for c, p, g in zip(confidences, preds, gts):
            for f in fields:
                pairs.append((c.get(f, 0.0), exact(p.get(f, ""), g.get(f, ""))))
        bins = 10
        # bucket into confidence deciles
        bucket_acc = [0.0] * bins
        bucket_conf = [0.0] * bins
        bucket_n = [0] * bins
        for conf, corr in pairs:
            b = min(int(conf * bins), bins - 1)
            bucket_n[b] += 1
            bucket_acc[b] += float(corr)
            bucket_conf[b] += conf
        tot = max(1, sum(bucket_n))
        ece = 0.0
        for b in range(bins):
            if bucket_n[b] == 0:
                continue
            acc = bucket_acc[b] / bucket_n[b]
            conf = bucket_conf[b] / bucket_n[b]
            ece += (bucket_n[b] / tot) * abs(acc - conf)
        low_frac = sum(1 for c in confidences for f in fields if c.get(f, 0) < 0.5) / max(
            1, len(confidences) * len(fields)
        )
    return Metrics(
        field_exact=fex / max(1, n * len(fields)),
        doc_full=ffull / max(1, n),
        field_f1=ff1 / max(1, n * len(fields)),
        n_calls_total=n_calls_total,
        n_docs=n,
        ece=ece,
        low_conf_frac=low_frac,
    )
