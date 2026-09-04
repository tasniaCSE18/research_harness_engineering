"""Run the harness comparison experiment on the SROIE sample.

Pipeline variants (the "harness" is everything around the frozen VLM):
  A. baseline      : single schema-guided extraction call (temperature 0)
  B. confidence    : N self-consistency samples + majority vote + per-field confidence
  C. self-correct  : B + targeted re-read/correction of low-confidence fields

Metrics per variant: field exact-match, document full-match, field F1, model
calls per document, and (for B/C) confidence calibration ECE + low-conf fraction.

Run:  cd research && MAX_DOCS=24 python run_experiment.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# allow `python run_experiment.py` and `python -m src...`
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from google import genai  # noqa: E402

import config  # noqa: E402
from evaluate import evaluate  # noqa: E402
from gemini_harness import (  # noqa: E402
    correct_low_confidence,
    extract_many,
    extract_one,
    majority_vote,
)

MAX_DOCS = int(os.environ.get("MAX_DOCS", "24"))
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", config.CONF_THRESHOLD))
WORKERS = int(os.environ.get("WORKERS", "3"))  # free tier: modest parallelism


def load_manifest() -> list[dict]:
    with open(config.MANIFEST) as f:
        return json.load(f)


def _safe_pred(r) -> dict[str, str]:
    return {k: (r.raw.get(k) or "").strip() if r.ok else "" for k in config.SCHEMA_FIELDS}


def process_doc(client, d: dict, idx: int):
    """Run A, B, C for one doc; return aligned records."""
    gt = d["ground_truth"]

    # A: single baseline call
    ra = extract_one(client, d["image_path"], temperature=0.0)

    # B + C: self-consistency samples
    samples = extract_many(client, d["image_path"], n=config.N_SAMPLES, temperature=0.5)
    agg, conf = majority_vote(samples)
    corrected, flagged = correct_low_confidence(
        client, d["image_path"], agg, conf, threshold=CONF_THRESHOLD
    )

    n_calls = 1 + len(samples) + (1 if flagged else 0)
    return {
        "idx": idx,
        "file": d["file_name"],
        "gt": gt,
        "A": _safe_pred(ra),
        "B": agg,
        "C": corrected,
        "conf": conf,
        "flagged": flagged,
        "n_calls": n_calls,
        "errors": {
            "A": ra.error,
            "samples": [s.error for s in samples if not s.ok],
        },
    }


def main() -> None:
    client = genai.Client(
        api_key=os.environ.get("gemini_api_key") or os.environ.get("GEMINI_API_KEY")
    )
    docs = load_manifest()[:MAX_DOCS]
    print(f"Docs={len(docs)} conf_threshold={CONF_THRESHOLD} workers={WORKERS}", flush=True)

    records = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_doc, client, d, i): d for i, d in enumerate(docs)}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()
            records.append(rec)
            done += 1
            el = time.time() - t0
            print(f"  [{done}/{len(docs)}] doc {rec['idx']} ({rec['file']}) "
                  f"calls={rec['n_calls']} flagged={rec['flagged']} elapsed={el:.0f}s", flush=True)
    records.sort(key=lambda r: r["idx"])

    # --- aggregate metrics ---
    gts = [r["gt"] for r in records]
    for variant in ["A", "B", "C"]:
        preds = [r[variant] for r in records]
        confs = [r["conf"] for r in records]
        calls = sum(r["n_calls"] for r in records)
        m = evaluate(preds, gts, calls, confidences=confs if variant != "A" else None)
        print(f"\nVariant {variant}:")
        print("  " + m.summary(), flush=True)

    # save
    out = {
        "config": {
            "model": config.MODEL,
            "n_samples": config.N_SAMPLES,
            "conf_threshold": CONF_THRESHOLD,
            "workers": WORKERS,
            "n_docs": len(records),
        },
        "records": records,
    }
    out_path = config.OUTPUTS / "results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
