# Research Notes — SROIE Harness Pilot

Date: 2026-09-03 · Model: `gemini-2.5-flash` (free tier) · Dataset: SROIE (real scanned receipts)

## What this pilot does

Tests the paper's central claim on a small real benchmark: **for a frozen VLM, the
"harness" (prompt + schema + sampling + correction policy) determines reliability.**
We compare three harnesses of increasing sophistication on SROIE key-value
extraction (fields: `company`, `date`, `address`, `total`):

- **A — baseline**: one schema-guided structured-extraction call (JSON Schema via `response_schema`).
- **B — confidence-aware**: 3 self-consistency samples → majority vote + per-field confidence (agreement rate).
- **C — self-corrective**: B + one targeted re-read of fields whose confidence < 0.5, merging corrections.

## Verified results (3-doc smoke run, before quota exhaustion)

| variant | field exact | field F1 | doc full | ECE | low-conf frac |
|---|---|---|---|---|---|
| A baseline | 0.833 | 0.983 | 0.667 | — | — |
| B confidence | 0.833 | 0.983 | 0.667 | 0.277 | 0.33 |
| C self-correct | 0.833 | 0.983 | 0.667 | 0.277 | 0.33 |

Error fields (variant C): `date` and `address` (1 doc each).

### What the 3 docs already show (directional, n=3)

1. **Schema-guided extraction is strong out of the box** — 0.83 field exact / 0.98 char-F1
   with a *single* call. The remaining errors are near-misses, not garbage.
2. **Confidence is informative**: the one doc where all 4 fields disagreed across
   samples (conf 0.33) was exactly the doc whose A-output contained case/format drift
   (`No. 76` vs `NO. 76`). Low self-consistency flagged the genuinely harder document.
3. **Confidence is miscalibrated as a raw probability**: ECE 0.28 — fields with
   conf ~0.67 are actually ~0.83 accurate. (Directionally *conservative*, which is
   what you want for a "route to human review" policy, but it must be calibrated to
   report meaningful numbers.)
4. **Blind self-correction did not help in this sample**: the correction call re-affirmed
   the majority values. This matches the literature (self-refine often plateaus); the
   open question is whether *targeted* correction helps on a larger/harder set.
5. **Date format is a real metric artifact**: `05/06/2018` vs GT `05-06-2018` is correct
   semantically but counts as an exact-match miss. A normalization layer is needed
   before exact-match is a fair headline metric.

## Free-tier quota findings (important for the paper's "reproducibility on a budget" angle)

- The Gemini free tier's per-day `generateContent` quota is **small**: we exhausted it after
  ~120 image calls (3-doc run + 24-doc run at 4 workers). After exhaustion, all calls return
  `429 RESOURCE_EXHAUSTED` until reset (midnight Pacific), while `countTokens` still works.
- `gemini-2.5-flash` on receipts: each image call is ~2–6 s (thinking model), so 24 docs ×
  5 calls ≈ 2–5 min *if* quota allows.
- Practical free-tier ceiling ≈ **30–50 documents/day** for this 5-call-per-doc design.
  This is itself a research-relevant constraint: a *budget-aware harness* that minimizes
  calls (e.g., only correct when confidence is low AND the doc is "hard") is a legitimate
  contribution axis.

## Robustness engineering added (so the full run is free to resume)

- **Disk cache** of every successful response in `research/outputs/cache/` (keyed by
  image + prompt kind + temperature). Re-runs skip cached docs → no re-spend.
- **429 retry with backoff** in the extraction layer.
- **Parallel worker pool** (default 3) with per-doc progress logging.
- `run_pilot.sh` = one command to run + summarize once quota is available.

## How to get full results

The 24-image sample is downloaded and manifests are ready. Once the daily quota resets
(or with any quota remaining), run:

```bash
./run_pilot.sh          # runs 24 docs (N_SAMPLES=2), then prints + saves FINDINGS.md
```

To control budget: `MAX_DOCS=10 N_SAMPLES=1 WORKERS=2 ./run_pilot.sh`.

## Mapping to the paper

| Paper claim | Evidence needed | This pilot provides |
|---|---|---|
| Schema-guided extraction is the reliability baseline | A vs. no-schema | A is strong; needs a no-schema control |
| Confidence routing beats blind processing | B/C calibration + human-review simulation | Confidence flags the hard doc; ECE measurable |
| Self-correction helps only when targeted | C vs. B on harder/larger set | n=3: no gain yet; design is ready to test |
| Harness optimization is the real lever | Meta-Harness outer loop over prompt/schema | Inner eval is now a clean reward function |
| Budget constraints matter | calls/doc vs. accuracy frontier | C costs ≤ 5 calls/doc; quota ceiling documented |

## Files

- `run_experiment.py` — pipeline comparison (A/B/C) with cache + retry + workers
- `src/gemini_harness.py` — extraction, self-consistency confidence, targeted correction
- `src/evaluate.py` — exact-match, char-F1, doc-full, ECE calibration
- `summarize.py` — results.json → FINDINGS.md comparison + error analysis
- `data/processed/sroie_sample_manifest.json` — 24-image pilot manifest with GT
