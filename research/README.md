# SROIE Harness Pilot — Schema-Guided, Confidence-Aware, Self-Corrective Document Extraction

Research scaffold for the paper:
**"Harness Engineering for Reliable Document Intelligence: A Schema-Guided,
Confidence-Aware, and Self-Corrective Vision-Language Model Approach"**

Implements the *harness* idea from the two anchor papers in this directory
(`2603.28052v1.pdf` = Meta-Harness, `2604.08224v1.pdf` = Externalization review):
everything around a frozen VLM is a **harness** that decides what context the model
sees. We compare three harnesses of increasing reliability on a fixed VLM
(Gemini free tier):

| variant | what it does |
|---|---|
| **A baseline** | single schema-guided structured-extraction call (JSON schema) |
| **B confidence** | N self-consistency samples + majority vote + per-field confidence |
| **C self-correct** | B + targeted re-read/correction of only low-confidence fields |

Metrics: field exact-match, field F1 (char-level), document full-match, model
calls/doc, and expected calibration error (ECE) for confidence.

## Dataset

**SROIE** (ICDAR 2019 scanned receipts), real-world key-value extraction benchmark.
We use the `podbilabs/sroie-donut` HuggingFace mirror's `val` split
(126 receipts; images + `metadata.jsonl` with ground truth `company/date/address/total`).
Pilot sample: first 24 images (~16 MB).

- Manifest: `data/processed/sroie_sample_manifest.json`
- Raw: `data/raw/sroie_val/{metadata.jsonl,images/}`
- Source: https://huggingface.co/datasets/podbilabs/sroie-donut

## Setup

```bash
python3 -m venv venv
venv/bin/pip install google-genai pillow pypdf
cp .env.example .env   # put GEMINI_API_KEY (or gemini_api_key)
```

Key is read from `.env` (`set -a; . ./.env; set +a`) or `GEMINI_API_KEY` env.

## Run the experiment

The runner is **cache-first and quota-resilient**: every successful API response is
cached in `research/outputs/cache/`; on a free-tier 429 it retries with backoff and
skips docs already cached. So you can stop and resume freely.

```bash
cd research
MAX_DOCS=24 N_SAMPLES=2 WORKERS=4 python run_experiment.py
python summarize.py          # prints metrics table + writes outputs/FINDINGS.md
```

### Quota notes (free tier — learned the hard way)

- Free tier `generateContent` quota is per-day (resets midnight Pacific) and
  *small*. A 24-doc × (1 + N + correct) run can exhaust it.
- Observed: after ~120 calls/day the API returns `429 RESOURCE_EXHAUSTED` for
  `generateContent` while `countTokens` still works.
- Keep `N_SAMPLES` small (2), use `WORKERS=2-4`, and re-run after reset — cached
  docs are free. Realistic free-tier throughput ≈ 30–50 docs/day.

## File map

```
research/
  data/raw/sroie_val/        downloaded SROIE images + metadata
  data/processed/            sample manifest (image + ground truth)
  outputs/results.json       full per-doc records (preds A/B/C, conf, flagged, errors)
  outputs/FINDINGS.md        paper-ready summary (after summarize.py)
  src/config.py              schema, model, constants
  src/gemini_harness.py      extraction + confidence + self-correction primitives
  src/evaluate.py            reliability metrics + calibration (ECE)
  run_experiment.py          pipeline comparison driver (cache + retry + parallel)
  summarize.py               results -> findings report
```

## Next steps for the paper

- Scale to all 126 val docs once quota allows (or a paid key); report CI.
- Add a **date/format normalizer** — early pilot shows near-miss date-format errors
  (e.g. `05/06/2018` vs `05-06-2018`) penalize exact-match without being real misses.
- Compare **verbalized confidence** vs self-consistency confidence.
- Run the **Meta-Harness outer loop**: use an LLM proposer to mutate the prompt/schema
  config and search for the best harness, using this eval as the reward.
