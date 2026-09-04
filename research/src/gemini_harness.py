"""Schema-guided extraction from receipt images via Gemini (free tier).

Implements the core building blocks of the "harness" for reliable document
intelligence:
  1. Schema-guided structured extraction (JSON schema / response_schema).
  2. Self-consistency confidence estimation (multiple samples + agreement).
  3. Confidence-aware, targeted self-correction (re-read only low-confidence fields).

All state lives in the caller; this module is deliberately stateless so the same
prompt/config can be evaluated under different "harness" configurations.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from google import genai
from google.genai import types

import config
from config import MODEL

SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "description": "Company/merchant name."},
        "date": {"type": "string", "description": "Receipt date (e.g. DD/MM/YYYY)."},
        "address": {"type": "string", "description": "Merchant address."},
        "total": {"type": "string", "description": "Total amount paid."},
    },
    "required": ["company", "date", "address", "total"],
}

EXTRACT_PROMPT = (
    "You are an expert document-understanding system. Read the receipt image "
    "carefully and extract exactly these fields as JSON:\n"
    "- company: the merchant/company name\n"
    "- date: the receipt date\n"
    "- address: the merchant's full address\n"
    "- total: the total amount\n"
    "Return only the JSON object."
)

CORRECT_PROMPT = (
    "You previously extracted the following fields from this receipt:\n"
    "{previous}\n\n"
    "An auditor flagged the field(s) {fields} as LOW CONFIDENCE / possibly wrong. "
    "Look again very carefully at the receipt image, focusing on the region and "
    "format for {fields}. Correct any mistake and return the FULL updated JSON "
    "object with all four fields (company, date, address, total), keeping the "
    "other fields unchanged if they were correct."
)


@dataclass
class SampleResult:
    """One structured extraction sample."""

    raw: dict[str, Any]
    ok: bool  # parsed + schema-valid
    error: Optional[str] = None


def _read_image_bytes(path) -> bytes:
    # Manifest paths are stored relative to the research/ base dir; resolve
    # against BASE_DIR so the runner works regardless of CWD.
    p = Path(path)
    if not p.exists() and not p.is_absolute():
        cand = Path(config.BASE_DIR) / p
        if cand.exists():
            p = cand
    with open(p, "rb") as f:
        return f.read()


def _cache_path(image_path: str, prompt_kind: str, temperature: float) -> Path:
    """Deterministic cache key from image filename + prompt kind + temperature."""
    stem = Path(image_path).stem
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return config.CACHE_DIR / f"{safe}__{prompt_kind}__t{int(temperature * 10)}.json"


def extract_one(
    client: genai.Client,
    image_path: str,
    model: str = MODEL,
    temperature: float = 0.2,
    prompt: str = EXTRACT_PROMPT,
    schema: dict = SCHEMA_JSON,
    extra_context: str = "",
    prompt_kind: str = "extract",
    use_cache: bool = True,
    max_retries: int = 3,
) -> SampleResult:
    """Single schema-guided extraction call, with disk cache + 429 retry/backoff.

    Free-tier tip: when the daily quota is exhausted the API returns 429
    RESOURCE_EXHAUSTED; we cache every successful response so a later re-run
    (after reset / on a paid key) does not re-spend calls.
    """
    user_text = prompt + ("\n" + extra_context if extra_context else "")
    cp = _cache_path(image_path, prompt_kind, temperature)
    if use_cache and cp.exists():
        try:
            return SampleResult(raw=json.loads(cp.read_text()), ok=True)
        except Exception:
            pass  # corrupt cache -> refetch
    last_err = ""
    for attempt in range(max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=model,
                contents=[
                    types.Part(text=user_text),
                    types.Part(
                        inline_data=types.Blob(
                            mime_type="image/jpeg", data=_read_image_bytes(image_path)
                        )
                    ),
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=temperature,
                ),
            )
            txt = (resp.text or "").strip()
            data = json.loads(txt)
            if use_cache:
                cp.parent.mkdir(parents=True, exist_ok=True)
                cp.write_text(json.dumps(data))
            return SampleResult(raw=data, ok=True)
        except Exception as e:  # noqa: BLE001 - provider errors vary
            last_err = str(e)[:300]
            if "429" in last_err and attempt < max_retries:
                wait = config.RETRY_BACKOFF_S * (attempt + 1)
                print(f"  [429 quota] retry {attempt + 1}/{max_retries} in {wait:.0f}s "
                      f"({Path(image_path).stem})", flush=True)
                time.sleep(wait)
                continue
            break
    return SampleResult(raw={}, ok=False, error=last_err)


def extract_many(
    client: genai.Client,
    image_path: str,
    n: int = 3,
    temperature: float = 0.5,
    model: str = MODEL,
    sleep_s: float = 0.0,
) -> list[SampleResult]:
    """Multiple samples for self-consistency (temp > 0)."""
    out = []
    for i in range(n):
        out.append(
            extract_one(
                client,
                image_path,
                model=model,
                temperature=temperature,
                prompt_kind="sample",
            )
        )
        if sleep_s:
            time.sleep(sleep_s)
    return out


# --- Confidence estimation ------------------------------------------------

def field_confidence_self_consistency(samples: list[SampleResult]) -> dict[str, float]:
    """Per-field confidence = fraction of valid samples agreeing on a value.

    Self-consistency confidence: fields where several independent samples give the
    same string are more trustworthy. Empty/missing field counts as disagreement.
    """
    vals: dict[str, list[str]] = {k: [] for k in SCHEMA_JSON["required"]}
    for s in samples:
        if not s.ok:
            continue
        for k in vals:
            v = (s.raw.get(k) or "").strip()
            vals[k].append(v)
    conf: dict[str, float] = {}
    for k, vs in vals.items():
        if not vs:
            conf[k] = 0.0
            continue
        # mode
        counts: dict[str, int] = {}
        for v in vs:
            counts[v] = counts.get(v, 0) + 1
        mode_n = max(counts.values())
        conf[k] = mode_n / len(vs)
    return conf


def majority_vote(samples: list[SampleResult]) -> tuple[dict[str, str], dict[str, float]]:
    """Majority-vote aggregate + per-field confidence."""
    fields = SCHEMA_JSON["required"]
    votes: dict[str, dict[str, int]] = {k: {} for k in fields}
    for s in samples:
        if not s.ok:
            continue
        for k in fields:
            v = (s.raw.get(k) or "").strip()
            if v:
                votes[k][v] = votes[k].get(v, 0) + 1
    agg: dict[str, str] = {}
    conf: dict[str, float] = {}
    for k in fields:
        if votes[k]:
            best = max(votes[k].items(), key=lambda kv: (kv[1], len(kv[0])))
            agg[k] = best[0]
            conf[k] = best[1] / len(samples)
        else:
            agg[k] = ""
            conf[k] = 0.0
    return agg, conf


# --- Self-correction ------------------------------------------------------

def correct_low_confidence(
    client: genai.Client,
    image_path: str,
    agg: dict[str, str],
    conf: dict[str, float],
    threshold: float = 0.5,
    model: str = MODEL,
) -> tuple[dict[str, str], list[str]]:
    """Re-read only fields below `threshold`; return corrected aggregate + flagged list."""
    flagged = [k for k in SCHEMA_JSON["required"] if conf.get(k, 0) < threshold]
    if not flagged:
        return agg, flagged
    prev = json.dumps(agg, indent=2)
    prompt = CORRECT_PROMPT.format(previous=prev, fields=", ".join(flagged))
    res = extract_one(
        client,
        image_path,
        model=model,
        temperature=0.0,
        prompt=prompt,
        prompt_kind="correct",
    )
    if res.ok:
        # take corrected values for flagged fields, keep others from agg
        merged = dict(agg)
        for k in flagged:
            v = (res.raw.get(k) or "").strip()
            if v:
                merged[k] = v
        return merged, flagged
    return agg, flagged
