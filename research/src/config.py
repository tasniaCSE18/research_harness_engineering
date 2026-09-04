"""Shared config / constants for the SROIE harness research pilot."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
OUTPUTS = BASE_DIR / "outputs"
LOGS = BASE_DIR / "logs"
for d in (OUTPUTS, LOGS):
    d.mkdir(parents=True, exist_ok=True)

MANIFEST = DATA_PROCESSED / "sroie_sample_manifest.json"
IMAGES_DIR = DATA_RAW / "sroie_val" / "images"

# SROIE key-value schema
SCHEMA_FIELDS = ["company", "date", "address", "total"]

# Gemini model for the pilot (free tier: flash = cheap + vision + thinking)
MODEL = "gemini-2.5-flash"
N_SAMPLES = int(__import__("os").environ.get("N_SAMPLES", "2"))  # self-consistency samples (quota-aware)
CONF_THRESHOLD = float(__import__("os").environ.get("CONF_THRESHOLD", "0.5"))
RETRY_BACKOFF_S = float(__import__("os").environ.get("RETRY_BACKOFF_S", "5"))
CACHE_DIR = OUTPUTS / "cache"
