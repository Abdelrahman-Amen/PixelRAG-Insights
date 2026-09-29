"""
PixelRAG Configuration
Edit this file only — everything else reads from here.
"""

import os
from pathlib import Path

# ── PDF to process ─────────────────────────────────────────
PDF_PATH = os.getenv("PDF_PATH", "/home/abdelrahman/pixel_rag/sample_2.pdf")

# ── PixelRAG Server ────────────────────────────────────────
PIXELRAG_PORT = int(os.getenv("PIXELRAG_PORT", "30001"))
PIXELRAG_URL = f"http://localhost:{PIXELRAG_PORT}"
N_DOCS = int(os.getenv("N_DOCS", "3"))

# ── vLLM Server ────────────────────────────────────────────
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:7834")
VLLM_MODEL = os.getenv("VLLM_MODEL", "google/gemma-4-31B-it")
VLLM_TEMPERATURE = float(os.getenv("VLLM_TEMPERATURE", "0.1"))
VLLM_TOP_P = float(os.getenv("VLLM_TOP_P", "0.9"))
VLLM_MAX_TOKENS = int(os.getenv("VLLM_MAX_TOKENS", "512"))
VLLM_TIMEOUT = int(os.getenv("VLLM_TIMEOUT", "180"))

# ── Render ─────────────────────────────────────────────────
DPI = int(os.getenv("DPI", "200"))
GPU_ID = int(os.getenv("GPU_ID", "1"))

# ── Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploads"
INDEX_DIR = BASE_DIR / "indexes"

for d in (UPLOAD_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Model paths ────────────────────────────────────────────
CONDA_ENV = Path(os.environ.get("CONDA_PREFIX", ""))
CUDA_BIN = CONDA_ENV / "lib/python3.12/site-packages/nvidia/cu13/bin"
ADAPTER = Path(
    os.getenv(
        "LORA_ADAPTER",
        "/home/abdelrahman/.cache/huggingface/hub/"
        "models--wiki-screenshot-embedding-lora/lora_vit/ckpt200",
    )
)
