"""
PixelRAG Pipeline
Handles: render → embed → index → serve → answer
"""

import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from config import (
    ADAPTER, CUDA_BIN, DPI, GPU_ID, N_DOCS,
    PIXELRAG_PORT, PIXELRAG_URL, VLLM_MAX_TOKENS,
    VLLM_MODEL, VLLM_TEMPERATURE, VLLM_TIMEOUT,
    VLLM_TOP_P, VLLM_URL,
)


# ── Helpers ───────────────────────────────────────────────

def log(msg):
    print(f"\n{'─'*60}\n{msg}\n{'─'*60}")


def run(cmd, check=True, env=None):
    print(f"$ {cmd}")
    merged = {**os.environ, **(env or {})}
    subprocess.run(cmd, shell=True, check=check, env=merged)


def cuda_env():
    return {
        "CUDA_HOME": str(CUDA_BIN.parent),
        "PATH": f"{CUDA_BIN}:{os.environ.get('PATH', '')}",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_TOKEN": os.environ.get("HF_TOKEN", "dummy"),
    }


def score_bar(score):
    filled = int(score * 10)
    color  = "🟢" if score >= 0.7 else "🟡" if score >= 0.4 else "🔴"
    return color + "█" * filled + "░" * (10 - filled)


def get_paths(pdf_path: Path):
    """Derive all folder paths from a PDF path."""
    stem        = pdf_path.stem
    base        = pdf_path.parent
    tiles_dir   = base / f"{stem}_tiles"
    article_dir = tiles_dir / "1.png.tiles"
    index_dir   = base / f"{stem}_index"
    emb_dir     = index_dir / "embeddings"
    return tiles_dir, article_dir, index_dir, emb_dir


def is_built(index_dir: Path) -> bool:
    return (index_dir / "index.faiss").exists()


def is_rendered(article_dir: Path) -> bool:
    return article_dir.exists() and len(list(article_dir.glob("tile_*.jpg"))) > 0


# ── Build steps ───────────────────────────────────────────


def render(pdf: Path, tiles_dir: Path, article_dir: Path):
    log(f"STEP 1 — Render PDF → tiles  (dpi={DPI})")

    # clean up any previous partial render
    if tiles_dir.exists():
        shutil.rmtree(tiles_dir)

    article_dir.mkdir(parents=True, exist_ok=True)
    run(f"pixelshot {pdf} -o {tiles_dir} --dpi {DPI}")

    # skip 1.png.tiles if it already exists from a previous run
    raw_sub = next(
        (p for p in tiles_dir.glob("*.tiles") if p.name != "1.png.tiles"), None
    )
    if not raw_sub:
        print("✗ No tiles found — check poppler"); sys.exit(1)

    for f in raw_sub.iterdir():
        shutil.copy(f, article_dir / f.name)
    shutil.rmtree(raw_sub)

    tiles_json = article_dir / "tiles.json"
    if tiles_json.exists():
        data = json.loads(tiles_json.read_text())
        data["article_id"] = 1
        tiles_json.write_text(json.dumps(data))

    tiles = list(article_dir.glob("tile_*.jpg"))
    print(f"\n✓ {len(tiles)} tiles ready")


def embed(tiles_dir: Path, emb_dir: Path):
    log("STEP 2 — Embed with Qwen3-VL + LoRA")
    emb_dir.mkdir(parents=True, exist_ok=True)
    run(
        f"pixelrag embed --shard-dir {tiles_dir} --output-dir {emb_dir} "
        f"--gpu-ids {GPU_ID} --backend direct_gpu --adapter {ADAPTER} --mode chunks",
        env=cuda_env(),
    )


def build_index(index_dir: Path, emb_dir: Path, stem: str):
    log("STEP 3 — Build FAISS index")
    run(
        f"{sys.executable} -m pixelrag_embed.index build "
        f"--embeddings-dir {emb_dir} --output-dir {index_dir} --nlist 1"
    )
    articles = index_dir / "articles.json"
    if not articles.exists():
        articles.write_text(json.dumps({"1": {"title": stem, "url": ""}}))
    print(f"\n✓ Index ready → {index_dir}")


def build(pdf_path: Path):
    """Full build pipeline for a PDF. Skips steps already done."""
    tiles_dir, article_dir, index_dir, emb_dir = get_paths(pdf_path)

    if is_built(index_dir):
        print(f"\n✓ Index already exists for {pdf_path.name} — skipping build")
        return index_dir, article_dir

    if not is_rendered(article_dir):
        render(pdf_path, tiles_dir, article_dir)
    else:
        print(f"\n✓ Tiles already exist — skipping render")

    embed(tiles_dir, emb_dir)
    build_index(index_dir, emb_dir, pdf_path.stem)
    return index_dir, article_dir


# ── Server ────────────────────────────────────────────────

_server_proc = None


def start_server(index_dir: Path) -> subprocess.Popen:
    global _server_proc

    log("Starting server")
    subprocess.run(f"fuser -k {PIXELRAG_PORT}/tcp",
                   shell=True, stderr=subprocess.DEVNULL)
    time.sleep(1)

    articles = index_dir / "articles.json"
    _server_proc = subprocess.Popen(
        f"pixelrag serve --index-dir {index_dir} "
        f"--articles-json {articles} --port {PIXELRAG_PORT}",
        shell=True,
        env={**os.environ, **cuda_env()},
    )

    for i in range(30):
        try:
            requests.get(f"{PIXELRAG_URL}/status", timeout=2)
            print(f"✓ Server up at {PIXELRAG_URL}")
            return _server_proc
        except Exception:
            pass
        time.sleep(2)
        print(f"  waiting... ({i+1}/30)")

    print(f"✓ Server started at {PIXELRAG_URL}")
    return _server_proc


def stop_server():
    global _server_proc
    if _server_proc:
        print("\nShutting down server...")
        _server_proc.terminate()
        _server_proc = None


# ── Search + Answer ───────────────────────────────────────

def retrieve(query: str, n_docs: int = N_DOCS) -> list:
    r = requests.post(
        f"{PIXELRAG_URL}/search",
        json={"queries": [{"text": query}], "n_docs": n_docs},
        timeout=30,
    )
    return r.json()["results"][0]["hits"]


def tile_to_b64(article_dir: Path, tile_index: int) -> str | None:
    for ext in ["jpg", "png"]:
        p = article_dir / f"tile_{tile_index:04d}.{ext}"
        if p.exists():
            return base64.b64encode(p.read_bytes()).decode()
    return None


def answer(query: str, article_dir: Path, n_docs: int = N_DOCS, save_dir: Path = None) -> dict:
    hits = retrieve(query, n_docs)

    print("\n[Retrieval]")
    for i, h in enumerate(hits, 1):
        print(f"  [{i}] page={h['tile_index']+1}  score={h['score']:.4f} {score_bar(h['score'])}")

    content = []
    for h in hits:
        b64 = tile_to_b64(article_dir, h["tile_index"])
        if b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    content.append({
        "type": "text",
        "text": (
            f"Based on the document page(s) shown, answer this question:\n\n{query}\n\n"
            "Format your answer as a numbered list with each point on a new line.\n"
            "Be concise and specific. "
            "If the answer is NOT visible in these pages, say so clearly instead of guessing."
        ),
    })

    r = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": VLLM_MAX_TOKENS,
            "temperature": VLLM_TEMPERATURE,
            "top_p": VLLM_TOP_P,
        },
        timeout=VLLM_TIMEOUT,
    )

    ans = r.json()["choices"][0]["message"]["content"]
    print(f"\n[Answer]\n{ans}")

    result = {
        "query": query,
        "pages": [h["tile_index"]+1 for h in hits],
        "scores": [h["score"] for h in hits],
        "answer": ans,
    }

    if save_dir:
        out = save_dir / "last_answer.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n✓ Saved → {out}")

    return result


# ── CLI entrypoint ────────────────────────────────────────

def main():
    from config import PDF_PATH

    pdf = Path(PDF_PATH).resolve()
    if not pdf.exists():
        print(f"✗ PDF not found: {pdf}"); sys.exit(1)

    print(f"\nPDF: {pdf}")

    index_dir, article_dir = build(pdf)
    server = start_server(index_dir)

    def shutdown(sig=None, frame=None):
        stop_server()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print("\n" + "═"*60)
    print("Ready! Type your questions. Type 'exit' to quit.")
    print("═"*60)

    while True:
        try:
            query = input("\nQuestion: ").strip()
            if not query:
                continue
            if query.lower() == "exit":
                shutdown()
            answer(query, article_dir, save_dir=pdf.parent)
        except EOFError:
            shutdown()


if __name__ == "__main__":
    main()