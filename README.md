# PixelRAG

> Visual PDF search and Q&A — no OCR, no text extraction, no layout parsing.

Traditional RAG breaks PDFs into text chunks. PixelRAG treats every page as an **image** — so it works on scanned documents, charts, tables, and complex layouts that text-based systems miss entirely.

Pages are rendered as images, embedded with a vision model (Qwen3-VL + LoRA), indexed with FAISS, and retrieved by visual similarity. A multimodal LLM then reads the top matching pages and answers your question directly from the visuals.


<img width="1536" height="1024" alt="Image" src="https://github.com/user-attachments/assets/1fbb02bc-d18c-4e38-a692-ea30cb83e5cf" />


```
PDF → render pages as images → embed with Qwen3-VL + LoRA → FAISS index
                                                                    ↓
Question → embed query → vector search → top-K pages → LLM → Answer
```



## When to use PixelRAG

| | PixelRAG | Traditional RAG |
|--|---------|-----------------|
| Scanned PDFs | ✅ Yes | ❌ Needs OCR |
| Charts / diagrams | ✅ Yes | ❌ Lost in text |
| Complex layouts | ✅ Yes | ⚠️ Often broken |
| Text-only PDFs | ✅ Good | ✅ Better / faster |
| GPU required | ✅ Yes | ⚠️ Optional |
| Setup complexity | Higher | Lower |

Use PixelRAG when your documents are scanned, image-heavy, or have complex visual layouts. Use traditional RAG when documents are clean text and you want lower infrastructure cost.

## Stack

| Component | What it does |
|-----------|-------------|
| **PixelRAG** | Renders PDF pages, manages tiles, runs the search server |
| **Qwen3-VL-Embedding-2B + LoRA** | Embeds page images into vectors |
| **FAISS** | Fast vector similarity search |
| **Gemma-4-31B via vLLM** | Reads retrieved page images and answers in natural language |
| **Streamlit** | Web UI — upload PDF, ask questions, see answers with source pages |

## Requirements

- Linux server with NVIDIA GPU
- Python 3.12 (via conda)
- poppler-utils installed (`apt install poppler-utils`)
- vLLM server running with your chosen model
- HuggingFace models cached locally (offline mode enabled)

## Setup

### 1. Clone

```bash
git clone https://github.com/Abdelrahman-Amen/pixel_rag.git
cd pixel_rag
```

### 2. Create conda environment

```bash
conda create -n pixel_rag python=3.12 -y
conda activate pixel_rag
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> `sglang==0.4.6.post1` is pinned — do not upgrade; newer versions break PixelRAG embedding.

### 4. Install poppler (PDF renderer)

```bash
sudo apt install poppler-utils -y
```

### 5. Download models

**Qwen3-VL-Embedding-2B** (vision encoder, ~4 GB):

```bash
pip install huggingface_hub
huggingface-cli download Qwen/Qwen3-VL-Embedding-2B
```

**LoRA adapter** (screenshot retrieval fine-tune):

```bash
huggingface-cli download Chrisyichuan/wiki-screenshot-embedding-lora --local-dir \
  ~/.cache/huggingface/hub/models--wiki-screenshot-embedding-lora
```

### 6. Configure

Edit `config.py` (or set environment variables):

```python
PDF_PATH    = "/path/to/your/document.pdf"
VLLM_URL    = "http://localhost:7834"       # where your vLLM server is running
VLLM_MODEL  = "google/gemma-4-31B-it"      # your vLLM model name
GPU_ID      = 0                             # GPU index to use for embedding
```

Or copy `.env.example` → `.env` and set values there for Docker.

## Running

### Prerequisites (must be running before starting the UI)

**1. vLLM server** (serves your model for answer generation):

```bash
vllm serve <your-model> --port 7834
```

**2. PixelRAG serve** (vector search server — auto-started by the pipeline after indexing):

This starts automatically when you upload a PDF through the UI.

### Option A — Streamlit directly (recommended)

```bash
conda activate pixel_rag
streamlit run streamlit.py
```

Open `http://localhost:8501`, upload a PDF, wait for indexing, then ask questions.

### Option B — Docker (UI only)

```bash
cp .env.example .env   # fill in VLLM_URL, VLLM_MODEL
docker compose up --build
```

> Docker runs the Streamlit UI only. vLLM and PixelRAG serve must run on the host GPU server (outside Docker). The container connects to them via `VLLM_URL` and `PIXELRAG_PORT`.

### Option C — CLI (pipeline only, no UI)

```bash
conda activate pixel_rag
# Edit PDF_PATH in config.py first, then:
python pixelrag_pipeline.py
```

Runs the full pipeline (render → embed → index → serve) and opens an interactive question loop in the terminal.

## Configuration reference

All settings live in `config.py` and can be overridden with environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PDF_PATH` | sample_2.pdf | PDF to process (CLI mode) |
| `PIXELRAG_PORT` | 30001 | Port for the search server |
| `N_DOCS` | 3 | Number of pages to retrieve per query |
| `VLLM_URL` | http://localhost:7834 | vLLM server endpoint |
| `VLLM_MODEL` | google/gemma-4-31B-it | Model name passed to vLLM |
| `VLLM_MAX_TOKENS` | 512 | Max tokens in the answer |
| `VLLM_TEMPERATURE` | 0.1 | Generation temperature |
| `VLLM_TIMEOUT` | 180 | Request timeout (seconds) |
| `DPI` | 200 | PDF render resolution |
| `GPU_ID` | 0 | GPU index to use for embedding |
| `LORA_ADAPTER` | ~/.cache/…/ckpt200 | Path to LoRA checkpoint |

## Project structure

```
pixel_rag/
├── streamlit.py            # Streamlit web UI
├── pixelrag_pipeline.py    # Full pipeline: render → embed → index → serve → answer
├── config.py               # All settings (edit this)
├── requirements.txt
├── Dockerfile              # Builds the Streamlit UI container
├── docker-compose.yml      # Runs the UI container
├── Makefile                # Shortcuts
├── .env.example            # Environment variable template
├── .streamlit/
│   └── config.toml         # Streamlit server and theme settings
├── ruff.toml               # Linter config
└── .github/
    └── workflows/
        └── ci.yml          # CI: lint → docker build → docker push
```

### What gets created when you process a PDF

```
uploads/
└── mydoc/
    ├── mydoc_tiles/             # page images in PixelRAG format
    │   └── 1.png.tiles/
    │       ├── tiles.json       # metadata (article_id, page count)
    │       ├── chunks.json      # chunk definitions
    │       └── tile_0000.jpg … tile_NNNN.jpg
    └── mydoc_index/             # FAISS index
        ├── index.faiss
        ├── index.pkl
        ├── articles.json
        └── embeddings/
            └── shard_000/
                └── embeddings.npy
```

## CI/CD

GitHub Actions runs on every push to `main` or `dev`:

1. **Lint** — `ruff check .` + `black --check .`
2. **Docker build** — builds the image (no push, just validates)
3. **Docker push** — pushes to Docker Hub (main branch only)

### GitHub Secrets required

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your Docker Hub username |
| `DOCKERHUB_TOKEN` | Docker Hub access token (Read & Write) |

## Common commands

```bash
make run          # streamlit run streamlit.py
make lint         # ruff check . && black --check .
make fmt          # black .
make docker-up    # docker compose up --build
make docker-down  # docker compose down
```


# Demo 📽

Below is a demonstration of how the application works:


![Demo of the Application](https://github.com/Abdelrahman-Amen/PixelRAG-Insights/blob/main/Demo.gif)



## 📄 License

MIT License


