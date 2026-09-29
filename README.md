# PixelRAG

Visual PDF search and Q&A using PixelRAG + Qwen3-VL + Gemma-4.

## Stack
- **PixelRAG** — visual page embedding and retrieval
- **Qwen3-VL-Embedding-2B** + LoRA — page image embedding
- **FAISS** — vector search index
- **Gemma-4-31B** via vLLM — answer generation
- **Streamlit** — web UI

## Requirements
- NVIDIA GPU
- vLLM server running with Gemma-4
- HuggingFace models cached locally

## Run locally
```bash
conda activate pixel_rag
streamlit run app.py
```

## Run with Docker
```bash
cp .env.example .env   # edit your values
docker compose up --build
```

## Config
Edit `config.py` to change settings, or set via environment variables — see `.env.example`.

## Project structure
```
├── app.py              # Streamlit UI
├── pipeline.py         # build + serve + answer logic
├── config.py           # all settings
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── Makefile            # shortcuts
├── .env.example        # env template
└── .github/
    └── workflows/
        └── ci.yml      # lint + docker build + push
```

## GitHub Secrets needed
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN`
