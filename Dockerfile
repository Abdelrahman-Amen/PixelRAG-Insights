FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY config.py pixelrag_pipeline.py streamlit.py ./
COPY .streamlit/ .streamlit/

RUN mkdir -p uploads

EXPOSE 8501

CMD ["streamlit", "run", "streamlit.py"]
