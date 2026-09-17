FROM python:3.12-slim
RUN apt-get update && apt-get upgrade -y && apt-get install -y \
    libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    redis \
    numpy \
    pyyaml \
    structlog \
    prometheus-client \
    pydantic \
    openai
COPY gateway ./gateway
COPY pool ./pool
COPY matrix ./matrix
ENV LLAMA_BIN=/usr/local/bin/llama-server
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
