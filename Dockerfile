FROM --platform=linux/arm64 ubuntu:24.04 AS llama
RUN apt-get update && apt-get install -y \
    build-essential cmake git libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /llama \
    && cd /llama \
    && cmake -B build -DLLAMA_CURL=ON -DCMAKE_BUILD_TYPE=Release \
    && cmake --build build --config Release -j$(nproc) --target llama-server

FROM --platform=linux/arm64 python:3.12-slim
RUN apt-get update && apt-get install -y libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=llama /llama/build/bin/llama-server /usr/local/bin/llama-server
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    httpx \
    redis \
    sentence-transformers \
    numpy \
    pyyaml \
    structlog \
    prometheus-client \
    pydantic \
    openai
COPY gateway ./gateway
COPY pool ./pool
COPY matrix ./matrix
EXPOSE 8000
CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
