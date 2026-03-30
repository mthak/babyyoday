FROM public.ecr.aws/docker/library/python:3.11-slim

RUN groupadd -r agent && useradd -r -g agent -m agent

WORKDIR /app

# System deps for llama-cpp-python CPU build + curl for health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python (CPU build)
# For GPU: set CMAKE_ARGS="-DGGML_CUDA=on" before this line
RUN pip install --no-cache-dir llama-cpp-python

COPY inference     ./inference
COPY agent         ./agent
COPY admin         ./admin
COPY data_pipeline ./data_pipeline
COPY sample_data   ./sample_data
COPY config.yaml .

# Create data and model dirs — data/ is mounted from EFS at runtime, models/ populated via S3
RUN mkdir -p /app/data/incoming /app/data/docs /app/models && \
    chown -R agent:agent /app

COPY builder/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh && chown agent:agent /entrypoint.sh

USER agent

# Port 8000: agent query API
# Port 8001: admin panel
EXPOSE 8000 8001

ENTRYPOINT ["/entrypoint.sh"]
