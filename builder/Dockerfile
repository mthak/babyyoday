FROM python:3.11-slim

RUN groupadd -r agent && useradd -r -g agent -m agent

WORKDIR /app

# System deps for llama-cpp-python CPU build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install llama-cpp-python (CPU build)
# For GPU: set CMAKE_ARGS="-DGGML_CUDA=on" before this line
RUN pip install --no-cache-dir llama-cpp-python

COPY app_code/inference     ./inference
COPY app_code/agent         ./agent
COPY app_code/admin         ./admin
COPY app_code/data_pipeline ./data_pipeline

COPY data    ./data
COPY models  ./models
COPY config.yaml .

# Create data dirs with correct ownership
RUN mkdir -p /app/data/incoming /app/data/docs && \
    chown -R agent:agent /app

USER agent

# Port 8000: agent query API
# Port 8001: admin panel
EXPOSE 8000 8001

# Start both services
CMD ["sh", "-c", \
  "uvicorn inference.server:app --host 0.0.0.0 --port 8000 & \
   uvicorn admin.app:admin_app --host 0.0.0.0 --port 8001 & \
   wait"]
