# BabyYoday — Technology Stack

## Languages & Runtimes

| Layer | Language | Version |
|-------|----------|---------|
| Application (inference, admin, agent, data pipeline, builder) | Python | 3.11 |
| Infrastructure (CDK) | TypeScript | ~5.4.0 |
| Container base | python:3.11-slim (ECR public) | — |

## Python Dependencies (`requirements.txt`)

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework for inference server and admin panel |
| `uvicorn[standard]` | ASGI server |
| `sentence-transformers` | `all-MiniLM-L6-v2` embedding model (384-dim, CPU-friendly) |
| `faiss-cpu` | Local vector store — no external service needed |
| `llama-cpp-python` | CPU inference for GGUF-quantized SLMs (Phi-3-mini, Mistral-7B, Llama-3-8B) |
| `python-multipart` | File upload support |
| `jinja2` | Admin panel HTML templates |
| `watchdog` | File system watcher for `/data/incoming/` |
| `pypdf` | PDF parsing |
| `python-docx` | DOCX parsing |
| `openpyxl` | XLSX parsing |
| `pyyaml` | Config file parsing (`config.yaml`) |
| `numpy` | Centroid computation, vector math |
| `httpx` | Async HTTP client |
| `pytest` | Test framework |

## Infrastructure Dependencies (`infra/package.json`)

| Package | Version | Purpose |
|---------|---------|---------|
| `aws-cdk-lib` | 2.x | CDK constructs for all AWS resources |
| `constructs` | ^10.0.0 | CDK construct base |
| `aws-cdk` (dev) | 2.x | CDK CLI |
| `typescript` (dev) | ~5.4.0 | TypeScript compiler |
| `@types/node` (dev) | 20.x | Node type definitions |

## AWS Services Used

| Service | Purpose |
|---------|---------|
| ECS Fargate | Runs InferenceService (4 vCPU / 16 GB) and AdminService (0.5 vCPU / 2 GB) |
| ECR | Docker image registry |
| ALB | HTTP routing — /query → inference, / /upload → admin |
| CloudFront | HTTPS termination, CDN |
| EFS | Persistent storage for FAISS index, docs, query logs (survives restarts) |
| S3 | Document uploads, model weight storage |
| Secrets Manager | API key (`babyyoday/api-key`, 32-char auto-generated) |
| CodePipeline | CI/CD — GitHub → Build → Deploy |
| CodeBuild | Docker build + ECR push (`STANDARD_7_0`, LARGE compute, privileged) |
| CloudWatch Logs | `/babyyoday/inference`, `/babyyoday/admin`, `/babyyoday/codebuild` |
| VPC | Private subnets for ECS tasks, public subnets for ALB |

## Configuration (`config.yaml`)

```yaml
model:
  path: ./models/model.gguf
  n_ctx: 2048
  n_gpu_layers: 0        # CPU inference by default
  temperature: 0.3

embedding:
  model_name: all-MiniLM-L6-v2

retrieval:
  top_k: 5
  relevance_threshold: 0.3

domain_gate:
  similarity_threshold: 0.20
  centroid_path: ./data/centroid.npy

faiss:
  index_path: ./data/faiss.index
  metadata_path: ./data/metadata.json

chunking:
  chunk_size: 400
  chunk_overlap: 50

server:
  host: 0.0.0.0
  port: 8000

admin:
  port: 8001
```

## Development Commands

### Local Development

```bash
# Start full agent locally (Docker Compose)
docker-compose up

# Run tests
pytest

# Local setup
python setup_local.py

# Build a customer image from data directory
python builder/build_customer.py --data ./sample_data/
```

### Infrastructure (CDK)

```bash
cd infra

# Install dependencies
npm install

# Compile TypeScript
npm run build

# Watch for changes
npm run watch

# Preview changes
cdk diff

# Deploy all stacks (in dependency order)
cdk deploy --all

# Deploy individual stacks
cdk deploy BabyYodayNetwork
cdk deploy BabyYodayStorage
cdk deploy BabyYodayEcr
cdk deploy BabyYodayEcs
cdk deploy BabyYodayCdn
cdk deploy BabyYodayPipeline

# Tear down
cdk destroy --all
```

### First Image Push (before pipeline runs)

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t babyyoday-agent -f builder/Dockerfile .
docker tag babyyoday-agent:latest <account>.dkr.ecr.us-east-1.amazonaws.com/babyyoday-agent:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/babyyoday-agent:latest
```

### Retrieve API Key

```bash
aws secretsmanager get-secret-value --secret-id babyyoday/api-key \
  --query SecretString --output text
```

## CI/CD Pipeline

GitHub (`main` branch) → CodeStar Connection → CodeBuild → ECR → ECS Deploy

- CodeBuild uses `IMAGE_TAG` = first 7 chars of commit SHA
- Pushes both `$IMAGE_TAG` and `latest` tags to ECR
- Produces two artifact files: `imagedefinitions-inference.json` and `imagedefinitions-admin.json`
- Deploy stage runs both EcsDeployActions in parallel with 20-minute timeout

## Fargate Sizing

| Service | CPU | Memory | Notes |
|---------|-----|--------|-------|
| InferenceService | 4 vCPU (4096) | 16 GB (16384 MiB) | sentence-transformers + FAISS + llama.cpp |
| AdminService | 0.5 vCPU (512) | 2 GB (2048 MiB) | Lightweight FastAPI + Jinja2, no ML models |

## Supported Document Formats

PDF, DOCX, TXT, XLSX — parsed by `data_pipeline/chunker.py`

## Supported SLM Models (GGUF format)

| Model | RAM (4-bit) | Hardware |
|-------|------------|---------|
| Phi-3-mini (3.8B) | ~3 GB | Mac Mini 8GB, $5 Lightsail |
| Mistral-7B | ~5 GB | Mac Mini 16GB, $10 Lightsail |
| Llama-3-8B | ~6 GB | Mac Mini 16GB+, $20 Lightsail |
