# BabyYoday — AWS Infrastructure (CDK TypeScript)

## Architecture

```
Internet → CloudFront (HTTPS) → ALB (HTTP) → ECS Fargate
                                                ├── InferenceService  :8000  /query /health
                                                └── AdminService      :8001  /  /upload
                                                        │
                                                       EFS (FAISS index, docs, query logs)
                                                       S3  (document uploads, model weights)
                                                       ECR (Docker image)
                                                       Secrets Manager (API key)
```

## Stacks

| Stack | Resources |
|-------|-----------|
| `BabyYodayNetwork` | VPC, subnets, NAT gateway, security groups |
| `BabyYodayStorage` | S3 bucket (docs/models), EFS filesystem |
| `BabyYodayEcr` | ECR repository |
| `BabyYodayEcs` | Fargate cluster, task defs, services, ALB, Secrets Manager |
| `BabyYodayCdn` | CloudFront distribution |
| `BabyYodayPipeline` | CodePipeline + CodeBuild (GitHub → ECR → ECS) |

## Prerequisites

```bash
npm install -g aws-cdk
cd infra && npm install
```

## One-time setup

1. Bootstrap CDK in your account/region (once per account):
   ```bash
   cdk bootstrap
   ```

2. Create a GitHub CodeStar connection in the AWS Console:
   - Go to CodePipeline → Settings → Connections → Create connection
   - Select GitHub, authorize, copy the connection ARN
   - Paste it into `lib/pipeline-stack.ts` replacing `YOUR_CONNECTION_ID`
   - Also update `YOUR_GITHUB_ORG` and repo name

## Deploy

```bash
# Build TypeScript
npm run build

# Preview changes
cdk diff

# Deploy all stacks
cdk deploy --all

# Or deploy individually in order
cdk deploy BabyYodayNetwork
cdk deploy BabyYodayStorage
cdk deploy BabyYodayEcr
cdk deploy BabyYodayEcs
cdk deploy BabyYodayCdn
cdk deploy BabyYodayPipeline
```

## First image push (before pipeline runs)

The ECS service needs at least one image in ECR before it can start:

```bash
# Build and push manually for the first deploy
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

docker build -t babyyoday-agent -f builder/Dockerfile .
docker tag babyyoday-agent:latest <account>.dkr.ecr.us-east-1.amazonaws.com/babyyoday-agent:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/babyyoday-agent:latest
```

## After deploy

CDK outputs:
- `BabyYodayCdn.DistributionDomainName` — your agent's public HTTPS endpoint
- `BabyYodayEcs.AlbDnsName` — ALB DNS (use CloudFront domain instead)
- `BabyYodayEcs.ApiKeySecretArn` — retrieve API key from Secrets Manager
- `BabyYodayEcr.RepositoryUri` — ECR image URI

Retrieve the API key:
```bash
aws secretsmanager get-secret-value --secret-id babyyoday/api-key --query SecretString --output text
```

## Fargate sizing

| Service | CPU | Memory | Notes |
|---------|-----|--------|-------|
| InferenceService | 2 vCPU | 8 GB | Fits Phi-3-mini (4-bit, ~3GB). Increase to 16GB for Mistral-7B |
| AdminService | 0.5 vCPU | 1 GB | Lightweight FastAPI + Jinja2 |

To run Mistral-7B, update `inferenceTaskDef` in `ecs-stack.ts`:
```typescript
memoryLimitMiB: 16384,
cpu: 4096,
```

## Teardown

```bash
cdk destroy --all
```

Note: S3 bucket and EFS have `RemovalPolicy.RETAIN` — delete manually to avoid data loss.
