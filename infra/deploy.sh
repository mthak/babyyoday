#!/bin/bash
# BabyYoday — idempotent deploy script
# Handles the case where stacks were deleted but AWS resources (S3, ECR, EFS) still exist.
set -e

REGION=${AWS_DEFAULT_REGION:-us-east-1}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)

echo "Deploying BabyYoday to account $ACCOUNT in $REGION"

cd "$(dirname "$0")"
npm run build

# ── Helper: import a resource into a stack if the stack doesn't exist yet ─────
import_if_needed() {
  local STACK=$1
  local RESOURCE_TYPE=$2
  local LOGICAL_ID=$3
  local IDENTIFIER_KEY=$4
  local IDENTIFIER_VAL=$5

  STATUS=$(aws cloudformation describe-stacks --stack-name "$STACK" \
    --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "DOES_NOT_EXIST")

  if [ "$STATUS" = "DOES_NOT_EXIST" ] || [ "$STATUS" = "REVIEW_IN_PROGRESS" ]; then
    echo "Stack $STACK does not exist — checking if $RESOURCE_TYPE $IDENTIFIER_VAL exists..."

    # Check if the resource exists in AWS
    EXISTS=false
    case $RESOURCE_TYPE in
      AWS::ECR::Repository)
        aws ecr describe-repositories --repository-names "$IDENTIFIER_VAL" > /dev/null 2>&1 && EXISTS=true || true
        ;;
      AWS::S3::Bucket)
        aws s3api head-bucket --bucket "$IDENTIFIER_VAL" > /dev/null 2>&1 && EXISTS=true || true
        ;;
      AWS::EFS::FileSystem)
        aws efs describe-file-systems --file-system-id "$IDENTIFIER_VAL" > /dev/null 2>&1 && EXISTS=true || true
        ;;
    esac

    if [ "$EXISTS" = "true" ]; then
      echo "  $RESOURCE_TYPE $IDENTIFIER_VAL exists — importing into $STACK..."

      # Build minimal template for import
      TEMPLATE=$(python3 -c "
import json, sys
t = json.loads(open('cdk.out/${STACK}.template.json').read())
keep = {'$LOGICAL_ID'}
t['Resources'] = {k: v for k, v in t['Resources'].items() if k in keep}
t.pop('Outputs', None)
print(json.dumps(t))
")

      echo "$TEMPLATE" > /tmp/${STACK}-import.json

      cat > /tmp/${STACK}-import-resources.json << RESEOF
[{"ResourceType":"$RESOURCE_TYPE","LogicalResourceId":"$LOGICAL_ID","ResourceIdentifier":{"$IDENTIFIER_KEY":"$IDENTIFIER_VAL"}}]
RESEOF

      # Delete REVIEW_IN_PROGRESS stack if needed
      if [ "$STATUS" = "REVIEW_IN_PROGRESS" ]; then
        aws cloudformation delete-stack --stack-name "$STACK" 2>/dev/null || true
        aws cloudformation wait stack-delete-complete --stack-name "$STACK" 2>/dev/null || true
      fi

      aws cloudformation create-change-set \
        --stack-name "$STACK" \
        --change-set-name "import-existing" \
        --change-set-type IMPORT \
        --resources-to-import "file:///tmp/${STACK}-import-resources.json" \
        --template-body "file:///tmp/${STACK}-import.json" \
        --capabilities CAPABILITY_IAM > /dev/null

      aws cloudformation wait change-set-create-complete \
        --stack-name "$STACK" --change-set-name "import-existing" 2>/dev/null || true

      aws cloudformation execute-change-set \
        --stack-name "$STACK" --change-set-name "import-existing" > /dev/null

      aws cloudformation wait stack-import-complete --stack-name "$STACK"
      echo "  Import complete."
    fi
  fi
}

# ── Synthesize first so templates exist in cdk.out ────────────────────────────
npx cdk synth --quiet > /dev/null

# ── Handle existing retained resources ────────────────────────────────────────
import_if_needed "BabyYodayEcr" "AWS::ECR::Repository" "AgentRepo85A4923D" "RepositoryName" "babyyoday-agent"
import_if_needed "BabyYodayStorage" "AWS::S3::Bucket" "DocsBucketECEA003F" "BucketName" "babyyoday-docs-${ACCOUNT}-${REGION}"

# EFS import requires knowing the filesystem ID — look it up by tag
EFS_ID=$(aws efs describe-file-systems \
  --query "FileSystems[?Tags[?Key=='aws:cloudformation:logical-id' && Value=='AgentEfsCCDAE4F6']].FileSystemId | [0]" \
  --output text 2>/dev/null || echo "None")

if [ "$EFS_ID" != "None" ] && [ -n "$EFS_ID" ]; then
  import_if_needed "BabyYodayStorage" "AWS::EFS::FileSystem" "AgentEfsCCDAE4F6" "FileSystemId" "$EFS_ID"
fi

# ── Deploy all stacks ─────────────────────────────────────────────────────────
echo "Deploying all stacks..."
npx cdk deploy --all --require-approval never

echo ""
echo "Deploy complete."
echo ""
aws cloudformation describe-stacks --stack-name BabyYodayCdn \
  --query "Stacks[0].Outputs[?OutputKey=='DistributionDomainName'].OutputValue" \
  --output text | xargs -I{} echo "Agent endpoint: https://{}"
