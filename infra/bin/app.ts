#!/usr/bin/env node
import "source-map-support/register";
import * as cdk from "aws-cdk-lib";
import { NetworkStack } from "../lib/network-stack";
import { StorageStack } from "../lib/storage-stack";
import { EcrStack } from "../lib/ecr-stack";
import { EcsStack } from "../lib/ecs-stack";
import { CdnStack } from "../lib/cdn-stack";
import { PipelineStack } from "../lib/pipeline-stack";

const app = new cdk.App();

const env: cdk.Environment = {
  account: process.env.CDK_DEFAULT_ACCOUNT,
  region: process.env.CDK_DEFAULT_REGION ?? "us-east-1",
};

const networkStack = new NetworkStack(app, "BabyYodayNetwork", { env });

const storageStack = new StorageStack(app, "BabyYodayStorage", {
  env,
  vpc: networkStack.vpc,
});

const ecrStack = new EcrStack(app, "BabyYodayEcr", { env });

const ecsStack = new EcsStack(app, "BabyYodayEcs", {
  env,
  vpc: networkStack.vpc,
  efsFileSystem: storageStack.fileSystem,
  docsBucket: storageStack.docsBucket,
  repository: ecrStack.repository,
});

const cdnStack = new CdnStack(app, "BabyYodayCdn", {
  env,
  alb: ecsStack.alb,
});

new PipelineStack(app, "BabyYodayPipeline", {
  env,
  repository: ecrStack.repository,
  ecsService: ecsStack.inferenceService,
  adminService: ecsStack.adminService,
});

app.synth();
