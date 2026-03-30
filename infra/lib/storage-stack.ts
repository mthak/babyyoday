import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as efs from "aws-cdk-lib/aws-efs";
import * as s3 from "aws-cdk-lib/aws-s3";
import { Construct } from "constructs";

interface StorageStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
}

export class StorageStack extends cdk.Stack {
  readonly docsBucket: s3.Bucket;
  readonly fileSystem: efs.FileSystem;

  constructor(scope: Construct, id: string, props: StorageStackProps) {
    super(scope, id, props);

    // S3: document uploads and model weights
    this.docsBucket = new s3.Bucket(this, "DocsBucket", {
      bucketName: `babyyoday-docs-${this.account}-${this.region}`,
      versioned: true,
      encryption: s3.BucketEncryption.S3_MANAGED,
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      autoDeleteObjects: false,
      lifecycleRules: [
        {
          noncurrentVersionExpiration: cdk.Duration.days(90),
        },
      ],
    });

    // EFS: persistent storage for FAISS index, docs dir, query logs
    // Shared across ECS task restarts — survives container replacement
    this.fileSystem = new efs.FileSystem(this, "AgentEfs", {
      vpc: props.vpc,
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      performanceMode: efs.PerformanceMode.GENERAL_PURPOSE,
      throughputMode: efs.ThroughputMode.BURSTING,
      encrypted: true,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      enableAutomaticBackups: true,
    });

    new cdk.CfnOutput(this, "DocsBucketName", { value: this.docsBucket.bucketName });
    new cdk.CfnOutput(this, "EfsFileSystemId", { value: this.fileSystem.fileSystemId });
  }
}
