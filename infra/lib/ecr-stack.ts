import * as cdk from "aws-cdk-lib";
import * as ecr from "aws-cdk-lib/aws-ecr";
import { Construct } from "constructs";

export class EcrStack extends cdk.Stack {
  readonly repository: ecr.Repository;

  constructor(scope: Construct, id: string, props: cdk.StackProps) {
    super(scope, id, props);

    this.repository = new ecr.Repository(this, "AgentRepo", {
      repositoryName: "babyyoday-agent",
      imageScanOnPush: true,
      encryption: ecr.RepositoryEncryption.AES_256,
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      emptyOnDelete: false,
    });

    new cdk.CfnOutput(this, "RepositoryUri", {
      value: this.repository.repositoryUri,
    });
  }
}
