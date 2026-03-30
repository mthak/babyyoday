import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import { Construct } from "constructs";

export class NetworkStack extends cdk.Stack {
  readonly vpc: ec2.Vpc;
  readonly albSg: ec2.SecurityGroup;
  readonly ecsSg: ec2.SecurityGroup;
  readonly efsSg: ec2.SecurityGroup;

  constructor(scope: Construct, id: string, props: cdk.StackProps) {
    super(scope, id, props);

    this.vpc = new ec2.Vpc(this, "Vpc", {
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        {
          name: "Public",
          subnetType: ec2.SubnetType.PUBLIC,
          cidrMask: 24,
        },
        {
          name: "Private",
          subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS,
          cidrMask: 24,
        },
      ],
    });

    // ALB: accepts HTTPS from internet
    this.albSg = new ec2.SecurityGroup(this, "AlbSg", {
      vpc: this.vpc,
      description: "ALB - allow HTTPS inbound",
      allowAllOutbound: true,
    });
    this.albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), "HTTPS");
    this.albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), "HTTP redirect");

    // ECS tasks: accept traffic only from ALB
    this.ecsSg = new ec2.SecurityGroup(this, "EcsSg", {
      vpc: this.vpc,
      description: "ECS tasks - allow traffic from ALB only",
      allowAllOutbound: true,
    });
    this.ecsSg.addIngressRule(this.albSg, ec2.Port.tcp(8000), "Inference API from ALB");
    this.ecsSg.addIngressRule(this.albSg, ec2.Port.tcp(8001), "Admin panel from ALB");

    // EFS: accept NFS only from ECS tasks
    this.efsSg = new ec2.SecurityGroup(this, "EfsSg", {
      vpc: this.vpc,
      description: "EFS - allow NFS from ECS tasks only",
      allowAllOutbound: false,
    });
    this.efsSg.addIngressRule(this.ecsSg, ec2.Port.tcp(2049), "NFS from ECS");

    new cdk.CfnOutput(this, "VpcId", { value: this.vpc.vpcId });
  }
}
