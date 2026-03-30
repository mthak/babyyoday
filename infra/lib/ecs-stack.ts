import * as cdk from "aws-cdk-lib";
import * as ec2 from "aws-cdk-lib/aws-ec2";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as efs from "aws-cdk-lib/aws-efs";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import * as s3 from "aws-cdk-lib/aws-s3";
import * as secretsmanager from "aws-cdk-lib/aws-secretsmanager";
import { Construct } from "constructs";

interface EcsStackProps extends cdk.StackProps {
  vpc: ec2.Vpc;
  efsFileSystem: efs.FileSystem;
  docsBucket: s3.Bucket;
  repository: ecr.Repository;
}

export class EcsStack extends cdk.Stack {
  readonly alb: elbv2.ApplicationLoadBalancer;
  readonly inferenceService: ecs.FargateService;
  readonly adminService: ecs.FargateService;

  constructor(scope: Construct, id: string, props: EcsStackProps) {
    super(scope, id, props);

    // ── Security groups ───────────────────────────────────────────────────────
    const albSg = new ec2.SecurityGroup(this, "AlbSg", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "ALB - allow HTTP and HTTPS inbound",
    });
    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(80), "HTTP");
    albSg.addIngressRule(ec2.Peer.anyIpv4(), ec2.Port.tcp(443), "HTTPS");

    const ecsSg = new ec2.SecurityGroup(this, "EcsSg", {
      vpc: props.vpc,
      allowAllOutbound: true,
      description: "ECS tasks - allow traffic from ALB",
    });
    ecsSg.addIngressRule(albSg, ec2.Port.tcp(8000), "Inference from ALB");
    ecsSg.addIngressRule(albSg, ec2.Port.tcp(8001), "Admin from ALB");

    // Add NFS ingress rule to the EFS filesystem's security group using a raw
    // CloudFormation resource to avoid a cross-stack dependency cycle.
    // The EFS SG ID is resolved at deploy time via Fn::Select on the EFS's SG list.
    new ec2.CfnSecurityGroupIngress(this, "EfsNfsFromEcs", {
      ipProtocol: "tcp",
      fromPort: 2049,
      toPort: 2049,
      sourceSecurityGroupId: ecsSg.securityGroupId,
      groupId: cdk.Fn.select(
        0,
        props.efsFileSystem.connections.securityGroups.map((sg) => sg.securityGroupId)
      ),
      description: "NFS from ECS tasks",
    });

    // ── API key secret ────────────────────────────────────────────────────────
    const apiKeySecret = new secretsmanager.Secret(this, "ApiKeySecret", {
      secretName: "babyyoday/api-key",
      generateSecretString: {
        excludePunctuation: true,
        passwordLength: 32,
      },
    });

    // ── EFS access point ──────────────────────────────────────────────────────
    const efsAccessPoint = new efs.AccessPoint(this, "AgentAccessPoint", {
      fileSystem: props.efsFileSystem,
      path: "/agent-data",
      createAcl: {
        ownerGid: "1000",
        ownerUid: "1000",
        permissions: "755",
      },
      posixUser: {
        gid: "1000",
        uid: "1000",
      },
    });

    // ── Cluster ───────────────────────────────────────────────────────────────
    const cluster = new ecs.Cluster(this, "AgentCluster", {
      vpc: props.vpc,
      containerInsightsV2: ecs.ContainerInsights.ENHANCED,
    });

    // ── Task execution role ───────────────────────────────────────────────────
    const executionRole = new iam.Role(this, "TaskExecutionRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName(
          "service-role/AmazonECSTaskExecutionRolePolicy"
        ),
      ],
    });
    apiKeySecret.grantRead(executionRole);

    // ── Task role (runtime permissions) ──────────────────────────────────────
    const taskRole = new iam.Role(this, "TaskRole", {
      assumedBy: new iam.ServicePrincipal("ecs-tasks.amazonaws.com"),
    });
    props.docsBucket.grantReadWrite(taskRole);

    // ── Log groups ────────────────────────────────────────────────────────────
    const inferenceLogGroup = new logs.LogGroup(this, "InferenceLogGroup", {
      logGroupName: "/babyyoday/inference",
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    const adminLogGroup = new logs.LogGroup(this, "AdminLogGroup", {
      logGroupName: "/babyyoday/admin",
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });

    // ── Shared task definition (both services use same image) ─────────────────
    // Inference service task
    const inferenceTaskDef = new ecs.FargateTaskDefinition(this, "InferenceTaskDef", {
      memoryLimitMiB: 16384, // 16GB — required for sentence-transformers + FAISS + llama.cpp
      cpu: 4096,             // 4 vCPU
      executionRole,
      taskRole,
      volumes: [
        {
          name: "agent-data",
          efsVolumeConfiguration: {
            fileSystemId: props.efsFileSystem.fileSystemId,
            transitEncryption: "ENABLED",
            authorizationConfig: {
              accessPointId: efsAccessPoint.accessPointId,
              iam: "ENABLED",
            },
          },
        },
      ],
    });

    const inferenceContainer = inferenceTaskDef.addContainer("InferenceContainer", {
      image: ecs.ContainerImage.fromEcrRepository(props.repository, "latest"),
      // entrypoint.sh handles: model download from S3, index seeding, then starts uvicorn
      entryPoint: ["/entrypoint.sh"],
      portMappings: [{ containerPort: 8000 }],
      environment: {
        PYTHONUNBUFFERED: "1",
      },
      secrets: {
        BABYYODAY_API_KEY: ecs.Secret.fromSecretsManager(apiKeySecret),
      },
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: "inference",
        logGroup: inferenceLogGroup,
      }),
      healthCheck: {
        command: ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(10),
        retries: 3,
        startPeriod: cdk.Duration.seconds(120), // model load time
      },
    });

    inferenceContainer.addMountPoints({
      containerPath: "/app/data",
      sourceVolume: "agent-data",
      readOnly: false,
    });

    // Admin service task
    const adminTaskDef = new ecs.FargateTaskDefinition(this, "AdminTaskDef", {
      memoryLimitMiB: 2048,  // 2GB — lightweight FastAPI, no ML models loaded
      cpu: 512,
      executionRole,
      taskRole,
      volumes: [
        {
          name: "agent-data",
          efsVolumeConfiguration: {
            fileSystemId: props.efsFileSystem.fileSystemId,
            transitEncryption: "ENABLED",
            authorizationConfig: {
              accessPointId: efsAccessPoint.accessPointId,
              iam: "ENABLED",
            },
          },
        },
      ],
    });

    const adminContainer = adminTaskDef.addContainer("AdminContainer", {
      image: ecs.ContainerImage.fromEcrRepository(props.repository, "latest"),
      // Override entrypoint to run only the admin server — not the full entrypoint.sh
      // which would also start the inference server and load the embedding model
      entryPoint: ["uvicorn", "admin.app:admin_app", "--host", "0.0.0.0", "--port", "8001"],
      portMappings: [{ containerPort: 8001 }],
      environment: {
        PYTHONUNBUFFERED: "1",
      },
      secrets: {
        BABYYODAY_API_KEY: ecs.Secret.fromSecretsManager(apiKeySecret),
      },
      logging: ecs.LogDrivers.awsLogs({
        streamPrefix: "admin",
        logGroup: adminLogGroup,
      }),
      healthCheck: {
        command: ["CMD-SHELL", "curl -f http://localhost:8001/health || exit 1"],
        interval: cdk.Duration.seconds(30),
        timeout: cdk.Duration.seconds(5),
        retries: 3,
        startPeriod: cdk.Duration.seconds(30),
      },
    });

    adminContainer.addMountPoints({
      containerPath: "/app/data",
      sourceVolume: "agent-data",
      readOnly: false,
    });

    // ── ALB ───────────────────────────────────────────────────────────────────
    this.alb = new elbv2.ApplicationLoadBalancer(this, "Alb", {
      vpc: props.vpc,
      internetFacing: true,
      securityGroup: albSg,
      vpcSubnets: { subnetType: ec2.SubnetType.PUBLIC },
    });

    const listener = this.alb.addListener("HttpListener", {
      port: 80,
      defaultAction: elbv2.ListenerAction.fixedResponse(404, {
        contentType: "text/plain",
        messageBody: "Not found",
      }),
    });

    // ── Fargate services ──────────────────────────────────────────────────────
    this.inferenceService = new ecs.FargateService(this, "InferenceService", {
      cluster,
      taskDefinition: inferenceTaskDef,
      desiredCount: 1,
      minHealthyPercent: 0,
      maxHealthyPercent: 200,
      circuitBreaker: { enable: false, rollback: false },
      securityGroups: [ecsSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      assignPublicIp: false,
      enableExecuteCommand: true,
    });
    (this.inferenceService.node.defaultChild as ecs.CfnService)
      .cfnOptions.updatePolicy = {};

    this.adminService = new ecs.FargateService(this, "AdminService", {
      cluster,
      taskDefinition: adminTaskDef,
      desiredCount: 1,
      minHealthyPercent: 0,
      maxHealthyPercent: 200,
      circuitBreaker: { enable: false, rollback: false },
      securityGroups: [ecsSg],
      vpcSubnets: { subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS },
      assignPublicIp: false,
      enableExecuteCommand: true,
    });
    (this.adminService.node.defaultChild as ecs.CfnService)
      .cfnOptions.updatePolicy = {};

    // ── Target groups + routing ───────────────────────────────────────────────
    const inferenceTargetGroup = new elbv2.ApplicationTargetGroup(this, "InferenceTg", {
      vpc: props.vpc,
      port: 8000,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      healthCheck: {
        path: "/health",
        interval: cdk.Duration.seconds(30),
        healthyHttpCodes: "200",
      },
      deregistrationDelay: cdk.Duration.seconds(30),
    });

    const adminTargetGroup = new elbv2.ApplicationTargetGroup(this, "AdminTg", {
      vpc: props.vpc,
      port: 8001,
      protocol: elbv2.ApplicationProtocol.HTTP,
      targetType: elbv2.TargetType.IP,
      healthCheck: {
        path: "/health",
        interval: cdk.Duration.seconds(30),
        healthyHttpCodes: "200",
      },
      deregistrationDelay: cdk.Duration.seconds(30),
    });

    this.inferenceService.attachToApplicationTargetGroup(inferenceTargetGroup);
    this.adminService.attachToApplicationTargetGroup(adminTargetGroup);

    // /query, /health → inference service
    listener.addAction("InferenceRouting", {
      priority: 10,
      conditions: [
        elbv2.ListenerCondition.pathPatterns(["/query", "/health", "/query/*"]),
      ],
      action: elbv2.ListenerAction.forward([inferenceTargetGroup]),
    });

    // /admin/* → admin service
    listener.addAction("AdminRouting", {
      priority: 20,
      conditions: [
        elbv2.ListenerCondition.pathPatterns(["/", "/upload", "/admin", "/admin/*"]),
      ],
      action: elbv2.ListenerAction.forward([adminTargetGroup]),
    });

    // Grant EFS access to task roles
    props.efsFileSystem.grantRootAccess(taskRole);

    new cdk.CfnOutput(this, "AlbDnsName", { value: this.alb.loadBalancerDnsName });
    new cdk.CfnOutput(this, "ApiKeySecretArn", { value: apiKeySecret.secretArn });
  }
}
