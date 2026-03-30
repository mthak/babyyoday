import * as cdk from "aws-cdk-lib";
import * as codebuild from "aws-cdk-lib/aws-codebuild";
import * as codepipeline from "aws-cdk-lib/aws-codepipeline";
import * as codepipeline_actions from "aws-cdk-lib/aws-codepipeline-actions";
import * as ecr from "aws-cdk-lib/aws-ecr";
import * as ecs from "aws-cdk-lib/aws-ecs";
import * as iam from "aws-cdk-lib/aws-iam";
import * as logs from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

interface PipelineStackProps extends cdk.StackProps {
  repository: ecr.Repository;
  ecsService: ecs.FargateService;
  adminService: ecs.FargateService;
}

export class PipelineStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: PipelineStackProps) {
    super(scope, id, props);

    // ── CodeBuild project ─────────────────────────────────────────────────────
    const buildProject = new codebuild.PipelineProject(this, "DockerBuild", {
      projectName: "babyyoday-docker-build",
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        privileged: true,
        computeType: codebuild.ComputeType.LARGE,
      },
      environmentVariables: {
        REPOSITORY_URI: { value: props.repository.repositoryUri },
        AWS_DEFAULT_REGION: { value: this.region },
        AWS_ACCOUNT_ID: { value: this.account },
      },
      buildSpec: codebuild.BuildSpec.fromObject({
        version: "0.2",
        phases: {
          pre_build: {
            commands: [
              "echo Logging in to Amazon ECR...",
              "aws ecr get-login-password --region $AWS_DEFAULT_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_DEFAULT_REGION.amazonaws.com",
              "IMAGE_TAG=$(echo $CODEBUILD_RESOLVED_SOURCE_VERSION | cut -c1-7)",
            ],
          },
          build: {
            commands: [
              "echo Building Docker image...",
              "docker build -t $REPOSITORY_URI:$IMAGE_TAG -t $REPOSITORY_URI:latest -f builder/Dockerfile .",
            ],
          },
          post_build: {
            commands: [
              "echo Pushing image to ECR...",
              "docker push $REPOSITORY_URI:$IMAGE_TAG",
              "docker push $REPOSITORY_URI:latest",
              // Single artifact with both files — each EcsDeployAction uses imageFile
              // to pick only the file relevant to its service
              `printf '[{"name":"InferenceContainer","imageUri":"%s:%s"}]' $REPOSITORY_URI $IMAGE_TAG > imagedefinitions-inference.json`,
              `printf '[{"name":"AdminContainer","imageUri":"%s:%s"}]' $REPOSITORY_URI $IMAGE_TAG > imagedefinitions-admin.json`,
            ],
          },
        },
        artifacts: {
          files: ["imagedefinitions-inference.json", "imagedefinitions-admin.json"],
        },
      }),
      logging: {
        cloudWatch: {
          logGroup: new logs.LogGroup(this, "BuildLogGroup", {
            logGroupName: "/babyyoday/codebuild",
            retention: logs.RetentionDays.ONE_WEEK,
            removalPolicy: cdk.RemovalPolicy.DESTROY,
          }),
        },
      },
    });

    props.repository.grantPullPush(buildProject);
    buildProject.addToRolePolicy(
      new iam.PolicyStatement({
        actions: ["ecr:GetAuthorizationToken"],
        resources: ["*"],
      })
    );

    // ── Pipeline ──────────────────────────────────────────────────────────────
    const sourceOutput = new codepipeline.Artifact("SourceOutput");
    const buildOutput = new codepipeline.Artifact("BuildOutput");

    const pipeline = new codepipeline.Pipeline(this, "Pipeline", {
      pipelineName: "babyyoday-deploy",
      restartExecutionOnUpdate: true,
    });

    pipeline.addStage({
      stageName: "Source",
      actions: [
        new codepipeline_actions.CodeStarConnectionsSourceAction({
          actionName: "GitHub_Source",
          owner: "ashwiniverma",
          repo: "babyyoday",
          branch: "main",
          connectionArn: `arn:aws:codeconnections:${this.region}:${this.account}:connection/6794ae1c-1712-4498-8f09-6afc03fb6ad6`,
          output: sourceOutput,
        }),
      ],
    });

    pipeline.addStage({
      stageName: "Build",
      actions: [
        new codepipeline_actions.CodeBuildAction({
          actionName: "Docker_Build_Push",
          project: buildProject,
          input: sourceOutput,
          outputs: [buildOutput],
        }),
      ],
    });

    // Each EcsDeployAction uses imageFile to pick only its own container definition
    pipeline.addStage({
      stageName: "Deploy",
      actions: [
        new codepipeline_actions.EcsDeployAction({
          actionName: "Deploy_Inference",
          service: props.ecsService,
          imageFile: buildOutput.atPath("imagedefinitions-inference.json"),
          deploymentTimeout: cdk.Duration.minutes(20),
        }),
        new codepipeline_actions.EcsDeployAction({
          actionName: "Deploy_Admin",
          service: props.adminService,
          imageFile: buildOutput.atPath("imagedefinitions-admin.json"),
          deploymentTimeout: cdk.Duration.minutes(20),
        }),
      ],
    });

    new cdk.CfnOutput(this, "PipelineName", { value: pipeline.pipelineName });
    new cdk.CfnOutput(this, "PipelineConsoleUrl", {
      value: `https://${this.region}.console.aws.amazon.com/codesuite/codepipeline/pipelines/${pipeline.pipelineName}/view`,
    });
  }
}
