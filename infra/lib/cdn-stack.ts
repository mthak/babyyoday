import * as cdk from "aws-cdk-lib";
import * as cloudfront from "aws-cdk-lib/aws-cloudfront";
import * as origins from "aws-cdk-lib/aws-cloudfront-origins";
import * as elbv2 from "aws-cdk-lib/aws-elasticloadbalancingv2";
import { Construct } from "constructs";

interface CdnStackProps extends cdk.StackProps {
  alb: elbv2.ApplicationLoadBalancer;
}

export class CdnStack extends cdk.Stack {
  readonly distribution: cloudfront.Distribution;

  constructor(scope: Construct, id: string, props: CdnStackProps) {
    super(scope, id, props);

    const albOrigin = new origins.LoadBalancerV2Origin(props.alb, {
      protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
      httpPort: 80,
    });

    this.distribution = new cloudfront.Distribution(this, "Distribution", {
      defaultBehavior: {
        origin: albOrigin,
        viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
        allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
        cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED, // API — never cache
        originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
      },
      additionalBehaviors: {
        // Admin panel — no cache, pass all headers
        "/admin/*": {
          origin: albOrigin,
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
        },
        // Health endpoint — short cache
        "/health": {
          origin: albOrigin,
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.REDIRECT_TO_HTTPS,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_GET_HEAD,
          cachePolicy: new cloudfront.CachePolicy(this, "HealthCachePolicy", {
            defaultTtl: cdk.Duration.seconds(10),
            maxTtl: cdk.Duration.seconds(30),
            minTtl: cdk.Duration.seconds(0),
          }),
        },
      },
      enableLogging: true,
      minimumProtocolVersion: cloudfront.SecurityPolicyProtocol.TLS_V1_2_2021,
      comment: "BabyYoday agent CDN",
    });

    new cdk.CfnOutput(this, "DistributionDomainName", {
      value: this.distribution.distributionDomainName,
      description: "CloudFront domain — use this as your agent endpoint",
    });

    new cdk.CfnOutput(this, "DistributionId", {
      value: this.distribution.distributionId,
    });
  }
}
