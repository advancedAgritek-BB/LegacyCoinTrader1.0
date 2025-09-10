# CI/CD Configuration Guide

This document outlines the required secrets and configuration for the LegacyCoinTrader CI/CD pipelines.

## Required GitHub Secrets

### AWS Configuration
```
AWS_ACCESS_KEY_ID          # Your AWS access key ID
AWS_SECRET_ACCESS_KEY      # Your AWS secret access key
```

### Docker Hub Configuration
```
DOCKER_USERNAME           # Your Docker Hub username
DOCKER_PASSWORD           # Your Docker Hub password/token
```

### Slack Notifications
```
SLACK_WEBHOOK_URL         # Slack webhook URL for notifications
```

### AWS Secrets Manager (for production)
The following secrets should be stored in AWS Secrets Manager:

#### Database Configuration
```
legacy-coin-trader/db-url
{
  "DATABASE_URL": "postgresql://username:password@host:port/database"
}
```

#### Redis Configuration
```
legacy-coin-trader/redis-url
{
  "REDIS_URL": "redis://host:port/db"
}
```

#### Kraken API Configuration
```
legacy-coin-trader/kraken-api-key
{
  "KRAKEN_API_KEY": "your-kraken-api-key"
}

legacy-coin-trader/kraken-api-secret
{
  "KRAKEN_API_SECRET": "your-kraken-api-secret"
}
```

## AWS Resources Required

### ECR Repositories
- `legacy-coin-trader/api-gateway`
- `legacy-coin-trader/trading-engine`

### ECS Clusters
- `legacy-coin-trader-staging` (for develop branch)
- `legacy-coin-trader-prod` (for main branch)

### ECS Services
- `legacy-coin-trader-service` (in both clusters)

### IAM Roles
- `legacy-coin-trader-task-role` (ECS task execution role)
- `legacy-coin-trader-execution-role` (ECS task role)

### CloudWatch Log Groups
- `/ecs/legacy-coin-trader` (for container logs)

## Required AWS Permissions

The AWS credentials should have the following permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "ecr:GetAuthorizationToken",
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload",
        "ecr:PutImage"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "ecs:RegisterTaskDefinition",
        "ecs:UpdateService",
        "ecs:DescribeServices",
        "ecs:DescribeTasks",
        "ecs:ListTasks"
      ],
      "Resource": "*"
    },
    {
      "Effect": "Allow",
      "Action": [
        "iam:PassRole"
      ],
      "Resource": [
        "arn:aws:iam::ACCOUNT-ID:role/legacy-coin-trader-task-role",
        "arn:aws:iam::ACCOUNT-ID:role/legacy-coin-trader-execution-role"
      ]
    },
    {
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity"
      ],
      "Resource": "*"
    }
  ]
}
```

## Environment Setup

### Staging Environment
- Domain: `staging.api.legacy-coin-trader.com`
- Health check endpoint: `/health`

### Production Environment
- Domain: `api.legacy-coin-trader.com`
- Health check endpoint: `/health`
- Status endpoint: `/api/trading/status`

## Deployment Flow

### Develop Branch (Staging)
1. Code quality checks
2. Unit tests
3. Integration tests
4. Docker build and ECR push
5. ECS deployment to staging
6. Health checks
7. Slack notification

### Main Branch (Production)
1. All staging steps
2. Production ECR push
3. ECS deployment to production
4. Extended health checks
5. Slack notification

## Monitoring

### CI/CD Metrics
- Pipeline success/failure rates
- Deployment duration
- Test coverage trends

### Application Metrics
- Health check status
- Error rates
- Performance metrics

## Troubleshooting

### Common Issues

1. **AWS Credentials Error**
   - Verify secrets are set correctly
   - Check AWS permissions
   - Ensure region is correct

2. **Docker Build Failures**
   - Check Docker Hub credentials
   - Verify Dockerfiles exist
   - Check build context paths

3. **ECS Deployment Failures**
   - Verify ECS cluster and service exist
   - Check task definition syntax
   - Ensure IAM roles are correct

4. **Slack Notifications Not Working**
   - Verify webhook URL is correct
   - Check Slack app permissions
   - Ensure webhook is active

### Debug Steps

1. Check GitHub Actions logs for detailed error messages
2. Verify all required secrets are set
3. Test AWS credentials locally
4. Check AWS resource existence and permissions
5. Validate Docker configurations
