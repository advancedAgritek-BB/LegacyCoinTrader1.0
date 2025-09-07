#!/usr/bin/env python3
"""
CI/CD Setup Validation Script

This script validates that all required components for the CI/CD pipelines are properly configured.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_file_exists(path, description):
    """Check if a file exists."""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: {path} - NOT FOUND")
        return False

def check_github_secrets():
    """Check if GitHub secrets are configured (simulated)."""
    print("\n🔐 GitHub Secrets Check:")
    required_secrets = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "DOCKER_USERNAME",
        "DOCKER_PASSWORD",
        "SLACK_WEBHOOK_URL"
    ]

    print("Note: GitHub secrets cannot be validated from local environment.")
    print("Please ensure these secrets are set in your GitHub repository:")
    for secret in required_secrets:
        print(f"  - {secret}")

def check_aws_resources():
    """Check AWS resource configuration."""
    print("\n☁️  AWS Resources Check:")

    # Check if AWS CLI is available
    try:
        result = subprocess.run(['aws', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ AWS CLI is available")
        else:
            print("❌ AWS CLI not found")
            return
    except FileNotFoundError:
        print("❌ AWS CLI not found")
        return

    # Check AWS credentials
    try:
        result = subprocess.run(['aws', 'sts', 'get-caller-identity'], capture_output=True, text=True)
        if result.returncode == 0:
            identity = json.loads(result.stdout)
            print(f"✅ AWS credentials configured for account: {identity['Account']}")
        else:
            print("❌ AWS credentials not configured or invalid")
    except Exception as e:
        print(f"❌ AWS credentials check failed: {e}")

def check_docker_configuration():
    """Check Docker configuration."""
    print("\n🐳 Docker Configuration Check:")

    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Docker is available")
        else:
            print("❌ Docker not found")
    except FileNotFoundError:
        print("❌ Docker not found")

def check_task_definition():
    """Check task definition file."""
    print("\n📋 Task Definition Check:")

    task_def_path = ".github/task-definition.json"
    if check_file_exists(task_def_path, "Task definition file"):
        try:
            with open(task_def_path, 'r') as f:
                task_def = json.load(f)

            # Check required fields
            required_fields = ['family', 'containerDefinitions']
            for field in required_fields:
                if field in task_def:
                    print(f"✅ Task definition has {field}")
                else:
                    print(f"❌ Task definition missing {field}")

            # Check container definitions
            containers = task_def.get('containerDefinitions', [])
            if len(containers) > 0:
                print(f"✅ Task definition has {len(containers)} container(s)")
                for i, container in enumerate(containers):
                    if 'name' in container and 'image' in container:
                        print(f"  ✅ Container {i+1}: {container['name']}")
                    else:
                        print(f"  ❌ Container {i+1}: missing name or image")
            else:
                print("❌ Task definition has no containers")

        except json.JSONDecodeError:
            print("❌ Task definition file is not valid JSON")
        except Exception as e:
            print(f"❌ Error reading task definition: {e}")

def check_workflow_files():
    """Check workflow files."""
    print("\n🔄 Workflow Files Check:")

    workflow_files = [
        ".github/workflows/ci.yml",
        ".github/workflows/ci-cd.yml"
    ]

    for wf_file in workflow_files:
        check_file_exists(wf_file, f"Workflow file")

def check_docker_compose_files():
    """Check Docker Compose files."""
    print("\n🐙 Docker Compose Files Check:")

    compose_files = [
        "docker-compose.yml",
        "docker-compose.dev.yml",
        "docker-compose.prod.yml",
        "docker-compose.test.yml"
    ]

    for compose_file in compose_files:
        check_file_exists(compose_file, f"Docker Compose file")

def main():
    """Main validation function."""
    print("🚀 LegacyCoinTrader CI/CD Setup Validation")
    print("=" * 50)

    # File existence checks
    check_workflow_files()
    check_task_definition()
    check_docker_compose_files()

    # Configuration checks
    check_github_secrets()
    check_aws_resources()
    check_docker_configuration()

    print("\n" + "=" * 50)
    print("📖 For detailed setup instructions, see: .github/CI_CD_SETUP.md")
    print("🔧 To fix any missing components, follow the setup guide.")

if __name__ == "__main__":
    main()
