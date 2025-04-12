#!/bin/bash
# Script to add files and commit with a progress message

# Create the checklist file
cat > SETUP-CHECKLIST.md <<'EOL'
# CI/CD Setup Progress Checklist

This checklist tracks our progress in setting up the CI/CD pipeline for the hazard modeling project. Use this to keep track of completed steps and what still needs to be done.

## Repository Setup

- [x] Create argocd branch
- [x] Add Dockerfile for containerization
- [x] Add Kubernetes manifests
- [x] Create GitHub Actions workflow files
- [x] Push changes to repositories

## Docker Hub Configuration

- [ ] Create Docker Hub account (if not already done)
- [ ] Create access token for GitHub Actions
- [ ] Add token to GitHub Secrets (DOCKERHUB_USERNAME, DOCKERHUB_TOKEN)

## GitHub Secrets Configuration

- [ ] Add DOCKERHUB_USERNAME secret
- [ ] Add DOCKERHUB_TOKEN secret
- [ ] Add ARGOCD_SERVER secret
- [ ] Add ARGOCD_USERNAME secret
- [ ] Add ARGOCD_PASSWORD secret

## Kubernetes Configuration

- [ ] Ensure Minikube/Kubernetes cluster is running
- [ ] Create hazard-modeling namespace
- [ ] Create Docker registry secret in Kubernetes
- [ ] Update secrets.yaml with actual values
- [ ] Apply secrets.yaml to the cluster

## ArgoCD Configuration

- [ ] Verify ArgoCD is installed and running
- [ ] Update application-prod.yaml with actual repository URL
- [ ] Apply ArgoCD application configuration

## Testing the Pipeline

- [ ] Make a test commit to trigger GitHub Actions
- [ ] Verify Docker image is built and pushed to Docker Hub
- [ ] Verify ArgoCD syncs the application
- [ ] Check if the application pods are running correctly

## Production Considerations (Future Work)

- [ ] Set up persistent storage for production
- [ ] Configure monitoring and alerting
- [ ] Implement backup and disaster recovery procedures
- [ ] Review and enhance security measures

## Notes

* Currently, we have created the framework and templates for CI/CD but haven't fully configured secrets and credentials
* Some steps require a running Kubernetes cluster, which is currently experiencing connection issues
* The GitHub Actions workflows will not work until all required secrets are configured
EOL

# Add all files
git add .

# Commit with progress message
git commit -m "Add CI/CD pipeline framework (completed repository setup, pending configurations)

- Created GitHub Actions workflows for Docker build/push and ArgoCD sync
- Added Kubernetes manifests for secrets and deployment
- Added ArgoCD configuration template
- Added detailed setup documentation
- Added setup progress checklist

Note: Secret configuration and credentials setup still pending. Minikube cluster currently experiencing connection issues."

echo "Commit created. Ready to push to repositories."
echo "Use: git push personal argocd"
echo "And: git push org argocd"
