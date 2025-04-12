# Production Deployment Guide for Hazard Modeling

This guide outlines the steps for setting up a production-ready deployment of the hazard modeling system using Docker Hub, GitHub Actions for CI/CD, and ArgoCD for continuous deployment.

## Prerequisites

- GitHub account with access to the repository
- Docker Hub account
- Kubernetes cluster with ArgoCD installed
- `kubectl` configured to access your cluster

## Setup Steps

### 1. Configure Docker Hub

1. Create a Docker Hub access token:
   - Go to [Docker Hub](https://hub.docker.com/settings/security)
   - Click "New Access Token"
   - Give it a name (e.g., "github-actions")
   - Copy the token (you won't see it again)

### 2. Configure GitHub Secrets

Add the following secrets to your GitHub repository:

1. Go to your repository on GitHub
2. Navigate to Settings > Secrets and variables > Actions
3. Add the following repository secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: The access token you created
   - `ARGOCD_SERVER`: Your ArgoCD server URL (e.g., `argocd.example.com`)
   - `ARGOCD_USERNAME`: Your ArgoCD username (usually `admin`)
   - `ARGOCD_PASSWORD`: Your ArgoCD password

### 3. Set Up Docker Registry Credentials in Kubernetes

Create a secret for pulling images from Docker Hub:

```bash
kubectl create secret docker-registry docker-registry-credentials \
  --namespace hazard-modeling \
  --docker-server=https://index.docker.io/v1/ \
  --docker-username=<your-dockerhub-username> \
  --docker-password=<your-dockerhub-token> \
  --docker-email=<your-email>
```

### 4. Configure Application Secrets

1. Edit the `k8s/secrets.yaml` file to include your actual secret values
2. Apply the secrets to your cluster:
   ```bash
   kubectl apply -f k8s/secrets.yaml
   ```

### 5. Update ArgoCD Application Configuration

1. Edit `argocd/application-prod.yaml` to set your:
   - GitHub repository URL
   - Docker Hub username
   - Other environment-specific settings

2. Apply the ArgoCD application:
   ```bash
   kubectl apply -f argocd/application-prod.yaml
   ```

### 6. Test the CI/CD Pipeline

1. Make a change to your repository
2. Push to the `argocd` branch
3. GitHub Actions will:
   - Build and push the Docker image to Docker Hub
   - Trigger ArgoCD to sync the application

4. Monitor the workflow in the GitHub Actions tab
5. Verify deployment in ArgoCD UI

## Production Considerations

### Storage

For production, consider:
- Using cloud-based storage solutions (AWS EBS, Azure Disk, GCP Persistent Disk)
- Implementing backup solutions for your persistent volumes

```yaml
# Example for AWS EBS
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: hazard-data-pvc
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: gp2  # AWS EBS storage class
  resources:
    requests:
      storage: 100Gi
```

### Monitoring and Alerting

1. Install Prometheus and Grafana:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm install prometheus prometheus-community/kube-prometheus-stack \
     --namespace monitoring --create-namespace
   ```

2. Create alerts for your application:
   - Job failures
   - Resource utilization
   - Storage capacity

### Backup and Disaster Recovery

1. Set up regular backups of:
   - Persistent volumes
   - Application configuration
   - Kubernetes resources

2. Document recovery procedures:
   - How to restore from backups
   - How to rebuild the environment from scratch

### Scaling

Consider horizontal and vertical scaling options:

- Increase resources for the jobs:
  ```yaml
  resources:
    requests:
      memory: "4Gi"
      cpu: "1000m"
    limits:
      memory: "8Gi"
      cpu: "2000m"
  ```

- Use node selectors or affinities to place workloads on appropriate nodes:
  ```yaml
  nodeSelector:
    compute-type: high-memory
  ```

### Security

1. Regularly update base images and dependencies
2. Implement network policies to restrict communication
3. Use Kubernetes RBAC to limit access
4. Consider running security scans on your Docker images

## Troubleshooting

### Common Issues

1. **Image pull errors**:
   - Verify Docker Hub credentials
   - Check image name and tag in the deployment

2. **Job failures**:
   - Check logs: `kubectl logs job/hazard-modeling-<id> -n hazard-modeling`
   - Verify resource limits are sufficient

3. **ArgoCD sync failures**:
   - Check ArgoCD logs
   - Verify GitHub repository access
   - Ensure Kubernetes manifests are valid