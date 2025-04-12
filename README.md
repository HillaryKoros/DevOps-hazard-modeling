# Hazard Modeling DevOps Setup

This repository contains the configuration for running hazard modeling scripts as a containerized application on Kubernetes, managed by ArgoCD, and deployed locally using Minikube.

## Project Structure

```
.
├── 01-pet-process-1km.py           # Python processing script
├── 02-gef-chirps-process-1km.py    # Python processing script
├── 03-imerg-process-1km.py         # Python processing script
├── data                            # Data directory
│   ├── geofsm-input                # Input data
│   ├── PET                         # PET data
│   ├── WGS                         # Shapefile data
│   └── zone_wise_txt_files         # Zone text files
├── utils.py                        # Utility functions
├── Dockerfile                      # Docker image definition
├── requirements.txt                # Python dependencies
├── k8s                             # Kubernetes manifests
│   ├── deployment.yaml             # CronJob definition
│   ├── namespace.yaml              # Namespace definition
│   ├── pvc.yaml                    # Persistent Volume Claims
│   └── kustomization.yaml          # Kustomize configuration
├── argocd                          # ArgoCD configuration
│   └── application.yaml            # ArgoCD Application
└── deploy-local.sh                 # Deployment script
```

## Prerequisites

- Docker
- Minikube
- kubectl
- Git

## Setup Instructions

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Make the deployment script executable:
   ```bash
   chmod +x deploy-local.sh
   ```

3. Run the deployment script:
   ```bash
   ./deploy-local.sh
   ```
   This will:
   - Start Minikube if it's not running
   - Build the Docker image
   - Configure Kubernetes manifests
   - Apply the manifests to create necessary resources
   - Install ArgoCD if it's not already installed
   - Create the ArgoCD application
   - Trigger an initial job run

4. Access ArgoCD UI:
   The script will output the URL, username, and password for ArgoCD.

## Configuration

### CronJob Schedule

The hazard modeling job is configured to run daily at midnight. You can modify the schedule in `k8s/deployment.yaml`:

```yaml
spec:
  schedule: "0 0 * * *"  # Cron schedule (current: daily at midnight)
```

### Resource Requirements

You can adjust the CPU and memory requirements in the `k8s/deployment.yaml` file:

```yaml
resources:
  requests:
    memory: "2Gi"
    cpu: "500m"
  limits:
    memory: "4Gi"
    cpu: "1000m"
```

### Storage

The application uses two Persistent Volume Claims:
- `hazard-data-pvc`: For input data (10Gi)
- `hazard-output-pvc`: For output data (5Gi)

You can adjust the storage sizes in `k8s/pvc.yaml`.

## Manual Job Execution

To manually trigger the job:

```bash
kubectl create job --from=cronjob/hazard-modeling hazard-modeling-manual -n hazard-modeling
```

## Viewing Logs

To check logs from the most recent job:

```bash
kubectl get pods -n hazard-modeling
kubectl logs <pod-name> -n hazard-modeling
```

## Scaling to Production

For production use:

1. Use a container registry like Docker Hub, GitHub Container Registry, or a private registry
2. Set up a CI/CD pipeline to build and push images automatically
3. Configure proper secrets management for sensitive data
4. Use dedicated persistent storage solutions
5. Consider implementing monitoring and alerting
6. Set up proper backup and disaster recovery procedures
## License

This project is licensed under the MIT License - see the LICENSE file for details.
