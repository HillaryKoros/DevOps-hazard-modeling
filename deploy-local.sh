#!/bin/bash
set -e

# Define variables
IMAGE_REPOSITORY="localhost:5000"
IMAGE_TAG=$(date +%Y%m%d-%H%M%S)

# Check if Minikube is running
if ! minikube status > /dev/null 2>&1; then
  echo "Starting Minikube..."
  minikube start --memory=4096 --cpus=2
fi

# Set docker to use Minikube's Docker daemon
eval $(minikube docker-env)

# Build the Docker image
echo "Building Docker image..."
docker build -t ${IMAGE_REPOSITORY}/hazard-modeling:${IMAGE_TAG} .

# Replace placeholders in Kubernetes manifests
echo "Configuring Kubernetes manifests..."
for file in k8s/*.yaml; do
  sed -i "s|\${IMAGE_REPOSITORY}|${IMAGE_REPOSITORY}|g" $file
  sed -i "s|\${IMAGE_TAG}|${IMAGE_TAG}|g" $file
done

# Create namespace if it doesn't exist
kubectl create namespace hazard-modeling --dry-run=client -o yaml | kubectl apply -f -

# Apply Kubernetes manifests
echo "Applying Kubernetes manifests..."
kubectl apply -k k8s/

# Check ArgoCD installation
if ! kubectl get namespace argocd > /dev/null 2>&1; then
  echo "ArgoCD namespace not found. Installing ArgoCD..."
  kubectl create namespace argocd
  kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml
  
  # Wait for ArgoCD to be ready
  echo "Waiting for ArgoCD to be ready..."
  kubectl wait --for=condition=available --timeout=300s deployment/argocd-server -n argocd
fi

# Apply ArgoCD application
echo "Creating ArgoCD application..."
kubectl apply -f argocd/application.yaml

# Get ArgoCD password
ARGOCD_PASSWORD=$(kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)
echo "ArgoCD is available at http://$(minikube ip):$(kubectl get svc argocd-server -n argocd -o jsonpath='{.spec.ports[0].nodePort}')"
echo "Username: admin"
echo "Password: $ARGOCD_PASSWORD"

# Trigger the job manually for initial run
echo "Triggering initial job run..."
kubectl create job --from=cronjob/hazard-modeling hazard-modeling-initial -n hazard-modeling

echo "Deployment completed successfully!"