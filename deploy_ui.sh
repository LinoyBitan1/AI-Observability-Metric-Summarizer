#!/bin/bash

# Script to build, update values, and deploy UI
set -e

# Configuration
UI_IMAGE="quay.io/rh-ee-lbitan/metric-ui"
NAMESPACE="linoy-metrics-summarizer-report"
TAG=test-report-dev-128  # Use timestamp as tag

echo "🚀 Starting UI deployment process..."

# Step 1: Build UI image
echo "🏗️ Building UI image..."
cd metric-ui
docker build -t ${UI_IMAGE}:${TAG} ui/
docker push ${UI_IMAGE}:${TAG}
cd ..

# Step 2: Update values.yaml with new tag
echo "📝 Updating values.yaml..."
sed -i "s/tag:.*/tag: ${TAG}/" deploy/helm/ui/values.yaml

# Step 3: Deploy UI
echo "🚀 Deploying UI..."
cd deploy/helm
make install-metric-ui NAMESPACE=${NAMESPACE}
cd ../..

echo "✅ UI deployment completed!"
echo "📊 New image tag: ${TAG}"
echo "🔍 Check status with: kubectl get pods -n ${NAMESPACE}" 