#!/bin/bash

# Exit on error
set -e

# Configuration
NAMESPACE="linoy-metrics-summarizer"
TOLERATION="g5-gpu"
MCP_IMAGE="quay.io/rh-ee-lbitan/metric-mcp:test-linoy16"
UI_IMAGE="quay.io/rh-ee-lbitan/metric-ui:test-linoy16"
HF_TOKEN=""  # Set your Hugging Face token here
UI_URL="https://ui-route-linoy-metrics-summarizer.apps.tsisodia-dev.51ty.p1.openshiftapps.com/"  # Your UI route URL

echo "🚀 Starting deployment process..."

# Clean up existing resources
echo "🧹 Cleaning up existing resources..."
cd deploy/helm/
make uninstall NAMESPACE=$NAMESPACE || true
oc delete project $NAMESPACE || true
cd ../..

# Build and push MCP image
echo "🏗️ Building MCP image..."
cd metric-ui
docker build -t $MCP_IMAGE mcp/
docker push $MCP_IMAGE

# Build and push UI image
echo "🏗️ Building UI image..."
docker build -t $UI_IMAGE ui/
docker push $UI_IMAGE
cd ..

# Deploy using Helm
echo "🚀 Deploying with Helm..."
cd deploy/helm/
HF_TOKEN=$HF_TOKEN make install NAMESPACE=$NAMESPACE TOLERATION=$TOLERATION

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
sleep 30  # Give some time for pods to start
oc get pods -n $NAMESPACE

echo "✅ Deployment completed!"
echo "📊 Check the status with: oc get pods -n $NAMESPACE"
echo "🌐 UI is available at: $UI_URL"
echo "🔍 To check the routes: oc get routes -n $NAMESPACE" 