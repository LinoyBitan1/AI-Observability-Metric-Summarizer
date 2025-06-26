#!/bin/bash

# Exit on error
set -e

# Configuration
NAMESPACE="linoy-metrics-summarizer-report"
TOLERATION="g5-gpu" 
MCP_IMAGE="quay.io/rh-ee-lbitan/metric-mcp:test-report-dev-114"
UI_IMAGE="quay.io/rh-ee-lbitan/metric-ui:test-report-dev-114"  
# HF_TOKEN should be set as environment variable: export HF_TOKEN="your_token_here"
UI_URL="https://ui-route-linoy-metrics-summarizer.apps.tsisodia-dev.51ty.p1.openshiftapps.com/"  # Your UI route URL

echo "🚀 Starting deployment process..."

# Check if HF_TOKEN is set
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable is not set"
    echo "Please set it with: export HF_TOKEN='your_huggingface_token_here'"
    exit 1
fi

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
HF_TOKEN=$HF_TOKEN make install NAMESPACE=$NAMESPACE 

# Wait for deployment
echo "⏳ Waiting for deployment to be ready..."
sleep 30  # Give some time for pods to start
oc get pods -n $NAMESPACE

echo "✅ Deployment completed!"
echo "📊 Check the status with: oc get pods -n $NAMESPACE"
echo "🌐 UI is available at: $UI_URL"
echo "🔍 To check the routes: oc get routes -n $NAMESPACE" 