#!/bin/bash

# Script to build, update values, and deploy MCP
set -e

# Configuration
MCP_IMAGE="quay.io/rh-ee-lbitan/metric-mcp"
NAMESPACE="linoy-metrics-summarizer-report"
TAG=test-report-dev-133  # Updated tag for wkhtmltopdf fix

echo "🚀 Starting MCP deployment process..."

# Step 1: Build MCP image
echo "🏗️ Building MCP image..."
cd metric-ui
docker build -t ${MCP_IMAGE}:${TAG} mcp/
docker push ${MCP_IMAGE}:${TAG}
cd ..

# Step 2: Update values.yaml with new tag
echo "📝 Updating values.yaml..."
sed -i "s/tag:.*/tag: ${TAG}/" deploy/helm/metric-mcp/values.yaml

# Step 3: Deploy MCP
echo "🚀 Deploying MCP..."
cd deploy/helm
make install-metric-mcp NAMESPACE=${NAMESPACE}
cd ../..

echo "✅ MCP deployment completed!"
echo "📊 New image tag: ${TAG}"
echo "🔍 Check status with: kubectl get pods -n ${NAMESPACE}" 