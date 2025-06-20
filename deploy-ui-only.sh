#!/bin/bash

# Exit on error
set -e

# Configuration
NAMESPACE="linoy-metrics-summarizer"
UI_IMAGE="quay.io/rh-ee-lbitan/metric-ui:test-linoy-multi-models8"

echo "🚀 Starting UI-only deployment process..."

# Build and push UI image
echo "🏗️ Building UI image..."
cd metric-ui
docker build -t $UI_IMAGE ui/
docker push $UI_IMAGE   
cd ..

# Deploy only UI using Helm
echo "🚀 Deploying UI with Helm..."
cd deploy/helm/
helm upgrade --install ui ./ui \
  --namespace $NAMESPACE \
  --create-namespace \
  --set image.tag=test-linoy-multi-models8

# Wait for deployment
echo "⏳ Waiting for UI deployment to be ready..."
sleep 10
oc get pods -n $NAMESPACE -l app=ui

echo "✅ UI deployment completed!"
echo "📊 Check the status with: oc get pods -n $NAMESPACE -l app=ui"
echo "🌐 UI is available at: https://ui-route-linoy-metrics-summarizer.apps.tsisodia-dev.51ty.p1.openshiftapps.com/"
echo "🔍 To check the routes: oc get routes -n $NAMESPACE" 