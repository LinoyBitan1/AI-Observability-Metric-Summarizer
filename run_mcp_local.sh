#!/bin/bash

# Local development script for MCP service only
set -e

# Configuration
NAMESPACE="linoy-metrics-summarizer-report"
LOCAL_PORT=8000
PROMETHEUS_PORT=9090
LLAMA_STACK_PORT=8321

echo "🚀 Starting MCP local development environment..."

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed. Please install kubectl first."
    exit 1
fi

# Check if we're connected to the cluster
if ! kubectl cluster-info &> /dev/null; then
    echo "❌ Not connected to Kubernetes cluster. Please connect first."
    exit 1
fi

# Check if namespace exists
if ! kubectl get namespace $NAMESPACE &> /dev/null; then
    echo "❌ Namespace $NAMESPACE does not exist. Please deploy the services first."
    exit 1
fi

echo "📡 Setting up port forwarding..."

# Kill existing port-forward processes
echo "🧹 Cleaning up existing port forwarding..."
pkill -f "kubectl port-forward" || true

# Start port forwarding in background
echo "🔗 Forwarding Prometheus (localhost:$PROMETHEUS_PORT -> cluster:9090)..."
kubectl port-forward -n $NAMESPACE svc/prometheus-operated $PROMETHEUS_PORT:9090 &
PROMETHEUS_PID=$!

echo "🔗 Forwarding Llama Stack (localhost:$LLAMA_STACK_PORT -> cluster:8321)..."
kubectl port-forward -n $NAMESPACE svc/llm-service $LLAMA_STACK_PORT:8321 &
LLAMA_PID=$!

# Function to cleanup background processes
cleanup() {
    echo "🧹 Cleaning up port forwarding..."
    kill $PROMETHEUS_PID 2>/dev/null || true
    kill $LLAMA_PID 2>/dev/null || true
    pkill -f "kubectl port-forward" || true
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Wait a moment for port forwarding to establish
echo "⏳ Waiting for port forwarding to establish..."
sleep 3

# Check if ports are accessible
if ! curl -s http://localhost:$PROMETHEUS_PORT/api/v1/status/config > /dev/null; then
    echo "❌ Prometheus is not accessible on localhost:$PROMETHEUS_PORT"
    exit 1
fi

echo "✅ Prometheus is accessible on localhost:$PROMETHEUS_PORT"

# Get the token for authentication
echo "🔐 Getting service account token..."
TOKEN=$(kubectl get secret -n $NAMESPACE $(kubectl get serviceaccount -n $NAMESPACE -o jsonpath='{.items[0].metadata.name}' | head -1) -o jsonpath='{.data.token}' | base64 -d)

# Set environment variables
export PROMETHEUS_URL="http://localhost:$PROMETHEUS_PORT"
export LLAMA_STACK_URL="http://localhost:$LLAMA_STACK_PORT/v1/openai/v1"
export THANOS_TOKEN="$TOKEN"
export LLM_API_TOKEN=""

# Load model configuration from the cluster
echo "📋 Loading model configuration from cluster..."
MODEL_CONFIG=$(kubectl get configmap -n $NAMESPACE metric-mcp-config -o jsonpath='{.data.model_config}' 2>/dev/null || echo '{}')
export MODEL_CONFIG="$MODEL_CONFIG"

echo "🔧 Environment variables set:"
echo "   PROMETHEUS_URL: $PROMETHEUS_URL"
echo "   LLAMA_STACK_URL: $LLAMA_STACK_URL"
echo "   THANOS_TOKEN: [REDACTED]"
echo "   MODEL_CONFIG: $MODEL_CONFIG"

# Install Python dependencies if needed
echo "📦 Checking Python dependencies..."
cd metric-ui/mcp
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
fi

echo "🔧 Activating virtual environment..."
source venv/bin/activate

echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "🚀 Starting MCP service on localhost:$LOCAL_PORT..."
echo "📊 API will be available at: http://localhost:$LOCAL_PORT"
echo "📖 API docs will be available at: http://localhost:$LOCAL_PORT/docs"
echo ""
echo "Press Ctrl+C to stop the service and cleanup port forwarding"

# Start the FastAPI server
uvicorn mcp:app --host 0.0.0.0 --port $LOCAL_PORT --reload 