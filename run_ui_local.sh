#!/bin/bash
# Local Development Setup Script for AI-Observability-Metric-Summarizer
# This script sets up port forwarding and runs local UI/MCP services
set -e
echo ":rocket: Setting up local development environment..."
# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color
# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}
print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}
print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}
# Check if oc is available
if ! command -v oc &> /dev/null; then
    print_error "OpenShift CLI (oc) is not installed or not in PATH"
    exit 1
fi
# Check if we're logged in to OpenShift
if ! oc whoami &> /dev/null; then
    print_error "Not logged in to OpenShift. Please run 'oc login' first"
    exit 1
fi
# Variables
NAMESPACE="linoy-metrics-summarizer-report"
MCP_POD="metric-mcp-app-858f45fb69-mhk2g"
LLAMA_POD="llamastack-7965b84c86-kscps"
THANOS_NAMESPACE="openshift-monitoring"
THANOS_SERVICE="thanos-querier"
# Kill existing port-forward processes
print_status "Killing existing port-forward processes..."
pkill -f "oc port-forward" || true
# Start port forwarding in background
print_status "Starting port forwarding..."
# MCP Service (Backend API)
print_status "Port forwarding MCP service (8000:8000)..."
oc port-forward pod/$MCP_POD 8000:8000 -n $NAMESPACE &
MCP_PF_PID=$!
# LlamaStack Service (LLM)
print_status "Port forwarding LlamaStack service (8321:8321)..."
oc port-forward pod/$LLAMA_POD 8321:8321 -n $NAMESPACE &
LLAMA_PF_PID=$!
# Thanos Querier (Prometheus access)
print_status "Port forwarding Thanos Querier service (9090:9091)..."
oc port-forward svc/$THANOS_SERVICE 9090:9091 -n $THANOS_NAMESPACE &
THANOS_PF_PID=$!
# Wait a moment for port forwarding to establish
sleep 3
# Check if port forwarding is working
print_status "Checking port forwarding status..."
if curl -s http://localhost:8000/health &> /dev/null; then
    print_success "MCP service is accessible on localhost:8000"
else
    print_warning "MCP service not yet accessible, may need more time..."
fi
# Set up local UI environment
print_status "Setting up local UI environment..."
# Navigate to UI directory
cd metric-ui/ui
# Check if virtual environment exists and recreate if there are issues
if [ ! -d ".venv" ] || [ ! -f ".venv/bin/python" ]; then
    print_status "Creating virtual environment..."
    rm -rf .venv
    python3 -m venv .venv
fi
# Activate virtual environment
print_status "Activating virtual environment..."
source .venv/bin/activate
# Install dependencies
print_status "Installing UI dependencies..."
pip install -r requirements.txt
# Set environment variables for local development
export PROMETHEUS_URL="http://localhost:9090"
export LLAMA_STACK_URL="http://localhost:8321/v1/openai/v1"
export API_URL="http://localhost:8000"
print_success "Environment variables set:"
echo "  PROMETHEUS_URL: $PROMETHEUS_URL"
echo "  LLAMA_STACK_URL: $LLAMA_STACK_URL"
echo "  API_URL: $API_URL"
# Function to cleanup on exit
cleanup() {
    print_status "Cleaning up..."
    kill $MCP_PF_PID 2>/dev/null || true
    kill $LLAMA_PF_PID 2>/dev/null || true
    kill $THANOS_PF_PID 2>/dev/null || true
    pkill -f "oc port-forward" || true
    print_success "Cleanup completed"
}
# Set trap to cleanup on script exit
trap cleanup EXIT
print_success "Local development environment is ready!"
echo ""
echo ":bar_chart: Services available:"
echo "  - MCP Backend API: http://localhost:8000"
echo "  - LlamaStack LLM: http://localhost:8321"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo ":globe_with_meridians: Starting Streamlit UI..."
echo "  - UI will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"
# Start Streamlit UI
streamlit run ui.py --server.port=8501 --server.address=0.0.0.0
