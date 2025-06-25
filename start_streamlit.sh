#!/bin/bash

# Simple script to start Streamlit UI without onboarding prompts
cd metric-ui/ui

# Set environment variables
export PROMETHEUS_URL="http://localhost:9090"
export LLAMA_STACK_URL="http://localhost:8321/v1/openai/v1"
export API_URL="http://localhost:8000"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Start Streamlit with configuration to skip onboarding
echo "Starting Streamlit UI..."
streamlit run ui.py --server.port=8501 --server.address=0.0.0.0 --global.developmentMode=false 