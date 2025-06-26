# Port forward to the Kubernetes MCP service
oc port-forward pod/metric-mcp-app-7cbd466db4-m65vc 8000:8000

# Port forward to Prometheus for metrics access
oc port-forward pod/prometheus-user-workload-0 -n openshift-user-workload-monitoring 9090:9090

# Port forward to Llama Stack service (needed for LLM summarization)
oc port-forward svc/llamastack -n linoy-metrics-summarizer-report 8321:8321

# Port forward to llama-3-2-3b-instruct predictor (needed for internal model)
oc port-forward svc/llama-3-2-3b-instruct-predictor -n linoy-metrics-summarizer-report 8080:80

#install and run ui locally
cd metric-ui/ui
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run ui.py --server.port=8501 --server.address=0.0.0.0

# Alternative: Run MCP locally instead of using Kubernetes
cd metric-ui/mcp/
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
export MODEL_CONFIG='{"meta-llama/Llama-3.2-3B-Instruct":{"external":false,"requiresApiKey":false,"serviceName":"llama-3-2-3b-instruct"},"openai/gpt-4o-mini":{"apiUrl":"https://api.openai.com/v1/chat/completions","external":true,"modelName":"gpt-4o-mini","provider":"openai","requiresApiKey":true,"serviceName":null}}'
export LLAMA_STACK_URL="http://localhost:8321/v1/openai/v1"
export PROMETHEUS_URL="http://localhost:9090"

uvicorn mcp:app --host 0.0.0.0 --port 8000 --reload