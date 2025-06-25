oc port-forward pod/metric-mcp-app-7cbd466db4-m65vc 8000:8000

#install and run ui locally
cd metric-ui/ui
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
streamlit run ui.py --server.port=8501 --server.address=0.0.0.0