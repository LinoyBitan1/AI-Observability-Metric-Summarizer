# Ingestion Pipeline Helm Chart

This Helm chart sets up an end-to-end RAG ingestion pipeline using OpenShift AI, MinIO, LlamaStack, and PGVector.

---

## Quick Start

### 1. Prerequisites

- OpenShift cluster with OpenShift AI installed
- `helm` and `oc` CLI configured
- Access to a namespace (default: `llama-stack-rag-2`)
- MinIO, PGVector, and LlamaStack components available

### Customization(Very Important Step before jumping to deployment)

Edit `values.yaml` to configure:

- Namespace
- MinIO settings
- LlamaStack base URL

### 2. Deploy the Pipeline

Clone the repository and run:

```bash
make install-ingestion-pipeline NAMESPACE = llama-stack-rag-2
```

This will:

- Set up secrets
- Deploy all components using Helm charts

---

## Run the Ingestion Pipeline

1. **Open OpenShift AI Workbench**  
   Navigate to **Workbench** and open the notebook named `rag-pipeline-notebook`.

2. **Run the Notebook Script**  
   Inside the notebook, execute the Python script. This will:
   - Create Pipeline and PipelineRun

3. **View Pipeline Runs**  
   Go to the **Pipelines** section in OpenShift AI.  
   You’ll see Kubeflow Pipelines triggered automatically by the notebook.
   Pipeline will -
   - Fetch PDF files from MinIO
   - Chunk and embed text using LlamaStack
   - Store embeddings in PGVector

---

## Components Deployed

- **MinIO** – Object storage for PDF documents
- **LlamaStack** – Embedding model for chunked documents
- **PGVector** – Vector database for storing embeddings
- **Kubeflow Pipelines** – Workflow to automate ingestion
- **Jupyter Notebook** – Entry point for triggering the pipeline

---

## Uninstall

To remove the deployment:

```bash
make uninstall-ingestion-pipeline
```

---
