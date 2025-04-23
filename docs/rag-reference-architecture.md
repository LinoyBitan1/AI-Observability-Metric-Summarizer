# RAG Blueprint - System Reference Architecture

## Introduction

Here is a representation of the architecture for the RAG Blueprint. Contains a basic identification of the components, the workflow and the mapping of those elements into possible micro-services.

## Architecture Diagram

![RAG System Architecture](img/rag-architecture.png)

*The architecture illustrates both the ingestion pipeline for document processing and the RAG pipeline for query handling.*

## System Components

The architecture consists of two main workflow pipelines:

1. **RAG Pipeline** - Handles user queries and generates responses
2. **Ingestion Pipeline** - Processes documents and updates the knowledge base

### 1. RAG Pipeline Components

#### Frontend UI
- Provides the user interface for submitting queries and viewing responses
- Communicates with the backend services via REST APIs
- Can be deployed as a separate pod from the main application logic

#### Query Input
- Captures user queries from the frontend
- Formats queries for downstream processing

#### Input Safety Shield
- Screens incoming queries for harmful content, manipulative prompts, or injection attacks
- Implements content moderation to detect inappropriate requests
- May use a combination of rule-based filters and ML models
- Rejects or sanitizes potentially harmful queries

#### Query Processing/Routing
- Routes queries to appropriate retrieval systems

#### Retriever and Embedding Service for user queries
- Converts queries into vector embeddings using chunks
- Interfaces with the vector database for similarity search
- May include cross-encoders for more accurate retrieval
- Can be implemented using frameworks like LangChain

#### Vector DB
- Stores document embeddings and metadata
- Performs efficient similarity searches
- Deployed as a separate container / Pod

#### Retriever and Reranking
- Takes initial retrieval results and improves them
- Reranks documents based on relevance to the query
- May filter out irrelevant or redundant information
- Optimizes context for the LLM
- Not yet implemented by Llama Stack

#### LLM Response Generation
- Processes the query and retrieved context to generate a response
- Formats prompts with appropriate instructions and context
- Interfaces with the LLM service (which could be vLLM running Llama models)

#### Output Content Safety Shield and Validator
- Screens generated responses for harmful content
- Verifies factual accuracy and alignment with retrieved information
- Checks for hallucinations or unsupported claims
- Ensures responses meet safety and compliance requirements

#### Generated Response
- The final, validated response delivered to the user
- Formatted appropriately for presentation in the UI
- May include citations or references to source material
- Could incorporate confidence scores or alternative answers

### 2. Ingestion Pipeline Components

#### Document Sources
- **S3 Bucket**: Cloud storage for document files
- **URL**: Documents for download
- **Uploads**: Direct file uploads from users via the frontend

#### Processing Methods
- **OpenShift AI Pipelines**: Orchestrated workflows for complex document processing
- **Python Script**: Custom scripts for specialized document handling
- **Frontend UI or Retriever Listener**: User-triggered document processing

#### Retriever and Embedding Service (Docling + embedding model)
- Chunks documents into appropriate segments
- Generates embeddings for each chunk
- Handles document metadata extraction
- Prepares data for insertion into the vector database

## Deployment Architecture in OpenShift

This reference architecture can be deployed in OpenShift with the following pod structure:

1. **Frontend Pod(s)**
   - Contains the user interface
   - Communicates with the Application Pod via APIs

2. **Input Safety Shield Pod(s)**

   - Dedicated pod for input content moderation
   - Screens incoming queries for harmful content
   - Implements query validation and sanitization
   - Can be independently scaled and updated
   - May connect to specialized content moderation APIs

3. **Application Pod(s)** (llama-stack)
   - Houses the core RAG logic and orchestration
   - Implements query processing, safety shields, and response generation
   - Contains the LangChain implementation for retrieval and reranking or other framework
   - Manages connections to other services

4. **LLM Service Pod(s)**
   - Runs the language model inference (e.g., vLLM with Llama models)
   - Optimized for GPU utilization
   - Potentially runs on specialized nodes with GPU resources
   - May be run locally through vLLM (to be confirmed)

5. **Vector Database Pod(s)**
   - Manages the vector store for document embeddings
   - Handles similarity search requests
   - Requires persistent storage
   - Deployed as a StatefulSet

6. **Output Safety Shield Pod(s)**

   - Dedicated pod for output content validation
   - Screens generated responses for harmful content
   - Verifies factual accuracy and alignment
   - Can be independently scaled and optimized
   - May connect to specialized validation models

7. **Embedding Service Pod(s)**
   - Generates vector embeddings for documents and queries
   - May be combined with document processing (retriever) components
   - Could be scaled independently based on embedding workload

8. **Ingestion Pipeline Pod(s)**
   - Handles document processing workflows
   - May use batch processing for large document sets
   - Connected to document sources like S3 (MinIO for example)
   - Can scale based on ingestion demand

## Implementation Technologies

Technologies chosen for the stack so far:

- **Application Framework**: llama-stack
- **LLM Service**: vLLM with meta-llama/Llama-3.2-3B-Instruct
- **Vector Database**: PGVector
- **Container Orchestration**: OpenShift + OpenShift AI
- **RAG Framework**: LangChain, LlamaIndex
- **Safety Models**: meta-llama/Llama-Guard-3-8B

