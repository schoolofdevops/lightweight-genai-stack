# Advanced Docker for GenAI: Building a Lightweight RAG Application

## Workshop Overview

In this hands-on workshop, you'll learn how to build and deploy a **memory-efficient Generative AI stack** using Docker. We'll create a Retrieval-Augmented Generation (RAG) application that can run on machines with as little as **6GB RAM** - perfect for laptops, edge devices, or cost-conscious cloud deployments.

### What You'll Build

A fully functional AI-powered document Q&A system featuring:
- **Ollama** - Local LLM inference server
- **ChromaDB** - Vector database for semantic search
- **LangChain** - AI orchestration framework
- **Streamlit** - Interactive web interface

### Prerequisites

- Docker Desktop installed (with Docker Compose v2)
- 6GB+ available RAM
- Basic familiarity with Docker concepts
- Terminal/command-line access

### Time Required

- **Full Workshop**: 60-90 minutes
- **Quick Start**: 15-20 minutes

---

## Part 1: Understanding the Architecture

### The GenAI Stack Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Browser                              │
│                   http://localhost:8501                      │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  Streamlit App (genai-app)                   │
│                     Port: 8501                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │  • File Upload & Processing                             │ │
│  │  • Chat Interface                                       │ │
│  │  • LangChain RAG Pipeline                              │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────┬──────────────────────────────────┬───────────────┘
           │                                  │
           ▼                                  ▼
┌─────────────────────────┐    ┌─────────────────────────────┐
│   Ollama LLM Server     │    │      ChromaDB               │
│      Port: 11434        │    │      Port: 8000             │
│  ┌───────────────────┐  │    │  ┌───────────────────────┐  │
│  │ tinyllama:1.1b    │  │    │  │ Vector Embeddings     │  │
│  │ nomic-embed-text  │  │    │  │ Document Chunks       │  │
│  └───────────────────┘  │    │  │ Semantic Search       │  │
└─────────────────────────┘    └─────────────────────────────┘
```

### Why This Stack?

| Component | Purpose | Memory Usage |
|-----------|---------|--------------|
| **Ollama** | Runs LLM locally | ~1-3GB (model dependent) |
| **ChromaDB** | Stores & searches embeddings | ~256-512MB |
| **Streamlit App** | Web UI + RAG logic | ~512MB-1GB |
| **Docker Overhead** | Container runtime | ~500MB |

**Total: ~4-6GB** vs. 20GB+ for full GenAI Stack

---

## Part 2: Project Setup

### Step 1: Create the Project Structure

```bash
# Create project directory
mkdir lightweight-genai-stack
cd lightweight-genai-stack

# Create subdirectories
mkdir -p app
```

### Step 2: Create the Docker Compose File

Create `docker-compose.yml`:

```yaml
services:
  # Ollama - LLM Server (lightweight model)
  ollama:
    image: ollama/ollama:latest
    container_name: ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        limits:
          memory: 3G  # Limit memory for Ollama (tinyllama is ~600MB)
    healthcheck:
      test: ["CMD", "ollama", "list"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 30s
    restart: unless-stopped

  # Model Puller - Downloads the lightweight model on startup
  model-puller:
    image: curlimages/curl:latest
    container_name: model-puller
    depends_on:
      ollama:
        condition: service_healthy
    entrypoint: ["/bin/sh", "-c"]
    command:
      - |
        echo "Waiting for Ollama to be ready..."
        sleep 5
        echo "Pulling tinyllama:1.1b model (ultra-lightweight, ~600MB)..."
        curl -X POST http://ollama:11434/api/pull -d '{"name": "tinyllama:1.1b"}'
        echo "Pulling nomic-embed-text for embeddings..."
        curl -X POST http://ollama:11434/api/pull -d '{"name": "nomic-embed-text"}'
        echo "Models downloaded successfully!"

  # ChromaDB - Lightweight Vector Database
  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    ports:
      - "8000:8000"
    volumes:
      - chroma_data:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    deploy:
      resources:
        limits:
          memory: 512M
    restart: unless-stopped

  # GenAI App - Streamlit RAG Application
  genai-app:
    build:
      context: ./app
      dockerfile: Dockerfile
    container_name: genai-app
    ports:
      - "8501:8501"
    environment:
      - OLLAMA_BASE_URL=http://ollama:11434
      - CHROMA_HOST=chromadb
      - CHROMA_PORT=8000
      - LLM_MODEL=tinyllama:1.1b
      - EMBEDDING_MODEL=nomic-embed-text
    depends_on:
      ollama:
        condition: service_healthy
      chromadb:
        condition: service_started
    volumes:
      - ./app:/app
      - uploaded_docs:/app/uploads
    deploy:
      resources:
        limits:
          memory: 1G
    restart: unless-stopped

volumes:
  ollama_data:
  chroma_data:
  uploaded_docs:

networks:
  default:
    name: genai-network
```

#### Key Docker Concepts Explained

**1. Service Dependencies with Health Checks**
```yaml
depends_on:
  ollama:
    condition: service_healthy
```
This ensures the app only starts after Ollama is fully ready, not just running.

**2. Resource Limits**
```yaml
deploy:
  resources:
    limits:
      memory: 3G
```
Prevents any single container from consuming all available memory.

**3. Named Volumes for Persistence**
```yaml
volumes:
  ollama_data:    # Persists downloaded models
  chroma_data:    # Persists vector embeddings
```

**4. Init Container Pattern (model-puller)**
```yaml
model-puller:
  # Runs once to download models, then exits
```
This pattern ensures models are ready before the app needs them.

---

## Part 3: Building the Application

### Step 3: Create the Dockerfile

Create `app/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Dockerfile Best Practices Demonstrated

1. **Use slim base images** - `python:3.11-slim` is ~150MB vs ~1GB for full image
2. **Layer caching** - Copy `requirements.txt` before code for faster rebuilds
3. **Clean up apt cache** - Reduces image size
4. **Health checks** - Enables Docker to monitor container health
5. **Non-root user** (optional enhancement for production)

### Step 4: Create Requirements File

Create `app/requirements.txt`:

```
streamlit==1.40.0
langchain==0.3.9
langchain-community==0.3.8
langchain-chroma==0.1.4
langchain-ollama==0.2.2
chromadb==0.5.20
pypdf==5.1.0
python-docx==1.1.2
unstructured==0.16.10
sentence-transformers==3.3.1
requests==2.32.3
```

### Step 5: Create the RAG Application

Create `app/main.py`:

```python
"""
Lightweight GenAI Stack - RAG Application
A memory-efficient GenAI stack using Ollama + ChromaDB + LangChain
Designed to run within 6GB RAM
"""

import os
import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
import tempfile
import time

# Configuration from environment
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
LLM_MODEL = os.getenv("LLM_MODEL", "tinyllama:1.1b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# Page config
st.set_page_config(
    page_title="Lightweight GenAI Stack",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_llm():
    """Initialize the Ollama LLM"""
    return OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.7,
        num_ctx=4096,  # Smaller context for memory efficiency
    )


@st.cache_resource
def init_embeddings():
    """Initialize Ollama embeddings"""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def init_vectorstore():
    """Initialize ChromaDB vector store"""
    embeddings = init_embeddings()
    return Chroma(
        collection_name="documents",
        embedding_function=embeddings,
        persist_directory="/app/chroma_db"
    )


def process_uploaded_file(uploaded_file):
    """Process uploaded file and return documents"""
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # Load based on file type
    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_extension == 'txt':
            loader = TextLoader(tmp_path)
        elif file_extension == 'md':
            loader = UnstructuredMarkdownLoader(tmp_path)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None

        documents = loader.load()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,  # Smaller chunks for memory efficiency
            chunk_overlap=50,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        return splits
    finally:
        os.unlink(tmp_path)


def check_ollama_health():
    """Check if Ollama is running and model is available"""
    import requests
    try:
        response = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            return LLM_MODEL in model_names or any(LLM_MODEL in m for m in model_names)
    except:
        pass
    return False


def main():
    st.title("🤖 Lightweight GenAI Stack")
    st.caption(f"Running on {LLM_MODEL} | Memory-efficient RAG with ChromaDB")

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")

        # Health check
        if check_ollama_health():
            st.success(f"✅ Connected to Ollama ({LLM_MODEL})")
        else:
            st.error("❌ Ollama not ready. Model may still be downloading...")
            st.info("Run `docker logs model-puller` to check progress")

        uploaded_files = st.file_uploader(
            "Upload documents for RAG",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or Markdown files to chat with"
        )

        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                vectorstore = init_vectorstore()

                with st.spinner("Processing documents..."):
                    for uploaded_file in uploaded_files:
                        docs = process_uploaded_file(uploaded_file)
                        if docs:
                            vectorstore.add_documents(docs)
                            st.success(f"✅ Processed: {uploaded_file.name}")

                st.session_state.docs_loaded = True
                st.rerun()

        st.divider()

        # Settings
        st.header("⚙️ Settings")

        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
        use_rag = st.checkbox("Use RAG (if documents loaded)", value=True)

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.session_state.memory = None
            st.rerun()

        st.divider()

        # System info
        st.header("📊 System Info")
        st.info(f"""
        **Model:** {LLM_MODEL}
        **Embeddings:** {EMBEDDING_MODEL}
        **Vector DB:** ChromaDB
        **Target RAM:** < 6GB
        """)

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=5  # Keep last 5 exchanges for memory efficiency
        )

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    llm = init_llm()

                    # Check if we should use RAG
                    use_retrieval = use_rag and st.session_state.get('docs_loaded', False)

                    if use_retrieval:
                        # RAG mode
                        vectorstore = init_vectorstore()
                        retriever = vectorstore.as_retriever(
                            search_kwargs={"k": 3}  # Retrieve top 3 chunks
                        )

                        chain = ConversationalRetrievalChain.from_llm(
                            llm=llm,
                            retriever=retriever,
                            memory=st.session_state.memory,
                            return_source_documents=True,
                        )

                        result = chain({"question": prompt})
                        response = result["answer"]

                        # Show sources
                        if result.get("source_documents"):
                            with st.expander("📚 Sources"):
                                for i, doc in enumerate(result["source_documents"], 1):
                                    st.markdown(f"**Source {i}:**")
                                    st.markdown(doc.page_content[:300] + "...")
                    else:
                        # Direct LLM mode
                        response = llm.invoke(prompt)

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})


if __name__ == "__main__":
    main()
```

---

## Part 4: Running the Stack

### Step 6: Start the Services

```bash
# Start all services in detached mode
docker compose up -d

# Watch the build and startup logs
docker compose logs -f
```

### Step 7: Monitor Model Downloads

The `model-puller` container downloads the AI models. This takes a few minutes:

```bash
# Watch model download progress
docker logs -f model-puller
```

Expected output:
```
Waiting for Ollama to be ready...
Pulling tinyllama:1.1b model (ultra-lightweight, ~600MB)...
{"status":"pulling manifest"}
{"status":"pulling 2af3b81862c6","total":637699456,"completed":...}
...
{"status":"success"}
Pulling nomic-embed-text for embeddings...
...
{"status":"success"}
Models downloaded successfully!
```

### Step 8: Verify All Services are Running

```bash
# Check container status
docker compose ps
```

Expected output:
```
NAME           IMAGE                               STATUS                   PORTS
chromadb       chromadb/chroma:latest              Up (healthy)             0.0.0.0:8000->8000/tcp
genai-app      lightweight-genai-stack-genai-app   Up (healthy)             0.0.0.0:8501->8501/tcp
ollama         ollama/ollama:latest                Up (healthy)             0.0.0.0:11434->11434/tcp
```

### Step 9: Access the Application

Open your browser and navigate to:
- **Main App**: http://localhost:8501
- **Ollama API**: http://localhost:11434
- **ChromaDB API**: http://localhost:8000/api/v2/heartbeat

---

## Part 5: Using the RAG Application

### Basic Chat (No Documents)

1. Open http://localhost:8501
2. Type a question in the chat input: "What is Docker?"
3. The LLM responds directly

### Document Q&A (RAG Mode)

1. Click "Browse files" in the sidebar
2. Upload a PDF, TXT, or Markdown file
3. Click "📥 Process Documents"
4. Wait for processing to complete
5. Ask questions about the document content

### Example Workflow

```
You: What are the main topics covered in this document?
AI: Based on the document, the main topics are... [contextual response]

You: Can you summarize section 3?
AI: Section 3 discusses... [pulls from document chunks]
```

---

## Part 6: Deep Dive - How RAG Works

### The RAG Pipeline

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Document   │────▶│   Chunking   │────▶│  Embedding   │
│   Upload     │     │  (500 chars) │     │   Model      │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   ChromaDB   │◀────│   Vector     │◀────│   Store      │
│   Storage    │     │   Index      │     │   Vectors    │
└──────┬───────┘     └──────────────┘     └──────────────┘
       │
       │  On Query:
       ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   User       │────▶│   Embed      │────▶│   Similarity │
│   Question   │     │   Query      │     │   Search     │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   LLM        │◀────│   Context +  │◀────│   Top K      │
│   Response   │     │   Question   │     │   Chunks     │
└──────────────┘     └──────────────┘     └──────────────┘
```

### Key Code Components Explained

**1. Document Chunking**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Characters per chunk
    chunk_overlap=50,    # Overlap for context continuity
)
```
Smaller chunks = faster search, less memory, but may lose context.

**2. Embedding Generation**
```python
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",  # 137M parameter model
    base_url=OLLAMA_BASE_URL,
)
```
Converts text to 768-dimensional vectors for semantic search.

**3. Similarity Search**
```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}  # Return top 3 most similar chunks
)
```

**4. Context-Augmented Generation**
```python
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # Show where answers came from
)
```

---

## Part 7: Monitoring and Debugging

### Check Resource Usage

```bash
# Real-time container stats
docker stats

# Expected output:
CONTAINER ID   NAME        CPU %     MEM USAGE / LIMIT
xxxx           ollama      0.50%     1.2GiB / 3GiB
xxxx           chromadb    0.10%     256MiB / 512MiB
xxxx           genai-app   0.30%     512MiB / 1GiB
```

### View Logs

```bash
# All services
docker compose logs -f

# Specific service
docker compose logs -f ollama

# Last 100 lines
docker compose logs --tail=100 genai-app
```

### Test Ollama Directly

```bash
# List available models
curl http://localhost:11434/api/tags | jq

# Generate text directly
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama:1.1b",
  "prompt": "What is Docker in one sentence?",
  "stream": false
}' | jq -r '.response'
```

### Test ChromaDB

```bash
# Health check
curl http://localhost:8000/api/v2/heartbeat

# List collections
curl http://localhost:8000/api/v2/collections
```

---

## Part 8: Customization Options

### Option 1: Change the LLM Model

Edit `docker-compose.yml`:

```yaml
# For better quality (needs 8GB RAM)
environment:
  - LLM_MODEL=phi3:mini

# For multilingual support
environment:
  - LLM_MODEL=qwen2.5:3b
```

Then update the model-puller command and restart:
```bash
docker compose down
docker compose up -d
```

### Option 2: Add GPU Support

```yaml
services:
  ollama:
    image: ollama/ollama:latest
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Option 3: Scale for Production

```yaml
services:
  genai-app:
    deploy:
      replicas: 3
    # Add load balancer in front
```

---

## Part 9: Exercises

### Exercise 1: Add a New Document Type (10 min)

Add support for `.docx` files:

1. The loader is already in requirements: `python-docx`
2. Add a new condition in `process_uploaded_file()`:
```python
elif file_extension == 'docx':
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(tmp_path)
```
3. Update the file uploader to accept `.docx`

### Exercise 2: Add Response Streaming (15 min)

Modify the chat to stream responses:

```python
# Instead of:
response = llm.invoke(prompt)

# Use:
for chunk in llm.stream(prompt):
    st.write(chunk)
```

### Exercise 3: Add API Endpoint (20 min)

Create a FastAPI wrapper for the RAG functionality:

1. Add `fastapi` and `uvicorn` to requirements
2. Create `api.py` with endpoints
3. Add a new service in docker-compose.yml

### Exercise 4: Implement Chat History Export (10 min)

Add a button to download chat history as JSON:

```python
import json

if st.button("📥 Export Chat"):
    chat_json = json.dumps(st.session_state.messages, indent=2)
    st.download_button(
        label="Download",
        data=chat_json,
        file_name="chat_history.json",
        mime="application/json"
    )
```

---

## Part 10: Cleanup

### Stop the Stack

```bash
# Stop containers (preserves data)
docker compose down

# Stop and remove all data
docker compose down -v

# Remove built images
docker compose down --rmi all
```

### Free Up Space

```bash
# Remove unused Docker resources
docker system prune -a

# Check space usage
docker system df
```

---

## Summary

### What You Learned

1. **Docker Compose** for multi-container AI applications
2. **Health checks** and service dependencies
3. **Resource limits** for memory-constrained environments
4. **Init containers** for one-time setup tasks
5. **RAG architecture** with LangChain
6. **Vector databases** with ChromaDB
7. **Local LLM inference** with Ollama

### Key Takeaways

| Concept | Docker Feature |
|---------|---------------|
| Service orchestration | `docker-compose.yml` |
| Startup order | `depends_on` + `condition: service_healthy` |
| Memory management | `deploy.resources.limits` |
| Data persistence | Named volumes |
| One-time tasks | Init container pattern |
| Health monitoring | `healthcheck` directive |

### Next Steps

- Explore the [Docker GenAI Stack](https://github.com/docker/genai-stack) for production features
- Add authentication with Streamlit-Authenticator
- Implement document versioning
- Add support for more file types (HTML, CSV, etc.)
- Deploy to cloud with Docker Swarm or Kubernetes

---

## Resources

- [Ollama Documentation](https://ollama.com/library)
- [LangChain Documentation](https://python.langchain.com/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Docker Compose Specification](https://docs.docker.com/compose/compose-file/)

---

## Troubleshooting Guide

### Problem: "Ollama not ready" error

**Solution:**
```bash
# Check if model is downloading
docker logs model-puller

# Manually pull the model
docker exec ollama ollama pull tinyllama:1.1b
```

### Problem: Out of memory errors

**Solution:**
```bash
# Use smaller model
docker exec ollama ollama pull tinyllama:1.1b

# Reduce container limits in docker-compose.yml
# Restart with: docker compose down && docker compose up -d
```

### Problem: Slow responses

**Solution:**
- Use GPU if available
- Reduce `num_ctx` in LLM configuration
- Use smaller/quantized models
- Increase container memory limits

### Problem: Documents not being found in search

**Solution:**
- Check if documents were processed (look for success message)
- Verify ChromaDB is running: `curl localhost:8000/api/v2/heartbeat`
- Try reprocessing documents
- Clear ChromaDB: `docker compose down -v` and restart

---

**Happy Building! 🚀**