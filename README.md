# 🚀 Lightweight GenAI Stack

A memory-efficient alternative to Docker's GenAI Stack, designed to run **within 6-8GB RAM**. Features an **educational Learning Mode** that visualizes the RAG pipeline in real-time.

## 📊 Comparison with Docker GenAI Stack

| Feature | Docker GenAI Stack | Lightweight GenAI Stack |
|---------|-------------------|------------------------|
| **Min RAM** | 20GB+ | **6-8GB** |
| **Vector DB** | Neo4j (heavy) | ChromaDB (light) |
| **Default Model** | Llama2 7B (~4.5GB) | TinyLlama 1.1B (~600MB) |
| **Embeddings** | Sentence Transformers | nomic-embed-text (768-dim) |
| **Framework** | LangChain + Streamlit | LangChain + Streamlit |
| **Features** | GraphRAG, Knowledge Graph | Simple RAG + Learning Mode |

## 🧠 Memory Breakdown

```
Component              | RAM Usage
-----------------------|----------
Ollama + tinyllama:1.1b| ~1-2GB
nomic-embed-text       | ~300MB
ChromaDB               | ~256-512MB
Streamlit App          | ~512MB-1GB
OS + Docker            | ~1-2GB
-----------------------|----------
Total                  | ~4-6GB
```

## 🚀 Quick Start

### 1. Clone or create the project

```bash
# Create directory
git clone https://github.com/schoolofdevops/lightweight-genai-stack.git
cd lightweight-genai-stack

# Copy the files from this project
```

### 2. Start the stack

```bash
# Start all services
docker compose up --build -d

# Watch the logs (model download takes a few minutes)
docker compose logs -f model-puller
```

### 3. Access the app

Open http://localhost:8501 in your browser.

## 📁 Project Structure

```
lightweight-genai-stack/
├── docker-compose.yml      # Main orchestration
├── .env.example            # Configuration template
├── README.md
├── WORKSHOP.md             # Detailed workshop guide
├── chroma_stats.py         # ChromaDB statistics script
├── rag_query.py            # RAG query testing script
├── test_chroma.py          # Full ChromaDB test suite
└── app/
    ├── Dockerfile          # Streamlit app image
    ├── requirements.txt    # Python dependencies
    └── main.py             # RAG application (Learning Mode)
```

## 🔧 Configuration

### Choose Your Model (by RAM availability)

Edit `docker-compose.yml` or create `.env`:

| Available RAM | Recommended Model | Notes |
|---------------|-------------------|-------|
| 6GB | `tinyllama:1.1b` | **Default** - Fastest, ~600MB |
| 8GB | `phi3:mini` | Better quality, ~2.3GB |
| 8GB | `llama3.2:3b` | Good general purpose |
| 8GB | `qwen2.5:3b` | Good for multilingual |

**Current Default Configuration:**
- **LLM Model:** `tinyllama:1.1b` (~600MB, fast inference)
- **Embedding Model:** `nomic-embed-text` (768-dimensional vectors)

### Reduce Memory Further

```yaml
# In docker-compose.yml, adjust limits:
services:
  ollama:
    deploy:
      resources:
        limits:
          memory: 3G  # Reduce if using tinyllama
```

## 📚 Features

### 1. **Chat Mode**
Direct conversation with the LLM without documents.

### 2. **RAG Mode** (Document Q&A)
- Upload PDF, TXT, or Markdown files
- Documents are chunked (500 chars) and embedded (768-dim vectors)
- Retrieval-augmented generation for accurate answers

### 3. **Learning Mode** (Educational)
- Real-time RAG pipeline visualization
- Step-by-step display: Query Embedding → Similarity Search → Context Retrieval → LLM Generation
- Timing information for each step
- View retrieved source chunks with page numbers

### 4. **Vector Database Stats**
- Live chunk and document counts in sidebar
- Document breakdown showing chunks per file

### 5. **Persistent Storage**
- ChromaDB stores embeddings persistently
- Chat history maintained in session

## 🛠️ Useful Commands

```bash
# Start the stack
docker compose up -d

# Check logs
docker compose logs -f

# Check Ollama models
docker exec ollama ollama list

# Pull a different model
docker exec ollama ollama pull llama3.2:3b

# Stop everything
docker compose down

# Stop and remove volumes (fresh start)
docker compose down -v

# Check memory usage
docker stats
```

## 🧪 Testing & Debugging Scripts

Three utility scripts are provided for inspecting ChromaDB and testing RAG queries:

### 1. `chroma_stats.py` - View Database Statistics

Shows document and chunk counts in ChromaDB:

```bash
docker exec genai-app python /app/chroma_stats.py
```

**Output:**
```
============================================================
CHROMADB STATISTICS
============================================================

Collection: documents
----------------------------------------
  Total chunks: 6,693
  Unique documents: 3

  Documents breakdown:
    - report.pdf: 2,231 chunks
    - manual.pdf: 2,231 chunks
    - guide.pdf: 2,231 chunks
============================================================
```

**When to use:** After uploading documents to verify they were processed correctly.

### 2. `rag_query.py` - Test RAG Searches

Run similarity searches against your documents:

```bash
# Single query
docker exec genai-app python /app/rag_query.py "What is the main topic?"

# Interactive mode
docker exec -it genai-app python /app/rag_query.py
```

**Output:**
```
Connected to ChromaDB | Collection: documents | Chunks: 6,693

============================================================
QUERY: What is the main topic?
============================================================
Found 3 results:

[1] Similarity: 0.510 | Source: report.pdf | Page: 12
------------------------------------------------------------
The main topic of this document covers...
```

**When to use:**
- Testing if documents are being retrieved correctly
- Debugging why certain queries aren't finding relevant content
- Comparing similarity scores for different query phrasings

### 3. `test_chroma.py` - Full Test Suite

Comprehensive ChromaDB inspection with sample queries:

```bash
docker exec genai-app python /app/test_chroma.py
```

**When to use:** Initial setup verification or troubleshooting RAG issues.

## 🔌 API Access

Ollama API is exposed on port 11434:

```bash
# Chat with the model directly
curl http://localhost:11434/api/generate -d '{
  "model": "tinyllama:1.1b",
  "prompt": "Explain Docker in 3 sentences",
  "stream": false
}'

# List available models
curl http://localhost:11434/api/tags
```

## 🐍 Python Integration

```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="tinyllama:1.1b",
    base_url="http://localhost:11434"
)

response = llm.invoke("What is Kubernetes?")
print(response)
```

## 🔄 Alternative: Use API Models (Zero RAM for LLM)

If you want to use OpenAI/Anthropic instead of local models:

1. Comment out `ollama` and `model-puller` services
2. Update `app/main.py` to use `ChatOpenAI` or `ChatAnthropic`
3. Add your API key to `.env`

## ❓ Troubleshooting

### Model not loading?
```bash
# Check if model is downloaded
docker exec ollama ollama list

# Manually pull model
docker exec ollama ollama pull phi3:mini
```

### Out of memory?
```bash
# Check what's using memory
docker stats

# Use a smaller model
docker exec ollama ollama pull tinyllama:1.1b
# Update LLM_MODEL in docker-compose.yml
```

### App can't connect to Ollama?
```bash
# Check Ollama health
curl http://localhost:11434/api/tags

# Restart Ollama
docker compose restart ollama
```

## 🎯 DevOps Use Cases

This stack is perfect for:

- **Local AI-assisted documentation** - Query your runbooks
- **Incident analysis** - RAG over incident reports
- **Code review assistant** - Analyze code files
- **Learning/demos** - Teach GenAI concepts without cloud costs

## 📈 Scaling Up

When you have more RAM available:

```yaml
# For 16GB RAM, use better models:
LLM_MODEL=llama3.2:8b
EMBEDDING_MODEL=nomic-embed-text

# For 32GB+ RAM, match Docker GenAI stack:
LLM_MODEL=llama2:13b
```

## 🙏 Credits

Inspired by:
- [Docker GenAI Stack](https://github.com/docker/genai-stack)
- [pi-genai-stack](https://github.com/bots-garden/pi-genai-stack) (Raspberry Pi version)
- [Ollama](https://ollama.com)
- [ChromaDB](https://www.trychroma.com)

## 📄 License

MIT License - Use freely!
