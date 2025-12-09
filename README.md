# 🚀 Lightweight GenAI Stack

A memory-efficient alternative to Docker's GenAI Stack, designed to run **within 8GB RAM**.

## 📊 Comparison with Docker GenAI Stack

| Feature | Docker GenAI Stack | Lightweight GenAI Stack |
|---------|-------------------|------------------------|
| **Min RAM** | 20GB+ | **6-8GB** |
| **Vector DB** | Neo4j (heavy) | ChromaDB (light) |
| **Default Model** | Llama2 7B (~4.5GB) | Phi3:mini (~2.3GB) |
| **Embeddings** | Sentence Transformers | Nomic-embed-text |
| **Framework** | LangChain + Streamlit | LangChain + Streamlit |
| **Features** | GraphRAG, Knowledge Graph | Simple RAG |

## 🧠 Memory Breakdown

```
Component           | RAM Usage
--------------------|----------
Ollama + phi3:mini  | ~3-4GB
ChromaDB            | ~256-512MB
Streamlit App       | ~512MB-1GB
OS + Docker         | ~1-2GB
--------------------|----------
Total               | ~5-7GB
```

## 🚀 Quick Start

### 1. Clone or create the project

```bash
# Create directory
mkdir lightweight-genai-stack && cd lightweight-genai-stack

# Copy the files from this project
```

### 2. Start the stack

```bash
# Start all services
docker compose up -d

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
└── app/
    ├── Dockerfile          # Streamlit app image
    ├── requirements.txt    # Python dependencies
    └── main.py            # RAG application
```

## 🔧 Configuration

### Choose Your Model (by RAM availability)

Edit `docker-compose.yml` or create `.env`:

| Available RAM | Recommended Model | Command |
|---------------|-------------------|---------|
| 6GB | `tinyllama:1.1b` | Fastest, basic capability |
| 8GB | `phi3:mini` | **Best balance** (default) |
| 8GB | `llama3.2:3b` | Good general purpose |
| 8GB | `qwen2.5:3b` | Good for multilingual |

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
- Documents are chunked and embedded
- Retrieval-augmented generation for accurate answers

### 3. **Persistent Storage**
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

## 🔌 API Access

Ollama API is exposed on port 11434:

```bash
# Chat with the model directly
curl http://localhost:11434/api/generate -d '{
  "model": "phi3:mini",
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
    model="phi3:mini",
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
