"""
Lightweight GenAI Stack - RAG Application
A memory-efficient GenAI stack using Docker Model Runner + ChromaDB + LangChain
Designed to run within 6GB RAM - Educational version with RAG insights
"""

import os
import time
import tempfile
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from collections import Counter

# Configuration from environment
# Docker Model Runner uses OpenAI-compatible API
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "http://model-runner.docker.internal/engines/llama.cpp/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "docker-model-runner")
CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = os.getenv("CHROMA_PORT", "8000")
LLM_MODEL = os.getenv("LLM_MODEL", "ai/llama3.2:1B-Q8_0")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

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
        max-width: 1400px;
        margin: 0 auto;
    }
    .rag-step {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
        margin: 0.25rem 0;
        font-size: 0.85rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        border-left: 4px solid #667eea;
    }
    .source-card {
        background: #fff3cd;
        border-radius: 0.5rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_llm():
    """Initialize the LLM using Docker Model Runner (OpenAI-compatible API)"""
    return ChatOpenAI(
        model=LLM_MODEL,
        base_url=OPENAI_API_BASE,
        api_key=OPENAI_API_KEY,
        temperature=0.7,
    )


@st.cache_resource
def init_embeddings():
    """Initialize local embeddings using HuggingFace sentence-transformers"""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def init_vectorstore():
    """Initialize ChromaDB vector store"""
    import chromadb
    from chromadb.config import Settings

    embeddings = init_embeddings()

    # Create ChromaDB client with proper settings
    client = chromadb.PersistentClient(
        path="/app/chroma_db",
        settings=Settings(anonymized_telemetry=False)
    )

    return Chroma(
        client=client,
        collection_name="documents",
        embedding_function=embeddings,
    )


def get_vectorstore_stats():
    """Get statistics about the vector store"""
    try:
        vectorstore = init_vectorstore()
        results = vectorstore.get()
        if not results["ids"]:
            return {"total_chunks": 0, "unique_docs": 0, "docs": {}}

        sources = [m.get("source", "unknown") for m in results["metadatas"]]
        source_counts = Counter(sources)

        return {
            "total_chunks": len(results["ids"]),
            "unique_docs": len(source_counts),
            "docs": {os.path.basename(k): v for k, v in source_counts.items()}
        }
    except:
        return {"total_chunks": 0, "unique_docs": 0, "docs": {}}


def process_uploaded_file(uploaded_file, progress_callback=None):
    """Process uploaded file and return documents with progress updates"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    file_extension = uploaded_file.name.split('.')[-1].lower()

    try:
        if progress_callback:
            progress_callback(f"Loading {uploaded_file.name}...")

        if file_extension == 'pdf':
            loader = PyPDFLoader(tmp_path)
        elif file_extension == 'txt':
            loader = TextLoader(tmp_path)
        elif file_extension == 'md':
            loader = UnstructuredMarkdownLoader(tmp_path)
        else:
            return None, f"Unsupported file type: {file_extension}"

        documents = loader.load()

        if progress_callback:
            progress_callback(f"Splitting into chunks (chunk_size=500, overlap=50)...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)

        return splits, None
    finally:
        os.unlink(tmp_path)


def check_model_runner_health():
    """Check if Docker Model Runner is running and model is available"""
    import requests
    try:
        # Docker Model Runner exposes OpenAI-compatible /models endpoint
        response = requests.get(f"{OPENAI_API_BASE}/models", timeout=10)
        if response.status_code == 200:
            return True
    except:
        pass
    return False


def display_rag_pipeline_info():
    """Display educational information about the RAG pipeline"""
    with st.expander("📖 How RAG Works", expanded=False):
        st.markdown("""
        **Retrieval-Augmented Generation (RAG)** enhances LLM responses with relevant context from your documents.

        ### The Pipeline:
        1. **Document Ingestion** → PDF/TXT/MD files are loaded
        2. **Chunking** → Documents split into 500-character chunks (with 50-char overlap)
        3. **Embedding** → Each chunk converted to 384-dimensional vector using `all-MiniLM-L6-v2`
        4. **Storage** → Vectors stored in ChromaDB for fast similarity search
        5. **Query** → Your question is also converted to a vector
        6. **Retrieval** → Top 3 most similar chunks are found
        7. **Generation** → LLM generates answer using retrieved context

        ### Docker Model Runner
        This app uses **Docker Model Runner** - Docker's native LLM inference engine built on llama.cpp.
        Models run directly on your host machine with an OpenAI-compatible API.
        """)


def main():
    st.title("🤖 Lightweight GenAI Stack")
    st.caption(f"Running on Docker Model Runner ({LLM_MODEL}) | Memory-efficient RAG with ChromaDB | **Learning Mode**")

    # Get vectorstore stats
    stats = get_vectorstore_stats()

    # Sidebar
    with st.sidebar:
        st.header("📁 Document Upload")

        # Health check
        if check_model_runner_health():
            st.success(f"✅ Model Runner ({LLM_MODEL})")
        else:
            st.error("❌ Model Runner not ready")
            st.info("Ensure Docker Desktop 4.40+ with Model Runner enabled")

        # Vector DB Status
        st.subheader("📊 Vector Database")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Chunks", f"{stats['total_chunks']:,}")
        with col2:
            st.metric("Documents", stats['unique_docs'])

        if stats['docs']:
            with st.expander("Document Details"):
                for doc, count in stats['docs'].items():
                    st.text(f"• {doc[:20]}... ({count:,} chunks)")

        st.divider()

        uploaded_files = st.file_uploader(
            "Upload documents for RAG",
            type=['pdf', 'txt', 'md'],
            accept_multiple_files=True,
            help="Upload PDF, TXT, or Markdown files"
        )

        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                vectorstore = init_vectorstore()

                progress_container = st.empty()

                for uploaded_file in uploaded_files:
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        def update_progress(msg):
                            progress_container.info(f"⚙️ {msg}")

                        docs, error = process_uploaded_file(uploaded_file, update_progress)

                        if error:
                            st.error(error)
                        elif docs:
                            update_progress(f"Generating embeddings for {len(docs)} chunks...")
                            vectorstore.add_documents(docs)
                            st.success(f"✅ {uploaded_file.name}: {len(docs)} chunks added")

                progress_container.empty()
                st.rerun()

        st.divider()

        # Settings
        st.header("⚙️ Settings")
        use_rag = st.checkbox("Enable RAG", value=True, help="Use document context for answers")
        show_rag_details = st.checkbox("Show RAG Details", value=True, help="Display retrieval process")

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

        if st.button("🗑️ Clear All Documents"):
            import shutil
            if os.path.exists("/app/chroma_db"):
                shutil.rmtree("/app/chroma_db")
            st.success("Documents cleared!")
            st.rerun()

    # Main content area
    display_rag_pipeline_info()

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show RAG details for assistant messages if available
            if message["role"] == "assistant" and "rag_details" in message:
                display_rag_details(message["rag_details"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            rag_details = {}

            try:
                llm = init_llm()
                vectorstore = init_vectorstore()
                doc_count = stats['total_chunks']
                use_retrieval = use_rag and doc_count > 0

                if use_retrieval:
                    # RAG Mode with detailed tracking
                    rag_details["mode"] = "RAG"
                    rag_details["steps"] = []

                    with st.status("🔍 RAG Pipeline Running...", expanded=show_rag_details) as status:
                        # Step 1: Embedding the query
                        st.write("**Step 1:** Converting query to embedding...")
                        start_time = time.time()
                        embeddings = init_embeddings()
                        query_embedding = embeddings.embed_query(prompt)
                        embed_time = time.time() - start_time
                        rag_details["steps"].append({
                            "name": "Query Embedding",
                            "time": f"{embed_time:.2f}s",
                            "details": f"384-dim vector using {EMBEDDING_MODEL}"
                        })
                        st.write(f"   ✅ Generated 384-dim vector ({embed_time:.2f}s)")

                        # Step 2: Similarity search
                        st.write("**Step 2:** Searching for similar chunks...")
                        start_time = time.time()
                        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                        docs = retriever.invoke(prompt)
                        search_time = time.time() - start_time
                        rag_details["steps"].append({
                            "name": "Similarity Search",
                            "time": f"{search_time:.2f}s",
                            "details": f"Found {len(docs)} relevant chunks from {doc_count:,} total"
                        })
                        st.write(f"   ✅ Found {len(docs)} relevant chunks ({search_time:.2f}s)")

                        # Step 3: Show retrieved context
                        st.write("**Step 3:** Retrieved Context:")
                        rag_details["sources"] = []
                        for i, doc in enumerate(docs):
                            source = os.path.basename(doc.metadata.get("source", "unknown"))
                            page = doc.metadata.get("page", "?")
                            preview = doc.page_content[:150].replace("\n", " ")
                            st.write(f"   📄 **Chunk {i+1}** (Page {page}): {preview}...")
                            rag_details["sources"].append({
                                "source": source,
                                "page": page,
                                "content": doc.page_content[:300]
                            })

                        # Step 4: Generate response
                        st.write("**Step 4:** Generating response with LLM...")
                        start_time = time.time()

                        # Build context from retrieved docs
                        context = "\n\n".join([doc.page_content for doc in docs])
                        augmented_prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {prompt}

Answer:"""

                        response = llm.invoke(augmented_prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        gen_time = time.time() - start_time
                        rag_details["steps"].append({
                            "name": "LLM Generation",
                            "time": f"{gen_time:.2f}s",
                            "details": f"Using {LLM_MODEL}"
                        })
                        st.write(f"   ✅ Response generated ({gen_time:.2f}s)")

                        total_time = embed_time + search_time + gen_time
                        rag_details["total_time"] = f"{total_time:.2f}s"
                        status.update(label=f"✅ RAG Complete ({total_time:.2f}s)", state="complete")

                else:
                    # Direct LLM mode
                    rag_details["mode"] = "Direct LLM"
                    rag_details["note"] = "No documents loaded or RAG disabled"

                    with st.status("🧠 Generating response...", expanded=show_rag_details) as status:
                        st.write(f"Using {LLM_MODEL} directly (no document context)")
                        start_time = time.time()
                        response = llm.invoke(prompt)
                        response_text = response.content if hasattr(response, 'content') else str(response)
                        gen_time = time.time() - start_time
                        rag_details["total_time"] = f"{gen_time:.2f}s"
                        status.update(label=f"✅ Complete ({gen_time:.2f}s)", state="complete")

                # Display response
                st.markdown("---")
                st.markdown(response_text)

                # Show sources expander
                if use_retrieval and rag_details.get("sources"):
                    with st.expander("📚 View Source Chunks"):
                        for i, src in enumerate(rag_details["sources"], 1):
                            st.markdown(f"**Source {i}** ({src['source']}, Page {src['page']})")
                            st.text(src['content'])
                            st.divider()

                # Store message with RAG details
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "rag_details": rag_details
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})


def display_rag_details(rag_details):
    """Display RAG details for stored messages"""
    if not rag_details:
        return

    with st.expander(f"🔍 RAG Details ({rag_details.get('mode', 'Unknown')})"):
        if rag_details.get("steps"):
            for step in rag_details["steps"]:
                st.markdown(f"**{step['name']}**: {step['time']} - {step['details']}")

        if rag_details.get("total_time"):
            st.markdown(f"**Total Time**: {rag_details['total_time']}")

        if rag_details.get("sources"):
            st.markdown("**Retrieved Sources:**")
            for i, src in enumerate(rag_details["sources"], 1):
                st.text(f"{i}. {src['source']} (Page {src['page']})")


if __name__ == "__main__":
    main()
