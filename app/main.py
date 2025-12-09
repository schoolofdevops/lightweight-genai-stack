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
                            output_key="answer",
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
