#!/usr/bin/env python3
"""
RAG Query Tool - Run similarity searches against ChromaDB
Usage:
  docker exec genai-app python /app/rag_query.py "your query here"
  docker exec -it genai-app python /app/rag_query.py   # interactive mode
"""

import chromadb
import os
import sys
from langchain_huggingface import HuggingFaceEmbeddings

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
COLLECTION_NAME = "documents"
N_RESULTS = 3


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def search(collection, query, n_results=N_RESULTS):
    """Run similarity search and display results"""
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print("=" * 60)

    embeddings = get_embeddings()
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if not results["documents"][0]:
        print("No results found.")
        return

    print(f"Found {len(results['documents'][0])} results:\n")

    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - distance
        source = os.path.basename(metadata.get("source", "unknown"))
        page = metadata.get("page", "?")

        print(f"[{i+1}] Similarity: {similarity:.3f} | Source: {source} | Page: {page}")
        print("-" * 60)
        print(doc[:400])
        print("...\n" if len(doc) > 400 else "\n")


def interactive_mode(collection):
    """Interactive query mode"""
    print("\n" + "=" * 60)
    print("INTERACTIVE RAG QUERY MODE")
    print("Type your queries (or 'quit' to exit)")
    print("=" * 60)

    while True:
        try:
            query = input("\nQuery> ").strip()
            if query.lower() in ["quit", "exit", "q"]:
                break
            if not query:
                continue
            search(collection, query)
        except (EOFError, KeyboardInterrupt):
            break

    print("\nGoodbye!")


def main():
    client = chromadb.PersistentClient(path=PERSIST_DIR)

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error: Collection '{COLLECTION_NAME}' not found.")
        print("Upload documents via the Streamlit app first.")
        sys.exit(1)

    count = collection.count()
    print(f"Connected to ChromaDB | Collection: {COLLECTION_NAME} | Chunks: {count:,}")

    # Check if query provided as argument
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search(collection, query)
    else:
        # Interactive mode
        interactive_mode(collection)


if __name__ == "__main__":
    main()
