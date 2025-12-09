#!/usr/bin/env python3
"""
ChromaDB Inspection & RAG Search Test Script
Run this to check if documents are vectorized and test search queries.
"""

import chromadb
from chromadb.config import Settings
import sys
import os
from langchain_ollama import OllamaEmbeddings

# Configuration
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
COLLECTION_NAME = "documents"


def get_embedding_function():
    """Get the same embedding function used by the app"""
    return OllamaEmbeddings(
        model=EMBEDDING_MODEL,
        base_url=OLLAMA_BASE_URL,
    )


def connect_to_chroma():
    """Connect to ChromaDB using persistent storage"""
    try:
        client = chromadb.PersistentClient(path=PERSIST_DIR)
        print(f"Connected to ChromaDB at {PERSIST_DIR}")
        return client
    except Exception as e:
        print(f"Failed to connect to ChromaDB: {e}")
        sys.exit(1)


def list_collections(client):
    """List all collections in ChromaDB"""
    collections = client.list_collections()
    print(f"\n{'='*50}")
    print("COLLECTIONS IN CHROMADB")
    print(f"{'='*50}")
    if not collections:
        print("No collections found. Upload some documents first!")
        return None

    for col in collections:
        print(f"  - {col.name}")
    return collections


def inspect_collection(client, collection_name):
    """Inspect a specific collection"""
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        print(f"Collection '{collection_name}' not found: {e}")
        return None

    count = collection.count()
    print(f"\n{'='*50}")
    print(f"COLLECTION: {collection_name}")
    print(f"{'='*50}")
    print(f"Document chunks: {count}")

    if count == 0:
        print("No documents in this collection. Upload documents via the Streamlit app.")
        return None

    # Peek at sample documents
    print(f"\n--- Sample Documents (first 3) ---")
    results = collection.peek(limit=3)

    for i, (doc_id, doc_content, metadata) in enumerate(zip(
        results.get('ids', []),
        results.get('documents', []),
        results.get('metadatas', [])
    )):
        print(f"\n[{i+1}] ID: {doc_id}")
        print(f"    Metadata: {metadata}")
        content_preview = doc_content[:200] if doc_content else "N/A"
        print(f"    Content: {content_preview}...")

    return collection


def run_similarity_search(collection, query, n_results=3):
    """Run a similarity search query using Ollama embeddings"""
    print(f"\n{'='*50}")
    print(f"QUERY: \"{query}\"")
    print(f"{'='*50}")

    # Generate embedding using the same model as the app
    embeddings = get_embedding_function()
    query_embedding = embeddings.embed_query(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    if not results['documents'][0]:
        print("No results found.")
        return

    print(f"Found {len(results['documents'][0])} results:\n")

    for i, (doc, metadata, distance) in enumerate(zip(
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        similarity = 1 - distance  # Convert distance to similarity score
        print(f"[{i+1}] Similarity: {similarity:.4f} (distance: {distance:.4f})")
        print(f"    Metadata: {metadata}")
        print(f"    Content: {doc[:300]}...")
        print()


def interactive_search(collection):
    """Interactive search mode"""
    print(f"\n{'='*50}")
    print("INTERACTIVE SEARCH MODE")
    print("Type your queries (or 'quit' to exit)")
    print(f"{'='*50}")

    while True:
        query = input("\nEnter query: ").strip()
        if query.lower() in ['quit', 'exit', 'q']:
            break
        if not query:
            continue
        run_similarity_search(collection, query)


def main():
    print("ChromaDB RAG Search Tester")
    print("="*50)

    # Connect
    client = connect_to_chroma()

    # List collections
    list_collections(client)

    # Inspect main collection
    collection = inspect_collection(client, COLLECTION_NAME)

    if collection and collection.count() > 0:
        # Run some test queries
        test_queries = [
            "What is this document about?",
            "main topic",
            "summary",
        ]

        print(f"\n{'#'*50}")
        print("RUNNING TEST QUERIES")
        print(f"{'#'*50}")

        for query in test_queries:
            run_similarity_search(collection, query)

        # Offer interactive mode
        response = input("\nEnter interactive search mode? (y/n): ").strip().lower()
        if response == 'y':
            interactive_search(collection)

    print("\nDone!")


if __name__ == "__main__":
    main()
