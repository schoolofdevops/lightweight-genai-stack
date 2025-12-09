#!/usr/bin/env python3
"""
ChromaDB Stats - Shows document and chunk counts
Usage: docker exec genai-app python /app/chroma_stats.py
"""

import chromadb
import os
from collections import Counter

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/app/chroma_db")


def main():
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collections = client.list_collections()

    print("=" * 60)
    print("CHROMADB STATISTICS")
    print("=" * 60)

    if not collections:
        print("No collections found.")
        return

    for col in collections:
        collection = client.get_collection(col.name)
        count = collection.count()

        print(f"\nCollection: {col.name}")
        print("-" * 40)
        print(f"  Total chunks: {count:,}")

        if count > 0:
            # Get all metadata to count unique sources
            results = collection.get(include=["metadatas"])
            sources = [m.get("source", "unknown") for m in results["metadatas"]]
            source_counts = Counter(sources)

            print(f"  Unique documents: {len(source_counts)}")
            print(f"\n  Documents breakdown:")
            for source, chunk_count in source_counts.most_common():
                # Extract just filename from path
                filename = os.path.basename(source)
                print(f"    - {filename}: {chunk_count:,} chunks")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
