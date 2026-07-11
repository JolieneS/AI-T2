"""
Stage 2: Embedding Generation & Indexing
Converts chunks to HuggingFace vectors and persists to ChromaDB.
"""

import os, json, shutil
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CHROMA_DIR  = "./models/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNKS_PATH = "./models/chunks.json"


def build_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )


def build_vector_store(chunks, embeddings):
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name="enterprise_kb"
    )


def load_vector_store(embeddings):
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="enterprise_kb"
    )


if __name__ == "__main__":
    with open(CHUNKS_PATH) as f:
        data = json.load(f)
    chunks = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]
    print("Loading embedding model...")
    embeddings   = build_embeddings()
    vector_store = build_vector_store(chunks, embeddings)
    results = vector_store.similarity_search(chunks[0].page_content[:30], k=2)
    print("\n── Top-2 sanity check ──")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {r.page_content[:100]}…")
    print(f"\n✅ Stage 2 done — index at {CHROMA_DIR}")
