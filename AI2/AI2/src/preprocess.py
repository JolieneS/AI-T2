"""
Stage 1: Document Ingestion & Text Segmentation
Loads PDFs and TXT files from ./data/, cleans and chunks them.
"""

import os, re, json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR      = "./data"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50
OUTPUT_PATH   = "./models/chunks.json"


def clean_text(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    text = re.sub(r'(\d+\s*/\s*\d+)', '', text)
    text = re.sub(r'[-─═]{5,}', '', text)
    return text.strip()


def classify_document(text):
    t = text.lower()
    if any(k in t for k in ["standard operating", "sop", "procedure"]): return "SOP"
    if any(k in t for k in ["compliance", "regulation"]): return "Compliance"
    if any(k in t for k in ["troubleshoot", "incident", "error log"]): return "TroubleshootingLog"
    if any(k in t for k in ["policy", "vpn", "security"]): return "Policy"
    return "General"


def load_documents():
    raw_docs = []
    pdf_files = list(Path(DATA_DIR).glob("**/*.pdf"))
    txt_files = list(Path(DATA_DIR).glob("**/*.txt"))
    print(f"Found {len(pdf_files)} PDF(s) and {len(txt_files)} TXT(s)")
    for fp in pdf_files + txt_files:
        try:
            loader = PyPDFLoader(str(fp)) if fp.suffix == ".pdf" else TextLoader(str(fp), encoding="utf-8")
            loaded = loader.load()
            for doc in loaded:
                doc.page_content = clean_text(doc.page_content)
                doc.metadata["doc_type"] = classify_document(doc.page_content)
            raw_docs.extend(loaded)
            print(f"  Loaded: {fp.name} ({len(loaded)} page(s))")
        except Exception as e:
            print(f"  Skipped {fp.name}: {e}")
    return raw_docs


def chunk_documents(raw_docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(raw_docs)


if __name__ == "__main__":
    os.makedirs("./models", exist_ok=True)
    raw_docs = load_documents()
    chunks   = chunk_documents(raw_docs)
    with open(OUTPUT_PATH, "w") as f:
        json.dump([{"content": c.page_content, "metadata": c.metadata} for c in chunks], f, indent=2)
    print(f"\n✅ Stage 1 done — {len(chunks)} chunks saved to {OUTPUT_PATH}")
