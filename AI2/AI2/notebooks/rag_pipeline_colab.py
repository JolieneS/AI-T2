# ╔══════════════════════════════════════════════════════════════════════╗
# ║  Enterprise Semantic Knowledge Management — Google Colab Notebook   ║
# ║  Stack : Groq (llama-3.1-8b-instant) + HuggingFace + ChromaDB      ║
# ╚══════════════════════════════════════════════════════════════════════╝
# Each ## CELL N ## block is one separate Colab cell.

# ── CELL 1 — Install dependencies ────────────────────────────────────────────
"""
!pip install -q \
    langchain \
    langchain-community \
    langchain-chroma \
    langchain-groq \
    langchain-text-splitters \
    "langchain-classic>=1.0.0" \
    sentence-transformers \
    "chromadb==0.4.24" \
    groq \
    pypdf \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    nest-asyncio \
    pandas \
    pyngrok
"""

# ── CELL 2 — Set API Key ─────────────────────────────────────────────────────
"""
import os

os.environ["GROQ_API_KEY"] = "your_groq_api_key_here"

print("GROQ_API_KEY set:", bool(os.environ.get("GROQ_API_KEY")))
"""

# ── CELL 3 — Test Groq connection ────────────────────────────────────────────
"""
from groq import Groq
import os

client = Groq(api_key=os.environ["GROQ_API_KEY"])
response = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Say hello in one sentence."}]
)
print("Groq works:", response.choices[0].message.content)
"""

# ── CELL 4 — Create folders and upload PDF ───────────────────────────────────
"""
import os, shutil
from google.colab import files

for d in ["./data", "./models"]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d, exist_ok=True)

print("Upload your PDF document(s) now:")
uploaded = files.upload()

for name, data in uploaded.items():
    dest = f"./data/{name}"
    with open(dest, "wb") as f:
        f.write(data)
    print(f"Saved: {dest}")

print("\nFiles in ./data/:", os.listdir("./data"))
"""

# ── CELL 5 — Stage 1: Document ingestion & chunking ──────────────────────────
"""
import os, re, json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

DATA_DIR      = "./data"
CHUNK_SIZE    = 400
CHUNK_OVERLAP = 50

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

raw_docs = []
pdf_files = list(Path(DATA_DIR).glob("**/*.pdf"))
txt_files = list(Path(DATA_DIR).glob("**/*.txt"))

print(f"Found {len(pdf_files)} PDF(s) and {len(txt_files)} TXT(s)")

for fp in pdf_files + txt_files:
    try:
        if fp.suffix == ".pdf":
            loader = PyPDFLoader(str(fp))
        else:
            loader = TextLoader(str(fp), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.page_content = clean_text(doc.page_content)
            doc.metadata["doc_type"] = classify_document(doc.page_content)
        raw_docs.extend(loaded)
        print(f"  Loaded: {fp.name} ({len(loaded)} page(s))")
    except Exception as e:
        print(f"  Skipped {fp.name}: {e}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " ", ""]
)
chunks = splitter.split_documents(raw_docs)

with open("./models/chunks.json", "w") as f:
    json.dump([{"content": c.page_content, "metadata": c.metadata} for c in chunks], f, indent=2)

print(f"\nS1 — {len(chunks)} chunks from {len(raw_docs)} page(s)")
"""

# ── CELL 6 — Stage 2: HuggingFace embeddings + ChromaDB ─────────────────────
"""
import json, os, shutil
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

os.system("pip install -q chromadb==0.5.0")

from langchain_chroma import Chroma

CHROMA_DIR  = "./models/chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

with open("./models/chunks.json") as f:
    data = json.load(f)
chunks = [Document(page_content=d["content"], metadata=d["metadata"]) for d in data]

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name=EMBED_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)

if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name="enterprise_kb"
)

results = vector_store.similarity_search(chunks[0].page_content[:30], k=2)
print("\n── Top-2 retrieval sanity check ──")
for i, r in enumerate(results, 1):
    print(f"  [{i}] {r.page_content[:100]}…")

print(f"\nS2 — index saved at {CHROMA_DIR}")
"""

# ── CELL 7 — Stage 3: Groq RAG chain ─────────────────────────────────────────
"""
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    api_key=os.environ["GROQ_API_KEY"],
    max_tokens=512
)

system_prompt = (
    "You are an enterprise knowledge management assistant.\n"
    "Answer questions ONLY using the retrieved context below.\n"
    "If the answer is not in the context, respond exactly:\n"
    "'I cannot answer this based on the provided corporate documentation.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_inner = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class RAGChain:
    def invoke(self, inputs):
        question = inputs["input"]
        docs = retriever.invoke(question)
        answer = rag_chain_inner.invoke(question)
        return {"answer": answer, "context": docs}

rag_chain = RAGChain()
print("OK")
"""

# ── CELL 8 — Test query ───────────────────────────────────────────────────────
"""
test_query = "Summarize the main policy covered in this document."

response = rag_chain.invoke({"input": test_query})

print("Answer:", response["answer"])
print("\nSources:")
for doc in response["context"]:
    src  = doc.metadata.get("source", "?")
    page = doc.metadata.get("page", "?")
    dtype = doc.metadata.get("doc_type", "?")
    print(f"  • {os.path.basename(str(src))} | page {page} | type: {dtype}")
"""

# ── CELL 9 — Stage 4: Pipeline evaluation ────────────────────────────────────
"""
import time
import pandas as pd

EVAL_SET = [
    {
        "question": "What is the main topic of this document?",
        "expected_keywords": ["policy", "compliance", "procedure"]
    },
    {
        "question": "What are the key rules or regulations mentioned?",
        "expected_keywords": ["rule", "regulation", "must", "shall"]
    },
    {
        "question": "What happens in case of a violation?",
        "expected_keywords": ["violation", "penalty", "action"]
    },
    {
        "question": "Who is responsible for enforcement?",
        "expected_keywords": ["manager", "team", "officer"]
    },
    {
        "question": "Tell me about cricket scores.",
        "expected_keywords": []
    },
]

def build_chain(search_type, search_kwargs):
    r = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    def invoke(inputs):
        question = inputs["input"]
        docs = r.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        filled_prompt = prompt.format_messages(context=context, input=question)
        answer = llm.invoke(filled_prompt).content
        return {"answer": answer, "context": docs}
    return type("Chain", (), {"invoke": staticmethod(invoke)})()

variants = {
    "standard":  build_chain("similarity", {"k": 3}),
    "reranked":  build_chain("mmr", {"k": 3, "fetch_k": 10, "lambda_mult": 0.6}),
    "threshold": build_chain("similarity_score_threshold", {"score_threshold": 0.35, "k": 3}),
}

all_rows = []

for variant_name, chain in variants.items():
    print(f"\nEvaluating: {variant_name.upper()}")
    for sample in EVAL_SET:
        start    = time.time()
        response = chain.invoke({"input": sample["question"]})
        latency  = round(time.time() - start, 3)
        answer   = response.get("answer", "")
        context  = " ".join(doc.page_content for doc in response.get("context", []))

        cr = "Relevant" if len(context.strip()) > 50 else "Not Relevant"

        tokens  = answer.lower().split()
        ctx_set = set(context.lower().split())
        f = round(sum(1 for t in tokens if t in ctx_set) / max(len(tokens), 1), 3)

        if sample["expected_keywords"]:
            ar = round(
                sum(1 for kw in sample["expected_keywords"] if kw.lower() in answer.lower())
                / len(sample["expected_keywords"]), 3
            )
        else:
            ar = 0.0

        qr = 0 if "i cannot answer" in answer.lower() else 1

        all_rows.append({
            "Variant":    variant_name,
            "Question":   sample["question"][:55] + "…",
            "CR":         cr,
            "F":          f,
            "AR":         ar,
            "Latency(s)": latency,
            "QR":         qr
        })
        print(f"  Q: {sample['question'][:45]}… | F={f} AR={ar} L={latency}s QR={qr}")

df = pd.DataFrame(all_rows)

summary = (
    df.groupby("Variant")
    .agg(
        Avg_Faithfulness=("F", "mean"),
        Avg_Ans_Relevance=("AR", "mean"),
        Avg_Latency_s=("Latency(s)", "mean"),
        Query_Resolution_Rate=("QR", "mean"),
        CR_Relevant_Pct=("CR", lambda x: round((x == "Relevant").mean(), 3))
    )
    .round(3)
    .reset_index()
)

print("\n\n══ PER-QUESTION RESULTS ══")
print(df.to_string(index=False))

print("\n\n══ VARIANT SUMMARY ══")
print(summary.to_string(index=False))
"""

# ── CELL 10 — Print metric summary ───────────────────────────────────────────
"""
print("   Pipeline eval summary")

for _, row in summary.iterrows():
    print(f"\n  Variant             : {row['Variant'].upper()}")
    print(f"  Faithfulness  (F)   : {row['Avg_Faithfulness']}")
    print(f"  Answer Relevance(AR): {row['Avg_Ans_Relevance']}")
    print(f"  Avg Latency         : {row['Avg_Latency_s']}s")
    print(f"  Query Resolution    : {row['Query_Resolution_Rate'] * 100:.0f}%")
    print(f"  Context Relevant    : {row['CR_Relevant_Pct'] * 100:.0f}%")
"""

# ── CELL 11 — Stage 5: FastAPI server ────────────────────────────────────────
"""
import nest_asyncio, os, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
from threading import Thread

nest_asyncio.apply()

app = FastAPI(
    title="Enterprise Semantic Knowledge Management API",
    description="Grounded RAG system — Groq + HuggingFace Embeddings + ChromaDB",
    version="2.0.0"
)

class QueryRequest(BaseModel):
    query: str
    variant: Optional[str] = "standard"

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]
    latency_s: float
    variant: str

@app.get("/health")
def health():
    return {"status": "ok", "pipeline_ready": rag_chain is not None}

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    active = rag_chain
    if request.variant == "reranked":
        r = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6})
        docs = r.invoke(request.query)
        context = "\n\n".join(doc.page_content for doc in docs)
        filled = prompt.format_messages(context=context, input=request.query)
        answer = llm.invoke(filled).content
        return QueryResponse(answer=answer, confidence=0.0, sources=[], latency_s=0.0, variant=request.variant)
    elif request.variant == "threshold":
        r = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.35, "k": 3})
        docs = r.invoke(request.query)
        context = "\n\n".join(doc.page_content for doc in docs)
        filled = prompt.format_messages(context=context, input=request.query)
        answer = llm.invoke(filled).content
        return QueryResponse(answer=answer, confidence=0.0, sources=[], latency_s=0.0, variant=request.variant)

    start    = time.time()
    response = active.invoke({"input": request.query})
    latency  = round(time.time() - start, 3)

    answer  = response.get("answer", "")
    context = ""
    sources = []

    for doc in response.get("context", []):
        src   = doc.metadata.get("source", "document")
        page  = doc.metadata.get("page", None)
        label = os.path.basename(str(src))
        if page is not None:
            label += f" (p.{int(page)+1})"
        sources.append(label)
        context += " " + doc.page_content

    sources    = list(dict.fromkeys(sources))
    tokens     = answer.lower().split()
    ctx_set    = set(context.lower().split())
    matched    = sum(1 for t in tokens if t in ctx_set)
    confidence = round(matched / max(len(tokens), 1), 2)

    if "i cannot answer" in answer.lower():
        confidence = 0.0

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        latency_s=latency,
        variant=request.variant or "standard"
    )

def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error")

thread = Thread(target=run_api, daemon=True)
thread.start()
print("FastAPI running at http://127.0.0.1:8000")
print("   Swagger UI → http://127.0.0.1:8000/docs")
"""

# ── CELL 12 — Test API locally ────────────────────────────────────────────────
"""
import requests

test_cases = [
    {"query": "What is the main topic of this document?",  "variant": "standard"},
    {"query": "What happens in case of a violation?",      "variant": "reranked"},
    {"query": "Tell me about cricket scores.",             "variant": "standard"},
]

for payload in test_cases:
    r    = requests.post("http://127.0.0.1:8000/query", json=payload)
    data = r.json()
    print(f"\nQ  : {payload['query']}")
    print(f"A  : {data['answer'][:200]}")
    print(f"     confidence={data['confidence']} | sources={data['sources']} | latency={data['latency_s']}s")
"""

# ── CELL 13 — ngrok public URL ────────────────────────────────────────────────
"""
from pyngrok import ngrok

NGROK_TOKEN = "your_ngrok_token_here"
ngrok.set_auth_token(NGROK_TOKEN)

public_url = ngrok.connect(8000, bind_tls=True)
print("Live URL  :", public_url.public_url)
print("   Swagger   :", public_url.public_url + "/docs")
print("   Endpoint  :", public_url.public_url + "/query")
"""

# ── CELL 14 — Live demo: upload unseen PDF and re-ingest ─────────────────────
"""
from google.colab import files
import shutil, os, json
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

CHROMA_DIR = "/tmp/chroma_db"

print("Upload a new unseen PDF for live demo:")
uploaded = files.upload()

for name, data in uploaded.items():
    with open(f"./data/{name}", "wb") as f:
        f.write(data)
    print(f"Saved: ./data/{name}")

raw_docs = []
for fp in list(Path("./data").glob("**/*.pdf")) + list(Path("./data").glob("**/*.txt")):
    try:
        loader = PyPDFLoader(str(fp)) if fp.suffix == ".pdf" else TextLoader(str(fp), encoding="utf-8")
        loaded = loader.load()
        for doc in loaded:
            doc.page_content = clean_text(doc.page_content)
        raw_docs.extend(loaded)
    except Exception as e:
        print(f"Skipped {fp.name}: {e}")

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks   = splitter.split_documents(raw_docs)

if os.path.exists(CHROMA_DIR):
    shutil.rmtree(CHROMA_DIR)
os.makedirs(CHROMA_DIR, exist_ok=True)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=CHROMA_DIR,
    collection_name="enterprise_kb"
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
rag_chain_inner = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class RAGChain:
    def invoke(self, inputs):
        question = inputs["input"]
        docs = retriever.invoke(question)
        answer = rag_chain_inner.invoke(question)
        return {"answer": answer, "context": docs}

rag_chain = RAGChain()

print(f"\nRe-indexed {len(chunks)} chunks. RAG chain updated — ready for demo queries!")
"""
