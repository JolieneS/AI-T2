"""
Stage 5: FastAPI Production Server
Endpoints: POST /query, GET /health, GET /docs (Swagger)
"""

import os, time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.features import build_embeddings, load_vector_store

app = FastAPI(
    title="Enterprise Semantic Knowledge Management API",
    description="Grounded RAG system — Groq + HuggingFace Embeddings + ChromaDB",
    version="2.0.0"
)

SYSTEM_PROMPT = (
    "You are an enterprise knowledge management assistant.\n"
    "Answer questions ONLY using the retrieved context below.\n"
    "If the answer is not in the context, respond exactly:\n"
    "'I cannot answer this based on the provided corporate documentation.'\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])

embeddings   = build_embeddings()
vector_store = load_vector_store(embeddings)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0,
               api_key=os.environ["GROQ_API_KEY"], max_tokens=512)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain_inner = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

class RAGChain:
    def invoke(self, inputs):
        question = inputs["input"]
        docs     = retriever.invoke(question)
        answer   = rag_chain_inner.invoke(question)
        return {"answer": answer, "context": docs}

rag_chain = RAGChain()


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
    return {"status": "ok", "pipeline_ready": True}


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if request.variant == "reranked":
        r = vector_store.as_retriever(search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6})
    elif request.variant == "threshold":
        r = vector_store.as_retriever(search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.35, "k": 3})
    else:
        r = retriever

    start    = time.time()
    docs     = r.invoke(request.query)
    context  = "\n\n".join(doc.page_content for doc in docs)
    filled   = prompt.format_messages(context=context, input=request.query)
    answer   = llm.invoke(filled).content
    latency  = round(time.time() - start, 3)

    sources = []
    for doc in docs:
        src   = doc.metadata.get("source", "document")
        page  = doc.metadata.get("page", None)
        label = os.path.basename(str(src))
        if page is not None:
            label += f" (p.{int(page)+1})"
        sources.append(label)
    sources = list(dict.fromkeys(sources))

    tokens     = answer.lower().split()
    ctx_set    = set(context.lower().split())
    confidence = round(sum(1 for t in tokens if t in ctx_set) / max(len(tokens), 1), 2)
    if "i cannot answer" in answer.lower():
        confidence = 0.0

    return QueryResponse(answer=answer, confidence=confidence,
                         sources=sources, latency_s=latency,
                         variant=request.variant or "standard")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.app:app", host="0.0.0.0", port=8000, reload=True)
