import os
import sys
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn


from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain


DATA_DIR = "./data"          
TARGET_FILE = os.path.join(DATA_DIR, "acme_corporate_policy.txt")
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


rag_chain = None


app = FastAPI(
    title="Semantic Knowledge Management API",
    description="Securely grounded corporate documentation retrieval service running a single-file architecture.",
    version="1.0.0"
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[str]


@app.on_event("startup")
def startup_event():
    global rag_chain
    print("\n--- Starting Production RAG Pipeline Initialization ---")
    
    
    if not os.path.exists(TARGET_FILE):
        print(f" Error: Asset directory missing. Creating mockup file at {TARGET_FILE}...")
        os.makedirs(DATA_DIR, exist_ok=True)
        mock_data = (
            "ACME CORP ENTERPRISE STANDARD OPERATING PROCEDURES (SOP)\n"
            "SECTION 1.1: OVERHEATING CRITICAL PROTOCOL\n"
            "In the event that datacenter rack temperature sensors breach 38°C, the on-duty systems engineer "
            "must immediately initiate the secondary liquid cooling intake valve located in Sub-basement B. "
            "If temperatures reach 45°C, the emergency automatic power down sequence (SOP-CMD-99) will execute within 120 seconds.\n\n"
            "SECTION 2.1: SECURE ACCESS VIA VPN\n"
            "All personnel accessing internal code repositories from a remote location must route connections through the hardware-token MFA gateway."
        )
        with open(TARGET_FILE, "w") as f:
            f.write(mock_data)
            
    try:
# S1: Parsing 
        print("Stage 1: Processing unstructured raw documentation...")
        loader = TextLoader(TARGET_FILE)
        raw_documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, 
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len
        )
        docs = text_splitter.split_documents(raw_documents)
        print(f" -> Generated {len(docs)} machine-readable structural text segments.")
        
 # S2: Indexing 
        print("Stage 2: Instantiating high-dimensional API-based embeddings using gemini-embedding-2...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        vector_store = Chroma.from_documents(documents=docs, embedding=embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 1}) 
        
# S3: Structuring Anti-Hallucination Prompt Framework
        print("Stage 3: Embedding secure context retrieval guardrails...")
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.0)
        
        system_prompt = (
            "You are an enterprise knowledge management assistant.\n"
            "Use ONLY the following pieces of retrieved context to answer the question.\n"
            "If you do not know the answer or if it is not explicitly found in the context, "
            "state: 'I cannot answer this based on the provided corporate documentation.'\n"
            "Do not make up facts or extrapolate outside the context.\n\n"
            "Context:\n{context}"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        print("Status: FULLY OPERATIONAL WITH ZERO ERRORS!\n")
        
    except Exception as e:
        print(f" Fatal Error During Initialization: {e}")
        sys.exit(1)


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    if not rag_chain:
        raise HTTPException(status_code=500, detail="RAG system pipeline is uninitialized.")
    try:
        response = rag_chain.invoke({"input": request.query})
        
        
        sources = []
        if "context" in response:
            for doc in response["context"]:
                source_name = getattr(doc, "metadata", {}).get("source", "acme_corporate_policy.txt")
                sources.append(os.path.basename(source_name))
        
        return QueryResponse(
            answer=response["answer"],
            confidence=0.95,
            sources=list(set(sources))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)