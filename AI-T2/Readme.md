# Enterprise Semantic Knowledge Management & Retrieval System

## Overview

This project is an end-to-end Retrieval-Augmented Generation (RAG) system that answers questions from internal corporate documentation.

The system processes an internal policy document, converts it into semantic embeddings, stores them in a Chroma vector database, retrieves the most relevant information for a user's query, and uses Google's Gemini LLM to generate grounded answers.

The project was built using LangChain, ChromaDB, Google Gemini, and FastAPI.

---

# Problem Statement

Organizations store a large amount of internal documentation, but searching through lengthy documents manually is inefficient.

The objective of this project was to build a semantic document retrieval system that can:

- Load corporate documentation
- Split documents into manageable chunks
- Generate semantic embeddings
- Store embeddings in a vector database
- Retrieve relevant information
- Generate grounded answers using an LLM
- Expose the system through a FastAPI endpoint

---

# Repository Structure

```
project/
│
├── data/
│   └── acme_corporate_policy.txt
│
├── api/
│   └── app.py
│
├── README.md
│
└── requirements.txt
```

---

# Technologies Used

- Python
- LangChain
- ChromaDB
- Google Gemini 1.5 Flash
- Gemini Embedding Model
- FastAPI
- Uvicorn

---

# Project Workflow

## Stage 1 – Document Loading

The corporate policy document is loaded using LangChain's `TextLoader`.

This converts the raw text file into LangChain Document objects.

---

## Stage 2 – Text Chunking

The document is split using `RecursiveCharacterTextSplitter`.

Configuration:

- Chunk Size: 400 characters
- Chunk Overlap: 50 characters

Chunk overlap helps preserve context when information spans across chunk boundaries.

---

## Stage 3 – Embedding Generation

Each chunk is converted into a semantic vector using Google's embedding model:

```
models/gemini-embedding-2
```

Unlike TF-IDF, embeddings capture the semantic meaning of text rather than only matching keywords.

---

## Stage 4 – Vector Database

The generated embeddings are stored inside an in-memory Chroma vector database.

When a user submits a query, the query is also converted into an embedding.

The retriever performs semantic similarity search and returns the most relevant chunk.

Retriever Configuration:

```
k = 1
```

meaning the most relevant chunk is retrieved.

---

## Stage 5 – Retrieval-Augmented Generation (RAG)

The retrieved context and the user's query are passed to the Gemini LLM.

A system prompt is used to reduce hallucinations by instructing the model to answer only from the retrieved context.

If the answer is unavailable, the model responds:

> "I cannot answer this based on the provided corporate documentation."

---

## Stage 6 – FastAPI Deployment

The RAG pipeline is exposed using FastAPI.

Endpoint:

```
POST /query
```

Input

```json
{
    "query":"What happens if temperatures reach 45°C?"
}
```

Output

```json
{
    "answer":"...",
    "confidence":0.95,
    "sources":[
        "acme_corporate_policy.txt"
    ]
}
```

---

# Evaluation

The system was evaluated using a set of representative corporate policy questions.

The following aspects were observed:

- Context Relevance
- Faithfulness (Groundedness)
- Answer Relevance
- Inference Latency

The evaluation confirmed that the retrieved context was relevant to the user query and that the generated answers remained grounded in the retrieved documentation.

---

# Anti-Hallucination Strategy

The project reduces hallucinations using Retrieval-Augmented Generation (RAG).

The language model receives only the retrieved document chunks as context and is explicitly instructed not to generate information outside the provided documentation.

Additionally,

- Temperature = 0.0
- Context-only prompting
- Retrieval before generation

help improve factual consistency.

---

# Running the Project

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 2. Set the API Key

```bash
export GOOGLE_API_KEY="YOUR_API_KEY"
```

---

## 3. Start the API

```bash
python -m uvicorn api.app:app --reload
```

---

## 4. Open Swagger

```
http://127.0.0.1:8000/docs
```

---

# Future Improvements

Possible improvements include:

- PDF document support
- Larger document collections
- Retrieval from multiple files
- Hybrid retrieval (BM25 + Embeddings)
- Cross-Encoder Re-ranking
- Automatic RAG evaluation using RAGAS
- Deployment on Render or Railway

---

# Key Learnings

Through this project, I learned:

- Document chunking strategies
- Semantic embeddings
- Vector databases
- Retrieval-Augmented Generation (RAG)
- Prompt engineering
- Reducing hallucinations
- FastAPI deployment
- End-to-end LLM application development

---

# Acknowledgements

This project was completed as part of the AI & Automation Task conducted by E-Cell.

It provided hands-on experience with modern Retrieval-Augmented Generation (RAG) systems, vector databases, semantic search, and FastAPI deployment.
