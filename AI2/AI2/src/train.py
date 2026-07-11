"""
Stage 3: LLM Inference & Context Orchestration
Builds RAG chain with Groq (llama-3.1-8b-instant).
Three retrieval variants: standard, reranked (MMR), threshold.
"""

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from src.features import build_embeddings, load_vector_store

GROQ_MODEL  = "llama-3.1-8b-instant"
TEMPERATURE = 0.0

SYSTEM_PROMPT = (
    "You are an enterprise knowledge management assistant.\n"
    "Answer questions ONLY using the retrieved context below.\n"
    "If the answer is not in the context, respond exactly:\n"
    "'I cannot answer this based on the provided corporate documentation.'\n\n"
    "Context:\n{context}"
)

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{input}"),
])


def build_llm():
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=TEMPERATURE,
        api_key=os.environ["GROQ_API_KEY"],
        max_tokens=512
    )


def build_pipeline(variant="standard"):
    embeddings   = build_embeddings()
    vector_store = load_vector_store(embeddings)
    llm          = build_llm()

    if variant == "reranked":
        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 3, "fetch_k": 10, "lambda_mult": 0.6}
        )
    elif variant == "threshold":
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.35, "k": 3}
        )
    else:
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain_inner = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | PROMPT | llm | StrOutputParser()
    )

    class RAGChain:
        def invoke(self, inputs):
            question = inputs["input"]
            docs     = retriever.invoke(question)
            answer   = chain_inner.invoke(question)
            return {"answer": answer, "context": docs}

    return RAGChain()


if __name__ == "__main__":
    chain = build_pipeline("standard")
    print("RAG chain ready. Type queries (or 'quit'):\n")
    while True:
        q = input("Query > ").strip()
        if q.lower() in ("quit", "q"): break
        r = chain.invoke({"input": q})
        print(f"Answer: {r['answer']}\n")
