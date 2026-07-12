"""
Stage 4: Pipeline Evaluation
Evaluates all 3 variants across CR, F, AR, L, QR metrics.
"""

import time, json, os
import pandas as pd
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from src.features import build_embeddings, load_vector_store

REPORT_PATH = "./models/evaluation_report.json"

EVAL_SET = [
    {"question": "What is the main topic of this document?",
     "expected_keywords": ["policy", "compliance", "procedure"]},
    {"question": "What are the key rules or regulations mentioned?",
     "expected_keywords": ["rule", "regulation", "must", "shall"]},
    {"question": "What happens in case of a violation?",
     "expected_keywords": ["violation", "penalty", "action"]},
    {"question": "Who is responsible for enforcement?",
     "expected_keywords": ["manager", "team", "officer"]},
    {"question": "Tell me about cricket scores.",
     "expected_keywords": []},
]

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


def build_chain(vector_store, llm, search_type, search_kwargs):
    r = vector_store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    def invoke(inputs):
        question = inputs["input"]
        docs     = r.invoke(question)
        context  = "\n\n".join(doc.page_content for doc in docs)
        filled   = PROMPT.format_messages(context=context, input=question)
        answer   = llm.invoke(filled).content
        return {"answer": answer, "context": docs}
    return type("Chain", (), {"invoke": staticmethod(invoke)})()


if __name__ == "__main__":
    embeddings   = build_embeddings()
    vector_store = load_vector_store(embeddings)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.0,
                   api_key=os.environ["GROQ_API_KEY"], max_tokens=512)

    variants = {
        "standard":  build_chain(vector_store, llm, "similarity", {"k": 3}),
        "reranked":  build_chain(vector_store, llm, "mmr", {"k": 3, "fetch_k": 10, "lambda_mult": 0.6}),
        "threshold": build_chain(vector_store, llm, "similarity_score_threshold", {"score_threshold": 0.35, "k": 3}),
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
            f  = round(sum(1 for t in tokens if t in ctx_set) / max(len(tokens), 1), 3)
            ar = round(sum(1 for kw in sample["expected_keywords"] if kw.lower() in answer.lower())
                       / max(len(sample["expected_keywords"]), 1), 3) if sample["expected_keywords"] else 0.0
            qr = 0 if "i cannot answer" in answer.lower() else 1

            all_rows.append({"Variant": variant_name, "Question": sample["question"][:55] + "…",
                             "CR": cr, "F": f, "AR": ar, "Latency(s)": latency, "QR": qr})
            print(f"  Q: {sample['question'][:45]}… | F={f} AR={ar} L={latency}s QR={qr}")

    df = pd.DataFrame(all_rows)
    summary = (df.groupby("Variant")
               .agg(Avg_Faithfulness=("F","mean"), Avg_Ans_Relevance=("AR","mean"),
                    Avg_Latency_s=("Latency(s)","mean"), Query_Resolution_Rate=("QR","mean"),
                    CR_Relevant_Pct=("CR", lambda x: round((x=="Relevant").mean(),3)))
               .round(3).reset_index())

    print("\n\n══ PER-QUESTION RESULTS ══")
    print(df.to_string(index=False))
    print("\n\n══ VARIANT SUMMARY ══")
    print(summary.to_string(index=False))

    os.makedirs("./models", exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump({"per_question": df.to_dict(orient="records"),
                   "summary": summary.to_dict(orient="records")}, f, indent=2)
    print(f"\n Report saved → {REPORT_PATH}")
