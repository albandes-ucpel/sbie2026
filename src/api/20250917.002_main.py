import os
from fastapi import FastAPI, Depends
from utils.auth import enforce_api_key
from .schemas import Query
import pandas as pd
from rag.index import get_index
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Psico-RAG PoC", version="0.1.0")
FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/features.parquet")

# constrói índice (em produção: criar fora e injetar)
index = get_index()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/search")
def search(q: Query, _: None = Depends(enforce_api_key)):
    retriever = index.as_retriever(similarity_top_k=q.k)
    passages = retriever.retrieve(q.query)
    return {
        "chunks":[
            {"text":p.get_text(), "meta":getattr(p.node, "metadata", {})}
            for p in passages
        ]
    }

@app.post("/answer")
def answer(q: Query, _: None = Depends(enforce_api_key)):
    df = pd.read_parquet(FEATURES_PATH)
    if q.class_id:
        df = df.query("class_id == @q.class_id").copy()
    df["risk_score"] = (1 - df["presence_30d"]) + (-df["grade_slope"]).clip(lower=0) + df["incidents_90d"]*0.5
    topk = df.nlargest(q.k, "risk_score")[["student_pid","class_id","risk_score","presence_30d","grade_slope","incidents_90d"]]

    retriever = index.as_retriever(similarity_top_k=3)
    passages = retriever.retrieve(f"indicadores de risco evasão escolar presença notas incidents {q.class_id or ''}")

    return {
        "class_id": q.class_id,
        "topk": topk.to_dict(orient="records"),
        "evidencias": [p.get_text() for p in passages]
    }

