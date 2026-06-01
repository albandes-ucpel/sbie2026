import os
from fastapi import FastAPI, Depends
from utils.auth import enforce_api_key
from .schemas import Query
import pandas as pd

app = FastAPI(title="Psico-RAG PoC", version="0.1.0")

FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/features.parquet")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/answer")
def answer(q: Query, _: None = Depends(enforce_api_key)):
    df = pd.read_parquet(FEATURES_PATH)
    if q.class_id:
        df = df.query("class_id == @q.class_id").copy()
    # score simples (baseline interpretável)
    df["risk_score"] = (1 - df["presence_30d"]) + (-df["grade_slope"]).clip(lower=0) + df["incidents_90d"]*0.5
    topk = df.nlargest(q.k, "risk_score")[["student_pid","class_id","risk_score","presence_30d","grade_slope","incidents_90d"]]
    return {
        "class_id": q.class_id,
        "topk": topk.to_dict(orient="records")
    }
