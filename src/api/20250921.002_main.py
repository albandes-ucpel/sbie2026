# src/api/main.py
import os, glob, re, uuid
from typing import List, Dict, Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import joblib  # para carregar o modelo treinado

# Auth (usa seu util atual; precisa aceitar alias="X-API-Key" lá)
from utils.auth import enforce_api_key

# RAG / Vetores
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# >>> ADIÇÃO: gerador de relatório em linguagem natural
from llm.report import generate_report

# ===== Carregar variáveis de ambiente =====
load_dotenv()

# ===== Config =====
APP_TITLE = "Psico-RAG PoC"
APP_VERSION = "0.1.0"

# Coleção de literatura / artigos (pública)
COLLECTION = os.getenv("QDRANT_COLLECTION", "literatura")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DIM = int(os.getenv("EMBED_DIM", "1536"))  # text-embedding-3-small

# Coleção de prontuário (privada)
NOTES_COLLECTION = os.getenv("QDRANT_NOTES_COLLECTION", "prontuario_privado")
NOTES_EMBED_MODEL = os.getenv("NOTES_EMBED_MODEL", "BAAI/bge-m3")  # local
NOTES_DIM = int(os.getenv("NOTES_EMBED_DIM", "1024"))  # ajuste ao modelo local

FEATURES_PATH = os.getenv("FEATURES_PATH", "data/processed/features.parquet")
MODEL_PATH = os.getenv("MODEL_PATH", "models/baseline.joblib")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Dependência de auth (aplicada nas rotas protegidas)
AUTH = [Depends(enforce_api_key)]

# ===== App =====
app = FastAPI(title=APP_TITLE, version=APP_VERSION)


# ---------- Helpers ----------
def _clean_text(t: str) -> str:
    t = t.replace("\r", "")
    t = re.sub(r"[ \t]+", " ", t)
    return t.strip()

def _chunk(text: str, size=800, overlap=120) -> List[str]:
    text = _clean_text(text)
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size]); i += (size - overlap)
    return [c for c in out if c]

def _get_client() -> QdrantClient:
    url = QDRANT_URL
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, api_key=QDRANT_API_KEY)
    return QdrantClient(path=url)

def _ensure_collection(client: QdrantClient, name: str, dim: int) -> None:
    cols = [c.name for c in client.get_collections().collections]
    if name not in cols:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

class _Emb:
    """Embeddings para literatura/artigos (pode usar OpenAI)."""
    def __init__(self, model: str | None = None):
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY não definido no .env ou ambiente.")
        self.client = OpenAI(api_key=key)
        self.model = model or EMBED_MODEL
    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

def _notes_embed(texts: List[str]) -> List[List[float]]:
    """Embeddings **locais** para prontuário (SentenceTransformers)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        # Fallback: nenhuma nota será retornada se modelo local não existir
        raise RuntimeError("sentence-transformers não instalado. pip install sentence-transformers")
    model_name = NOTES_EMBED_MODEL
    model = SentenceTransformer(model_name)
    vecs = model.encode(texts)
    return [v.tolist() for v in vecs]

def _redact_pii(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    patterns = [
        re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b"),  # CPF
        re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
        re.compile(r"\+?\d{2}\s?\(?\d{2}\)?\s?\d{4,5}-?\d{4}")
    ]
    out = text
    for p in patterns:
        out = p.sub("[REDACTED]", out)
    return out


# ---------- Schemas ----------
class ReindexIn(BaseModel):
    corpus_dir: str = "data/corpus"

class SearchIn(BaseModel):
    query: str
    k: int = 5

class AnswerIn(BaseModel):
    query: str
    k: int = 5
    class_id: str | None = None
    mode: str = "baseline"   # "baseline" ou "model"
    include_notes: bool = False  # incluir "glimpses" do prontuário (RBAC exigido)


# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    # Torna o startup resiliente: não derruba a API se Qdrant estiver off
    try:
        client = _get_client()
        _ensure_collection(client, COLLECTION, DIM)
        # Não cria coleção de notas aqui; criaremos sob demanda quando necessário
    except Exception as e:
        print(f"[WARN] Qdrant indisponível no startup: {e}")


# ---------- Rotas ----------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/rag/reindex", dependencies=AUTH)
def rag_reindex(body: ReindexIn):
    try:
        paths = sorted(
            glob.glob(os.path.join(body.corpus_dir, "**/*.txt"), recursive=True) +
            glob.glob(os.path.join(body.corpus_dir, "**/*.md"),  recursive=True)
        )

        client = _get_client()
        _ensure_collection(client, COLLECTION, DIM)

        if not paths:
            return {"ok": True, "chunks_indexed": 0}

        emb = _Emb()
        chunks: List[str] = []
        payloads: List[Dict] = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read()
            for idx, ch in enumerate(_chunk(raw)):
                chunks.append(ch)
                payloads.append({"text": ch, "source": os.path.relpath(p), "meta": {"chunk": idx}})

        total = 0
        for i in range(0, len(chunks), 64):
            vecs = emb.embed(chunks[i:i+64])
            points = [
                PointStruct(id=str(uuid.uuid4()), vector=v, payload=pl)
                for v, pl in zip(vecs, payloads[i:i+64])
            ]
            client.upsert(collection_name=COLLECTION, points=points)
            total += len(points)

        return {"ok": True, "chunks_indexed": total}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"reindex failed: {e}")


@app.post("/search", dependencies=AUTH)
def search(body: SearchIn):
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query vazia")

    try:
        client = _get_client()
        _ensure_collection(client, COLLECTION, DIM)

        emb = _Emb()
        qvec = emb.embed([body.query])[0]

        res = client.search(
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=body.k,
            with_payload=True,
        )
        results = [
            {
                "id": r.id,
                "score": float(r.score),
                "text": (r.payload or {}).get("text"),
                "source": (r.payload or {}).get("source"),
                "meta": (r.payload or {}).get("meta", {}),
            }
            for r in res
        ]
        return {"query": body.query, "k": body.k, "results": results}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"search failed: {e}")


@app.post("/answer", dependencies=AUTH)
def answer(
    body: AnswerIn,
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role")
):
    """
    Retorna Top-K alunos por risco + evidências da literatura e (opcional) glimpses do prontuário.
    mode:
      - baseline: regra interpretável
      - model:    modelo treinado (joblib)
    include_notes:
      - true  → busca um trecho do prontuário por aluno (apenas se X-User-Role=psych)
      - false → não inclui notas
    """
    # ---- features
    if not os.path.exists(FEATURES_PATH):
        raise HTTPException(status_code=500, detail=f"Features não encontradas em {FEATURES_PATH}. Rode o ETL.")

    try:
        df = pd.read_parquet(FEATURES_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Falha ao ler features: {e}")

    if body.class_id:
        df = df.query("class_id == @body.class_id").copy()
        if df.empty:
            # >>> também retornaremos report_text=None aqui para manter contrato
            return {
                "class_id": body.class_id,
                "mode": body.mode,
                "topk": [],
                "evidencias": [],
                "report_text": None,
                "msg": "Nenhum registro para esta turma."
            }

    # ---- risco
    if body.mode.lower() == "model":
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(status_code=500, detail=f"Modelo não encontrado em {MODEL_PATH}. Treine primeiro.")
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Falha ao carregar modelo: {e}")

        if hasattr(model, "feature_names_in_"):
            feats_needed = list(model.feature_names_in_)
            faltantes = [c for c in feats_needed if c not in df.columns]
            if faltantes:
                raise HTTPException(status_code=500, detail=f"Features ausentes no parquet: {faltantes}")
            X = df[feats_needed]
        else:
            X = df.drop(columns=[c for c in ["label","student_pid","class_id"] if c in df.columns], errors="ignore")

        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Falha no predict_proba: {e}")
        df["risk_score"] = proba
    else:
        cols_esperadas = {"student_pid","class_id","presence_30d","grade_slope","incidents_90d"}
        if not cols_esperadas.issubset(df.columns):
            faltando = list(cols_esperadas - set(df.columns))
            raise HTTPException(status_code=500, detail=f"Colunas ausentes: {faltando}")
        df["risk_score"] = (
            (1 - df["presence_30d"])
            + (-df["grade_slope"]).clip(lower=0)
            + df["incidents_90d"] * 0.5
        )

    k = body.k or 5
    topk_df = df.nlargest(k, "risk_score")[
        ["student_pid","class_id","risk_score","presence_30d","grade_slope","incidents_90d"]
    ]
    topk_records = topk_df.to_dict(orient="records")

    # ---- evidências da literatura (coleção pública)
    evidencias = []
    try:
        client = _get_client()
        _ensure_collection(client, COLLECTION, DIM)

        query_str = body.query.strip() if body.query else "indicadores de risco evasão escolar"
        if body.class_id:
            query_str += f" {body.class_id}"

        emb = _Emb()
        qvec = emb.embed([query_str])[0]

        res = client.search(
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=3,
            with_payload=True,
        )
        evidencias = [(r.payload or {}).get("text") for r in res]
    except Exception:
        evidencias = []

    # ---- glimpses do prontuário (coleção privada) — apenas para psicologia
    if body.include_notes:
        if x_user_role != "psych":
            raise HTTPException(status_code=403, detail="Acesso negado: include_notes requer X-User-Role=psych")

        try:
            client = _get_client()
            _ensure_collection(client, NOTES_COLLECTION, NOTES_DIM)

            # vetor da consulta para notas
            q_text = body.query.strip() if body.query else "resumo notas recentes"
            try:
                qvec_notes = _notes_embed([q_text])[0]
            except Exception as e:
                # não bloquear a resposta por falha em embeddings locais
                qvec_notes = None
                print(f"[WARN] notas: embeddings locais indisponíveis: {e}")

            for rec in topk_records:
                pid = rec["student_pid"]
                rec["notes_glimpse"] = None
                if qvec_notes is None:
                    continue
                res = client.search(
                    collection_name=NOTES_COLLECTION,
                    query_vector=qvec_notes,
                    limit=1,
                    with_payload=True,
                    query_filter={"must":[{"key":"pid","match":{"value": pid}}]}
                )
                if res:
                    # redigir PII por segurança adicional
                    txt = (res[0].payload or {}).get("text", "")
                    rec["notes_glimpse"] = _redact_pii(txt)
        except HTTPException:
            raise
        except Exception as e:
            print(f"[WARN] notas: falha ao buscar glimpses: {e}")

    # >>> ADIÇÃO: gerar relatório textual (usa GEN_MODE/openai se configurado)
    report_text = generate_report(
        body.class_id,
        topk_records,
        [e for e in evidencias if e]
    )

    return {
        "class_id": body.class_id,
        "mode": body.mode,
        "topk": topk_records,
        "evidencias": [e for e in evidencias if e],
        "report_text": report_text,  # pode ser None se GEN_MODE != 'openai'
    }
