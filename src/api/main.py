# src/api/main.py
import os, glob, re, uuid, tempfile, shutil
from typing import List, Dict, Optional, Literal

from fastapi import FastAPI, HTTPException, Depends, Header, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import joblib  # para carregar o modelo treinado

# Auth (usa seu util atual; precisa aceitar alias="X-API-Key" lá)
from utils.auth import enforce_api_key

# RAG / Vetores
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Gerador de relatório em linguagem natural (templates + cache + force_refresh)
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

# Limites/custos para ingestão híbrida
MAX_TOKENS_ABSTRACT = int(os.getenv("MAX_TOKENS_ABSTRACT", "1200"))
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS", "700"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS", "100"))

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

def _approx_token_chunk(text: str, max_tokens=700, overlap=100) -> List[str]:
    # usa caracteres como proxy (~4 chars/token)
    n = max_tokens * 4
    o = overlap * 4
    s = re.sub(r"\s+", " ", (text or "")).strip()
    out, i = [], 0
    while i < len(s):
        j = min(len(s), i + n)
        out.append(s[i:j])
        if j == len(s): break
        i = max(0, j - o)
    return out

def _get_client() -> QdrantClient:
    url = QDRANT_URL
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if url.startswith(("http://", "https://")):
        return QdrantClient(url=url, api_key=QDRANT_API_KEY)
    return QdrantClient(path=url)

def _ensure_collection_hybrid(client: QdrantClient, name: str, dim: int) -> None:
    """
    Coleção pública com vetores nomeados:
      - vec_abstract  (metadados+resumo)
      - vec_fulltext  (chunks do texto completo)
    """
    cols = [c.name for c in client.get_collections().collections]
    if name not in cols:
        client.recreate_collection(
            collection_name=name,
            vectors_config={
                "vec_abstract": VectorParams(size=dim, distance=Distance.COSINE),
                "vec_fulltext": VectorParams(size=dim, distance=Distance.COSINE),
            },
        )

def _ensure_collection_simple(client: QdrantClient, name: str, dim: int) -> None:
    """Coleção simples (uma só dimensão/vetor) — manter para compat/privado."""
    cols = [c.name for c in client.get_collections().collections]
    if name not in cols:
        client.recreate_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )

class _Emb:
    """Embeddings para literatura/artigos (usa OpenAI)."""
    def __init__(self, model: str | None = None):
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY não definido no .env ou ambiente.")
        self.client = OpenAI()
        self.model = model or EMBED_MODEL
    def embed(self, texts: List[str]) -> List[List[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]

def _notes_embed(texts: List[str]) -> List[List[float]]:
    """Embeddings **locais** para prontuário (SentenceTransformers)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
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

def _bib_norm_list(x):
    if not x: return []
    if isinstance(x, list): return x
    return [i.strip() for i in str(x).replace(" and ", ";").split(";") if i.strip()]

def _build_abstract_block(entry: Dict) -> str:
    fields = []
    for k in ["title","author","year","journal","booktitle","keywords","doi","url","abstract"]:
        v = entry.get(k) or entry.get(k.upper())
        if v: fields.append(f"{k}: {v}")
    txt = " | ".join(fields)
    return txt[:MAX_TOKENS_ABSTRACT*4]  # corte leve por custo


# ---------- Schemas ----------
class ReindexIn(BaseModel):
    corpus_dir: str = "data/corpus"

class SearchIn(BaseModel):
    query: str
    k: int = 5
    mode: Literal["abstract","full","hybrid"] = "hybrid"

class AnswerIn(BaseModel):
    # parâmetros principais
    query: str
    k: int = 5
    class_id: str | None = None
    mode: str = "baseline"           # "baseline" ou "model"
    include_notes: bool = False      # incluir "glimpses" do prontuário (RBAC exigido)

    # Templates + cache + force_refresh
    task: str = "turma_topk"         # "turma_topk" | "aluno" | "intervencao"
    force_refresh: bool = False      # ignora cache do report_text

    # extras por task
    pid: Optional[str] = None        # para "aluno"
    aluno_data: Optional[Dict] = None# para "aluno"
    description: Optional[str] = None# para "intervencao"

class IngestBibRequest(BaseModel):
    bibtex: str


# ---------- Startup ----------
@app.on_event("startup")
def _startup():
    # Torna o startup resiliente: não derruba a API se Qdrant estiver off
    try:
        client = _get_client()
        _ensure_collection_hybrid(client, COLLECTION, DIM)
        # coleção privada segue simples e sob demanda
    except Exception as e:
        print(f"[WARN] Qdrant indisponível no startup: {e}")


# ---------- Rotas utilitárias ----------
@app.get("/health")
def health():
    return {"status": "ok"}


# ---------- Ingestão "legada" (txt/md) ----------
@app.post("/rag/reindex", dependencies=AUTH)
def rag_reindex(body: ReindexIn):
    """
    Mantida para compatibilidade. Os textos reindexados vão para vec_fulltext.
    """
    try:
        paths = sorted(
            glob.glob(os.path.join(body.corpus_dir, "**/*.txt"), recursive=True) +
            glob.glob(os.path.join(body.corpus_dir, "**/*.md"),  recursive=True)
        )

        client = _get_client()
        _ensure_collection_hybrid(client, COLLECTION, DIM)

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
                payloads.append({
                    "kind": "fulltext",
                    "text": ch,
                    "source": os.path.relpath(p),
                    "meta": {"chunk": idx, "access_level": "fulltext", "is_open_access": True}
                })

        total = 0
        for i in range(0, len(chunks), 64):
            vecs = emb.embed(chunks[i:i+64])
            points = [
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector={"vec_fulltext": v},
                    payload=pl
                )
                for v, pl in zip(vecs, payloads[i:i+64])
            ]
            client.upsert(collection_name=COLLECTION, points=points)
            total += len(points)

        return {"ok": True, "chunks_indexed": total}

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"reindex failed: {e}")


# ---------- Ingestão híbrida: BibTeX + PDF ----------
@app.post("/ingest/bib", dependencies=AUTH)
def ingest_bib(req: IngestBibRequest):
    """
    Ingestão de entradas BibTeX. Armazena metadados/abstract em vec_abstract.
    """
    try:
        import bibtexparser
    except Exception:
        raise HTTPException(status_code=400, detail="bibtexparser não instalado. pip install bibtexparser")

    try:
        db = bibtexparser.loads(req.bibtex)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"BibTeX inválido: {e}")

    emb = _Emb()
    client = _get_client()
    _ensure_collection_hybrid(client, COLLECTION, DIM)

    count = 0
    for entry in db.entries:
        title = entry.get("title") or entry.get("TITLE") or ""
        doi   = (entry.get("doi") or entry.get("DOI") or "").lower().strip()
        url   = entry.get("url") or entry.get("URL") or ""
        year  = int(entry.get("year") or entry.get("YEAR") or 0) or None
        authors = _bib_norm_list(entry.get("author") or entry.get("AUTHOR"))
        keywords = _bib_norm_list(entry.get("keywords"))

        block = _build_abstract_block(entry)
        vec = emb.embed([block])[0]

        payload = {
            "kind": "abstract",
            "doi": doi or f"no-doi:{uuid.uuid4()}",
            "title": title,
            "authors": authors,
            "year": year,
            "venue": entry.get("journal") or entry.get("booktitle"),
            "url": url,
            "keywords": keywords,
            "license": entry.get("license","unknown"),
            "is_open_access": str(entry.get("is_open_access","false")).lower() == "true",
            "access_level": "metadata",
            "source": "bibtex",
            "abstract_block": block
        }

        client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(id=str(uuid.uuid4()), vector={"vec_abstract": vec}, payload=payload)]
        )
        count += 1

    return {"status": "ok", "count": count}


@app.post("/ingest/pdf", dependencies=AUTH)
def ingest_pdf(
    doi: str = Form(...),
    title: str = Form(...),
    is_open_access: bool = Form(False),
    file: UploadFile = File(...),
):
    """
    Faz upload de PDF e indexa chunks em vec_fulltext.
    Somente use fulltext se open-access/autorizado.
    """
    if not is_open_access:
        # Permitimos armazenar, mas marcamos access_level=metadata para evitar exposição de chunks em /answer
        # (Se quiser bloquear por completo, troque para 403.)
        pass

    emb = _Emb()
    client = _get_client()
    _ensure_collection_hybrid(client, COLLECTION, DIM)

    # salvar temporariamente
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        try:
            import fitz  # PyMuPDF
        except Exception:
            raise HTTPException(status_code=400, detail="PyMuPDF não instalado. pip install PyMuPDF")

        doc = fitz.open(tmp_path)
        pages = [p.get_text() or "" for p in doc]
        full = "\n".join(pages)

        chunks = _approx_token_chunk(full, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
        total = 0
        for i in range(0, len(chunks), 64):
            vecs = emb.embed(chunks[i:i+64])
            points = []
            for v, ck in zip(vecs, chunks[i:i+64]):
                payload = {
                    "kind": "fulltext",
                    "doi": doi,
                    "title": title,
                    "chunk_index": total,
                    "chunk_of": doi,
                    "license": "CC-BY" if is_open_access else "unknown",
                    "is_open_access": bool(is_open_access),
                    "access_level": "fulltext" if is_open_access else "metadata",
                    "source": "pdf",
                    "text": ck if is_open_access else None  # opcional ocultar texto bruto se não-open
                }
                points.append(PointStruct(id=str(uuid.uuid4()), vector={"vec_fulltext": v}, payload=payload))
                total += 1

            if points:
                client.upsert(collection_name=COLLECTION, points=points)

        return {"status": "ok", "doi": doi, "chunks": total}
    finally:
        try: os.remove(tmp_path)
        except Exception: pass


# ---------- Busca híbrida ----------
def _search_hybrid(query: str, k: int):
    emb = _Emb()
    qvec = emb.embed([query])[0]
    client = _get_client()
    _ensure_collection_hybrid(client, COLLECTION, DIM)

    # 1) abstracts sempre
    res_abs = client.search(
        collection_name=COLLECTION,
        query_vector=("vec_abstract", qvec),
        limit=k,
        with_payload=True
    )

    # 2) expande em fulltext apenas dos DOIs achados
    dois = [r.payload.get("doi") for r in res_abs if r.payload and r.payload.get("doi")]
    res_full = []
    if dois:
        f = qm.Filter(
            must=[qm.FieldCondition(key="kind", match=qm.MatchValue(value="fulltext")),
                  qm.FieldCondition(key="doi", match=qm.MatchAny(any=dois))]
        )
        res_full = client.search(
            collection_name=COLLECTION,
            query_vector=("vec_fulltext", qvec),
            limit=k,
            with_payload=True,
            query_filter=f
        )

    return res_abs, res_full


# ---------- Busca ----------
@app.post("/search", dependencies=AUTH)
def search(body: SearchIn):
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query vazia")

    try:
        client = _get_client()
        _ensure_collection_hybrid(client, COLLECTION, DIM)

        emb = _Emb()
        qvec = emb.embed([body.query])[0]

        if body.mode == "abstract":
            res = client.search(COLLECTION, ("vec_abstract", qvec), body.k, with_payload=True)
            results = [
                {"id": r.id, "score": float(r.score), "payload": (r.payload or {})}
                for r in res
            ]
            return {"mode": "abstract", "query": body.query, "k": body.k, "results": results}

        if body.mode == "full":
            res = client.search(COLLECTION, ("vec_fulltext", qvec), body.k, with_payload=True)
            results = [
                {"id": r.id, "score": float(r.score), "payload": (r.payload or {})}
                for r in res
            ]
            return {"mode": "full", "query": body.query, "k": body.k, "results": results}

        # hybrid
        res_abs, res_full = _search_hybrid(body.query, body.k)
        return {
            "mode": "hybrid",
            "query": body.query,
            "k": body.k,
            "abstract_hits": [
                {"id": r.id, "score": float(r.score), "payload": (r.payload or {})} for r in res_abs
            ],
            "fulltext_hits": [
                {"id": r.id, "score": float(r.score), "payload": (r.payload or {})} for r in res_full
            ],
        }

    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"search failed: {e}")


# ---------- Answer (mantém sua lógica + evidências híbridas) ----------
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

    task:
      - turma_topk   → relatório para turma com Top-K
      - aluno        → parecer individual (usa pid/aluno_data)
      - intervencao  → plano de ação (usa description)
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
            return {
                "class_id": body.class_id,
                "mode": body.mode,
                "task": body.task,
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

    # ---- evidências da literatura (coleção pública) — híbrido
    evidencias = []
    citations = []
    try:
        query_str = (body.query or "indicadores de risco evasão escolar").strip()
        if body.class_id:
            query_str += f" {body.class_id}"

        res_abs, res_full = _search_hybrid(query_str, 3)

        # construir contexto/citações; respeitar acesso
        for r in res_abs:
            p = r.payload or {}
            citations.append({
                "title": p.get("title") or p.get("source"),
                "year": p.get("year"),
                "doi": p.get("doi"),
                "url": p.get("url"),
                "access_level": p.get("access_level", "metadata"),
                "kind": "abstract"
            })
            # usar bloco de resumo/metadados
            ab = p.get("abstract_block") or p.get("text")
            if ab:
                evidencias.append(ab)

        for r in res_full:
            p = r.payload or {}
            if not p.get("is_open_access", False):
                # Não expor conteúdo granular de PDFs fechados
                continue
            txt = p.get("text")
            if txt:
                evidencias.append(txt)
            citations.append({
                "title": p.get("title"),
                "year": p.get("year"),
                "doi": p.get("doi"),
                "url": p.get("url"),
                "access_level": p.get("access_level", "fulltext" if p.get("is_open_access") else "metadata"),
                "kind": "fulltext",
                "chunk_index": p.get("chunk_index")
            })
    except Exception as e:
        print(f"[WARN] evidências híbridas falharam: {e}")

    # ---- glimpses do prontuário (coleção privada) — apenas para psicologia
    if body.include_notes:
        if x_user_role != "psych":
            raise HTTPException(status_code=403, detail="Acesso negado: include_notes requer X-User-Role=psych")

        try:
            client = _get_client()
            _ensure_collection_simple(client, NOTES_COLLECTION, NOTES_DIM)

            q_text = body.query.strip() if body.query else "resumo notas recentes"
            try:
                qvec_notes = _notes_embed([q_text])[0]
            except Exception as e:
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
                    txt = (res[0].payload or {}).get("text", "")
                    rec["notes_glimpse"] = _redact_pii(txt)
        except HTTPException:
            raise
        except Exception as e:
            print(f"[WARN] notas: falha ao buscar glimpses: {e}")

    # -------- Geração do relatório textual com templates + cache + force_refresh --------
    evids = [e for e in evidencias if e]

    # monta 'extra' conforme o tipo de template solicitado
    extra: Optional[Dict] = None
    task = body.task if body.task in {"turma_topk", "aluno", "intervencao"} else "turma_topk"
    if task == "aluno":
        pid = body.pid or (topk_records[0]["student_pid"] if topk_records else None)
        aluno_data = body.aluno_data or (topk_records[0] if topk_records else {})
        extra = {"pid": pid, "aluno_data": aluno_data}
    elif task == "intervencao":
        extra = {"description": body.description or body.query}

    # prompt com proveniência clara fica dentro do template do generate_report
    report_text = generate_report(
        task=task,
        class_id=body.class_id,
        topk=topk_records,
        evidencias=evids,
        extra=extra,
        force_refresh=body.force_refresh,
    )

    return {
        "class_id": body.class_id,
        "mode": body.mode,
        "task": task,
        "topk": topk_records,
        "evidencias": evids,
        "citations": citations,
        "report_text": report_text,  # None se GEN_MODE != 'openai'
    }
