# src/rag/ingest_pdf.py
import os, uuid, fitz  # PyMuPDF
from dotenv import load_dotenv
from rag.common import ensure_collection, embed, chunk_text, qdrant, COLLECTION

load_dotenv()
MAX_CHUNK_TOKENS = int(os.getenv("MAX_CHUNK_TOKENS","700"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("CHUNK_OVERLAP_TOKENS","100"))

def extract_text_pdf(path: str) -> str:
    doc = fitz.open(path)
    out = []
    for p in doc:
        out.append(p.get_text())
    return "\n".join(out)

def upsert_pdf(doi: str, title: str, is_open_access: bool, pdf_path: str):
    ensure_collection()
    full = extract_text_pdf(pdf_path)
    chunks = chunk_text(full, MAX_CHUNK_TOKENS, CHUNK_OVERLAP_TOKENS)
    points = []
    for i, ck in enumerate(chunks):
        vec = embed(ck)
        payload = {
            "kind": "fulltext",
            "doi": doi,
            "title": title,
            "chunk_id": str(uuid.uuid4()),
            "chunk_index": i,
            "chunk_of": doi,
            "license": "CC-BY" if is_open_access else "unknown",
            "is_open_access": bool(is_open_access),
            "access_level": "fulltext" if is_open_access else "metadata",
            "source": "pdf"
        }
        points.append({
            "id": str(uuid.uuid4()),
            "vector": {"vec_fulltext": vec},
            "payload": payload
        })
    if points:
        qdrant.upsert(collection_name=COLLECTION, points=points)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--doi", required=True)
    ap.add_argument("--title", required=True)
    ap.add_argument("--pdf", required=True)
    ap.add_argument("--open", action="store_true", help="PDF é open access?")
    args = ap.parse_args()
    upsert_pdf(args.doi, args.title, args.open, args.pdf)
    print("OK: PDF ingerido.")
