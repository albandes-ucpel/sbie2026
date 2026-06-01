# src/rag/ingest_bib.py
import os, json, glob, uuid
from typing import List, Dict
from dotenv import load_dotenv
import bibtexparser
from rag.common import ensure_collection, embed, qdrant, COLLECTION

load_dotenv()
MAX_TOKENS_ABSTRACT = int(os.getenv("MAX_TOKENS_ABSTRACT", "1200"))

def _norm_list(x):
    if not x: return []
    if isinstance(x, list): return x
    return [i.strip() for i in str(x).replace(" and ", ";").split(";") if i.strip()]

def build_abstract_block(entry: Dict) -> str:
    fields = []
    for k in ["title","author","year","journal","booktitle","keywords","doi","url","abstract"]:
        v = entry.get(k) or entry.get(k.upper())
        if v: fields.append(f"{k}: {v}")
    txt = " | ".join(fields)
    # corte para custo ↓
    return txt[:MAX_TOKENS_ABSTRACT*4]

def upsert_bib_entry(entry: Dict):
    title = entry.get("title") or entry.get("TITLE") or ""
    doi   = (entry.get("doi") or entry.get("DOI") or "").lower().strip()
    url   = entry.get("url") or entry.get("URL") or ""
    year  = int(entry.get("year") or entry.get("YEAR") or 0) or None
    authors = _norm_list(entry.get("author") or entry.get("AUTHOR"))

    abstract_block = build_abstract_block(entry)
    vec = embed(abstract_block)

    payload = {
        "kind": "abstract",
        "doi": doi or f"no-doi:{uuid.uuid4()}",
        "title": title,
        "authors": authors,
        "year": year,
        "venue": entry.get("journal") or entry.get("booktitle"),
        "url": url,
        "keywords": _norm_list(entry.get("keywords")),
        "license": entry.get("license","unknown"),
        "is_open_access": entry.get("is_open_access","false").lower() == "true",
        "access_level": "metadata",
        "source": "bibtex"
    }

    qdrant.upsert(
        collection_name=COLLECTION,
        points=[{
            "id": str(uuid.uuid4()),
            "vector": {"vec_abstract": vec},
            "payload": payload
        }]
    )

def ingest_bib_files(pattern: str):
    ensure_collection()
    for path in glob.glob(pattern):
        with open(path, "r", encoding="utf-8") as f:
            db = bibtexparser.load(f)
        for entry in db.entries:
            upsert_bib_entry(entry)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="./data/bib/*.bib")
    args = ap.parse_args()
    ingest_bib_files(args.glob)
    print("OK: BibTeX ingerido.")
