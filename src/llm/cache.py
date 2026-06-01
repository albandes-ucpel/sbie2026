# src/llm/cache.py
import os, json, hashlib, time
from pathlib import Path
from typing import Any, Dict

CACHE_DIR = Path(os.getenv("LLM_CACHE_DIR", "artifacts/llm_cache"))
CACHE_TTL = int(os.getenv("LLM_CACHE_TTL_SECONDS", "86400"))  # 24h padrão

def _key(payload: Dict[str, Any]) -> str:
    data = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()

def get(payload: Dict[str, Any]) -> str | None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = _key(payload)
    fp = CACHE_DIR / f"{k}.json"
    if not fp.exists():
        return None
    try:
        obj = json.loads(fp.read_text(encoding="utf-8"))
        ts = obj.get("_ts", 0)
        if (time.time() - ts) > CACHE_TTL:
            return None
        return obj.get("text")
    except Exception:
        return None

def put(payload: Dict[str, Any], text: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    k = _key(payload)
    fp = CACHE_DIR / f"{k}.json"
    obj = {"_ts": time.time(), "text": text}
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
