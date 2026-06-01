import os
from fastapi import Header, HTTPException
from dotenv import load_dotenv

# Garanta que o .env é carregado na importação do módulo
load_dotenv()

def get_api_key() -> str:
    key = os.getenv("API_KEY")
    if not key:
        raise RuntimeError("API_KEY não definido no ambiente (.env ou variável de sistema).")
    return key

def enforce_api_key(x_api_key: str = Header(..., alias="X-API-Key")):
    api_key = get_api_key()
    if x_api_key != api_key:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")
