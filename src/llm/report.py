# src/llm/report.py
import os
from typing import List, Dict, Literal
from openai import OpenAI

from .templates import (
    SYSTEM_BASE,
    build_context_turma, user_prompt_turma,
    build_context_aluno, user_prompt_aluno,
    build_context_intervencao, user_prompt_intervencao,
)
from .cache import get as cache_get, put as cache_put

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEN_MODE = os.getenv("GEN_MODE", "none").lower()  # "none" | "openai"

Task = Literal["turma_topk", "aluno", "intervencao"]

def _build_messages(task: Task, class_id: str | None, topk: List[Dict], evidencias: List[str], extra: Dict | None):
    if task == "turma_topk":
        context = build_context_turma(class_id, topk, evidencias)
        user_prompt = user_prompt_turma()
    elif task == "aluno":
        # espera extra={"pid":"pid_xxx","aluno_data":{...}}
        if not extra or "pid" not in extra or "aluno_data" not in extra:
            raise ValueError("extra deve conter 'pid' e 'aluno_data' para task='aluno'")
        context = build_context_aluno(extra["pid"], extra["aluno_data"], evidencias)
        user_prompt = user_prompt_aluno()
    else:  # "intervencao"
        if not extra or "description" not in extra:
            raise ValueError("extra deve conter 'description' para task='intervencao'")
        context = build_context_intervencao(extra["description"], evidencias)
        user_prompt = user_prompt_intervencao()

    return [
        {"role": "system", "content": SYSTEM_BASE},
        {"role": "user", "content": context + "\n\n" + user_prompt},
    ]

def generate_report(
    task: Task,
    class_id: str | None,
    topk: List[Dict],
    evidencias: List[str],
    extra: Dict | None = None,
    force_refresh: bool = False,
) -> str | None:
    """
    Retorna o relatório textual (ou None se GEN_MODE != 'openai' ou faltar API key).
    Usa cache por 24h (configurável via LLM_CACHE_TTL_SECONDS).
    """
    if GEN_MODE != "openai" or not os.getenv("OPENAI_API_KEY"):
        return None

    payload_key = {
        "task": task,
        "class_id": class_id,
        "topk": topk,
        "evidencias": evidencias,
        "extra": extra or {},
        "model": OPENAI_MODEL,
    }

    if not force_refresh:
        cached = cache_get(payload_key)
        if cached:
            return cached

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    messages = _build_messages(task, class_id, topk, evidencias, extra)

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=500,
    )
    text = resp.choices[0].message.content.strip() if resp and resp.choices else None
    if text:
        cache_put(payload_key, text)
    return text
