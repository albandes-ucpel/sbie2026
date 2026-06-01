# src/llm/templates.py
from typing import List, Dict

SYSTEM_BASE = """Você é um assistente para psicologia escolar.
Respeite a LGPD: nunca use identificadores pessoais (apenas student_pid).
Evite rótulos; indique hipóteses e ações de apoio.
Seja concreto, claro e ético."""

def fmt_topk(topk: List[Dict]) -> str:
    lines = []
    for r in topk:
        lines.append(
            f"- {r.get('student_pid')} | risco={r.get('risk_score'):.3f} | "
            f"presença_30d={r.get('presence_30d')} | slope_notas={r.get('grade_slope')} | incidents_90d={r.get('incidents_90d')}"
        )
    return "\n".join(lines)

def fmt_evidencias(evidencias: List[str], max_chars=1400) -> str:
    joined = "\n\n".join([f"• {e}" for e in evidencias if e])
    return joined[:max_chars] if joined else "—"

def build_context_turma(class_id: str | None, topk: List[Dict], evidencias: List[str]) -> str:
    return f"""CONTEXTO (agregado e anônimo)
Turma: {class_id or 'N/D'}
Top-K (PID | risco | principais features):
{fmt_topk(topk)}

Trechos da literatura:
{fmt_evidencias(evidencias)}
"""

def user_prompt_turma() -> str:
    return """TAREFA:
1) Liste os principais SINAIS na turma (máx 5 bullets).
2) Dê HIPÓTESES (2–3) coerentes com dados e literatura.
3) Proponha AÇÕES coletivas (sala, família, coordenação) em curto prazo (1–2 semanas) e acompanhamento (1–3 meses).
4) Inclua nota de cautela/limitações.

Formato: bullets claros; tom técnico e acolhedor; sem PII."""

def build_context_aluno(pid: str, aluno_data: Dict, evidencias: List[str]) -> str:
    return f"""CONTEXT (aluno, anônimo)
student_pid: {pid}
Dados principais: {aluno_data}

Trechos da literatura:
{fmt_evidencias(evidencias)}
"""

def user_prompt_aluno() -> str:
    return """Gere um parecer breve (10–14 linhas):
- Sinais de risco (máx 5 bullets)
- 2–3 hipóteses
- Plano de ação: imediato (1–2 semanas) e acompanhamento (1–3 meses)
- Nota ética/limitações
Sem PII, use apenas student_pid."""

def build_context_intervencao(description: str, evidencias: List[str]) -> str:
    return f"""CENÁRIO:
{description}

Literatura relevante:
{fmt_evidencias(evidencias)}
"""

def user_prompt_intervencao() -> str:
    return """Sugira 3 estratégias de intervenção:
Para cada uma:
- Objetivo
- Passos práticos
- Quem executa (psicologia, coordenação, família, professor)
- Prazo
- Como medir
Tom técnico, claro e realista. Sem PII."""
