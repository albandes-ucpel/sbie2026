# src/etl/build_features.py
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import re

OUT = Path("data/processed")
RAW = Path("data/raw")
OUT.mkdir(parents=True, exist_ok=True)
RAW.mkdir(parents=True, exist_ok=True)

# ------------ util: sentimento e keywords (bem simples p/ PoC) ------------
RISK_TERMS = [
    r"\bisolament[oa]\b", r"\sansiedad[ea]\b", r"\bdepress[aã]o\b",
    r"\bbullying\b", r"\bviol[eê]ncia\b", r"\bide[aç]?[çc]?[aã]o\b",
    r"\bauto[- ]?les[aã]o\b", r"\btrabalho infantil\b"
]
RISK_RE = re.compile("|".join(RISK_TERMS), flags=re.IGNORECASE)

def simple_sentiment(text: str) -> float:
    """Placeholder simples (troque por modelo local quando quiser)."""
    if not isinstance(text, str):
        return 0.0
    t = text.lower()
    pos = sum(w in t for w in ["melhora", "evolução", "evolucao", "positivo", "engajado", "engajada"])
    neg = sum(w in t for w in ["piora", "crise", "negativo", "isolado", "isolada", "triste", "ansioso", "ansiosa"])
    tot = pos + neg
    return (pos - neg) / tot if tot else 0.0

# ------------ ETL prontuário → features agregadas por student_pid ------------
def build_prontuario_features(df_services: pd.DataFrame, df_notes: pd.DataFrame, today=None) -> pd.DataFrame:
    """
    Espera:
      df_services: columns ['student_pid','service_date','service_type','priority','outcome']
      df_notes:    columns ['student_pid','note_ts','text']
    Retorna colunas agregadas por student_pid:
      services_30d, services_90d, days_since_last_service, referrals_count,
      escalations_90d, no_show_services_30d, notes_sentiment_90d, risk_keywords_90d
    """
    today = pd.Timestamp(today or datetime.utcnow().date())

    # --- services ---
    svc = df_services.copy()
    if not svc.empty:
        svc["service_date"] = pd.to_datetime(svc["service_date"], errors="coerce")
        svc = svc.dropna(subset=["service_date"])
        # janelas
        win30 = today - pd.Timedelta(days=30)
        win90 = today - pd.Timedelta(days=90)

        svc_agg = svc.groupby("student_pid").agg(
            services_30d=("service_date", lambda s: (s >= win30).sum()),
            services_90d=("service_date", lambda s: (s >= win90).sum()),
            last_service=("service_date", "max"),
            referrals_count=("outcome", lambda s: (s == "encaminhado").sum()),
            escalations_90d=("priority", lambda s: ((svc.loc[s.index, "service_date"] >= win90) & (s == "alta")).sum()),
            no_show_services_30d=("outcome", lambda s: ((svc.loc[s.index, "service_date"] >= win30) & (s == "sem_comparecimento")).sum()),
        ).reset_index()
        svc_agg["days_since_last_service"] = (today - svc_agg["last_service"]).dt.days.fillna(9999).astype(int)
        svc_agg = svc_agg.drop(columns=["last_service"])
    else:
        svc_agg = pd.DataFrame(columns=[
            "student_pid","services_30d","services_90d","days_since_last_service",
            "referrals_count","escalations_90d","no_show_services_30d"
        ])

    # --- notes ---
    nt = df_notes.copy()
    if not nt.empty:
        nt["note_ts"] = pd.to_datetime(nt["note_ts"], errors="coerce")
        nt = nt.dropna(subset=["note_ts"])
        win90 = today - pd.Timedelta(days=90)
        nt_recent = nt[nt["note_ts"] >= win90].copy()
        if not nt_recent.empty:
            nt_recent["sent"] = nt_recent["text"].map(simple_sentiment)
            nt_recent["kw"] = nt_recent["text"].map(lambda t: len(RISK_RE.findall(t or "")))
            nt_agg = nt_recent.groupby("student_pid").agg(
                notes_sentiment_90d=("sent", "mean"),
                risk_keywords_90d=("kw", "sum")
            ).reset_index()
        else:
            nt_agg = pd.DataFrame(columns=["student_pid","notes_sentiment_90d","risk_keywords_90d"])
    else:
        nt_agg = pd.DataFrame(columns=["student_pid","notes_sentiment_90d","risk_keywords_90d"])

    # join e preenchimento
    feat = pd.merge(svc_agg, nt_agg, on="student_pid", how="outer")
    if feat.empty:
        # cria linha vazia para compatibilidade de schema (será preenchida no merge final)
        feat = pd.DataFrame(columns=[
            "student_pid","services_30d","services_90d","days_since_last_service",
            "referrals_count","escalations_90d","no_show_services_30d",
            "notes_sentiment_90d","risk_keywords_90d"
        ])
    defaults = {
        "services_30d": 0, "services_90d": 0, "days_since_last_service": 9999,
        "referrals_count": 0, "escalations_90d": 0, "no_show_services_30d": 0,
        "notes_sentiment_90d": 0.0, "risk_keywords_90d": 0
    }
    feat = feat.fillna(defaults)
    return feat

def safe_read_csv(path: Path, expected_cols: list) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] arquivo não encontrado: {path} — seguindo sem este insumo.")
        return pd.DataFrame(columns=expected_cols)
    df = pd.read_csv(path)
    # garante colunas mínimas
    for c in expected_cols:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df[expected_cols]

# ------------ build principal (mantém seus dados sintéticos + merge prontuário) ------------
def main():
    # base sintética (seu exemplo)
    base = pd.DataFrame([
        {"student_pid":"pid_001","class_id":"91","presence_30d":0.72,"grade_slope":-0.8,"incidents_90d":2,"label":1},
        {"student_pid":"pid_002","class_id":"91","presence_30d":0.95,"grade_slope":-0.1,"incidents_90d":0,"label":0},
        {"student_pid":"pid_003","class_id":"91","presence_30d":0.81,"grade_slope":-0.4,"incidents_90d":1,"label":0},
        {"student_pid":"pid_004","class_id":"82","presence_30d":0.69,"grade_slope":-0.9,"incidents_90d":3,"label":1},
        {"student_pid":"pid_005","class_id":"82","presence_30d":0.88,"grade_slope":-0.2,"incidents_90d":0,"label":0},
    ])

    # lê prontuário (se houver)
    svc_cols = ["student_pid","service_date","service_type","priority","outcome"]
    notes_cols = ["student_pid","note_ts","text"]
    df_services = safe_read_csv(RAW / "services.csv", svc_cols)
    df_notes = safe_read_csv(RAW / "notes.csv", notes_cols)

    pront_feat = build_prontuario_features(df_services, df_notes)

    # merge por student_pid
    df = base.merge(pront_feat, on="student_pid", how="left")

    # preencher defaults se não havia prontuário
    fill_defaults = {
        "services_30d": 0, "services_90d": 0, "days_since_last_service": 9999,
        "referrals_count": 0, "escalations_90d": 0, "no_show_services_30d": 0,
        "notes_sentiment_90d": 0.0, "risk_keywords_90d": 0
    }
    df = df.fillna(fill_defaults)

    # salva parquet
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "features.parquet"
    df.to_parquet(out_path, index=False)
    print("features →", out_path)
    print("colunas finais:", list(df.columns))

if __name__ == "__main__":
    main()
