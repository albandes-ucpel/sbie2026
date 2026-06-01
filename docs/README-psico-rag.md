# Psico-RAG — Guia para Contribuidores

Sistema de **Análise Preditiva e Suporte à Decisão** para Psicologia Escolar usando **RAG + Modelos Preditivos** e uma **API FastAPI**.

> Este README explica **como clonar, preparar o ambiente (`.venv`)**, **reconstruir modelos treinados**, **reindexar o RAG**, e **subir a API**. Também cobre boas práticas de versionamento e troubleshooting.

---

## 🧱 Estrutura recomendada do repositório

```
.
├── api/                 # FastAPI (endpoints /search e /answer)
│   ├── main.py
│   └── schemas.py
├── etl/                 # extract/transform/load (gera data/processed/features.parquet)
├── models/              # treino e artefatos (ex.: baseline.joblib)  [IGNORAR ARTEFATOS NO GIT]
├── rag/                 # indexação/busca, pipeline de embeddings
│   └── index.py
├── scripts/             # utilitários de manutenção
├── config/              # YAML de configs (sem segredos)
│   └── config.yaml
├── data/                # dados (raw/interim/processed) [NÃO VERSIONAR]
│   ├── raw/
│   ├── interim/
│   └── processed/
├── tests/               # testes automatizados
├── .gitignore
├── .env.example         # modelo de variáveis de ambiente
├── requirements.txt     # dependências Python (ou pyproject.toml)
├── Dockerfile / docker-compose.yml (opcional)
└── README.md
```

> **Importante**: dados, vetores e modelos **não** devem ir para o Git. Armazene-os em S3/MinIO/Drive, ou use DVC/MLflow (ver seção abaixo).

---

## 🚀 Primeira vez aqui? (Setup rápido)

### 1) Pré-requisitos
- Python **3.11+**
- `git`, `make` (opcional), e Docker (opcional, p/ Qdrant local)

### 2) Clonar o repositório
```bash
git clone <URL_DO_REPO>
cd <PASTA_DO_REPO>
```

### 3) Criar o ambiente virtual e instalar dependências
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Configurar variáveis de ambiente
Copie o arquivo de exemplo e preencha:

```bash
cp .env.example .env
# edite .env com as chaves/endpoints corretos
```

Valores típicos (ver `.env.example`):
- `OPENAI_API_KEY=`
- `QDRANT_URL=` e `QDRANT_API_KEY=` (ou usar docker-compose local)
- `DATABASE_URL=` (se necessário para ETL)
- `FEATURES_PATH=data/processed/features.parquet`

---

## 🔁 Reconstruindo tudo do zero (cold start)

> Use este fluxo quando **não houver modelos/índices prontos** ou você deseja reconstruí-los localmente.

1) **Rodar o ETL** e gerar `features.parquet`:
```bash
python -m etl.extract
python -m etl.transform
python -m etl.load
# ou, se existir Makefile:
make etl
```
- Saída esperada: `data/processed/features.parquet`

2) **Treinar o modelo baseline** (gera, por exemplo, `models/baseline.joblib`):
```bash
python -m models.train --cfg ./config/config.yaml
# ou:
make train
```

3) **Subir Qdrant local** (opcional; se usar Qdrant gerenciado, pule para o passo 4):
```bash
docker run -p 6333:6333 -p 6334:6334 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant:latest
# QDRANT_URL=http://localhost:6333
```

4) **Construir o índice RAG** (gerar embeddings e gravar no Qdrant):
```bash
python -m rag.index
# ou:
make index
```

5) **Subir a API FastAPI**:
```bash
# exemplo usado anteriormente pelo time
python -m uvicorn api.main:app --app-dir src --host 0.0.0.0 --port 8000 --reload
# OU, se o main está na raiz de api/:
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## ♻️ Restaurando a partir de artefatos (modelos/índices) já existentes

Se o time **já treina e publica modelos/índices** (ex.: em S3), siga:

### Opção A — DVC (recomendado para dados/modelos grandes)
1) Instale DVC:
```bash
pip install dvc dvc-s3
```
2) Configure o remote (S3/MinIO):
```bash
dvc remote add -d storage s3://meu-bucket-psico-rag
dvc remote modify storage access_key_id <AKIA...>
dvc remote modify storage secret_access_key <SECRET>
dvc remote modify storage endpointurl https://s3.amazonaws.com  # ou seu endpoint
```
3) Baixe artefatos versionados:
```bash
dvc pull
```
Isso deve restaurar `data/processed/*`, `models/*.joblib` e, opcionalmente, dumps do índice.

### Opção B — Download manual
- Baixe os arquivos do local acordado (S3/Drive).
- Coloque **datasets** em `data/` (p. ex. `data/processed/features.parquet`).
- Coloque **modelos** em `models/` (p. ex. `models/baseline.joblib`).
- Se houver **snapshot de índice** (Qdrant), restaure a pasta `qdrant_storage/` ou reindexe via `python -m rag.index`.

---

## 🧪 Testes rápidos

- **Saúde da API**: `GET /` (se implementado)  
- **Busca**: `POST /search` com `{"query": "aluno com suspensões", "k": 5}`  
- **Resposta RAG**: `POST /answer` com `{"query": "fatores de risco para evasão", "k": 3}`

> Dica: mantenha **payloads de exemplo** em `tests/fixtures/` para smoke tests.

---

## 🔐 Boas práticas de segurança

- **NUNCA** commitar `.env`, chaves, certificados ou dumps com dados pessoais.
- Use **segregação de variáveis**: configs públicas em `config/*.yaml`, segredos em `.env`.
- Aplique **princípio do menor privilégio** nas chaves (OpenAI, S3, DB).
- Se LGPD: anonimizar/pseudonimizar datasets de treino; manter registros de consentimento e bases legais.

---

## 🧰 Troubleshooting

- **`ModuleNotFoundError`**: garanta que o `.venv` esteja ativo e `pip install -r requirements.txt` foi executado.
- **`features.parquet` ausente**: rode o ETL (ver “Cold start”).
- **Qdrant não responde**: confira `QDRANT_URL` e se os ports `6333/6334` estão acessíveis; em Docker, confirme o volume `qdrant_storage/`.
- **OpenAI chave inválida**: verifique `OPENAI_API_KEY` no `.env`.
- **Timezone/agenda**: se houver uso de eventos, defina TZ do container/host (`America/Sao_Paulo`).

---

## 📦 Convenções de Makefile (opcional, mas útil)

```Makefile
.PHONY: etl train index api test

etl:
\tpython -m etl.extract && python -m etl.transform && python -m etl.load

train:
\tpython -m models.train --cfg ./config/config.yaml

index:
\tpython -m rag.index

api:
\tuvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
\tpytest -q
```

---

## 🗂️ Versionamento de dados e modelos (DVC/MLflow)

- **DVC**: rastreia datasets e artefatos de modelo (arquivos grandes) fora do Git.
- **MLflow** (opcional): rastreia **experimentos**, métricas e parâmetros; armazena modelos com versionamento.

> Recomendação: usar **DVC** para arquivos (dados/modelos) + **MLflow** para experimentos.

---

## 🤝 Contribuições

1. Crie branch: `git checkout -b feat/minha-feature`
2. Commits pequenos e descritivos
3. Pull Request com resumo do que foi feito, passos de teste e impacto

---

## 📄 Licença

Defina a licença do projeto (ex.: MIT, Apache-2.0) conforme política da Escola MQ.
