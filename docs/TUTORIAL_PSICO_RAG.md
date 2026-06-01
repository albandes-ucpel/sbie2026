
# Tutorial — Psico‑RAG PoC (Professor Edition)

Este guia foi pensado para alunos(as) e novos(as) membros do time. Ao final, você saberá:
- **Iniciar o sistema**: ativar o `venv`, subir a **API** e o **Qdrant (Docker)**.
- **Usar a CLI** (antes e depois da ingestão híbrida: BibTeX + PDF).
- **Chamar a API**: o que cada endpoint faz, parâmetros e exemplos práticos.

---

## 1) Pré‑requisitos

- **Python 3.10+** (recomendado 3.11)
- **Docker** instalado e ativo
- **Chave OpenAI** (para embeddings do acervo público): `OPENAI_API_KEY`
- (Opcional) **sentence-transformers** para a coleção privada de prontuário

> Dica: em máquinas com pouca RAM (2GB), feche programas; PyMuPDF e SentenceTransformers podem consumir memória durante ingestão.

---

## 2) Criar/ativar o ambiente virtual (venv) e instalar dependências

```bash
# 2.1 criar e ativar o venv
python3 -m venv .venv
source .venv/bin/activate        # (Linux/Mac)
# .venv\Scripts\activate         # (Windows PowerShell)

# 2.2 atualizar pip (opcional, mas ajuda)
pip install --upgrade pip

# 2.3 instalar requisitos principais do projeto
pip install -r requirements.txt

# 2.4 instalar extras para ingestão híbrida
pip install bibtexparser PyMuPDF sentence-transformers
```

> Se **não** for usar a coleção de prontuário (privada), `sentence-transformers` é opcional.

---

## 3) Variáveis de ambiente (`.env`)

Crie um arquivo `.env` na raiz do projeto:

```ini
# OpenAI (acervo público - embeddings)
OPENAI_API_KEY=sk-...

# Qdrant
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=literatura
QDRANT_NOTES_COLLECTION=prontuario_privado

# Modelos de embedding e dimensões
EMBEDDING_MODEL=text-embedding-3-small
EMBED_DIM=1536
NOTES_EMBED_MODEL=BAAI/bge-m3
NOTES_EMBED_DIM=1024

# Limites/custos da ingestão híbrida
MAX_TOKENS_ABSTRACT=1200
MAX_CHUNK_TOKENS=700
CHUNK_OVERLAP_TOKENS=100

# Caminhos dos artefatos locais (já usados no projeto)
FEATURES_PATH=data/processed/features.parquet
MODEL_PATH=models/baseline.joblib
```

---

## 4) Subir o Qdrant (Docker)

```bash
docker run -d --name qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -v ~/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

- **Porta**: `6333` (HTTP)
- **Volume**: persiste os vetores/chunks em `~/qdrant_storage`
- Se já existir um container chamado `qdrant`, use:
  ```bash
  docker start qdrant
  # ou remova/renomeie o antigo:
  # docker rm -f qdrant
  # docker run ... (como acima)
  ```

---

## 5) Subir a API (FastAPI + Uvicorn)

Com o `venv` **ativado**:

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

Teste rápido:

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok"}
```

> Algumas rotas exigem `X-API-Key` (veja `utils/auth.py`). Use, por exemplo, `-H "X-API-Key: dev-key"` nas chamadas protegidas.

---

## 6) CLI — Antes vs. Agora (com ingestão híbrida)

### 6.1 Versão **anterior** (legado)

- **Reindex de textos locais (`.txt`/`.md`)** para a coleção pública:
  ```bash
  curl -X POST http://127.0.0.1:8000/rag/reindex \
    -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
    -d '{"corpus_dir":"data/corpus"}'
  ```
  - **O que faz**: lê arquivos em `data/corpus`, fatia em chunks e indexa.
  - **Uso didático**: útil para começar com materiais próprios (textos livres).

### 6.2 Versão **atual** (com ingestão híbrida)

> Agora temos **duas novas peças** de ingestão: **BibTeX** e **PDF**.

- **Ingestão de BibTeX (sempre)** — via **CLI**:
  ```bash
  python -m src.rag.ingest_bib --glob "./data/bib/*.bib"
  ```
  - **O que faz**: lê `.bib`, monta um *abstract block* (título+metadados+resumo) e indexa em `vec_abstract`.
  - **Por quê**: baixo custo, legalmente seguro (abstract é geralmente aberto).

- **Ingestão de PDF (quando open access)** — via **CLI**:
  ```bash
  python -m src.rag.ingest_pdf \
    --doi 10.1234/abcd.2025.001 \
    --title "Meu Artigo" \
    --pdf ./data/pdfs/meu_artigo.pdf \
    --open
  ```
  - **O que faz**: extrai texto do PDF, fatia em chunks e indexa em `vec_fulltext`.
  - **Política**: use `--open` apenas para PDFs abertos/autorizados. Em PDFs fechados, prefira ficar só no BibTeX/abstract.

---

## 7) API — Endpoints e exemplos (professor mode)

### 7.1 `GET /health`
- **Para quê**: checar se a API está viva.
- **Exemplo**:
  ```bash
  curl http://127.0.0.1:8000/health
  ```

---

### 7.2 `POST /rag/reindex` (legado)
- **Para quê**: indexar corpus local (`.txt`/`.md`) como **fulltext**.
- **Body (JSON)**:
  - `corpus_dir` (str): pasta raiz dos textos. Default: `data/corpus`.
- **Exemplo**:
  ```bash
  curl -X POST http://127.0.0.1:8000/rag/reindex \
    -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
    -d '{"corpus_dir":"data/corpus"}'
  ```
- **Retorno**: `chunks_indexed` com o total indexado.

> Obs.: Na coleção híbrida, esses textos entram em `vec_fulltext`.

---

### 7.3 `POST /ingest/bib` (novo)
- **Para quê**: ingerir entradas **BibTeX** (metadados + abstract → `vec_abstract`).
- **Headers**: `X-API-Key: ...`, `Content-Type: application/json`
- **Body (JSON)**:
  - `bibtex` (str): conteúdo BibTeX completo.
- **Exemplo**:
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest/bib \
    -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
    -d @bib_payload.json
  ```
  `bib_payload.json`:
  ```json
  {
    "bibtex": "@article{key2025, title={Meu Artigo}, author={Silva, Joao and Souza, Maria}, year={2025}, journal={Revista X}, doi={10.1234/abcd.2025.001}, url={https://...}, abstract={Resumo aqui}, keywords={evasao; psicologia}}"
  }
  ```
- **Retorno**: `{ "status": "ok", "count": N }`

---

### 7.4 `POST /ingest/pdf` (novo)
- **Para quê**: anexar **PDF** (open access) e indexar chunks em `vec_fulltext`.
- **Headers**: `X-API-Key: ...`
- **Form‑Data**:
  - `doi` (str, obrigatório)
  - `title` (str, obrigatório)
  - `is_open_access` (bool, **prefira true** quando permitido)
  - `file` (arquivo .pdf)
- **Exemplo**:
  ```bash
  curl -X POST http://127.0.0.1:8000/ingest/pdf \
    -H "X-API-Key: dev-key" \
    -F "doi=10.1234/abcd.2025.001" \
    -F "title=Meu Artigo" \
    -F "is_open_access=true" \
    -F "file=@./data/pdfs/meu_artigo.pdf"
  ```
- **Retorno**: `{ "status": "ok", "doi": "...", "chunks": N }`

> **Boas práticas**: para PDFs **fechados**, **não** armazene conteúdo integral. Fique no BibTeX/abstract (rota `/ingest/bib`).

---

### 7.5 `POST /search`
- **Para quê**: buscar evidências no acervo público.
- **Headers**: `X-API-Key: ...`, `Content-Type: application/json`
- **Body (JSON)**:
  - `query` (str): sua pergunta/pesquisa
  - `k` (int): quantidade de resultados (default `5`)
  - `mode` (str): `"abstract" | "full" | "hybrid"` (default `"hybrid"`)
- **Exemplos**:
  - **Híbrido (recomendado)**:
    ```bash
    curl -X POST http://127.0.0.1:8000/search \
      -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
      -d '{"query":"fatores de risco de evasão escolar","k":5,"mode":"hybrid"}'
    ```
  - **Somente abstracts** (rápido/barato):
    ```bash
    -d '{"query":"intervenções psicológicas na evasão","k":5,"mode":"abstract"}'
    ```
  - **Somente fulltext** (global — pode ser mais caro):
    ```bash
    -d '{"query":"assiduidade e nota como preditores","k":5,"mode":"full"}'
    ```
- **Retorno**:
  - Para `hybrid`: `abstract_hits[]` e `fulltext_hits[]` (payloads incluem DOI, título, etc.).

---

### 7.6 `POST /answer`
- **Para quê**: gerar relatório (turma/aluno/intervenção) + **evidências híbridas**.
- **Headers**: `X-API-Key: ...`, `Content-Type: application/json`
- **Body (JSON)** (principais campos):
  - `query` (str): pesquisa/tema do relatório
  - `k` (int): tamanho do Top‑K e hits
  - `class_id` (str|opcional): filtra a turma
  - `mode` (str): `"baseline"` ou `"model"` (usa o `models/baseline.joblib`)
  - `include_notes` (bool): se *true*, **exige** `X-User-Role: psych` e busca “glimpses” no prontuário
  - `task` (str): `"turma_topk" | "aluno" | "intervencao"`
  - `pid` / `aluno_data` / `description`: opcionais conforme `task`
- **Exemplo**:
  ```bash
  curl -X POST http://127.0.0.1:8000/answer \
    -H "X-API-Key: dev-key" -H "Content-Type: application/json" \
    -d '{
          "query":"quais preditores de evasão aparecem na literatura recente?",
          "k":4,
          "class_id":"91",
          "mode":"model",
          "include_notes":false,
          "task":"turma_topk"
        }'
  ```
- **Retorno**:
  - `topk` (alunos com maior `risk_score`), `evidencias` (abstracts/trechos OA), `citations` (proveniência), `report_text` (texto final).

> **RBAC**: Para `include_notes=true`, envie também o header `X-User-Role: psych`. A API mascara PII nos trechos retornados.

---

## 8) Fluxo recomendado (do zero ao resultado)

1. Ativar `venv` e instalar deps (Seção 2)  
2. Subir **Qdrant** (Seção 4)  
3. Configurar **.env** (Seção 3)  
4. Subir a **API** (Seção 5)  
5. Ingerir **BibTeX** do seu acervo (`/ingest/bib` *ou* CLI)  
6. (Opcional) Ingerir **PDFs open access** (`/ingest/pdf` *ou* CLI)  
7. Testar **/search** em `mode:"hybrid"`  
8. Rodar **/answer** com a `class_id` e escolher `mode:"baseline"` ou `"model"`

---

## 9) Dicas de professor

- **Compare modos**: peça para os alunos rodarem `/search` com `abstract` vs `full` vs `hybrid` e discutirem custo x qualidade.
- **Ética & LGPD**: reforce por que **não** indexamos PDFs fechados sem permissão e como o `include_notes` respeita RBAC e PII.
- **Reprodutibilidade**: peça um `requirements.txt` “congelado” e um `.env.example` no repositório da turma.
- **Benchmark rápido**: peça 3 consultas típicas e registrem o tempo médio e tokens (quando possível) por modo.

---

## 10) Troubleshooting

- **`docker: name "qdrant" already in use`** → `docker rm -f qdrant` e suba novamente (ou use outro nome).
- **`OPENAI_API_KEY` ausente** → confira `.env` e variáveis do shell (`printenv | grep OPENAI`).
- **Falha ao ler PDF** → verifique se é PDF texto (não imagem) e se `PyMuPDF` está instalado.
- **`sentence-transformers` não instalado** → instale apenas se for usar `include_notes`.
- **Dimensão de embedding** → `EMBED_DIM=1536` para `text-embedding-3-small` (ajuste se trocar o modelo).

---

**Bom estudo!** Qualquer comando acima pode virar um *script* de conveniência (ex.: `make ingest-bib`, `make api`, `make qdrant`).
