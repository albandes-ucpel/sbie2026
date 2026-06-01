
# README — Ingestão Híbrida (BibTeX + PDF) e Retrieval

## Visão geral
Este projeto suporta um fluxo **híbrido** de literatura:
- **Sempre** indexa **BibTeX + abstract/metadados** (barato e legalmente seguro) em `vec_abstract`;
- **Opcionalmente** indexa **PDF completo** (apenas *open access* ou autorizado) em `vec_fulltext`;
- As buscas/answers usam **retrieval híbrido**: começam pelos abstracts e, se houver fulltext dos mesmos DOIs, expandem nos chunks do PDF.

---

## Pré-requisitos

- **Qdrant** rodando (ex.: Docker)
  ```bash
  docker run -d --name qdrant     -p 6333:6333     -v ~/qdrant_storage:/qdrant/storage     qdrant/qdrant:latest
  ```
- **Variáveis de ambiente** (`.env`)
  ```ini
  OPENAI_API_KEY=sk-...
  QDRANT_URL=http://localhost:6333
  QDRANT_COLLECTION=literatura
  QDRANT_NOTES_COLLECTION=prontuario_privado

  EMBEDDING_MODEL=text-embedding-3-small
  EMBED_DIM=1536

  # Híbrido
  MAX_TOKENS_ABSTRACT=1200
  MAX_CHUNK_TOKENS=700
  CHUNK_OVERLAP_TOKENS=100

  # Caminhos locais (já usados no projeto)
  FEATURES_PATH=data/processed/features.parquet
  MODEL_PATH=models/baseline.joblib
  ```
- **Dependências Python** (além das já existentes no projeto):
  ```bash
  pip install bibtexparser PyMuPDF sentence-transformers
  ```

> Observação: `sentence-transformers` é usado **apenas** para a coleção privada de prontuário; se não for usar *notes*, pode omitir.

---

## Estrutura das coleções (Qdrant)

- **Coleção pública** `literatura` (híbrida com vetores nomeados):
  - `vec_abstract`  → embedding de título+metadados+abstract (1 ponto por artigo);
  - `vec_fulltext`  → embedding de chunks do PDF (vários pontos por artigo, quando permitido).
  - **Payload comum (exemplos):**
    ```json
    {
      "kind": "abstract|fulltext",
      "doi": "10.1234/abcd.2025.001",
      "title": "Título do artigo",
      "authors": ["Sobrenome, Nome", "…"],
      "year": 2025,
      "venue": "Periódico/Conferência",
      "url": "https://...",
      "keywords": ["..."],
      "license": "CC-BY|closed|unknown",
      "is_open_access": true,
      "access_level": "metadata|fulltext",
      "source": "bibtex|pdf",
      "abstract_block": "...",     
      "text": "...",               
      "chunk_index": 0,            
      "chunk_of": "10.1234/..."    
    }
    ```

---

## Uso por **CLI**

### 1) Ingerir BibTeX (sempre)
Coloque seus arquivos `.bib` em `data/bib/` e rode:
```bash
python -m src.rag.ingest_bib --glob "./data/bib/*.bib"
```

### 2) Ingerir PDF (quando open access)
```bash
python -m src.rag.ingest_pdf   --doi 10.1234/abcd.2025.001   --title "Meu Artigo"   --pdf ./data/pdfs/meu_artigo.pdf   --open
```

> Sem `--open`, o script marca o PDF como **não-open**; por padrão, evite armazenar texto integral de PDFs fechados.

---

## Uso por **API**

### Subir a API
```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Healthcheck
```bash
curl http://127.0.0.1:8000/health
```

### Autenticação
Todas as rotas protegidas exigem `X-API-Key` (ver `utils/auth.py`).

### Ingestão via HTTP

#### 1) **POST /ingest/bib**
```bash
curl -X POST http://127.0.0.1:8000/ingest/bib   -H "X-API-Key: dev-key" -H "Content-Type: application/json"   -d @bib_payload.json
```
`bib_payload.json`:
```json
{
  "bibtex": "@article{key2025, title={Meu Artigo}, author={Silva, Joao and Souza, Maria}, year={2025}, journal={Revista X}, doi={10.1234/abcd.2025.001}, url={https://...}, abstract={Resumo aqui}, keywords={evasão; psicologia}}"
}
```

#### 2) **POST /ingest/pdf**
```bash
curl -X POST http://127.0.0.1:8000/ingest/pdf   -H "X-API-Key: dev-key"   -F "doi=10.1234/abcd.2025.001"   -F "title=Meu Artigo"   -F "is_open_access=true"   -F "file=@./data/pdfs/meu_artigo.pdf"
```

### Busca e Resposta

#### 3) **POST /search**
```bash
curl -X POST http://127.0.0.1:8000/search   -H "X-API-Key: dev-key" -H "Content-Type: application/json"   -d '{"query":"fatores de risco de evasão escolar","k":5,"mode":"hybrid"}'
```

#### 4) **POST /answer**
```bash
curl -X POST http://127.0.0.1:8000/answer   -H "X-API-Key: dev-key" -H "Content-Type: application/json"   -d '{
        "query":"quais preditores de evasão aparecem na literatura recente?",
        "k":4,
        "class_id":"91",
        "mode":"model",
        "include_notes":false,
        "task":"turma_topk"
      }'
```

---

## Boas práticas

- Para artigos fechados: ingira apenas **BibTeX/abstract**.
- Sempre verifique `citations` no `/answer` → indica se a evidência veio de `abstract` ou `fulltext`.
- Controle custo ajustando `MAX_TOKENS_ABSTRACT`, `MAX_CHUNK_TOKENS`, `CHUNK_OVERLAP_TOKENS`.

---

## Troubleshooting rápido

- `OPENAI_API_KEY` ausente → defina no `.env`.
- `PyMuPDF` ou `bibtexparser` não instalados → instale via `pip`.
- Qdrant off → suba o container.
- Confirme que `EMBED_DIM` corresponde ao modelo de embedding usado.

---
