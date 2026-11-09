# dev (backend - FastAPI)

Arquitetura em camadas, organizada por módulos em `app/modules/<modulo>/`:

- `router.py`     → apresentação (HTTP)
- `service.py`    → regras de negócio
- `repository.py` → acesso a dados
- `schemas.py`    → contratos (Pydantic)

## Rodar
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Docs: http://localhost:8000/docs

## Dados e índice vetorial
- `data/` → base de conhecimento (documentos-fonte `.txt`/`.pdf`), versionada.
- `chroma_db/` → índice ChromaDB, **não versionado**. Gere/atualize rodando `pesquisa/rag_pipeline_semana2.ipynb`.
- Variáveis de ambiente (`OPENAI_API_KEY` ou `GOOGLE_API_KEY`) ficam em `dev/.env` (veja `.env.example`).
