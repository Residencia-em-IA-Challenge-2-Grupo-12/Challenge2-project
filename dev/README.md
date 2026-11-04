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
