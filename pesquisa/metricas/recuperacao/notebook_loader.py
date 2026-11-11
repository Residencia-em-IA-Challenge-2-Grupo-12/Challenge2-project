"""Executa as células do notebook que montam o índice e devolve o namespace resultante.

O runner depende do notebook `pesquisa/rag_pipeline_semana2.ipynb` como fonte do pipeline,
em vez de um módulo Python separado. Assim, o que é medido é exatamente o que o notebook faz.

Células executadas (identificadas pelo conteúdo, não pelo índice, para sobreviver a edições):
  - setup do ambiente      -> define ACTIVE_PROVIDER
  - ingestão               -> define raw_documents
  - chunking               -> define document_chunks, CHUNK_SIZE, CHUNK_OVERLAP
  - embeddings + ChromaDB  -> define embedding_model, vectorstore, TOP_K (REINDEXA dev/chroma_db)
  - chain RAG (opcional)   -> define llm, rag_chain, RAG_SYSTEM_PROMPT (só com ate="geracao")

Células ignoradas de propósito:
  - a que grava os 20 artigos em ../dev/data (sobrescreveria edições manuais da base)
  - a de `%pip install` (magic do Jupyter, e a instalação é feita fora)
  - a da demonstração com as 3 personas (chama o LLM só para exibir respostas)
"""

import contextlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List

NOTEBOOK_PATH = Path(__file__).resolve().parents[2] / "rag_pipeline_semana2.ipynb"

# (rótulo, trecho que identifica a célula, nome que ela precisa deixar no namespace)
ETAPAS = [
    ("setup", "ACTIVE_PROVIDER = setup_environment()", "ACTIVE_PROVIDER"),
    ("ingestao", "raw_documents = load_knowledge_base(", "raw_documents"),
    ("chunking", "document_chunks = split_documents(", "document_chunks"),
    ("indice", "vectorstore = build_or_update_vectorstore(", "vectorstore"),
    ("geracao", "rag_chain = RunnableParallel(", "rag_chain"),
]


def selecionar_celulas(path: Path = NOTEBOOK_PATH) -> List[Dict[str, str]]:
    """Devolve, na ordem do notebook, as células de código das ETAPAS (rótulo + código)."""
    with open(path, encoding="utf-8") as fh:
        celulas = [c for c in json.load(fh)["cells"] if c["cell_type"] == "code"]

    selecionadas = []
    for rotulo, marcador, _ in ETAPAS:
        achadas = [c for c in celulas if marcador in "".join(c["source"])]
        if len(achadas) != 1:
            raise RuntimeError(
                f"Etapa '{rotulo}': esperava 1 célula com '{marcador}', achei {len(achadas)}. "
                "O notebook mudou? Ajuste ETAPAS em notebook_loader.py."
            )
        selecionadas.append({"rotulo": rotulo, "codigo": "".join(achadas[0]["source"])})
    return selecionadas


def executar_notebook(path: Path = NOTEBOOK_PATH, silencioso: bool = True, ate: str = "indice") -> Dict[str, Any]:
    """Executa as etapas num namespace único e devolve esse namespace.

    O notebook usa caminhos relativos a `pesquisa/` (ex.: ../dev/data), então o diretório
    de trabalho é trocado para lá durante a execução e restaurado depois.
    `ate` para depois da etapa indicada (ex.: "chunking" devolve os chunks sem reindexar o ChromaDB).
    """
    rotulos = [r for r, _, _ in ETAPAS]
    etapas = ETAPAS[: rotulos.index(ate) + 1]
    celulas = selecionar_celulas(path)[: len(etapas)]
    ns: Dict[str, Any] = {"__name__": "__notebook__"}
    cwd = os.getcwd()
    os.chdir(path.parent)
    try:
        for c in celulas:
            buf = io.StringIO()
            ctx = contextlib.redirect_stdout(buf) if silencioso else contextlib.nullcontext()
            with ctx:
                exec(compile(c["codigo"], f"<notebook:{c['rotulo']}>", "exec"), ns)
    finally:
        os.chdir(cwd)

    faltando = [nome for _, _, nome in etapas if nome not in ns]
    if faltando:
        raise RuntimeError(f"Notebook executado, mas não definiu: {faltando}")
    return ns
