"""Teste manual: faz uma pergunta e mostra, lado a lado, o que cada embedding recupera.

Uso (a partir de pesquisa/):
    python -m metricas.recuperacao.perguntar                          # modo interativo
    python -m metricas.recuperacao.perguntar "como parar de procrastinar?"
    python -m metricas.recuperacao.perguntar --id q05                 # pergunta do gabarito, marca acertos
    python -m metricas.recuperacao.perguntar --chunk 300 --k 5 --modelos bge-m3

Usa a ingestão e o chunking do notebook, mas monta um índice EM MEMÓRIA por modelo:
não toca em dev/chroma_db nem em relatorios/. Modelos via Ollama.
"""

import argparse
import os
from typing import Dict, List, Optional, Sequence

from metricas.dataset.loader import load_gabarito, nome_fonte

from .notebook_loader import executar_notebook


def montar_indices(modelos: Sequence[str], chunk_size: int) -> Dict[str, object]:
    from langchain_chroma import Chroma
    from langchain_ollama import OllamaEmbeddings

    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_size // 10)
    chunks = executar_notebook(ate="chunking")["document_chunks"]
    print(f"{len(chunks)} chunks (chunk_size={chunk_size}). Indexando...", flush=True)

    indices = {}
    for modelo in modelos:
        store = Chroma(collection_name=f"manual_{modelo.replace(':', '_')}", embedding_function=OllamaEmbeddings(model=modelo))
        store.add_documents(chunks)
        indices[modelo] = store
    return indices


def mostrar(indices: Dict[str, object], pergunta: str, k: int, esperadas: Optional[List[str]] = None) -> None:
    print(f"\nPERGUNTA: {pergunta}")
    if esperadas is not None:
        print(f"ESPERADO: {', '.join(esperadas) or '(fora de escopo: nenhum artigo deveria responder)'}")
    for modelo, store in indices.items():
        print(f"\n  [{modelo}]")
        for pos, (doc, dist) in enumerate(store.similarity_search_with_score(pergunta, k=k), start=1):
            fonte = nome_fonte(doc.metadata.get("source", ""))
            marca = ("✓ " if fonte in esperadas else "  ") if esperadas else ""
            trecho = " ".join(doc.page_content.split())[:110]
            print(f"   {pos}. {marca}d={dist:.3f}  {fonte}")
            print(f"        \"{trecho}...\"")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pergunta", nargs="?", help="texto da pergunta (sem ela, entra no modo interativo)")
    ap.add_argument("--id", help="usa uma pergunta do gabarito (ex.: q05) e marca os acertos")
    ap.add_argument("--modelos", nargs="+", default=["bge-m3", "qwen3-embedding:0.6b"])
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    gabarito = {p.id: p for p in load_gabarito()}
    indices = montar_indices(args.modelos, args.chunk)

    if args.id:
        p = gabarito[args.id]
        mostrar(indices, p.pergunta, args.k, p.fontes_esperadas)
        return
    if args.pergunta:
        mostrar(indices, args.pergunta, args.k)
        return

    print("\nDigite uma pergunta, um id do gabarito (q01..q20) ou 'sair'.")
    while True:
        entrada = input("\n> ").strip()
        if entrada.lower() in {"", "sair", "q"}:
            break
        if entrada in gabarito:
            mostrar(indices, gabarito[entrada].pergunta, args.k, gabarito[entrada].fontes_esperadas)
        else:
            mostrar(indices, entrada, args.k)


if __name__ == "__main__":
    main()
