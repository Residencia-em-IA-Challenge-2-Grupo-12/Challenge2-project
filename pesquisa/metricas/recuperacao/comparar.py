"""Roda o runner para várias combinações de embedding x chunking e compara lado a lado.

Uso (a partir de pesquisa/):
    python -m metricas.recuperacao.comparar
    python -m metricas.recuperacao.comparar --modelos bge-m3 qwen3-embedding:0.6b --chunks 300 500 800

Cada combinação reindexa dev/chroma_db (via notebook) e salva seu próprio JSON/CSV em relatorios/
com o nome `<modelo>_c<chunk_size>`. No fim, imprime uma tabela e grava `comparativo.csv`.

Os modelos são do Ollama (EMBEDDING_PROVIDER=ollama precisa estar no dev/.env).
O overlap acompanha o chunk (10%), como no notebook (500/50).
"""

import argparse
import csv
import os
import time
from typing import Dict, List

from metricas.dataset.loader import load_gabarito

from .robustez import avaliar_robustez
from .runner import RELATORIOS_DIR, agregar, avaliar, make_search_from_notebook, salvar

KS = [1, 3, 5]


def rodar(modelo: str, chunk_size: int) -> Dict:
    os.environ["OLLAMA_EMBEDDING_MODEL"] = modelo
    os.environ["CHUNK_SIZE"] = str(chunk_size)
    os.environ["CHUNK_OVERLAP"] = str(chunk_size // 10)

    inicio = time.perf_counter()
    search, config = make_search_from_notebook()
    perguntas = load_gabarito()
    resultados = avaliar(search, perguntas, KS)
    segundos = round(time.perf_counter() - inicio, 1)

    resumo = agregar(resultados)
    resumo["robustez"] = rb = avaliar_robustez(search, perguntas, k=3)
    config["ks"] = KS
    nome = f"{modelo.replace(':', '-')}_c{chunk_size}"
    salvar(resultados, resumo, nome, config)

    g = resumo["geral"]
    fora = [d for d in resumo["diagnostico_fora_de_escopo"].values() if d is not None]
    return {
        "modelo": modelo,
        "chunk": chunk_size,
        "chunks": config["chunks_no_indice"],
        "hit@1": g["hit@1"],
        "hit@3": g["hit@3"],
        "mrr@3": g["rr@3"],
        "ndcg@3": g["ndcg@3"],
        "ndcg@5": g["ndcg@5"],
        "prec@3": g["precision@3"],
        "rec@3": g["recall@3"],
        "rec@3_multi": resumo["por_tipo"].get("multi_fonte", {}).get("recall@3"),
        "rec@5": g["recall@5"],
        "cob@3": g["cobertura@3"],
        "cob@5": g["cobertura@5"],
        # Margem > 0: dá para separar fora de escopo por um corte de distância. Só compare dentro do mesmo modelo.
        "margem": round(min(fora) - resumo["distancia_top1_em_escopo_max"], 4) if fora else None,
        "auc": resumo.get("auc_distancia"),
        "recusa_ok": resumo.get("recusa_sem_erro"),
        "hit@3_ruido": rb["ruido"]["hit"],
        "hit@3_paraf": rb["parafrases"]["hit"],
        "mrr@3_paraf": rb["parafrases"]["mrr"],
        "jaccard": rb["parafrases"]["jaccard_medio"],
        "lat_p50_ms": resumo["custo"]["latencia_ms_p50"],
        "lat_p95_ms": resumo["custo"]["latencia_ms_p95"],
        "tokens@3": round(g["tokens@3"]),
        "tokens@5": round(g["tokens@5"]),
        "index_s": config["indexacao_s"],
        "vetores_kb": config["vetores_kb"],
        "seg": segundos,
    }


GRUPOS = {
    "QUALIDADE": ["hit@1", "hit@3", "mrr@3", "ndcg@3", "prec@3", "rec@3", "rec@3_multi", "cob@3", "cob@5"],
    "RECUSA E ROBUSTEZ": ["margem", "auc", "recusa_ok", "hit@3_ruido", "hit@3_paraf", "mrr@3_paraf", "jaccard"],
    "CUSTO": ["chunks", "lat_p50_ms", "lat_p95_ms", "tokens@3", "tokens@5", "index_s", "vetores_kb"],
}


def imprimir(linhas: List[Dict]) -> None:
    for titulo, grupo in GRUPOS.items():
        cols = ["modelo", "chunk", *grupo]
        larg = {c: max(len(c), *(len(str(l[c])) for l in linhas)) + 2 for c in cols}
        print(f"\n{titulo}")
        print("".join(c.ljust(larg[c]) for c in cols))
        for l in linhas:
            print("".join(str(l[c]).ljust(larg[c]) for c in cols))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelos", nargs="+", default=["bge-m3", "qwen3-embedding:0.6b"])
    ap.add_argument("--chunks", type=int, nargs="+", default=[300, 500, 800])
    args = ap.parse_args()

    linhas = []
    for modelo in args.modelos:
        for chunk in args.chunks:
            print(f"> {modelo} | chunk_size={chunk}", flush=True)
            linhas.append(rodar(modelo, chunk))

    imprimir(linhas)
    destino = RELATORIOS_DIR / "comparativo.csv"
    with open(destino, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)
    print(f"\nSalvo em: {destino}")


if __name__ == "__main__":
    main()
