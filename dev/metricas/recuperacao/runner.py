"""Executa o retriever sobre o gabarito, calcula as métricas e salva os resultados.

Uso (a partir de dev/):
    python -m metricas.recuperacao.runner --nome baseline
    python -m metricas.recuperacao.runner --nome baseline --ks 1 3 5

Fluxo, passo a passo:
  1. Carrega o gabarito (20 perguntas).
  2. Para cada pergunta, busca os `max(ks)` chunks mais próximos UMA vez (com distância L2).
  3. Reduz cada chunk ao nome do arquivo de origem (`nome_fonte`).
  4. Calcula as métricas para cada k usando só os k primeiros (um prefixo do mesmo ranking).
  5. Perguntas fora de escopo ficam de fora das médias (não há "acerto" possível);
     guardamos só a distância do melhor chunk, que serve de diagnóstico.
  6. Agrega (geral e por tipo de pergunta) e grava em relatorios/ como JSON e CSV.
"""

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

from metricas.dataset.loader import Pergunta, load_gabarito, nome_fonte

from . import metrics as m

RELATORIOS_DIR = Path(__file__).resolve().parents[1] / "relatorios"

# search(pergunta, n) -> [(source_do_chunk, distancia_L2), ...] ordenado do mais próximo ao mais distante
SearchFn = Callable[[str, int], List[Tuple[str, float]]]


@dataclass
class ResultadoPergunta:
    id: str
    tipo: str
    pergunta: str
    esperadas: List[str]
    recuperadas: List[str]  # nomes de arquivo, na ordem do ranking
    distancias: List[float]
    metricas: Dict[str, float] = field(default_factory=dict)  # ex.: {"hit@3": 1.0, ...}


def avaliar(search: SearchFn, perguntas: Sequence[Pergunta], ks: Sequence[int] = (1, 3, 5)) -> List[ResultadoPergunta]:
    max_k = max(ks)
    resultados: List[ResultadoPergunta] = []

    for p in perguntas:
        bruto = search(p.pergunta, max_k)  # passo 2
        recuperadas = [nome_fonte(src) for src, _ in bruto]  # passo 3
        distancias = [round(float(d), 4) for _, d in bruto]
        esperadas = set(p.fontes_esperadas)

        valores: Dict[str, float] = {}
        if not p.fora_de_escopo:  # passo 5
            for k in ks:  # passo 4
                valores[f"hit@{k}"] = m.hit_at_k(recuperadas, esperadas, k)
                valores[f"rr@{k}"] = m.reciprocal_rank(recuperadas, esperadas, k)
                valores[f"precision@{k}"] = m.precision_at_k(recuperadas, esperadas, k)
                valores[f"recall@{k}"] = m.recall_at_k(recuperadas, esperadas, k)
                valores[f"distintas@{k}"] = float(m.fontes_distintas_at_k(recuperadas, k))

        resultados.append(
            ResultadoPergunta(p.id, p.tipo, p.pergunta, sorted(esperadas), recuperadas, distancias, valores)
        )
    return resultados


def _media_por_metrica(resultados: Sequence[ResultadoPergunta]) -> Dict[str, float]:
    if not resultados:
        return {}
    chaves = resultados[0].metricas.keys()
    return {c: round(m.media(r.metricas[c] for r in resultados), 4) for c in chaves}


def agregar(resultados: Sequence[ResultadoPergunta]) -> Dict:
    """Médias gerais e por tipo. `rr@k` médio é o MRR@k."""
    em_escopo = [r for r in resultados if r.metricas]
    fora = [r for r in resultados if not r.metricas]

    resumo: Dict = {
        "n_em_escopo": len(em_escopo),
        "geral": _media_por_metrica(em_escopo),
        "por_tipo": {},
        # Distância do melhor chunk nas perguntas fora de escopo: se for tão baixa quanto a das
        # perguntas em escopo, não dá para separar "sem resposta" só pela distância.
        "diagnostico_fora_de_escopo": {r.id: (r.distancias[0] if r.distancias else None) for r in fora},
        "distancia_top1_em_escopo_media": round(m.media(r.distancias[0] for r in em_escopo if r.distancias), 4),
    }
    for tipo in sorted({r.tipo for r in em_escopo}):
        grupo = [r for r in em_escopo if r.tipo == tipo]
        resumo["por_tipo"][tipo] = {"n": len(grupo), **_media_por_metrica(grupo)}
    return resumo


def salvar(resultados: Sequence[ResultadoPergunta], resumo: Dict, nome: str, config: Dict) -> Tuple[Path, Path]:
    RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RELATORIOS_DIR / f"recuperacao_{nome}.json"
    csv_path = RELATORIOS_DIR / f"recuperacao_{nome}.csv"

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"config": config, "resumo": resumo, "perguntas": [asdict(r) for r in resultados]},
            fh, ensure_ascii=False, indent=2,
        )

    colunas = sorted({c for r in resultados for c in r.metricas})
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "tipo", "esperadas", "recuperadas_top", *colunas])
        for r in resultados:
            w.writerow([r.id, r.tipo, "|".join(r.esperadas), "|".join(r.recuperadas), *[r.metricas.get(c, "") for c in colunas]])
    return json_path, csv_path


def imprimir(resultados: Sequence[ResultadoPergunta], resumo: Dict, ks: Sequence[int]) -> None:
    k_ref = 3 if 3 in ks else ks[0]
    print(f"\nPOR PERGUNTA (k={k_ref})  ✓ = fonte esperada no top-{k_ref}")
    print(f"{'id':<5}{'tipo':<15}{'hit':<5}{'rr':<7}{'prec':<7}{'rec':<7}recuperadas")
    for r in resultados:
        if not r.metricas:
            print(f"{r.id:<5}{r.tipo:<15}{'—':<5}{'—':<7}{'—':<7}{'—':<7}{', '.join(x[:18] for x in r.recuperadas[:k_ref])}  (d1={r.distancias[0]})")
            continue
        mt = r.metricas
        marca = "✓" if mt[f"hit@{k_ref}"] else "✗"
        print(f"{r.id:<5}{r.tipo:<15}{marca:<5}{mt[f'rr@{k_ref}']:<7.2f}{mt[f'precision@{k_ref}']:<7.2f}{mt[f'recall@{k_ref}']:<7.2f}{', '.join(x[:18] for x in r.recuperadas[:k_ref])}")

    print(f"\nGERAL ({resumo['n_em_escopo']} perguntas em escopo)")
    print(f"{'k':<4}{'Hit':<8}{'MRR':<8}{'Prec':<8}{'Recall':<8}{'Distintas'}")
    for k in ks:
        g = resumo["geral"]
        print(f"{k:<4}{g[f'hit@{k}']:<8.3f}{g[f'rr@{k}']:<8.3f}{g[f'precision@{k}']:<8.3f}{g[f'recall@{k}']:<8.3f}{g[f'distintas@{k}']:.2f}")

    print(f"\nPOR TIPO (k={k_ref})")
    for tipo, v in resumo["por_tipo"].items():
        print(f"  {tipo:<15} n={v['n']}  hit={v[f'hit@{k_ref}']:.2f}  mrr={v[f'rr@{k_ref}']:.2f}  prec={v[f'precision@{k_ref}']:.2f}  recall={v[f'recall@{k_ref}']:.2f}")

    print("\nDISTÂNCIA L2 do melhor chunk (menor = mais próximo)")
    print(f"  em escopo (média): {resumo['distancia_top1_em_escopo_media']}")
    print(f"  fora de escopo   : {resumo['diagnostico_fora_de_escopo']}")


def make_search_from_rag(provider: str = None) -> Tuple[SearchFn, Dict]:
    """Liga o runner ao índice ChromaDB persistido (import tardio: exige langchain instalado)."""
    from app.modules.rag.config import CHUNK_OVERLAP, CHUNK_SIZE, setup_environment
    from app.modules.rag.embeddings import get_embedding_model
    from app.modules.rag.vectorstore import open_vectorstore

    provider = provider or setup_environment()
    emb = get_embedding_model(provider)
    store = open_vectorstore(emb)
    n_chunks = len(store.get().get("ids", []))
    if n_chunks == 0:
        raise RuntimeError("Índice vazio. Rode antes: python -m scripts.build_index")

    def search(pergunta: str, n: int) -> List[Tuple[str, float]]:
        return [(d.metadata.get("source", ""), s) for d, s in store.similarity_search_with_score(pergunta, k=n)]

    config = {
        "provider": provider,
        "embedding": type(emb).__name__,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "chunks_no_indice": n_chunks,
    }
    return search, config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", default="baseline", help="rótulo da execução (vira parte do nome do arquivo)")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 3, 5])
    args = ap.parse_args()

    search, config = make_search_from_rag()
    config["ks"] = args.ks
    perguntas = load_gabarito()
    resultados = avaliar(search, perguntas, args.ks)
    resumo = agregar(resultados)
    imprimir(resultados, resumo, args.ks)
    jp, cp = salvar(resultados, resumo, args.nome, config)
    print(f"\nSalvo em:\n  {jp}\n  {cp}")


if __name__ == "__main__":
    main()
