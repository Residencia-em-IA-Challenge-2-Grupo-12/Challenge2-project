"""Avalia a GERAÇÃO do RAG: o LLM usou bem o contexto que recebeu?

Uso (a partir de pesquisa/):
    python -m metricas.geracao.runner --nome qwen3_bge                 # gera e avalia
    python -m metricas.geracao.runner --nome qwen3_bge --so-gerar      # só gera as respostas
    python -m metricas.geracao.runner --nome qwen3_bge --so-avaliar    # reavalia respostas já geradas
    python -m metricas.geracao.runner --nome teste --limite 3          # rodada rápida com 3 perguntas

Fase 1 (gerar): executa o notebook até a chain RAG (reindexa dev/chroma_db) e responde todas as
perguntas do gabarito. Salva pergunta, chunks recuperados, contexto exato enviado ao LLM e resposta
em relatorios/geracao_respostas_<nome>.json.

Fase 2 (avaliar): sobre esse arquivo, calcula
  - sem LLM:  recusa (correta e indevida), presença e precisão das citações [Fonte: ...];
  - com juiz: fidelidade (afirmações sustentadas pelo contexto) e completude (pontos-chave cobertos).
Salva em relatorios/geracao_<nome>.json e .csv.

LLM gerador: LLM_PROVIDER / OLLAMA_LLM_MODEL no dev/.env (padrão qwen3:8b).
Juiz: OLLAMA_JUIZ_MODEL (padrão gemma3:12b), sempre diferente do gerador.
"""

import argparse
import csv
import json
import time
from typing import Dict, List, Optional

from metricas.dataset.loader import load_gabarito, nome_fonte
from metricas.recuperacao import metrics as mr
from metricas.recuperacao.runner import RELATORIOS_DIR, contar_tokens

from . import metrics as mg


def gerar(nome: str, limite: Optional[int] = None) -> List[Dict]:
    from metricas.recuperacao.notebook_loader import executar_notebook

    ns = executar_notebook(ate="geracao")
    chain, formatar = ns["rag_chain"], ns["format_retrieved_docs"]
    config = {
        "llm": getattr(ns["llm"], "model", type(ns["llm"]).__name__),
        "embedding": getattr(ns["embedding_model"], "model", type(ns["embedding_model"]).__name__),
        "chunk_size": ns["CHUNK_SIZE"],
        "top_k": ns["TOP_K"],
    }
    perguntas = load_gabarito()[:limite] if limite else load_gabarito()

    registros = []
    for n, p in enumerate(perguntas, start=1):
        print(f"  [{n}/{len(perguntas)}] {p.id} gerando...", flush=True)
        inicio = time.perf_counter()
        saida = chain.invoke(p.pergunta)
        segundos = round(time.perf_counter() - inicio, 2)
        docs = saida["context_docs"]
        registros.append({
            "id": p.id, "tipo": p.tipo, "pergunta": p.pergunta,
            "esperadas": p.fontes_esperadas, "pontos_chave": p.pontos_chave,
            "recuperados": [nome_fonte(d.metadata.get("source", "")) for d in docs],
            "contexto": formatar(docs),  # exatamente o texto que o LLM recebeu
            "resposta": saida["answer"],
            "latencia_s": segundos,
            "tokens_resposta": contar_tokens(saida["answer"]),
        })

    destino = RELATORIOS_DIR / f"geracao_respostas_{nome}.json"
    with open(destino, "w", encoding="utf-8") as fh:
        json.dump({"config": config, "respostas": registros}, fh, ensure_ascii=False, indent=2)
    print(f"Respostas salvas em {destino}")
    return registros


def avaliar_registro(r: Dict, juiz) -> Dict:
    em_escopo = r["tipo"] != "fora_de_escopo"
    recusou = mg.e_recusa(r["resposta"])
    citados = mg.citacoes(r["resposta"])
    recuperados = {mg.numero_artigo(f) for f in r["recuperados"]}
    esperados = {mg.numero_artigo(f) for f in r["esperadas"]}

    av: Dict = {"recusou": recusou, "citados": citados, "citacoes_brutas": mg.citacoes_brutas(r["resposta"])}
    if citados:
        av["precisao_citacao"] = mg.precisao_citacao(citados, recuperados)
    if em_escopo and not recusou:
        av["tem_citacao"] = 1.0 if citados else 0.0
        av["cita_esperada"] = mg.cita_esperada(citados, esperados)

    # Fidelidade: só faz sentido se a resposta afirmou algo (uma recusa pura não tem o que checar)
    if not recusou:
        try:
            afirmacoes = juiz.fidelidade(r["contexto"], r["resposta"])
            av["afirmacoes"] = afirmacoes
            if afirmacoes:
                av["fidelidade"] = sum(a["sustentada"] for a in afirmacoes) / len(afirmacoes)
        except Exception as exc:
            av["erro_fidelidade"] = str(exc)

    # Completude: uma recusa em pergunta válida não cobre nada (vale 0 sem consultar o juiz)
    if em_escopo:
        if recusou:
            av["pontos_cobertos"] = [False] * len(r["pontos_chave"])
        else:
            try:
                av["pontos_cobertos"] = juiz.completude(r["resposta"], r["pontos_chave"])
            except Exception as exc:
                av["erro_completude"] = str(exc)
        if "pontos_cobertos" in av:
            av["completude"] = sum(av["pontos_cobertos"]) / len(av["pontos_cobertos"])
    return av


def _media(valores) -> Optional[float]:
    valores = [v for v in valores if v is not None]
    return round(mr.media(valores), 4) if valores else None


def agregar(registros: List[Dict]) -> Dict:
    dentro = [r for r in registros if r["tipo"] != "fora_de_escopo"]
    fora = [r for r in registros if r["tipo"] == "fora_de_escopo"]
    av = lambda r, chave: r["avaliacao"].get(chave)

    resumo: Dict = {
        "em_escopo": {
            "n": len(dentro),
            "recusa_indevida": _media(1.0 if av(r, "recusou") else 0.0 for r in dentro),
            "tem_citacao": _media(av(r, "tem_citacao") for r in dentro),
            "precisao_citacao": _media(av(r, "precisao_citacao") for r in dentro),
            "cita_esperada": _media(av(r, "cita_esperada") for r in dentro),
            "fidelidade": _media(av(r, "fidelidade") for r in dentro),
            "completude": _media(av(r, "completude") for r in dentro),
        },
        "fora_de_escopo": {
            "n": len(fora),
            "recusa_correta": _media(1.0 if av(r, "recusou") else 0.0 for r in fora),
            # Quando NÃO recusou, o que disse estava no contexto? (baixo = inventou resposta)
            "fidelidade_quando_responde": _media(av(r, "fidelidade") for r in fora),
        },
        "por_tipo": {},
        "custo": {
            "latencia_s_p50": round(mr.percentil([r["latencia_s"] for r in registros], 50), 2),
            "latencia_s_p95": round(mr.percentil([r["latencia_s"] for r in registros], 95), 2),
            "tokens_resposta_media": round(mr.media(r["tokens_resposta"] for r in registros)),
        },
    }
    for tipo in sorted({r["tipo"] for r in dentro}):
        grupo = [r for r in dentro if r["tipo"] == tipo]
        resumo["por_tipo"][tipo] = {
            "n": len(grupo),
            "fidelidade": _media(av(r, "fidelidade") for r in grupo),
            "completude": _media(av(r, "completude") for r in grupo),
        }
    return resumo


def imprimir(registros: List[Dict], resumo: Dict) -> None:
    fmt = lambda v: "—" if v is None else f"{v:.2f}"
    print(f"\n{'id':<5}{'tipo':<15}{'recusou':<9}{'citados':<14}{'cit.ok':<8}{'fidel.':<8}{'complet.':<9}lat(s)")
    for r in registros:
        a = r["avaliacao"]
        print(f"{r['id']:<5}{r['tipo']:<15}{'sim' if a['recusou'] else 'não':<9}{','.join(a['citados']) or '—':<14}"
              f"{fmt(a.get('precisao_citacao')):<8}{fmt(a.get('fidelidade')):<8}{fmt(a.get('completude')):<9}{r['latencia_s']}")

    e, f, c = resumo["em_escopo"], resumo["fora_de_escopo"], resumo["custo"]
    print(f"\nEM ESCOPO ({e['n']})")
    print(f"  fidelidade      {fmt(e['fidelidade'])}   (afirmações sustentadas pelo contexto)")
    print(f"  completude      {fmt(e['completude'])}   (pontos-chave cobertos)")
    print(f"  tem citação     {fmt(e['tem_citacao'])}   precisão da citação {fmt(e['precisao_citacao'])}   cita esperada {fmt(e['cita_esperada'])}")
    print(f"  recusa indevida {fmt(e['recusa_indevida'])}")
    print(f"FORA DE ESCOPO ({f['n']})")
    print(f"  recusa correta  {fmt(f['recusa_correta'])}   fidelidade quando responde {fmt(f['fidelidade_quando_responde'])}")
    print("POR TIPO")
    for tipo, v in resumo["por_tipo"].items():
        print(f"  {tipo:<15} n={v['n']}  fidelidade={fmt(v['fidelidade'])}  completude={fmt(v['completude'])}")
    print(f"CUSTO  latência p50={c['latencia_s_p50']} s  p95={c['latencia_s_p95']} s  tokens/resposta={c['tokens_resposta_media']}")


def avaliar(nome: str) -> None:
    from .juiz import Juiz

    with open(RELATORIOS_DIR / f"geracao_respostas_{nome}.json", encoding="utf-8") as fh:
        dados = json.load(fh)
    registros = dados["respostas"]
    juiz = Juiz()
    if juiz.modelo == dados["config"]["llm"]:
        raise SystemExit(f"O juiz ({juiz.modelo}) não pode ser o mesmo modelo do gerador.")

    for n, r in enumerate(registros, start=1):
        print(f"  [{n}/{len(registros)}] {r['id']} avaliando...", flush=True)
        r["avaliacao"] = avaliar_registro(r, juiz)

    resumo = agregar(registros)
    imprimir(registros, resumo)

    config = {**dados["config"], "juiz": juiz.modelo}
    json_path = RELATORIOS_DIR / f"geracao_{nome}.json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump({"config": config, "resumo": resumo, "perguntas": registros}, fh, ensure_ascii=False, indent=2)
    csv_path = RELATORIOS_DIR / f"geracao_{nome}.csv"
    colunas = ["recusou", "citados", "precisao_citacao", "tem_citacao", "cita_esperada", "fidelidade", "completude"]
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "tipo", *colunas, "latencia_s", "tokens_resposta"])
        for r in registros:
            a = r["avaliacao"]
            w.writerow([r["id"], r["tipo"], *[a.get(c) for c in colunas], r["latencia_s"], r["tokens_resposta"]])
    print(f"\nSalvo em:\n  {json_path}\n  {csv_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nome", default="baseline")
    ap.add_argument("--limite", type=int, help="só as N primeiras perguntas (teste rápido)")
    fase = ap.add_mutually_exclusive_group()
    fase.add_argument("--so-gerar", action="store_true")
    fase.add_argument("--so-avaliar", action="store_true")
    args = ap.parse_args()

    if not args.so_avaliar:
        gerar(args.nome, args.limite)
    if not args.so_gerar:
        avaliar(args.nome)


if __name__ == "__main__":
    main()
