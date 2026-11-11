"""Gera as respostas das 4 formas de usar (ou não) o RAG, no mesmo formato da camada de geração.

Uso (a partir de pesquisa/):
    python -m metricas.comparacao.modos calibrar              # distâncias e corte do porteiro
    python -m metricas.comparacao.modos llm_puro              # T1: só o LLM, sem base
    python -m metricas.comparacao.modos so_busca              # T2: só a busca, sem LLM
    python -m metricas.comparacao.modos rag_llm               # T3: RAG + LLM (k=5, prompt do notebook)
    python -m metricas.comparacao.modos rag_otimizado         # T4: porteiro de distância + prompt revisado
    python -m metricas.comparacao.modos rag_llm_k3            # referência: rodada k=3 da semana passada
    python -m metricas.comparacao.modos llm_puro --limite 3   # teste rápido

T3 reaproveita relatorios/geracao_respostas_qwen3_bge_k5.json (mesma configuração, temperature=0).
Para gerar de novo, use --regerar.

Todos usam o notebook como fonte do pipeline (chunking 500, bge-m3, ChromaDB) e o mesmo gerador
(OLLAMA_LLM_MODEL, padrão qwen3:8b). Saída: relatorios/comparacao_respostas_<modo>.json.
"""

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence, Tuple

from metricas.dataset.loader import Pergunta, load_gabarito, nome_fonte
from metricas.geracao.metrics import FRASE_RECUSA
from metricas.recuperacao import metrics as mr
from metricas.recuperacao.runner import RELATORIOS_DIR, contar_tokens

K = 5
RECUSA = FRASE_RECUSA + " para responder a essa pergunta."
MODOS = ("llm_puro", "so_busca", "rag_llm", "rag_otimizado")
REFERENCIA = "rag_llm_k3"  # rodada da semana passada (k=3), só como referência

# T1: mesmo papel e mesmas regras de forma do notebook, mas sem contexto. Ele ganha a mesma
# chance de recusar (frase exata), para a comparação de recusa ser justa.
PROMPT_LLM_PURO = f"""Você é um Mentor Técnico de Estudos e Produtividade altamente rigoroso.
Sua tarefa é responder à pergunta do estudante sobre técnicas de estudo, aprendizagem e produtividade.

Regras obrigatórias:
1. Se a pergunta não for sobre estudos, aprendizagem ou produtividade, ou se você não souber responder com segurança, responda exatamente: "{RECUSA}"
2. Organize a resposta em tópicos claros e, quando possível, mencione a fonte ([Fonte: ...]) da informação.
"""

# T4: mantém as regras 1 e 2 do notebook e ataca a diluição observada com k=5 (q12, q03):
# responder primeiro o que foi perguntado e priorizar as instruções mais ligadas à pergunta.
PROMPT_OTIMIZADO = f"""Você é um Mentor Técnico de Estudos e Produtividade altamente rigoroso.
Sua tarefa é responder à pergunta do estudante utilizando ESTRITAMENTE as informações presentes nos fragmentos do CONTEXTO RECUPERADO abaixo.

Regras obrigatórias:
1. Não invente informações, não extrapole além do texto e não utilize conhecimentos externos.
2. Se a resposta não estiver contida no contexto recuperado, responda exatamente: "{RECUSA}"
3. Comece com 1 ou 2 frases que respondem diretamente ao que foi perguntado.
4. Depois, liste em tópicos as recomendações práticas MAIS ligadas à pergunta, priorizando instruções concretas (números, passos, quando/onde/como fazer).
5. Ignore os fragmentos que não tratam do que foi perguntado, mesmo que estejam no contexto.
6. Termine cada tópico com a fonte utilizada no formato [Fonte: nome_do_arquivo].
"""


# ------------------------------------------------------------------------------------------
# Pipeline do notebook
# ------------------------------------------------------------------------------------------

def carregar_pipeline(ate: str = "geracao") -> Dict:
    """Executa o notebook (reindexa dev/chroma_db com a config padrão: bge-m3, chunk 500)."""
    from metricas.recuperacao.notebook_loader import executar_notebook

    os.environ["TOP_K"] = str(K)
    return executar_notebook(ate=ate)


def buscar(ns: Dict, pergunta: str, k: int = K) -> Tuple[list, List[float]]:
    pares = ns["vectorstore"].similarity_search_with_score(pergunta, k=k)
    return [d for d, _ in pares], [float(s) for _, s in pares]


def llm_gerador():
    from langchain_ollama import ChatOllama

    modelo = os.getenv("OLLAMA_LLM_MODEL", "qwen3:8b")
    return ChatOllama(model=modelo, temperature=0, reasoning=False), modelo


# ------------------------------------------------------------------------------------------
# Porteiro de distância (recusa antes do LLM)
# ------------------------------------------------------------------------------------------

def calibrar(ns: Optional[Dict] = None, verbose: bool = True) -> Dict:
    """Escolhe o corte tau SEM usar as 28 perguntas de teste.

    Calibração: as 36 paráfrases (variações) das perguntas em escopo, que não são avaliadas
    nos 4 testes. tau = maior d1 entre elas, ou seja, o corte não recusa nenhuma paráfrase válida.
    Depois, só para diagnóstico, mostra onde ficam as 28 perguntas de teste em relação ao corte.
    """
    ns = ns or carregar_pipeline(ate="indice")
    gabarito = load_gabarito()
    d1 = lambda texto: buscar(ns, texto, 1)[1][0]

    variacoes = [(p.id, v, d1(v)) for p in gabarito if not p.fora_de_escopo for v in p.variacoes]
    tau = max(d for _, _, d in variacoes)
    dentro = [(p.id, d1(p.pergunta)) for p in gabarito if not p.fora_de_escopo]
    fora = [(p.id, d1(p.pergunta)) for p in gabarito if p.fora_de_escopo]

    res = {
        "tau": round(tau, 4),
        "calibracao": "maior d1 entre as 36 paráfrases em escopo (fora do conjunto de teste)",
        "variacao_mais_distante": max(variacoes, key=lambda x: x[2])[:2],
        "teste_em_escopo_recusadas": [i for i, d in dentro if d > tau],
        "teste_fora_barradas": [i for i, d in fora if d > tau],
        "d1_em_escopo": {i: round(d, 4) for i, d in dentro},
        "d1_fora": {i: round(d, 4) for i, d in fora},
        "d1_variacoes": sorted(round(d, 4) for _, _, d in variacoes),
    }
    if verbose:
        print(f"tau = {res['tau']}  ({res['calibracao']})")
        print(f"  paráfrase mais distante: {res['variacao_mais_distante']}")
        print(f"  em escopo (teste): máx d1 = {max(d for _, d in dentro):.4f}; recusadas pelo porteiro: {res['teste_em_escopo_recusadas']}")
        print(f"  fora de escopo:    mín d1 = {min(d for _, d in fora):.4f}; barradas pelo porteiro: "
              f"{len(res['teste_fora_barradas'])}/{len(fora)} {res['teste_fora_barradas']}")
        for i, d in sorted(fora, key=lambda x: x[1]):
            print(f"    {i} d1={d:.4f} {'BARRADA' if d > tau else 'passa para o LLM'}")
    with open(RELATORIOS_DIR / "comparacao_porteiro.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    return res


# ------------------------------------------------------------------------------------------
# Os 4 modos. Cada um devolve (contexto, resposta, recuperados, d1) para uma pergunta.
# ------------------------------------------------------------------------------------------

def _registro(p: Pergunta, contexto: str, resposta: str, recuperados: List[str], segundos: float,
              d1: Optional[float] = None, chamou_llm: bool = True) -> Dict:
    return {
        "id": p.id, "tipo": p.tipo, "pergunta": p.pergunta,
        "esperadas": p.fontes_esperadas, "pontos_chave": p.pontos_chave,
        "recuperados": recuperados, "contexto": contexto, "resposta": resposta,
        "latencia_s": round(segundos, 2), "tokens_resposta": contar_tokens(resposta),
        "tokens_contexto": contar_tokens(contexto), "d1": d1, "chamou_llm": chamou_llm,
    }


def modo_llm_puro(perguntas: Sequence[Pergunta]) -> Tuple[Dict, List[Dict]]:
    from langchain_core.messages import HumanMessage, SystemMessage

    llm, modelo = llm_gerador()
    regs = []
    for n, p in enumerate(perguntas, 1):
        print(f"  [{n}/{len(perguntas)}] {p.id}", flush=True)
        t = time.perf_counter()
        resp = llm.invoke([SystemMessage(PROMPT_LLM_PURO), HumanMessage(p.pergunta)]).content
        regs.append(_registro(p, "", resp, [], time.perf_counter() - t))
    return {"llm": modelo, "embedding": None, "top_k": 0, "prompt": "PROMPT_LLM_PURO"}, regs


def _resposta_extrativa(docs) -> str:
    """O que o estudante veria sem LLM: os trechos crus, na ordem da busca, com a fonte de cada um."""
    partes = []
    for i, d in enumerate(docs, 1):
        fonte = nome_fonte(d.metadata.get("source", ""))
        partes.append(f"{i}. {d.page_content.strip()} [Fonte: {fonte}]")
    return "\n\n".join(partes)


def modo_so_busca(perguntas: Sequence[Pergunta], ns: Dict, tau: float) -> Tuple[Dict, List[Dict]]:
    regs = []
    for p in perguntas:
        t = time.perf_counter()
        docs, dists = buscar(ns, p.pergunta)
        barrada = dists[0] > tau
        resp = RECUSA if barrada else _resposta_extrativa(docs)
        ctx = ns["format_retrieved_docs"](docs)
        regs.append(_registro(p, ctx, resp, [nome_fonte(d.metadata.get("source", "")) for d in docs],
                              time.perf_counter() - t, dists[0], chamou_llm=False))
    cfg = {"llm": None, "embedding": ns["embedding_model"].model, "chunk_size": ns["CHUNK_SIZE"], "top_k": K, "tau": tau}
    return cfg, regs


def modo_rag(perguntas: Sequence[Pergunta], ns: Dict, prompt_sistema: str, tau: Optional[float]) -> Tuple[Dict, List[Dict]]:
    """RAG + LLM. Com `tau`, o porteiro recusa antes de chamar o LLM quando d1 > tau."""
    from langchain_core.prompts import ChatPromptTemplate

    llm, modelo = llm_gerador()
    prompt = ChatPromptTemplate.from_messages([("system", prompt_sistema), ("human", ns["RAG_USER_PROMPT"])])
    chain = prompt | llm
    regs = []
    for n, p in enumerate(perguntas, 1):
        print(f"  [{n}/{len(perguntas)}] {p.id}", flush=True)
        t = time.perf_counter()
        docs, dists = buscar(ns, p.pergunta)
        ctx = ns["format_retrieved_docs"](docs)
        fontes = [nome_fonte(d.metadata.get("source", "")) for d in docs]
        if tau is not None and dists[0] > tau:
            regs.append(_registro(p, ctx, RECUSA, fontes, time.perf_counter() - t, dists[0], chamou_llm=False))
            continue
        resp = chain.invoke({"context": ctx, "question": p.pergunta}).content
        regs.append(_registro(p, ctx, resp, fontes, time.perf_counter() - t, dists[0]))
    cfg = {"llm": modelo, "embedding": ns["embedding_model"].model, "chunk_size": ns["CHUNK_SIZE"], "top_k": K, "tau": tau}
    return cfg, regs


def rag_llm_reaproveitado(perguntas: Sequence[Pergunta], arquivo: str = "geracao_respostas_qwen3_bge_k5.json") -> Tuple[Dict, List[Dict]]:
    with open(RELATORIOS_DIR / arquivo, encoding="utf-8") as fh:
        dados = json.load(fh)
    ids = {p.id for p in perguntas}
    regs = []
    for r in dados["respostas"]:
        if r["id"] not in ids:
            continue
        if isinstance(r["recuperados"], str):  # gravado como repr de lista numa versão antiga
            import ast
            r["recuperados"] = ast.literal_eval(r["recuperados"])
        r.setdefault("tokens_contexto", contar_tokens(r["contexto"]))
        r.setdefault("chamou_llm", True)
        r.setdefault("d1", None)
        regs.append(r)
    return {**dados["config"], "prompt": "RAG_SYSTEM_PROMPT (notebook)", "reaproveitado_de": arquivo}, regs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modo", choices=("calibrar", *MODOS, REFERENCIA))
    ap.add_argument("--limite", type=int)
    ap.add_argument("--regerar", action="store_true", help="rag_llm: gera de novo em vez de reaproveitar")
    args = ap.parse_args()

    perguntas = load_gabarito()[: args.limite] if args.limite else load_gabarito()

    if args.modo == "calibrar":
        calibrar()
        return
    if args.modo == "llm_puro":
        cfg, regs = modo_llm_puro(perguntas)
    elif args.modo == REFERENCIA:
        cfg, regs = rag_llm_reaproveitado(perguntas, "geracao_respostas_qwen3_bge.json")
    elif args.modo == "rag_llm" and not args.regerar:
        cfg, regs = rag_llm_reaproveitado(perguntas)
    else:
        ns = carregar_pipeline()
        if args.modo == "rag_llm":
            cfg, regs = modo_rag(perguntas, ns, ns["RAG_SYSTEM_PROMPT"], tau=None)
        else:
            tau = calibrar(ns, verbose=False)["tau"]
            if args.modo == "so_busca":
                cfg, regs = modo_so_busca(perguntas, ns, tau)
            else:
                cfg, regs = modo_rag(perguntas, ns, PROMPT_OTIMIZADO, tau)
                cfg["prompt"] = "PROMPT_OTIMIZADO"

    cfg["modo"] = args.modo
    destino = RELATORIOS_DIR / f"comparacao_respostas_{args.modo}.json"
    with open(destino, "w", encoding="utf-8") as fh:
        json.dump({"config": cfg, "respostas": regs}, fh, ensure_ascii=False, indent=2)
    lat = [r["latencia_s"] for r in regs]
    print(f"{len(regs)} respostas em {destino}  (latência p50={mr.percentil(lat, 50):.2f}s)")


if __name__ == "__main__":
    main()
