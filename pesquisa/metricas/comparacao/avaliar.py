"""Avalia e compara as formas de usar o RAG com as MESMAS métricas e o MESMO juiz.

Uso (a partir de pesquisa/):
    python -m metricas.comparacao.avaliar                  # avalia o que falta (retoma de onde parou)
    python -m metricas.comparacao.avaliar rag_otimizado    # só um modo
    python -m metricas.comparacao.avaliar --so-tabela      # só remonta a tabela com o que já foi avaliado
    python -m metricas.comparacao.avaliar --refazer        # avalia tudo de novo

Lê relatorios/comparacao_respostas_<modo>.json (gerado por metricas.comparacao.modos) e grava
relatorios/comparacao_<modo>.json com o veredito por pergunta, mais relatorios/comparacao.csv e .md.

Métricas da semana anterior (regra e juiz), mantidas:
  recusa correta / indevida, tem citação, precisão da citação, cita a esperada,
  fidelidade ao CONTEXTO (juiz) e completude (juiz).
Métricas novas:
  - fidelidade à BASE (juiz): cada afirmação é checada contra os 20 artigos inteiros. Funciona para
    qualquer modo, inclusive o LLM sem RAG, que não tem contexto para a fidelidade antiga medir.
  - relevância (juiz): fração das afirmações que ajudam a responder o que foi perguntado.
  - taxa de sucesso: % das 28 perguntas com um resultado "bom para o produto" (ver `sucesso`).
  - chamadas ao LLM e tokens de contexto (custo).
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

from metricas.dataset.loader import DATA_DIR
from metricas.geracao import metrics as mg
from metricas.recuperacao import metrics as mr
from metricas.recuperacao.runner import RELATORIOS_DIR

ORDEM = ["llm_puro", "so_busca", "rag_llm", "rag_otimizado", "rag_llm_k3"]
ROTULOS = {
    "llm_puro": "T1 · Só LLM (sem RAG)",
    "so_busca": "T2 · Só busca (sem LLM)",
    "rag_llm": "T3 · RAG + LLM (k=5)",
    "rag_otimizado": "T4 · RAG + LLM otimizado",
    "rag_llm_k3": "Ref · RAG + LLM k=3 (semana passada)",
}

# Limiares da taxa de sucesso (por pergunta em escopo)
MIN_COMPLETUDE = 2 / 3
MIN_FIDELIDADE_BASE = 0.8


def texto_base(data_dir: Path = DATA_DIR) -> str:
    return "\n\n".join(f"=== {f.name} ===\n{f.read_text(encoding='utf-8').strip()}" for f in sorted(data_dir.glob("*.txt")))


def avaliar_registro(r: Dict, juiz, base: str, modo: str) -> Dict:
    em_escopo = r["tipo"] != "fora_de_escopo"
    recusou = mg.e_recusa(r["resposta"])
    citados = mg.citacoes(r["resposta"])
    recuperados = {mg.numero_artigo(f) for f in r["recuperados"]}
    esperados = {mg.numero_artigo(f) for f in r["esperadas"]}

    av: Dict = {"recusou": recusou, "citados": citados}
    if em_escopo and not recusou:
        av["tem_citacao"] = 1.0 if citados else 0.0
        av["cita_esperada"] = mg.cita_esperada(citados, esperados)
        if citados and r["recuperados"]:
            av["precisao_citacao"] = mg.precisao_citacao(citados, recuperados)

    if not recusou:
        # Fidelidade ao contexto: não existe sem contexto (T1); é 1 por construção quando a resposta
        # é o próprio contexto (T2). Nos modos com LLM, é o juiz de sempre.
        if modo == "so_busca":
            av["fidelidade"] = 1.0
        elif r["contexto"]:
            afirm = juiz.fidelidade(r["contexto"], r["resposta"])
            av["afirmacoes_contexto"] = afirm
            if afirm:
                av["fidelidade"] = sum(a["sustentada"] for a in afirm) / len(afirm)

        afirm = juiz.relevancia(r["pergunta"], r["resposta"])
        av["afirmacoes_relevancia"] = afirm
        if afirm:
            av["relevancia"] = sum(a["relevante"] for a in afirm) / len(afirm)

        # Fidelidade à base. Com contexto, ela já está medida: o contexto É um pedaço da base, então
        # "sustentada pelo contexto" implica "sustentada pela base" (a fidelidade ao contexto é um piso).
        # Sem contexto (T1), cada afirmação é verificada contra a base inteira, uma por chamada.
        if modo == "so_busca":
            av["fidelidade_base"], av["origem_fidelidade_base"] = 1.0, "por construção (trechos da base)"
        elif r["contexto"]:
            if "fidelidade" in av:
                av["fidelidade_base"], av["origem_fidelidade_base"] = av["fidelidade"], "piso: fidelidade ao contexto"
        elif afirm:
            for a in afirm:
                a.update(juiz.sustentada_na_base(base, a["afirmacao"]))
            av["fidelidade_base"] = sum(a["sustentada"] for a in afirm) / len(afirm)
            av["origem_fidelidade_base"] = "juiz, afirmação por afirmação contra a base inteira"

    if em_escopo:
        av["pontos_cobertos"] = [False] * len(r["pontos_chave"]) if recusou else juiz.completude(r["resposta"], r["pontos_chave"])
        av["completude"] = sum(av["pontos_cobertos"]) / len(av["pontos_cobertos"])

    av["sucesso"] = sucesso(r, av)
    return av


def sucesso(r: Dict, av: Dict) -> bool:
    """O resultado serve ao estudante?

    Fora de escopo: recusou. Em escopo: não recusou, cobre ≥ 2/3 dos pontos-chave, ≥ 80% das afirmações
    estão na base (não inventou) e citou um artigo correto (o estudante consegue conferir).
    """
    if r["tipo"] == "fora_de_escopo":
        return av["recusou"]
    return (not av["recusou"]
            and av.get("completude", 0) >= MIN_COMPLETUDE - 1e-9
            and av.get("fidelidade_base", 0) >= MIN_FIDELIDADE_BASE
            and av.get("cita_esperada", 0) == 1.0)


def _m(valores) -> Optional[float]:
    valores = [v for v in valores if v is not None]
    return round(mr.media(valores), 3) if valores else None


def agregar(regs: List[Dict]) -> Dict:
    dentro = [r for r in regs if r["tipo"] != "fora_de_escopo"]
    fora = [r for r in regs if r["tipo"] == "fora_de_escopo"]
    a = lambda r, c: r["avaliacao"].get(c)
    respondidas = [r for r in regs if not a(r, "recusou")]
    res = {
        "taxa_sucesso": _m(1.0 if a(r, "sucesso") else 0.0 for r in regs),
        "sucesso_em_escopo": _m(1.0 if a(r, "sucesso") else 0.0 for r in dentro),
        "completude": _m(a(r, "completude") for r in dentro),
        "fidelidade_contexto": _m(a(r, "fidelidade") for r in dentro),
        "fidelidade_base": _m(a(r, "fidelidade_base") for r in dentro),
        "relevancia": _m(a(r, "relevancia") for r in dentro),
        "tem_citacao": _m(a(r, "tem_citacao") for r in dentro),
        "cita_esperada": _m(a(r, "cita_esperada") for r in dentro),
        "precisao_citacao": _m(a(r, "precisao_citacao") for r in dentro),
        "recusa_indevida": _m(1.0 if a(r, "recusou") else 0.0 for r in dentro),
        "recusa_correta": _m(1.0 if a(r, "recusou") else 0.0 for r in fora),
        "fora_fidelidade_base_quando_responde": _m(a(r, "fidelidade_base") for r in fora),
        "latencia_p50": round(mr.percentil([r["latencia_s"] for r in regs], 50), 2),
        "latencia_p95": round(mr.percentil([r["latencia_s"] for r in regs], 95), 2),
        "latencia_p50_respostas": round(mr.percentil([r["latencia_s"] for r in respondidas], 50), 2) if respondidas else None,
        "tokens_resposta": round(mr.media(r["tokens_resposta"] for r in dentro)),
        "tokens_contexto": round(mr.media(r.get("tokens_contexto", 0) for r in regs)),
        "chamadas_llm": sum(1 for r in regs if r.get("chamou_llm", True)),
        "por_tipo": {},
    }
    for tipo in ("direta", "parafraseada", "multi_fonte"):
        g = [r for r in dentro if r["tipo"] == tipo]
        res["por_tipo"][tipo] = {
            "n": len(g),
            "completude": _m(a(r, "completude") for r in g),
            "fidelidade_base": _m(a(r, "fidelidade_base") for r in g),
            "relevancia": _m(a(r, "relevancia") for r in g),
            "sucesso": _m(1.0 if a(r, "sucesso") else 0.0 for r in g),
        }
    return res


def _avaliado_com_juiz_atual(r: Dict) -> bool:
    """O registro foi avaliado pela versão atual do juiz (v3)? A v2 não tinha a chamada de relevância."""
    av = r.get("avaliacao")
    return bool(av) and (av["recusou"] or "afirmacoes_relevancia" in av)


def modo_pronto(modo: str) -> bool:
    p = RELATORIOS_DIR / f"comparacao_{modo}.json"
    if not p.exists():
        return False
    regs = json.loads(p.read_text(encoding="utf-8"))["perguntas"]
    return len(regs) == 28 and all(_avaliado_com_juiz_atual(r) for r in regs)


def avaliar_modo(modo: str, juiz, base: str) -> Dict:
    """Avalia um modo, gravando o progresso a cada pergunta em comparacao_<modo>.parcial.json.

    Se a execução for interrompida (ou continuar em outra máquina), rodar de novo retoma do ponto
    em que parou: as perguntas já avaliadas pelo juiz atual são lidas do arquivo parcial.
    """
    with open(RELATORIOS_DIR / f"comparacao_respostas_{modo}.json", encoding="utf-8") as fh:
        dados = json.load(fh)
    if dados["config"].get("llm") == juiz.modelo:
        raise SystemExit(f"O juiz ({juiz.modelo}) não pode ser o mesmo modelo do gerador.")
    parcial = RELATORIOS_DIR / f"comparacao_{modo}.parcial.json"
    feitos: Dict[str, Dict] = {}
    if parcial.exists():
        feitos = {r["id"]: r["avaliacao"] for r in json.loads(parcial.read_text(encoding="utf-8"))
                  if _avaliado_com_juiz_atual(r)}
        print(f"  [{modo}] retomando: {len(feitos)} perguntas já avaliadas", flush=True)

    regs = dados["respostas"]
    for n, r in enumerate(regs, 1):
        if r["id"] in feitos:
            r["avaliacao"] = feitos[r["id"]]
            continue
        print(f"  [{modo} {n}/{len(regs)}] {r['id']}", flush=True)
        r["avaliacao"] = avaliar_registro(r, juiz, base, modo)
        prontos = [x for x in regs if "avaliacao" in x]
        parcial.write_text(json.dumps(prontos, ensure_ascii=False, indent=2), encoding="utf-8")

    resumo = agregar(regs)
    saida = {"config": {**dados["config"], "juiz": juiz.modelo}, "resumo": resumo, "perguntas": regs}
    with open(RELATORIOS_DIR / f"comparacao_{modo}.json", "w", encoding="utf-8") as fh:
        json.dump(saida, fh, ensure_ascii=False, indent=2)
    parcial.unlink(missing_ok=True)
    return saida


LINHAS = [
    ("taxa_sucesso", "Taxa de sucesso (28)"),
    ("completude", "Completude"),
    ("fidelidade_base", "Fidelidade à base"),
    ("fidelidade_contexto", "Fidelidade ao contexto"),
    ("relevancia", "Relevância"),
    ("cita_esperada", "Cita artigo correto"),
    ("recusa_correta", "Recusa correta (10)"),
    ("recusa_indevida", "Recusa indevida (18)"),
    ("latencia_p50", "Latência p50, todas as 28 (s)"),
    ("latencia_p50_respostas", "Latência p50, só respostas (s)"),
    ("tokens_resposta", "Tokens por resposta"),
    ("tokens_contexto", "Tokens de contexto"),
    ("chamadas_llm", "Chamadas ao LLM (28)"),
]


def tabela() -> None:
    resumos = {}
    for modo in ORDEM:
        p = RELATORIOS_DIR / f"comparacao_{modo}.json"
        if p.exists():
            resumos[modo] = json.loads(p.read_text(encoding="utf-8"))["resumo"]
    if not resumos:
        raise SystemExit("Nada avaliado ainda.")
    fmt = lambda v: "—" if v is None else (f"{v:.2f}" if isinstance(v, float) else str(v))
    cab = ["Métrica", *[ROTULOS[m] for m in resumos]]
    linhas = [[rot, *[fmt(resumos[m].get(ch)) for m in resumos]] for ch, rot in LINHAS]
    for tipo in ("direta", "parafraseada", "multi_fonte"):
        linhas.append([f"Completude {tipo}", *[fmt(resumos[m]["por_tipo"][tipo]["completude"]) for m in resumos]])
    md = ["| " + " | ".join(cab) + " |", "| " + " | ".join([":--"] * len(cab)) + " |"]
    md += ["| " + " | ".join(l) + " |" for l in linhas]
    (RELATORIOS_DIR / "comparacao.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(RELATORIOS_DIR / "comparacao.csv", "w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows([cab, *linhas])
    print("\n".join(md))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modos", nargs="*", default=[], help=f"padrão: os que faltam, na ordem {ORDEM}")
    ap.add_argument("--so-tabela", action="store_true")
    ap.add_argument("--refazer", action="store_true", help="avalia de novo mesmo os modos já prontos")
    args = ap.parse_args()
    if not args.so_tabela:
        from metricas.geracao.juiz import Juiz

        modos = args.modos or ORDEM
        pendentes = [m for m in modos if args.refazer or not modo_pronto(m)]
        print(f"Prontos: {[m for m in modos if m not in pendentes]} | a avaliar: {pendentes}", flush=True)
        juiz, base = Juiz(), texto_base()
        for modo in pendentes:
            avaliar_modo(modo, juiz, base)
    tabela()


if __name__ == "__main__":
    main()
