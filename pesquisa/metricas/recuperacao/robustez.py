"""Robustez da busca: a mesma pergunta, escrita de outro jeito, ainda acha os mesmos artigos?

Dois testes, só com as perguntas em escopo:
  1. Paráfrases: as `variacoes` do gabarito (escritas à mão, linguagem de estudante).
  2. Ruído: a pergunta original sem acentos, em minúsculas e com erros de digitação gerados
     de forma reprodutível (mesma semente = mesmos erros em toda execução).

Para cada teste: Hit@k e MRR@k das versões alteradas, e a queda em relação à original.
Para as paráfrases, também o Jaccard entre os artigos do top-k da original e de cada variação.
"""

import random
import unicodedata
from typing import Dict, List, Sequence

from metricas.dataset.loader import Pergunta, nome_fonte

from . import metrics as m


def com_ruido(texto: str, semente: int, taxa: float = 0.3) -> str:
    """Tira acentos, põe em minúsculas e troca duas letras vizinhas em ~`taxa` das palavras longas.

    Ex.: "Como funciona a técnica Pomodoro?" -> "como funcoina a tecnica pomodoro?"
    """
    rng = random.Random(semente)
    base = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    palavras = []
    for palavra in base.split(" "):
        if len(palavra) > 4 and rng.random() < taxa:
            i = rng.randrange(1, len(palavra) - 2)  # preserva a 1ª letra, como num erro de digitação real
            palavra = palavra[:i] + palavra[i + 1] + palavra[i] + palavra[i + 2:]
        palavras.append(palavra)
    return " ".join(palavras)


def _top(search, texto: str, k: int) -> List[str]:
    return [nome_fonte(r[0]) for r in search(texto, k)]


def avaliar_robustez(search, perguntas: Sequence[Pergunta], k: int = 3) -> Dict:
    em_escopo = [p for p in perguntas if not p.fora_de_escopo]
    orig_hit, orig_rr, ruido_hit, ruido_rr, var_hit, var_rr, jaccards = [], [], [], [], [], [], []

    for n, p in enumerate(em_escopo):
        esperadas = set(p.fontes_esperadas)
        top_orig = _top(search, p.pergunta, k)
        orig_hit.append(m.hit_at_k(top_orig, esperadas, k))
        orig_rr.append(m.reciprocal_rank(top_orig, esperadas, k))

        top_ruido = _top(search, com_ruido(p.pergunta, semente=n), k)
        ruido_hit.append(m.hit_at_k(top_ruido, esperadas, k))
        ruido_rr.append(m.reciprocal_rank(top_ruido, esperadas, k))

        for v in p.variacoes:
            top_v = _top(search, v, k)
            var_hit.append(m.hit_at_k(top_v, esperadas, k))
            var_rr.append(m.reciprocal_rank(top_v, esperadas, k))
            jaccards.append(m.jaccard(top_orig, top_v))

    r = lambda xs: round(m.media(xs), 4)
    return {
        "k": k,
        "original": {"hit": r(orig_hit), "mrr": r(orig_rr)},
        "ruido": {"hit": r(ruido_hit), "mrr": r(ruido_rr), "queda_hit": round(r(orig_hit) - r(ruido_hit), 4)},
        "parafrases": {
            "n": len(var_hit), "hit": r(var_hit), "mrr": r(var_rr),
            "queda_hit": round(r(orig_hit) - r(var_hit), 4),
            "jaccard_medio": r(jaccards),
        },
    }
