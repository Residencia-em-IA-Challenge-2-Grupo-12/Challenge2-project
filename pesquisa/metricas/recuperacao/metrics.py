"""Métricas de recuperação. Funções puras (só stdlib), sem dependência de langchain.

Entrada comum a todas:
  - `recuperadas`: nomes de arquivo das fontes dos chunks, NA ORDEM do ranking
                   (posição 0 = chunk mais próximo da pergunta). Pode repetir fonte,
                   porque um artigo tem vários chunks.
  - `esperadas`:   conjunto de artigos que o gabarito diz serem os corretos.

Passo 1 de qualquer métrica: marcar cada chunk recuperado como relevante (True) ou
não (False), conforme sua fonte esteja ou não em `esperadas`.
"""

import math
import re
import unicodedata
from typing import Dict, Iterable, List, Sequence, Set


def relevancia(recuperadas: Sequence[str], esperadas: Set[str]) -> List[bool]:
    """Ex.: recuperadas=[a, b, a], esperadas={a} -> [True, False, True]."""
    return [fonte in esperadas for fonte in recuperadas]


def hit_at_k(recuperadas: Sequence[str], esperadas: Set[str], k: int) -> float:
    """1.0 se ALGUM dos top-k chunks vem de um artigo esperado, senão 0.0.

    Responde: "o retriever chegou perto o suficiente para o LLM ter alguma chance?"
    """
    return 1.0 if any(relevancia(recuperadas[:k], esperadas)) else 0.0


def reciprocal_rank(recuperadas: Sequence[str], esperadas: Set[str], k: int) -> float:
    """1 / posição (começando em 1) do primeiro chunk relevante nos top-k; 0.0 se nenhum.

    1º lugar -> 1.0 | 2º -> 0.5 | 3º -> 0.333. A média disso sobre as perguntas é o MRR.
    Responde: "o acerto vem no topo ou escondido lá embaixo?"
    """
    for pos, relevante in enumerate(relevancia(recuperadas[:k], esperadas), start=1):
        if relevante:
            return 1.0 / pos
    return 0.0


def precision_at_k(recuperadas: Sequence[str], esperadas: Set[str], k: int) -> float:
    """Fração dos k slots ocupada por chunks relevantes (nível de CHUNK).

    O denominador é sempre k: se vieram menos de k chunks, os que faltam contam como erro.
    Responde: "quanto do contexto entregue ao LLM é útil, e quanto é ruído?"
    """
    if k <= 0:
        raise ValueError("k deve ser positivo")
    return sum(relevancia(recuperadas[:k], esperadas)) / k


def recall_at_k(recuperadas: Sequence[str], esperadas: Set[str], k: int) -> float:
    """Fração dos ARTIGOS esperados que aparecem nos top-k (nível de ARTIGO, sem repetição).

    Importa nas perguntas multi_fonte: se o gabarito pede 3 artigos e vieram 2, recall = 0.67.
    """
    if not esperadas:
        raise ValueError("recall não se aplica sem fontes esperadas")
    return len(set(recuperadas[:k]) & esperadas) / len(esperadas)


def ndcg_at_k(recuperadas: Sequence[str], ganhos: Dict[str, int], k: int) -> float:
    """nDCG@k com relevância graduada (nível de ARTIGO): ex. principal = 2, complementar = 1.

    DCG = soma de ganho / log2(posição + 1), contando cada artigo só na 1ª vez que aparece
    (chunk repetido não soma de novo). Divide pelo DCG ideal: os melhores artigos nas primeiras
    posições. 1.0 = ordem perfeita; 0.0 = nenhum artigo relevante nos top-k.
    Responde: "os artigos mais importantes vêm antes dos menos importantes?"
    """
    vistos: Set[str] = set()
    dcg = 0.0
    for pos, fonte in enumerate(recuperadas[:k], start=1):
        if fonte in ganhos and fonte not in vistos:
            vistos.add(fonte)
            dcg += ganhos[fonte] / math.log2(pos + 1)
    ideais = sorted(ganhos.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(pos + 1) for pos, g in enumerate(ideais, start=1))
    return dcg / idcg if idcg else 0.0


def fontes_distintas_at_k(recuperadas: Sequence[str], k: int) -> int:
    """Quantos artigos diferentes aparecem nos top-k (diagnóstico de diversidade).

    Se Top-K=3 traz 3 chunks do mesmo artigo, isso vale 1, e o contexto é redundante.
    """
    return len(set(recuperadas[:k]))


def media(valores: Iterable[float]) -> float:
    valores = list(valores)
    return sum(valores) / len(valores) if valores else 0.0


# ---------------------------------------------------------------------------
# Cobertura de pontos-chave (nível de TEXTO do chunk, não de artigo)
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "para", "como", "pois", "mais", "menos", "quando", "entre", "sobre", "pelo", "pela", "pelos", "pelas",
    "isso", "esse", "essa", "este", "esta", "cada", "ainda", "mesmo", "mesma", "seus", "suas", "dele", "dela",
    "também", "muito", "muita", "sem", "com", "que", "uma", "uns", "umas", "dos", "das", "nos", "nas", "não",
}


def termos(texto: str) -> Set[str]:
    """Normaliza o texto em termos comparáveis: sem acento, minúsculo, sem palavras vazias.

    Palavras viram um radical de 5 letras (ex.: "consolida", "consolidação" -> "conso"), para tolerar
    flexões. Números ficam inteiros, porque "25 minutos" e "7 horas" são o conteúdo da resposta.
    """
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    saida: Set[str] = set()
    for palavra in re.findall(r"\w+", sem_acento):
        if palavra.isdigit():
            saida.add(palavra)
        elif len(palavra) > 3 and palavra not in _STOPWORDS:
            saida.add(palavra[:5])
    return saida


def cobertura_pontos_chave(textos: Sequence[str], pontos: Sequence[str], limiar: float = 0.5) -> float:
    """Fração dos pontos-chave presentes no contexto recuperado (os k chunks juntos).

    Um ponto conta como coberto se pelo menos `limiar` dos seus termos aparecem no contexto.
    Responde: "o LLM recebeu o CONTEÚDO que a resposta precisa?" (o Hit só garante o artigo certo).
    Medida lexical: subestima quando o chunk diz a mesma coisa com outras palavras.
    """
    if not pontos:
        raise ValueError("cobertura não se aplica sem pontos_chave")
    contexto = termos(" ".join(textos))
    cobertos = 0
    for ponto in pontos:
        t = termos(ponto)
        if t and len(t & contexto) / len(t) >= limiar:
            cobertos += 1
    return cobertos / len(pontos)


# ---------------------------------------------------------------------------
# Separação em escopo x fora de escopo pela distância do 1º chunk
# ---------------------------------------------------------------------------

def auc_separacao(dist_em_escopo: Sequence[float], dist_fora: Sequence[float]) -> float:
    """AUC-ROC da distância como classificador "tem resposta na base?" (Mann-Whitney).

    É a probabilidade de uma pergunta em escopo sorteada estar MAIS PERTO da base do que uma fora
    de escopo sorteada (empate vale 0.5). 1.0 = separação perfeita; 0.5 = a distância não ajuda.
    """
    if not dist_em_escopo or not dist_fora:
        raise ValueError("AUC exige perguntas em escopo e fora de escopo")
    pares = 0.0
    for d_in in dist_em_escopo:
        for d_out in dist_fora:
            pares += 1.0 if d_in < d_out else 0.5 if d_in == d_out else 0.0
    return pares / (len(dist_em_escopo) * len(dist_fora))


def recusa_sem_erro(dist_em_escopo: Sequence[float], dist_fora: Sequence[float]) -> float:
    """Fração das fora de escopo recusáveis com um corte que não recusa NENHUMA pergunta válida.

    O corte fica logo acima da maior distância em escopo. Complementa a AUC com um uso prático.
    """
    if not dist_fora:
        raise ValueError("sem perguntas fora de escopo")
    corte = max(dist_em_escopo)
    return sum(d > corte for d in dist_fora) / len(dist_fora)


# ---------------------------------------------------------------------------
# Robustez e custo
# ---------------------------------------------------------------------------

def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Sobreposição entre dois conjuntos de artigos: 1.0 = idênticos, 0.0 = nada em comum."""
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 1.0


def percentil(valores: Sequence[float], p: float) -> float:
    """Percentil por interpolação linear (p entre 0 e 100). Ex.: p=95 -> latência p95."""
    if not valores:
        raise ValueError("lista vazia")
    ordenados = sorted(valores)
    pos = (len(ordenados) - 1) * p / 100
    baixo, alto = math.floor(pos), math.ceil(pos)
    return ordenados[baixo] + (ordenados[alto] - ordenados[baixo]) * (pos - baixo)
