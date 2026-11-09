"""Métricas de recuperação. Funções puras (só stdlib), sem dependência de langchain.

Entrada comum a todas:
  - `recuperadas`: nomes de arquivo das fontes dos chunks, NA ORDEM do ranking
                   (posição 0 = chunk mais próximo da pergunta). Pode repetir fonte,
                   porque um artigo tem vários chunks.
  - `esperadas`:   conjunto de artigos que o gabarito diz serem os corretos.

Passo 1 de qualquer métrica: marcar cada chunk recuperado como relevante (True) ou
não (False), conforme sua fonte esteja ou não em `esperadas`.
"""

from typing import Iterable, List, Sequence, Set


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


def fontes_distintas_at_k(recuperadas: Sequence[str], k: int) -> int:
    """Quantos artigos diferentes aparecem nos top-k (diagnóstico de diversidade).

    Se Top-K=3 traz 3 chunks do mesmo artigo, isso vale 1, e o contexto é redundante.
    """
    return len(set(recuperadas[:k]))


def media(valores: Iterable[float]) -> float:
    valores = list(valores)
    return sum(valores) / len(valores) if valores else 0.0
