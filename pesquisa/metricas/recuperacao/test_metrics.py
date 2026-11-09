"""Testes das fórmulas e do fluxo do runner com um retriever FALSO (sem langchain/API).

Rodar (a partir de pesquisa/): python -m metricas.recuperacao.test_metrics   (ou pytest)
"""

import math

from metricas.dataset.loader import load_gabarito
from metricas.recuperacao import metrics as m
from metricas.recuperacao.runner import agregar, avaliar

A, B, C = "a.txt", "b.txt", "c.txt"


def test_hit():
    assert m.hit_at_k([B, A, C], {A}, 1) == 0.0
    assert m.hit_at_k([B, A, C], {A}, 2) == 1.0


def test_reciprocal_rank():
    assert m.reciprocal_rank([A, B], {A}, 3) == 1.0
    assert m.reciprocal_rank([B, A], {A}, 3) == 0.5
    assert math.isclose(m.reciprocal_rank([B, C, A], {A}, 3), 1 / 3)
    assert m.reciprocal_rank([B, C, A], {A}, 2) == 0.0  # acerto fora do corte


def test_precision_conta_chunks_repetidos():
    # 2 chunks do artigo A + 1 do B, esperado {A}: 2 de 3 slots úteis
    assert math.isclose(m.precision_at_k([A, A, B], {A}, 3), 2 / 3)
    assert m.precision_at_k([A], {A}, 3) == 1 / 3  # faltantes contam como erro


def test_recall_conta_artigos_distintos():
    assert m.recall_at_k([A, A, A], {A, B}, 3) == 0.5  # A repetido não conta 2x
    assert m.recall_at_k([A, B, C], {A, B}, 3) == 1.0


def test_distintas():
    assert m.fontes_distintas_at_k([A, A, A], 3) == 1


def test_runner_com_retriever_perfeito_e_cego():
    qs = load_gabarito()
    esperadas = {q.pergunta: q.fontes_esperadas for q in qs}

    def perfeito(pergunta, n):
        fontes = esperadas[pergunta] or ["artigo_01_active_recall_karpicke.txt"]
        return [(f"C:\\x\\dev\\data\\{f}", 0.5) for f in (fontes * n)[:n]]

    def cego(pergunta, n):
        return [("C:\\x\\dev\\data\\artigo_99_inexistente.txt", 9.0)] * n

    ok = agregar(avaliar(perfeito, qs, (1, 3)))
    assert ok["geral"]["hit@1"] == 1.0 and ok["geral"]["rr@3"] == 1.0
    assert ok["n_em_escopo"] == 18  # as 2 fora de escopo não entram

    ruim = agregar(avaliar(cego, qs, (1, 3)))
    assert ruim["geral"]["hit@3"] == 0.0 and ruim["geral"]["recall@3"] == 0.0


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("ok ", nome)
    print("todos passaram")
