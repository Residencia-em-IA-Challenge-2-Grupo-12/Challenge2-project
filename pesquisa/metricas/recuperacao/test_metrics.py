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


def test_ndcg_graduado():
    ganhos = {A: 2, B: 1}  # A principal, B complementar
    assert m.ndcg_at_k([A, B, C], ganhos, 3) == 1.0  # ordem ideal
    assert m.ndcg_at_k([C, C, C], ganhos, 3) == 0.0
    # complementar antes do principal vale menos que a ordem ideal, mas mais que zero
    assert 0.0 < m.ndcg_at_k([B, A, C], ganhos, 3) < 1.0
    # chunk repetido do mesmo artigo não soma duas vezes
    assert m.ndcg_at_k([A, A, B], ganhos, 3) < m.ndcg_at_k([A, B, A], ganhos, 3)
    assert math.isclose(m.ndcg_at_k([C, A], {A: 2}, 3), 1 / math.log2(3))


def test_cobertura_pontos_chave():
    pontos = ["Intervalos de 25 minutos com pausas de 5 minutos", "Registrar o progresso diário"]
    assert m.cobertura_pontos_chave(["Use intervalos de 25 minutos e pausas curtas de 5 minutos."], pontos) == 0.5
    assert m.cobertura_pontos_chave(["Nada a ver com o assunto."], pontos) == 0.0
    # tolera acento e flexão: "consolidação" e "consolida" viram o mesmo radical
    assert m.termos("consolidação") == m.termos("consolida")


def test_auc_e_recusa():
    assert m.auc_separacao([0.5, 0.6], [0.9, 1.0]) == 1.0  # todas as válidas mais perto
    assert m.auc_separacao([0.9, 1.0], [0.5, 0.6]) == 0.0
    assert m.auc_separacao([0.5], [0.5]) == 0.5  # empate
    assert m.recusa_sem_erro([0.5, 0.8], [0.7, 0.9]) == 0.5  # só 0.9 passa do corte 0.8


def test_jaccard_e_percentil():
    assert m.jaccard([A, B], [B, C]) == 1 / 3
    assert m.percentil([10, 20, 30, 40], 50) == 25
    assert m.percentil([1, 2, 3], 100) == 3


def test_ruido_reprodutivel():
    from metricas.recuperacao.robustez import com_ruido

    assert com_ruido("Técnica Pomodoro", 1) == com_ruido("Técnica Pomodoro", 1)
    assert "é" not in com_ruido("Técnica Pomodoro", 1)


def test_runner_com_retriever_perfeito_e_cego():
    qs = load_gabarito()
    esperadas = {q.pergunta: q.fontes_esperadas for q in qs}

    def perfeito(pergunta, n):
        fontes = esperadas[pergunta] or ["artigo_01_active_recall_karpicke.txt"]
        return [(f"C:\\x\\dev\\data\\{f}", 0.5, "") for f in (fontes * n)[:n]]

    def cego(pergunta, n):
        return [("C:\\x\\dev\\data\\artigo_99_inexistente.txt", 9.0, "")] * n

    ok = agregar(avaliar(perfeito, qs, (1, 3)))
    assert ok["geral"]["hit@1"] == 1.0 and ok["geral"]["rr@3"] == 1.0
    assert ok["n_em_escopo"] == 18  # as fora de escopo não entram

    ruim = agregar(avaliar(cego, qs, (1, 3)))
    assert ruim["geral"]["hit@3"] == 0.0 and ruim["geral"]["recall@3"] == 0.0


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("ok ", nome)
    print("todos passaram")
