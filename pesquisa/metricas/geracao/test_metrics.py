"""Testes das métricas determinísticas da geração e da agregação (com juiz FALSO, sem LLM).

Rodar (a partir de pesquisa/): python -m metricas.geracao.test_metrics   (ou pytest)
"""

from metricas.geracao import metrics as mg
from metricas.geracao.runner import agregar, avaliar_registro

RECUSA = "Não encontrei informações suficientes na base de conhecimento para responder a essa pergunta."


def test_recusa_tolera_acento_e_caixa():
    assert mg.e_recusa(RECUSA)
    assert mg.e_recusa("nao encontrei   informacoes suficientes na BASE de conhecimento.")
    assert not mg.e_recusa("A técnica Pomodoro usa blocos de 25 minutos.")


def test_citacoes():
    r = "Use 25 min [Fonte: ../dev/data/artigo_10_pomodoro.txt]. Limite o tempo [Fonte: artigo_09_x.txt; artigo_10_y.txt]."
    assert mg.citacoes(r) == ["10", "09"]  # sem repetição, na ordem
    assert mg.citacoes_brutas(r) == 2
    assert mg.citacoes("[Fonte: Técnica Pomodoro]") == []  # sem artigo_NN: não rastreável


def test_precisao_e_cita_esperada():
    assert mg.precisao_citacao(["10", "05"], {"10", "09"}) == 0.5  # citou o 05 sem tê-lo recebido
    assert mg.cita_esperada(["09"], {"10"}) == 0.0
    assert mg.numero_artigo("C:\\x\\artigo_07_attention.txt") == "07"


class JuizFalso:
    modelo = "falso"

    def fidelidade(self, contexto, resposta):
        return [{"afirmacao": "a", "sustentada": True}, {"afirmacao": "b", "sustentada": False}]

    def completude(self, resposta, pontos):
        return [True] + [False] * (len(pontos) - 1)


def _registro(tipo, resposta, pontos=("p1", "p2")):
    return {
        "id": "qx", "tipo": tipo, "pergunta": "?", "esperadas": ["artigo_10_a.txt"], "pontos_chave": list(pontos),
        "recuperados": ["artigo_10_a.txt", "artigo_09_b.txt"], "contexto": "ctx", "resposta": resposta,
        "latencia_s": 1.0, "tokens_resposta": 10,
    }


def test_avaliacao_e_agregacao():
    boa = _registro("direta", "Faça X [Fonte: artigo_10_a.txt].")
    recusa_indevida = _registro("direta", RECUSA)
    recusa_certa = _registro("fora_de_escopo", RECUSA, pontos=())
    inventou = _registro("fora_de_escopo", "Invista em ações.", pontos=())
    for r in (boa, recusa_indevida, recusa_certa, inventou):
        r["avaliacao"] = avaliar_registro(r, JuizFalso())

    assert boa["avaliacao"]["fidelidade"] == 0.5 and boa["avaliacao"]["completude"] == 0.5
    assert boa["avaliacao"]["precisao_citacao"] == 1.0 and boa["avaliacao"]["cita_esperada"] == 1.0
    assert recusa_indevida["avaliacao"]["completude"] == 0.0  # recusar pergunta válida não cobre nada
    assert "fidelidade" not in recusa_certa["avaliacao"]  # recusa pura não tem afirmação a checar

    res = agregar([boa, recusa_indevida, recusa_certa, inventou])
    assert res["em_escopo"]["recusa_indevida"] == 0.5
    assert res["fora_de_escopo"]["recusa_correta"] == 0.5


if __name__ == "__main__":
    for nome, fn in list(globals().items()):
        if nome.startswith("test_"):
            fn()
            print("ok ", nome)
    print("todos passaram")
