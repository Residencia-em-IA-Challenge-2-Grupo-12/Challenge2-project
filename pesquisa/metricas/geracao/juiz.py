"""LLM-juiz: avalia o que regra não consegue (fidelidade ao contexto e completude).

O juiz é um modelo DIFERENTE do gerador (padrão: gemma3:12b julgando o qwen3:8b), para não
"corrigir a própria prova". Ele responde em JSON com esquema fixo (structured output do Ollama),
então a saída é sempre parseável, e a nota é calculada aqui em Python, não pelo juiz.
"""

import json
import os
from typing import Dict, List, Sequence

MODELO_JUIZ = os.getenv("OLLAMA_JUIZ_MODEL", "gemma3:12b")

PROMPT_FIDELIDADE = """Você é um avaliador rigoroso de sistemas RAG.

Tarefa: verificar se a RESPOSTA é fiel ao CONTEXTO.
1. Quebre a RESPOSTA em afirmações factuais curtas e independentes.
   Ignore: saudações, frases de transição, as marcações [Fonte: ...] e frases dizendo que não há informação.
2. Para cada afirmação, decida se ela é SUSTENTADA pelo CONTEXTO:
   - true: o contexto diz isso explicitamente ou é uma paráfrase direta.
   - false: o contexto não diz isso (mesmo que seja verdade no mundo real) ou diz o contrário.
Não use conhecimento externo. Julgue apenas pelo CONTEXTO.

CONTEXTO:
{contexto}

RESPOSTA:
{resposta}"""

ESQUEMA_FIDELIDADE = {
    "type": "object",
    "properties": {
        "afirmacoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "afirmacao": {"type": "string"},
                    "sustentada": {"type": "boolean"},
                },
                "required": ["afirmacao", "sustentada"],
            },
        }
    },
    "required": ["afirmacoes"],
}

PROMPT_COMPLETUDE = """Você é um avaliador rigoroso de sistemas RAG.

Tarefa: para cada PONTO-CHAVE numerado, decida se a RESPOSTA o cobre.
1. Procure na RESPOSTA a frase que transmite a ideia do ponto e copie-a LITERALMENTE em "trecho".
   Se nenhuma frase da resposta transmite essa ideia, deixe "trecho" vazio.
2. "coberto" é true só se você copiou um trecho que realmente transmite a mesma ideia (mesmo com outras
   palavras). Ideia ausente, vaga demais ou apenas parecida conta como false.
Não presuma que a resposta cobre um ponto só porque o tema é parecido.

PONTOS-CHAVE:
{pontos}

RESPOSTA:
{resposta}"""

ESQUEMA_COMPLETUDE = {
    "type": "object",
    "properties": {
        "pontos": {
            "type": "array",
            "items": {
                "type": "object",
                # trecho ANTES de coberto: o modelo precisa achar a evidência antes de dar o veredito
                "properties": {"numero": {"type": "integer"}, "trecho": {"type": "string"}, "coberto": {"type": "boolean"}},
                "required": ["numero", "trecho", "coberto"],
            },
        }
    },
    "required": ["pontos"],
}


# Fidelidade à BASE: uma afirmação por chamada. Com a base inteira e ~15 afirmações numa chamada só,
# o juiz de 12B errava para os dois lados (aprovava "Matriz de Eisenhower", reprovava o Pomodoro de
# 25 min, que está no artigo 10). A base vem PRIMEIRO no prompt, então o Ollama reaproveita o cache
# desse prefixo e cada chamada só processa a afirmação nova.
PROMPT_BASE = """Você é um verificador rigoroso. Abaixo está a BASE DE CONHECIMENTO completa de um assistente de estudos.

BASE DE CONHECIMENTO:
{base}

---
Tarefa: decida se a AFIRMAÇÃO é sustentada pela BASE.
1. Procure na BASE a frase que diz a mesma coisa que a afirmação e copie-a LITERALMENTE em "evidencia" (vazio se não houver).
2. "sustentada": true se a evidência diz a mesma coisa (paráfrase vale). false se a BASE não diz isso, mesmo que seja verdade no mundo real. Técnicas, nomes, números ou horários ausentes da BASE contam como false.

AFIRMAÇÃO: {afirmacao}"""

ESQUEMA_BASE = {
    "type": "object",
    # evidência ANTES do veredito: achar o trecho antes de decidir
    "properties": {"evidencia": {"type": "string"}, "sustentada": {"type": "boolean"}},
    "required": ["evidencia", "sustentada"],
}

PROMPT_RELEVANCIA = """Você é um avaliador rigoroso de um assistente de estudos.

Tarefa: verificar se a RESPOSTA se mantém no que a PERGUNTA pede.
1. Quebre a RESPOSTA em afirmações curtas e independentes.
   Ignore: saudações, frases de transição, as marcações [Fonte: ...] e frases dizendo que não há informação.
2. Para cada afirmação, "relevante" é true só se ela ajuda DIRETAMENTE a responder o que a PERGUNTA pede.
   É false se trata de outro assunto, se é um conselho genérico que serviria para qualquer pergunta,
   ou se responde a uma pergunta parecida, mas diferente da que foi feita.

PERGUNTA:
{pergunta}

RESPOSTA:
{resposta}"""

ESQUEMA_RELEVANCIA = {
    "type": "object",
    "properties": {
        "afirmacoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"afirmacao": {"type": "string"}, "relevante": {"type": "boolean"}},
                "required": ["afirmacao", "relevante"],
            },
        }
    },
    "required": ["afirmacoes"],
}


class Juiz:
    def __init__(self, modelo: str = MODELO_JUIZ) -> None:
        from langchain_ollama import ChatOllama

        self.modelo = modelo
        self._fid = ChatOllama(model=modelo, temperature=0, format=ESQUEMA_FIDELIDADE)
        self._comp = ChatOllama(model=modelo, temperature=0, format=ESQUEMA_COMPLETUDE)
        self._rel = ChatOllama(model=modelo, temperature=0, format=ESQUEMA_RELEVANCIA)
        # A base inteira (~5 mil tokens) não cabe na janela padrão do Ollama: amplia só para esta tarefa
        self._base = ChatOllama(model=modelo, temperature=0, format=ESQUEMA_BASE, num_ctx=8192)

    def relevancia(self, pergunta: str, resposta: str) -> List[Dict]:
        """Lista de {"afirmacao", "relevante"}. A nota é relevantes / total (calculada no avaliador)."""
        saida = self._rel.invoke(PROMPT_RELEVANCIA.format(pergunta=pergunta, resposta=resposta))
        return json.loads(saida.content)["afirmacoes"]

    def sustentada_na_base(self, base: str, afirmacao: str) -> Dict:
        """{"evidencia", "veredito_juiz", "sustentada"} para UMA afirmação, contra a base inteira.

        "sustentada" só é true se o juiz aprovou E a evidência copiada existe mesmo na base (checagem em Python).
        """
        saida = json.loads(self._base.invoke(PROMPT_BASE.format(base=base, afirmacao=afirmacao)).content)
        veredito = bool(saida["sustentada"])
        return {"evidencia": saida.get("evidencia", ""), "veredito_juiz": veredito,
                "sustentada": veredito and trecho_existe(saida.get("evidencia", ""), base)}

    def fidelidade(self, contexto: str, resposta: str) -> List[Dict]:
        """Lista de {"afirmacao", "sustentada"}. A nota é sustentadas / total (calculada no runner)."""
        saida = self._fid.invoke(PROMPT_FIDELIDADE.format(contexto=contexto, resposta=resposta))
        return json.loads(saida.content)["afirmacoes"]

    def completude(self, resposta: str, pontos: Sequence[str]) -> List[bool]:
        """Um booleano por ponto-chave, na ordem do gabarito.

        O veredito do juiz só vale se o trecho que ele citou existe mesmo na resposta (checagem em Python):
        evita o "sim" por semelhança de tema, sem evidência.
        """
        lista = "\n".join(f"{i}. {p}" for i, p in enumerate(pontos, start=1))
        saida = self._comp.invoke(PROMPT_COMPLETUDE.format(pontos=lista, resposta=resposta))
        por_numero = {
            item["numero"]: bool(item["coberto"]) and trecho_existe(item.get("trecho", ""), resposta)
            for item in json.loads(saida.content)["pontos"]
        }
        return [por_numero.get(i, False) for i in range(1, len(pontos) + 1)]  # ponto omitido conta como não coberto


def trecho_existe(trecho: str, resposta: str, limiar: float = 0.8) -> bool:
    """O trecho citado pelo juiz está na resposta? Tolera pequenas diferenças (acento, pontuação, flexão)."""
    from metricas.recuperacao.metrics import termos

    t = termos(trecho)
    return bool(t) and len(t & termos(resposta)) / len(t) >= limiar
