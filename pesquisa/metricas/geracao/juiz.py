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


PROMPT_BASE = """Você é um avaliador rigoroso de um assistente de estudos.

Tarefa: avaliar cada afirmação da RESPOSTA contra a BASE DE CONHECIMENTO inteira e contra a PERGUNTA.
1. Quebre a RESPOSTA em afirmações factuais curtas e independentes.
   Ignore: saudações, frases de transição, as marcações [Fonte: ...] e frases dizendo que não há informação.
2. Para cada afirmação, procure na BASE a frase que diz a mesma coisa e copie-a LITERALMENTE em "evidencia".
   Se nenhuma frase da BASE diz isso, deixe "evidencia" vazio.
3. "sustentada": true só se você copiou uma evidência que realmente diz a mesma coisa que a afirmação
   (mesmo com outras palavras). Termos, técnicas, números ou horários que não aparecem na BASE contam como false,
   mesmo que sejam verdade no mundo real. Tema parecido não basta.
4. "relevante": true se a afirmação ajuda a responder o que a PERGUNTA pede;
   false se é sobre outro assunto, genérica demais ou não tem relação com a pergunta.
Não use conhecimento externo. Julgue apenas pela BASE.

BASE DE CONHECIMENTO:
{base}

PERGUNTA:
{pergunta}

RESPOSTA:
{resposta}"""

ESQUEMA_BASE = {
    "type": "object",
    "properties": {
        "afirmacoes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "afirmacao": {"type": "string"},
                    # evidência ANTES do veredito, como na completude: achar o trecho antes de decidir
                    "evidencia": {"type": "string"},
                    "sustentada": {"type": "boolean"},
                    "relevante": {"type": "boolean"},
                },
                "required": ["afirmacao", "evidencia", "sustentada", "relevante"],
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
        # A base inteira (~5 mil tokens) não cabe na janela padrão do Ollama: amplia só para esta tarefa
        self._base = ChatOllama(model=modelo, temperature=0, format=ESQUEMA_BASE, num_ctx=16384)

    def base_e_relevancia(self, base: str, pergunta: str, resposta: str) -> List[Dict]:
        """Lista de {"afirmacao", "sustentada", "relevante"}, julgada contra a base INTEIRA.

        Diferente de `fidelidade` (que julga contra o contexto recuperado), esta funciona para qualquer
        modo, inclusive o LLM sem RAG, que não recebe contexto nenhum.
        O veredito "sustentada" só vale se a evidência copiada existe mesmo na base (checagem em Python).
        """
        saida = self._base.invoke(PROMPT_BASE.format(base=base, pergunta=pergunta, resposta=resposta))
        afirmacoes = json.loads(saida.content)["afirmacoes"]
        for a in afirmacoes:
            a["veredito_juiz"] = bool(a["sustentada"])
            a["sustentada"] = a["veredito_juiz"] and trecho_existe(a.get("evidencia", ""), base)
        return afirmacoes

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
