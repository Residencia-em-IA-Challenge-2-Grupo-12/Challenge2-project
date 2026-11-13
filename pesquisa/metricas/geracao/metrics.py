"""Métricas determinísticas da geração (sem LLM). Funções puras, só stdlib.

O prompt do notebook exige duas coisas verificáveis por regra:
  - citar a fonte no formato [Fonte: ...];
  - recusar com uma frase exata quando o contexto não responde.
"""

import re
import unicodedata
from typing import List, Optional, Sequence, Set

# Trecho da frase de recusa definida em RAG_SYSTEM_PROMPT (regra 2 do notebook)
FRASE_RECUSA = "Não encontrei informações suficientes na base de conhecimento"

_CITACAO = re.compile(r"\[\s*Fonte\s*:\s*([^\]]+)\]", re.IGNORECASE)
_ARTIGO = re.compile(r"artigo_(\d{2})")


def _normalizar(texto: str) -> str:
    sem_acento = unicodedata.normalize("NFKD", texto.lower()).encode("ascii", "ignore").decode()
    return " ".join(sem_acento.split())


def e_recusa(resposta: str, frase: str = FRASE_RECUSA) -> bool:
    """True se a resposta contém a frase de recusa (ignora acentos, maiúsculas e espaços)."""
    return _normalizar(frase) in _normalizar(resposta)


def citacoes(resposta: str) -> List[str]:
    """Números dos artigos citados em [Fonte: ...], na ordem, sem repetição. Ex.: ["10", "09"].

    Uma citação sem "artigo_NN" (ex.: só o título) não é rastreável e fica de fora;
    `citacoes_brutas` conta quantas foram feitas no total.
    """
    vistos: List[str] = []
    for bloco in _CITACAO.findall(resposta):
        for num in _ARTIGO.findall(bloco):
            if num not in vistos:
                vistos.append(num)
    return vistos


def citacoes_brutas(resposta: str) -> int:
    return len(_CITACAO.findall(resposta))


def numero_artigo(nome_arquivo: str) -> Optional[str]:
    m = _ARTIGO.search(nome_arquivo)
    return m.group(1) if m else None


def precisao_citacao(citados: Sequence[str], recuperados: Set[str]) -> float:
    """Fração dos artigos citados que estavam de fato no contexto entregue ao LLM.

    1.0 = só citou o que recebeu. Abaixo disso, o LLM citou algo que não leu (citação inventada).
    """
    if not citados:
        raise ValueError("sem citações")
    return sum(c in recuperados for c in citados) / len(citados)


def cita_esperada(citados: Sequence[str], esperados: Set[str]) -> float:
    """1.0 se pelo menos um artigo citado é um dos esperados pelo gabarito."""
    return 1.0 if set(citados) & esperados else 0.0
