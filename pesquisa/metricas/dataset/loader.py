"""Carrega e valida o gabarito. Só biblioteca padrão (sem langchain).

Uso (a partir de pesquisa/): python -m metricas.dataset.loader
"""

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PureWindowsPath
from typing import List, Optional

DATASET_DIR = Path(__file__).resolve().parent
GABARITO_PATH = DATASET_DIR / "gabarito.json"
DATA_DIR = DATASET_DIR.parents[2] / "dev" / "data"  # dev/data (base de conhecimento)

TIPOS = {"direta", "parafraseada", "multi_fonte", "fora_de_escopo"}
ORIGENS = {"notebook", "novo"}


@dataclass(frozen=True)
class Pergunta:
    id: str
    pergunta: str
    fontes_esperadas: List[str]
    tipo: str
    origem: str
    persona: Optional[str] = None
    pontos_chave: List[str] = field(default_factory=list)

    @property
    def fora_de_escopo(self) -> bool:
        return self.tipo == "fora_de_escopo"


def nome_fonte(source: str) -> str:
    """Reduz o `metadata['source']` do chunk ao nome do arquivo.

    O TextLoader grava o caminho completo (ex.: C:\\...\\dev\\data\\artigo_01.txt),
    então a comparação com o gabarito deve ser pelo nome do arquivo.
    """
    return PureWindowsPath(source).name


def validar(perguntas: List[Pergunta], data_dir: Path = DATA_DIR) -> List[str]:
    """Devolve a lista de problemas encontrados (vazia se o gabarito está consistente)."""
    erros: List[str] = []
    ids = [p.id for p in perguntas]
    for dup in {i for i in ids if ids.count(i) > 1}:
        erros.append(f"id duplicado: {dup}")

    existentes = {f.name for f in data_dir.glob("*.txt")} if data_dir.exists() else set()

    for p in perguntas:
        if p.tipo not in TIPOS:
            erros.append(f"{p.id}: tipo inválido '{p.tipo}'")
        if p.origem not in ORIGENS:
            erros.append(f"{p.id}: origem inválida '{p.origem}'")
        if not p.pergunta.strip():
            erros.append(f"{p.id}: pergunta vazia")
        if p.fora_de_escopo:
            if p.fontes_esperadas:
                erros.append(f"{p.id}: fora_de_escopo não deve ter fontes_esperadas")
        else:
            if not p.fontes_esperadas:
                erros.append(f"{p.id}: sem fontes_esperadas")
            if not p.pontos_chave:
                erros.append(f"{p.id}: sem pontos_chave")
            if p.tipo == "multi_fonte" and len(p.fontes_esperadas) < 2:
                erros.append(f"{p.id}: multi_fonte exige 2+ fontes")
        for f in p.fontes_esperadas:
            if existentes and f not in existentes:
                erros.append(f"{p.id}: fonte inexistente em data/: {f}")
    return erros


def load_gabarito(path: Path = GABARITO_PATH, validate: bool = True) -> List[Pergunta]:
    with open(path, encoding="utf-8") as fh:
        perguntas = [Pergunta(**item) for item in json.load(fh)]
    if validate:
        erros = validar(perguntas)
        if erros:
            raise ValueError("Gabarito inválido:\n  - " + "\n  - ".join(erros))
    return perguntas


if __name__ == "__main__":
    qs = load_gabarito()
    print(f"{len(qs)} perguntas válidas")
    print("por tipo  :", dict(Counter(q.tipo for q in qs)))
    print("por origem:", dict(Counter(q.origem for q in qs)))
    cobertas = {f for q in qs for f in q.fontes_esperadas}
    print(f"artigos cobertos: {len(cobertas)}/20")
