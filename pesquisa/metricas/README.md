# metricas

Localização: `pesquisa/metricas/`. Todos os comandos abaixo rodam a partir de `pesquisa/`.

Avaliação do RAG em camadas. **Candidata a ir para a `main`.** O módulo `app/modules/rag/` é só andaime experimental (extraído do notebook) e não deve ser a base definitiva.

Para não acoplar as métricas ao andaime, tudo aqui consome o RAG por uma fronteira estreita: uma chain que recebe a pergunta e devolve `{"context_docs", "question", "answer"}`. Se o `rag/` for substituído, só o ponto de entrada muda.

## Camadas

| Pasta | Responsabilidade | Precisa de LLM? |
| :--- | :--- | :--- |
| `dataset/` | Gabarito: perguntas, fonte(s) esperada(s), perguntas fora de escopo | não |
| `recuperacao/` | Hit Rate@K, MRR, Precision@K, Recall@K | não |
| `geracao/` | Citação válida (regex), faithfulness e relevância (LLM-juiz) | parcial |
| `relatorios/` | Saídas das execuções (CSV/JSON), para comparar configurações | não |

## Ordem sugerida

1. `dataset/`: ~20 perguntas, com `source` esperado igual ao nome do arquivo em `dev/data/`.
2. `recuperacao/`: roda sem custo de API e já permite comparar `chunk_size` e `K`.
3. `geracao/`: checagens determinísticas primeiro, juiz LLM depois (modelo diferente do gerador).

## Status

- `dataset/`: pronto. 20 perguntas em `gabarito.json`, validadas por `python -m metricas.dataset.loader` (só stdlib).
- `recuperacao/`: `metrics.py` (fórmulas), `runner.py` (executa e salva em `relatorios/`), `test_metrics.py`.
  Rodar: `python -m metricas.recuperacao.runner --nome baseline`. O runner depende do notebook: executa as células de índice de `rag_pipeline_semana2.ipynb` (`notebook_loader.py`), o que **reindexa** `dev/chroma_db`. Requer as dependências da célula `%pip` instaladas.
- `geracao/`: ainda vazio.

## Notas sobre o gabarito

- O gabarito é por **artigo** (fonte), não por chunk. Os chunks mudam quando `chunk_size` muda; os artigos não.
- O `metadata["source"]` dos chunks é o caminho completo do arquivo. Compare sempre com `nome_fonte(source)`.
- Perguntas `fora_de_escopo` não têm fonte esperada: o comportamento correto é a recusa definida no prompt do RAG.
- `pontos_chave` alimentam a checagem de completude na camada de geração.
