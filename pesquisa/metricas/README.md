# metricas

Localização: `pesquisa/metricas/`. Todos os comandos abaixo rodam a partir de `pesquisa/`.

Avaliação do RAG em camadas, com modelos locais (Ollama). A avaliação da Semana 3 via API do Gemini (RAGAS) está em [`../RELATORIO_COMPARACAO_SEMANA3.md`](../RELATORIO_COMPARACAO_SEMANA3.md); a comparação entre as duas está em [`../relatorio/final/README.md`](../relatorio/final/README.md#duas-avaliações-da-semana-3).

O RAG medido é o do notebook `rag_pipeline_semana2.ipynb`: o `recuperacao/notebook_loader.py` executa as células de índice dele e as métricas usam o `vectorstore` resultante. Não há módulo RAG separado em `dev/`.

## Camadas

| Pasta | Responsabilidade | Precisa de LLM? |
| :--- | :--- | :--- |
| `dataset/` | Gabarito: perguntas, fonte(s) esperada(s), perguntas fora de escopo | não |
| `recuperacao/` | Hit Rate@K, MRR, Precision@K, Recall@K | não |
| `geracao/` | Citação válida (regex), fidelidade, completude e fidelidade à base + relevância (LLM-juiz) | parcial |
| `comparacao/` | Os 4 modos de uso (só LLM, só busca, RAG + LLM, RAG + LLM otimizado) avaliados com as mesmas métricas | parcial |
| `relatorios/` | Saídas das execuções (CSV/JSON), para comparar configurações | não |

## Ordem sugerida

1. `dataset/`: ~20 perguntas, com `source` esperado igual ao nome do arquivo em `dev/data/`.
2. `recuperacao/`: roda sem custo de API e já permite comparar `chunk_size` e `K`.
3. `geracao/`: checagens determinísticas primeiro, juiz LLM depois (modelo diferente do gerador).

## Status

- `dataset/`: pronto. 20 perguntas em `gabarito.json`, validadas por `python -m metricas.dataset.loader` (só stdlib).
- `recuperacao/`: `metrics.py` (fórmulas), `runner.py` (executa e salva em `relatorios/`), `test_metrics.py`.
  Rodar: `python -m metricas.recuperacao.runner --nome baseline`. O runner depende do notebook: executa as células de índice de `rag_pipeline_semana2.ipynb` (`notebook_loader.py`), o que **reindexa** `dev/chroma_db`. Requer as dependências da célula `%pip` instaladas.
- `geracao/`: `metrics.py` (regras: recusa e citação), `juiz.py` (LLM-juiz `gemma3:12b`: fidelidade, completude, fidelidade à base e relevância, sempre com evidência conferida em Python), `runner.py`.
  Rodar: `python -m metricas.geracao.runner --nome qwen3_bge` (opcional `TOP_K=5`).
- `comparacao/`: `modos.py` gera as respostas de cada modo (`calibrar`, `llm_puro`, `so_busca`, `rag_llm`, `rag_otimizado`, `rag_llm_k3`); `avaliar.py` avalia todos e monta `relatorios/comparacao.md`.
  Rodar: `python -m metricas.comparacao.modos <modo>` e depois `python -m metricas.comparacao.avaliar <modos...>`.

## Notas sobre o gabarito

- O gabarito é por **artigo** (fonte), não por chunk. Os chunks mudam quando `chunk_size` muda; os artigos não.
- O `metadata["source"]` dos chunks é o caminho completo do arquivo. Compare sempre com `nome_fonte(source)`.
- Perguntas `fora_de_escopo` não têm fonte esperada: o comportamento correto é a recusa definida no prompt do RAG.
- `pontos_chave` alimentam a checagem de completude na camada de geração.
