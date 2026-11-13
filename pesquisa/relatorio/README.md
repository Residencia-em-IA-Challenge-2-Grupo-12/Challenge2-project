# Relatório de pesquisa: métricas para validar um RAG

Mini relatório de apoio ao experimento de métricas do StudyBuddy (branch `metricas-experimento`, pasta `pesquisa/metricas/` do repositório).

## Ordem de leitura

1. [O que é um RAG](01-o-que-e-rag.md): o problema que ele resolve e as peças que o compõem.
2. [O que são métricas](02-o-que-sao-metricas.md): o que é medir, o que é um gabarito e quais métricas usamos.
3. [Por que usar métricas para validar um RAG](03-por-que-metricas-no-rag.md): os limites de "olhar as respostas" e o que os números revelam.
4. [Exemplo simples](04-exemplo-simples.md): uma base de 4 artigos e 5 perguntas, com cada métrica calculada à mão.
5. [Resultados: bge-m3 x qwen3-embedding](05-resultados.md): o comparativo de embeddings e chunking no RAG do StudyBuddy.
6. [Como testar na mão](06-teste-manual.md): o script de perguntas lado a lado, como ler a saída e um roteiro de testes.
7. [Métricas da busca](07-metricas-da-busca.md): todas as métricas da camada de busca, com a matemática detalhada e exemplos calculados.
8. [Geração e LLM-juiz](08-geracao-e-llm-juiz.md): as métricas da resposta (fidelidade, completude, citação, recusa), por que usar um modelo como juiz e como ele foi calibrado.

## Semana 3: LLM x RAG (relatório final)

O relatório da entrega da semana 3 está na pasta [`final/`](final/README.md):

1. [Metodologia](final/01-metodologia.md): os 4 testes (só LLM, só busca, RAG + LLM, RAG + LLM otimizado), as métricas novas (fidelidade à base, relevância, taxa de sucesso) e a calibração do porteiro e do juiz.
2. [Resultados](final/02-resultados.md): os números dos 4 testes, o que muda com e sem LLM e com e sem RAG, e por que muda.
3. [Decisão](final/03-decisao.md): qual teste o StudyBuddy adota, por quê, e os links do projeto.

## Glossário rápido

| Termo | Significado |
| :-- | :-- |
| **LLM** | Modelo de linguagem que gera texto (ex.: GPT, Gemini, Qwen). |
| **Chunk** | Pedaço de um documento. A base é cortada em chunks antes de ser indexada. |
| **Embedding** | Vetor de números que representa o significado de um texto. Textos parecidos geram vetores próximos. |
| **Banco vetorial** | Banco que guarda os embeddings e encontra os mais próximos de uma pergunta (aqui: ChromaDB). |
| **Top-K** | Quantos chunks a busca devolve para cada pergunta. |
| **Gabarito** | Lista de perguntas com a resposta "certa" já conhecida, usada como régua. |
| **Porteiro de distância** | Regra que recusa a pergunta antes de chamar o LLM quando o trecho mais próximo está longe demais da base ($d_1 > \tau$). |
| **LLM-juiz** | Um segundo modelo de linguagem, diferente do que gera as respostas, usado para avaliar o que regra não mede (ex.: se a resposta inventou algo). |
