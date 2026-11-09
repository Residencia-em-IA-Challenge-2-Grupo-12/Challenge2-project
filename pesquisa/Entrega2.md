# Entrega 2 — Base de Conhecimento e Protótipo RAG

Resumo da entrega da Semana 2 da Residência em IA (Grupo 12): o primeiro mecanismo de **Retrieval-Augmented Generation (RAG)** do projeto, que fundamenta orientações de estudo e produtividade em evidências científicas.

A especificação técnica completa vive no repositório de documentação, que é a fonte única:
[Challenge2-documentation › Semana 2](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation/tree/main/docs/semana-2).

## O que foi entregue

| Componente | Onde está |
| :--- | :--- |
| Base de conhecimento (20 artigos científicos, 4 blocos temáticos) | `dev/data/` |
| Pipeline RAG (ingestão, chunking, embeddings, ChromaDB, chain LCEL) | `pesquisa/rag_pipeline_semana2.ipynb` |
| Índice vetorial (gerado localmente, fora do git) | `dev/chroma_db/` |
| Visão geral, decisões técnicas e como executar | [`README.md`](../README.md) |

## Arquitetura em uma frase

Ingestão em lote (`.txt`/`.pdf` → chunks de 500 caracteres com overlap de 50 → embeddings → ChromaDB) e consulta em tempo real (pergunta → Top-K=3 por distância L2 → prompt com *grounding* → LLM com `temperature=0.0` → resposta com citação de fonte).

## Documentação detalhada

- [Visão geral da Semana 2](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation/tree/main/docs/semana-2/index.md)
- [Arquitetura da solução](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation/tree/main/docs/semana-2/arquitetura.md)
- [Pipeline de ingestão](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation/tree/main/docs/semana-2/pipeline-ingestao.md)
- [Estratégias de chunking e embeddings](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation/tree/main/docs/semana-2/estrategias-rag.md)
