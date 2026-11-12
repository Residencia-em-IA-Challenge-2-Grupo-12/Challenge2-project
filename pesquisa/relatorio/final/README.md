# Relatório final · Semana 3: LLM x RAG

> **Status: final (12/11).** Metodologia e os quatro testes (T1 a T4) estão avaliados por completo. A Ref (k=3) não foi reavaliada e usa os valores da semana passada.

## A entrega

> Comparar o uso de LLM x RAG, melhorar as métricas e testar diferentes formas de usar o RAG (ex.: com e sem LLM: o que muda e por que muda). **Critério de pronto:** fazer 4 testes diferentes e escolher o melhor para o produto.

| Pedido | Onde está |
| :-- | :-- |
| 4 testes diferentes | [Metodologia](01-metodologia.md#os-4-testes) e [Resultados](02-resultados.md) |
| Metodologia | [Metodologia](01-metodologia.md) |
| Melhoria das métricas | [Metodologia › Métricas novas](01-metodologia.md#métricas-novas) e [calibração do juiz](01-metodologia.md#calibração-do-juiz-novo-três-versões-até-um-método-utilizável) |
| O que muda com e sem LLM, e por quê | [Resultados › T1](02-resultados.md#t1--só-llm-o-que-muda-sem-a-base-e-por-quê) e [T2](02-resultados.md#t2--só-busca-o-que-muda-sem-o-llm-e-por-quê) |
| Escolha do melhor para o produto | [Decisão](03-decisao.md) |
| Links do projeto | [Decisão › Links](03-decisao.md#links) |

## Os 4 testes em uma linha cada

| # | Teste | Resultado (prévia) |
| :-- | :-- | :-- |
| T1 | **Só LLM** (sem RAG) | ❌ Taxa de sucesso 0,32: não cita fonte nenhuma vez, cobre menos (0,59 contra 0,91) e é o mais lento (55 s) |
| T2 | **Só busca** (sem LLM) | ❌ como produto: instantâneo e fiel, mas entrega trechos sem relação para 4 das 10 perguntas fora de escopo |
| T3 | **RAG + LLM** (k=5) | ✅ Passa em tudo: completude 0,91†, fidelidade 0,97†, recusa 10/10 |
| T4 | **RAG + LLM otimizado** (porteiro + prompt) | ⚠️ O porteiro funciona (6/10 recusas sem LLM), mas o prompt novo derrubou a completude para 0,76 |
| **Escolha** | **T3 + porteiro** | ✅ A completude do T3 com a economia do porteiro do T4. Confirmado pela reavaliação do T3 (taxa de sucesso 0,93) |

## Ordem de leitura

1. [Metodologia](01-metodologia.md): os 4 testes, o que fica fixo, o porteiro, as métricas novas e a calibração do juiz.
2. [Resultados](02-resultados.md): os números e, para cada teste, o que muda e por quê.
3. [Decisão](03-decisao.md): os critérios do produto, a escolha, os próximos passos e os links.

Os capítulos de fundamentos e das semanas anteriores (o que é RAG, métricas da busca, LLM-juiz) estão na [pasta acima](../README.md).
