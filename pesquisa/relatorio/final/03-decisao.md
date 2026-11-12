# 3. Decisão para o produto

> **Final (12/11).** T1, T2, T3 e T4 estão completos. A reavaliação do T3 **confirmou** a escolha: **T3 + porteiro**, com taxa de sucesso de 0,93 (T4: 0,75; T2: 0,75; T1: 0,32). A Ref (k=3) não foi reavaliada.

## O que o StudyBuddy precisa

O StudyBuddy é um mentor de estudos que promete **orientação baseada em evidência científica**. Disso saem os critérios, em ordem de importância:

| # | Critério | Métrica | Por quê |
| :-- | :-- | :-- | :-- |
| 1 | Não inventar | Fidelidade à base | Um conselho inventado com cara de científico é o pior erro possível para o produto |
| 2 | Deixar conferir | Cita um artigo correto | É o que diferencia o StudyBuddy de um chatbot genérico |
| 3 | Saber dizer "não sei" | Recusa correta / indevida | Responder remédio para dormir é um risco; recusar uma pergunta válida frustra o estudante |
| 4 | Responder o que importa | Completude, relevância | Cobrir os pontos-chave dos artigos, sem desviar do assunto |
| 5 | Custo | Latência, chamadas ao LLM, tokens | Importa, mas só entre opções que passam nos critérios 1 a 4 |

A **taxa de sucesso** (ver [Metodologia](01-metodologia.md#métricas-novas)) junta os critérios 1 a 4 numa regra só, por pergunta.

## Eliminatórios

| Teste | Elimina? | Motivo |
| :-- | :-- | :-- |
| T1 · Só LLM | **Sim** | Não cita fonte em nenhuma resposta (critério 2) e cobre bem menos dos pontos-chave (0,59 contra 0,91). Também é o mais lento |
| T2 · Só busca | **Sim, como produto principal** | Entrega trechos sem relação para 4 das 10 perguntas fora de escopo (critério 3) e não sintetiza a resposta |
| T3 · RAG + LLM | Não | Passa em tudo |
| T4 · RAG + LLM otimizado | Não | Passa em tudo |

## Recomendação: T3 + porteiro (RAG + LLM, k=5, prompt do notebook, com o porteiro de distância)

A prévia das 15h apontava o T4, **com uma condição**: a completude dele tinha de ficar no nível da do T3 (diferença menor que ~0,05). **A condição falhou:** a completude do T4 foi 0,76, contra 0,91 do T3, com 8 perguntas piores e 2 melhores ([Resultados › T4](02-resultados.md#t4--rag--llm-otimizado-o-que-mudou)). O prompt novo corrigiu a q12, mas custou mais do que resolveu.

As duas mudanças do T4 tiveram efeitos opostos, e só uma delas fica:

| Mudança do T4 | Efeito medido | Fica? |
| :-- | :-- | :-- |
| **Porteiro de distância** | Barra 6/10 fora de escopo sem chamar o LLM (0,14 s contra 5,8 s), sem recusar nenhuma válida | **Sim** |
| **Prompt revisado** | Corrige a q12, mas a completude cai de 0,91 para 0,76 e a fidelidade de 0,97 para 0,86 | **Não** |

| | T3 | T4 | **T3 + porteiro** |
| :-- | :-- | :-- | :-- |
| Completude | 0,91 | 0,76 | **0,91** |
| Fidelidade ao contexto | 0,98 | 0,86 | **0,98** |
| Recusa correta / indevida | 10/10 · 0/18 | 10/10 · 0/18 | **10/10 · 0/18** |
| Citação de artigo correto | 100% | 100% | **100%** |
| Chamadas ao LLM (de 28) | 28 | 22 | **22** |
| Tempo de uma recusa | 5,8 s | 0,14 s | **0,14 s** |

O "T3 + porteiro" é **deduzido** de T3 e T4, não medido como um quinto teste. A dedução é válida porque o porteiro não recusou nenhuma pergunta válida: em escopo, o pipeline é idêntico ao do T3 ([Resultados › T3 + porteiro](02-resultados.md#t3--porteiro-a-combinação-que-sai-dos-dados)).

Valores do T3 da reavaliação de 12/11 (mesmo juiz). Eles confirmam os da semana passada (completude ~0,9), e a escolha está fechada.

**E a diluição da q12?** Continua sendo um problema conhecido do k=5. Ela merece uma correção mais estreita que a do T4, por exemplo testar só a regra 3 do prompt novo, sem a regra 5, medindo de novo nas 18 perguntas.

**E o T2?** Ele não serve como produto principal, mas os números mostram um uso: como **modo de contingência** quando o LLM estiver fora do ar (é o que o `GroundedExtractiveFallbackLLM` do notebook já faz). Ele é instantâneo, 100% fiel e sempre cita a fonte. Nesse modo, o porteiro é a única defesa contra perguntas fora de escopo (4 das 10 passaram, inclusive "remédio para dormir"), e isso precisa ficar claro para o estudante.

## Próximos passos

1. **Levar o T3 + porteiro para o notebook:** `TOP_K=5`, o prompt atual e o porteiro com $\tau = 1{,}026$ antes da chamada ao LLM.
2. **Atacar a diluição da q12 com uma mudança menor no prompt** (só a regra 3, sem a regra 5) e medir de novo.
3. **Recalibrar o porteiro com perguntas reais** de estudantes, assim que houver uso.
4. **Medir a latência T3 × T4 na mesma sessão**, para fechar a dúvida dos 21 s contra 29 s.
5. **Testar um gerador via API** (ex.: GPT-4o-mini), com o mesmo gabarito e o mesmo juiz, para ver se a completude e a latência compensam o custo por token.

## Links

| O quê | Onde |
| :-- | :-- |
| Repositório (branch do experimento) | <https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy/tree/metricas-experimento> |
| Código dos 4 testes | [`pesquisa/metricas/comparacao/`](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy/tree/metricas-experimento/pesquisa/metricas/comparacao) (`modos.py` gera, `avaliar.py` avalia) |
| LLM-juiz | [`pesquisa/metricas/geracao/juiz.py`](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy/blob/metricas-experimento/pesquisa/metricas/geracao/juiz.py) |
| Gabarito (28 perguntas) | [`pesquisa/metricas/dataset/gabarito.json`](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy/blob/metricas-experimento/pesquisa/metricas/dataset/gabarito.json) |
| Respostas e avaliações de cada teste | [`pesquisa/metricas/relatorios/`](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy/tree/metricas-experimento/pesquisa/metricas/relatorios) (`comparacao_respostas_<teste>.json`, `comparacao_<teste>.json`, `comparacao.md`) |
| Pipeline RAG (notebook) | [`pesquisa/rag_pipeline_semana2.ipynb`](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy/blob/metricas-experimento/pesquisa/rag_pipeline_semana2.ipynb) |
| Documentação do projeto | <https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation> |

> As avaliações de T1, T2, T3 e T4 estão no repositório. A da Ref (k=3) não foi refeita (ver `pesquisa/metricas/comparacao/RETOMAR.md`).
