# 3. Decisão para o produto

> **⏳ Prévia (11/11, 15h).** A recomendação abaixo usa as métricas definitivas (recusa, citação, custo) e as medidas do juiz de rodadas anteriores. Ela será confirmada, ou revista, quando a avaliação final terminar.

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

## Recomendação (prévia): T4 · RAG + LLM otimizado

**Entre T3 e T4**, o T4 ganha naquilo que já é definitivo e empata no resto:

| | T3 | T4 |
| :-- | :-- | :-- |
| Recusa correta / indevida | 10/10 · 0/18 | 10/10 · 0/18 |
| Citação | 100% | 100% |
| Recusas sem chamar o LLM | 0 de 10 | **6 de 10** |
| Tempo de uma recusa | 5,8 s | **0,14 s** |
| Diluição da q12 | Perdeu o "quando, onde e como" | **Corrigida** |
| Completude, fidelidade, relevância | 0,91† · 0,97† · ⏳ | ⏳ |

**Condição para confirmar:** a completude e a fidelidade do T4 precisam ficar **no mesmo nível das do T3** (diferença menor que ~0,05, ou seja, menos de uma pergunta). Se o prompt novo tiver custado pontos-chave em outras perguntas, a escolha volta a ser o T3 com o porteiro, mas sem o prompt novo.

**E o T2?** Ele não serve como produto principal, mas os números mostram um uso: como **modo de contingência** quando o LLM estiver fora do ar (é o que o `GroundedExtractiveFallbackLLM` do notebook já faz). Ele é instantâneo, 100% fiel e sempre cita a fonte. Nesse modo, o porteiro é a única defesa contra perguntas fora de escopo, e isso precisa ficar claro para o estudante.

## Próximos passos

1. **Levar o T4 para o notebook:** `TOP_K=5`, o porteiro com $\tau = 1{,}026$ e o prompt revisado.
2. **Recalibrar o porteiro com perguntas reais** de estudantes, assim que houver uso.
3. **Medir a latência T3 × T4 na mesma sessão**, para fechar a dúvida dos 21 s contra 29 s.
4. **Testar um gerador via API** (ex.: GPT-4o-mini), com o mesmo gabarito e o mesmo juiz, para ver se a completude e a latência compensam o custo por token.

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

> As avaliações finais (`comparacao_<teste>.json` e `comparacao.md`) e a versão atual do juiz ainda não estão no GitHub. Elas entram num commit quando a rodada terminar.
