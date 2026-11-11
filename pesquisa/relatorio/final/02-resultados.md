# 2. LLM x RAG: resultados

> **⏳ Parcial (11/11, 17h30).** **T1 e T4 estão completos** (todas as métricas, avaliados no Mac). Para T2, T3 e Ref, as métricas que não dependem do juiz (recusa, citação, latência, tokens, chamadas ao LLM) são definitivas; as do juiz serão preenchidas quando a avaliação terminar (no Galaxy Book4). Onde já existe uma medida da mesma régua de uma rodada anterior, ela aparece marcada com †.

Configuração e métricas: ver [Metodologia](01-metodologia.md). 28 perguntas: 18 em escopo (4 diretas, 9 parafraseadas, 5 multi-fonte) e 10 fora de escopo.

## Visão geral

| Métrica | T1 · Só LLM | T2 · Só busca | T3 · RAG + LLM | T4 · RAG + LLM otimizado | Ref · k=3 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| **Taxa de sucesso (28)** | **0,32** | ⏳ | ⏳ | **0,75** | ⏳ |
| Completude | 0,59 | ⏳ | 0,91† | 0,76 | 0,81† |
| Completude multi-fonte | 0,53 | ⏳ | 0,87† | 0,73 | 0,53† |
| Fidelidade à base | 0,81 (juiz v3)‡ | 1,00 (construção) | 0,97† (piso) | 0,86 (piso) | 0,93† (piso) |
| Relevância | 0,89 | ⏳ | ⏳ | 0,79 | ⏳ |
| Cita a fonte | **0,00** | 1,00 | 1,00 | 1,00 | 1,00 |
| Cita um artigo correto | **0,00** | 1,00 | 1,00 | 1,00 | 1,00 |
| Precisão da citação | — | 1,00 | 1,00 | 1,00 | 1,00 |
| Recusa correta (10 fora de escopo) | 0,90 | **0,60** | 1,00 | 1,00 | 1,00 |
| Recusa indevida (18 em escopo) | 0,00 | 0,00 | 0,00 | 0,00 | 0,00 |
| Latência p50 das respostas | **54,7 s** | **0,1 s** | 20,8 s | 29,3 s* | 19,7 s |
| Latência p50 das recusas | 2,1 s | 0,15 s | 5,8 s | **0,14 s** | 3,8 s |
| Tokens por resposta | 844 | 542 | 315 | 310 | 288 |
| Tokens de contexto | 0 | 598 | 598 | 598 | 352 |
| Chamadas ao LLM (de 28) | 28 | 0 | 28 | **22** | 28 |

† Mesma régua (juiz `gemma3:12b`, mesmo prompt de completude e de fidelidade ao contexto), medida numa rodada anterior: a avaliação do T1 de hoje cedo e as rodadas k=5 e k=3 do [cap. 8](../08-geracao-e-llm-juiz.md). Serão substituídas pelos valores da rodada final.

\* Ver "Latência do T4" abaixo: a diferença para o T3 não vem do tamanho da resposta.

‡ O juiz v3 tende a ser generoso em afirmações vagas (ver [Metodologia](01-metodologia.md#calibração-do-juiz-novo-três-versões-até-um-método-utilizável)): é um teto, não um piso.

## T1 · Só LLM: o que muda sem a base, e por quê

**1. Cobre menos do que importa.** Completude de 0,59, contra 0,91 do RAG com k=5. O LLM escreve **quase 3 vezes mais** (844 tokens contra 315), mas com conselhos genéricos de produtividade, e não com o que os artigos dizem. Exemplo (q01, rotina de quem trabalha 8 h): a resposta do T1 traz "técnica SMART", "Matriz de Eisenhower" e "horário de pico entre 9h e 12h". Não fala dos **micro-blocos matinais de 45 a 60 min antes do expediente**, que é a recomendação do artigo 19.

> **Por quê:** o modelo responde com o que aprendeu no treino, uma "média" da internet sobre produtividade. Ele não tem como saber quais 20 artigos o StudyBuddy escolheu, nem o que eles recomendam.

**2. Nunca cita a fonte.** Tem citação = 0,00. O prompt pede "mencione a fonte quando possível", e ele nunca o faz em formato rastreável. Para um produto que promete orientação **baseada em evidência**, isso é eliminatório: o estudante não tem como conferir nada.

**3. Recusa bem, mas pelo motivo errado.** Recusou 9 das 10 perguntas fora de escopo, inclusive as "perigosas" (remédio para dormir, Bhaskara). A recusa vem do **prompt** ("se não for sobre estudos, recuse"), e não da base: ele recusa pelo **tema**. A única que respondeu foi a q20 (café para concentração), que é do tema, mas não está na base. Essa é justamente a recusa que só um sistema com base consegue fazer.

**4. É o mais lento.** 54,7 s por resposta (p50), contra 20,8 s do T3. Mais texto gerado significa mais tempo: no LLM local, a latência é dominada pela escrita, não pela leitura do contexto.

**5. Fala coisas que estão na base, mas não as que importam.** Fidelidade à base de 0,81 (medida afirmação por afirmação, com um juiz que tende a ser generoso) e relevância de 0,89. Ou seja, boa parte do que o LLM diz tem algum respaldo nos artigos, porque a base trata de técnicas de estudo conhecidas (Pomodoro, repetição espaçada, sono). O problema não é inventar muito: é **deixar de fora o que a base recomenda especificamente** (completude 0,59) e **não deixar o estudante conferir** (0 citações). Nas multi-fonte, a fidelidade à base cai para 0,63: quanto mais específica a pergunta, mais ele preenche com conhecimento genérico.

**6. Taxa de sucesso de 0,32.** São só as 9 recusas corretas. **Nenhuma** das 18 perguntas válidas passa na regra de sucesso, porque nenhuma resposta cita um artigo.

## T2 · Só busca: o que muda sem o LLM, e por quê

**1. É instantâneo e sempre rastreável.** 0,1 s por pergunta, mais de 200 vezes mais rápido que qualquer modo com LLM, e cada trecho já vem com `[Fonte: artigo_NN]`. Fidelidade à base de 1,00 por construção: o texto **é** a base, nada é inventado.

**2. Não sabe dizer "não sei".** Sem o LLM, só o porteiro de distância recusa, e ele barrou **6 das 10** perguntas fora de escopo. As outras 4 receberam 5 trechos sem relação com o que foi perguntado:

| Pergunta fora de escopo | $d_1$ | Artigos dos 5 trechos que o estudante recebeu |
| :-- | :-- | :-- |
| q26 · "Como faço para emagrecer mesmo passando o dia sentado estudando?" | 0,915 | 02 repetição espaçada, 04 carga cognitiva, 05 sono, 19 rotina de quem trabalha |
| q20 · "Quanto café posso tomar para me concentrar melhor nos estudos?" | 0,931 | 02, 05 sono, 06 trabalho profundo, 19 |
| q28 · "Qual cadeira ergonômica é melhor para estudar muitas horas?" | 0,988 | 02, 04, 17 ciclo de estudos, 19 |
| **q21 · "Qual o melhor remédio para dormir quando estou ansioso antes da prova?"** | 1,011 | 02, **05 sono e consolidação da memória**, 19 |

A q21 é o caso mais grave: uma pergunta sobre **medicamento** recebe trechos sobre sono, que um estudante ansioso pode ler como resposta.

> **Por quê:** a busca só mede **parecença**, não **resposta**. Uma pergunta sobre café para concentração fica perto dos artigos de concentração, mesmo que nenhum deles fale de café. Só quem **lê** o trecho percebe que ele não responde, e esse é o papel do LLM. No T4, essas mesmas 4 perguntas passam pelo porteiro, mas o LLM as recusa.

**3. Não é uma resposta, são 5 recortes.** O estudante recebe ~540 tokens de trechos crus, cortados no meio da frase, na ordem da busca, e precisa juntar as peças sozinho. Numa pergunta multi-fonte (q01), os trechos dos artigos 19, 17, 06 e 14 chegam misturados, sem a síntese "estude antes do expediente, em blocos de 45 a 60 min".

**4. Completude e relevância: ⏳.** A régua lexical da busca já sugeria que o conteúdo está lá: a **cobertura de pontos-chave@5 é 0,89** ([cap. 5](../05-resultados.md)). A dúvida é quanto disso é útil e quanto é ruído (relevância).

## T3 · RAG + LLM: a referência

É a configuração escolhida na semana passada ([cap. 8](../08-geracao-e-llm-juiz.md)): completude 0,91, fidelidade ao contexto 0,97, recusa de 10/10 e citação de 100%. As métricas novas (relevância e taxa de sucesso) estão ⏳.

**A comparação com o k=3 (Ref)** mostra de onde veio o ganho da semana passada: mais contexto (352 → 598 tokens) levou a completude multi-fonte de 0,53 para 0,87, com respostas quase do mesmo tamanho (288 → 315 tokens).

## T4 · RAG + LLM otimizado: o que mudou

**1. O porteiro funcionou como esperado.** 6 das 10 perguntas fora de escopo foram recusadas **sem chamar o LLM**: em 0,14 s, contra 5,8 s no T3. São 6 das 28 chamadas ao LLM economizadas (21%). As 4 que passaram (q20, q21, q26, q28) foram recusadas pelo LLM. Recusa correta 10/10 e recusa indevida 0/18, iguais às do T3.

**2. O prompt corrigiu a diluição da q12.** No T3, a resposta para "como sair do 'vou estudar amanhã'?" tinha perdido a instrução central do artigo 08. No T4, ela voltou para a primeira linha:

> *"Para sair do 'vou estudar amanhã' e realmente começar, é essencial automatizar o início da conduta através da formulação de intenções de implementação [Fonte: artigo_08]. [...] 1. Defina com precisão 'Quando', 'Onde' e 'Como' você iniciará o estudo."*

**3. Mas o prompt novo custou completude em outras perguntas.** A completude caiu de 0,91 (T3) para **0,76**, e a fidelidade ao contexto, de 0,97 para **0,86**. Pergunta a pergunta, contra a avaliação do T3 com a mesma régua:

| | Perguntas |
| :-- | :-- |
| Completude **subiu** | q03 (0,67 → 1,00), **q12** (0,50 → 1,00) |
| Completude **caiu** | q01, q04, q16 (1,00 → 0,67); q11, q13, q15, q18 (1,00 → 0,50); q08 (0,50 → 0,00) |
| Igual | as outras 8 |

O saldo é **2 perguntas melhores e 8 piores**. As respostas não ficaram mais curtas (310 contra 315 tokens), então o modelo não está escrevendo menos: está **escolhendo outros pontos**. A hipótese mais provável, **não testada**, é a regra 5 ("ignore os fragmentos que não tratam do que foi perguntado"): ela corrige a q12, mas faz o modelo descartar trechos que tinham pontos-chave em perguntas mais abertas. A regra 3 ("comece com 1 ou 2 frases que respondem diretamente") também gera sínteses que o juiz de fidelidade às vezes reprova (q09: fidelidade 0,67 → 0,00).

> **Ressalva:** o T3 desta comparação foi avaliado na semana passada, com o mesmo juiz e os mesmos prompts de avaliação. A reavaliação do T3 na rodada atual (pendente) confirma, ou não, que a diferença é do prompt e não de variação do juiz.

**4. Taxa de sucesso de 0,75.** As 10 recusas corretas mais 11 das 18 válidas. Das 7 que falham, 5 ficaram com completude abaixo de 2/3 (q08, q11, q13, q15, q18) e 2 com fidelidade abaixo de 0,8 (q09 e q17, que cobrem todos os pontos-chave, mas acrescentam sínteses que o juiz não encontrou no contexto).

## T3 + porteiro: a combinação que sai dos dados

O porteiro **não recusou nenhuma pergunta válida** (a maior $d_1$ em escopo é 0,901, e o corte é 1,026). Então, nas 18 perguntas em escopo, "T3 + porteiro" se comporta **exatamente** como o T3: busca igual, prompt igual, resposta igual. Nas 10 fora de escopo, se comporta **exatamente** como o T4: 6 barradas pelo porteiro e 4 recusadas pelo prompt do notebook, que também recusou as 10 no T3.

| | T3 | T4 | **T3 + porteiro** |
| :-- | :-- | :-- | :-- |
| Completude | 0,91† | 0,76 | **0,91** (= T3) |
| Fidelidade ao contexto | 0,97† | 0,86 | **0,97** (= T3) |
| Recusa correta / indevida | 10/10 · 0/18 | 10/10 · 0/18 | **10/10 · 0/18** |
| Recusas sem chamar o LLM | 0 | 6 | **6** (= T4) |
| Diluição da q12 | sim | corrigida | sim (= T3) |

Esta linha é **deduzida** dos testes T3 e T4, não medida como um quinto teste. A dedução vale porque as duas peças não interagem: o porteiro só age quando $d_1 > \tau$, o que não aconteceu em nenhuma pergunta válida.

### Latência do T4

As respostas do T4 levaram 29,3 s (p50), contra 20,8 s do T3, **com o mesmo tamanho** (310 contra 315 tokens). Então não é o prompt fazendo o modelo escrever mais. As respostas do T3 foram geradas em outro dia, com o Mac em outro estado (temperatura, outros programas abertos, modelo já carregado). **Tratamos essa diferença como inconclusiva:** para medi-la de verdade, os dois teriam de ser gerados em sequência, na mesma sessão.

## Conferência manual do juiz

⏳ Amostra de afirmações do T1 conferida no texto dos artigos, para estimar quanto a v3 do juiz superestima a fidelidade (ver [Metodologia, calibração](01-metodologia.md#calibração-do-juiz-novo-três-versões-até-um-método-utilizável)).

## Limitações

- **Amostra pequena:** 18 perguntas em escopo; cada uma vale ~0,056. Diferenças menores que ~0,1 entre testes podem ser ruído.
- **Fidelidade à base com réguas diferentes por teste:** no T1 ela é medida afirmação por afirmação; no T3, T4 e Ref, ela é o piso dado pela fidelidade ao contexto; no T2, é 1,00 por construção. A comparação é justa na direção (o RAG não pode ter fidelidade à base **menor** que a medida), mas não é idêntica.
- **O juiz da fidelidade à base (v3) tende a ser generoso** em afirmações vagas, o que favorece o T1.
- **O porteiro foi calibrado em paráfrases escritas por nós**, não em perguntas reais de estudantes. Com perguntas reais, o corte pode precisar de ajuste.
- **Tudo local:** `qwen3:8b` gerando e `gemma3:12b` julgando, num M4. Um LLM via API (GPT-4o-mini, Gemini) seria mais rápido e provavelmente mais completo. Os resultados valem para esta configuração.
