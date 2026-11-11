# 1. LLM x RAG: metodologia dos 4 testes

Até a semana passada, toda rodada de avaliação usava **RAG + LLM**: a busca escolhia os trechos e o LLM escrevia a resposta. Nunca medimos o que acontece **sem** uma das duas peças. Esta etapa responde a três perguntas:

1. **O RAG vale a pena?** O que muda quando o LLM responde sozinho, sem a base?
2. **O LLM vale a pena?** O que muda quando o estudante recebe só os trechos da busca, sem um LLM para escrever?
3. **Qual forma de usar o RAG é a melhor para o StudyBuddy?**

## Os 4 testes

| # | Teste | Busca | LLM | O que muda em relação ao T3 |
| :-- | :-- | :-- | :-- | :-- |
| **T1** | **Só LLM** (sem RAG) | — | qwen3:8b | Tira a busca: o LLM responde com o que aprendeu no treino |
| **T2** | **Só busca** (sem LLM) | bge-m3, k=5 | — | Tira o LLM: o estudante recebe os 5 trechos crus, com a fonte |
| **T3** | **RAG + LLM** | bge-m3, k=5 | qwen3:8b | Nada: é a melhor configuração da semana passada ([cap. 8](../08-geracao-e-llm-juiz.md)) |
| **T4** | **RAG + LLM otimizado** | bge-m3, k=5 + porteiro | qwen3:8b | Porteiro de distância antes do LLM + prompt revisado |
| Ref | RAG + LLM com k=3 | bge-m3, k=3 | qwen3:8b | Só para mostrar a evolução desde a semana passada |

Para os testes serem comparáveis, **mudamos uma peça por vez** e mantivemos fixo todo o resto:

| Item fixo | Valor |
| :-- | :-- |
| Base | Os 20 artigos de `dev/data/` (~14 mil caracteres, ~4 mil tokens) |
| Gabarito | As mesmas 28 perguntas: 4 diretas, 9 parafraseadas, 5 multi-fonte, 10 fora de escopo |
| Chunking | 500 caracteres, overlap 50 (o do notebook) |
| Gerador | `qwen3:8b` via Ollama, `temperature=0`, sem modo pensamento |
| Juiz | `gemma3:12b` via Ollama, `temperature=0`, outra família e maior que o gerador |
| Máquina | Apple M4, 24 GB, tudo local (custo de API = 0) |

### T1: Só LLM

O LLM recebe só a pergunta. Para a comparação ser justa, ele recebe **o mesmo papel** ("Mentor Técnico de Estudos e Produtividade"), **o mesmo formato** (tópicos, citar a fonte quando possível) e **a mesma chance de recusar**: se a pergunta não for sobre estudos ou se ele não souber, deve responder a mesma frase de recusa do RAG. A única coisa que ele não tem é a base.

### T2: Só busca

A resposta é o que o notebook já fazia no modo offline (`GroundedExtractiveFallbackLLM`): os 5 trechos mais próximos, na ordem da busca, cada um com `[Fonte: artigo_NN]`. Sem LLM, não existe quem diga "não sei", então a recusa fica com o **porteiro de distância** (abaixo).

### T3: RAG + LLM

É o pipeline do notebook com `TOP_K=5`. As respostas são as mesmas da comparação k=3 × k=5 do [cap. 8](../08-geracao-e-llm-juiz.md) (`geracao_respostas_qwen3_bge_k5.json`): com `temperature=0`, gerar de novo daria o mesmo texto. Só a avaliação foi refeita, com as métricas novas.

### T4: RAG + LLM otimizado

Duas mudanças, cada uma ligada a um problema medido antes:

**1. Porteiro de distância.** Antes de chamar o LLM, olhamos a distância $d_1$ do trecho mais próximo. Se $d_1 > \tau$, a pergunta está longe demais da base, e o sistema recusa **sem chamar o LLM**. Motivo: no cap. 8, cada recusa custava ~4 a 6 s de LLM, e a recusa dependia só do prompt.

**2. Prompt revisado** contra a *diluição* observada com k=5 (a q12 perdeu "defina quando, onde e como" quando entrou um 5º trecho fora do assunto). As regras 1 e 2 do notebook (não inventar, frase de recusa) ficaram iguais. As novas:

> 3. Comece com 1 ou 2 frases que respondem diretamente ao que foi perguntado.
> 4. Depois, liste em tópicos as recomendações práticas MAIS ligadas à pergunta, priorizando instruções concretas (números, passos, quando/onde/como fazer).
> 5. Ignore os fragmentos que não tratam do que foi perguntado, mesmo que estejam no contexto.
> 6. Termine cada tópico com a fonte utilizada no formato [Fonte: nome_do_arquivo].

## Calibração do porteiro (sem "colar" do gabarito)

Escolher $\tau$ olhando as 28 perguntas de teste seria trapaça: o corte ficaria perfeito no teste e falharia com perguntas novas. Por isso, calibramos com dados que **não entram na avaliação**: as **36 paráfrases** das perguntas em escopo (as `variacoes` do gabarito, escritas como um estudante escreveria).

$$
\tau = \max_{v \,\in\, \text{paráfrases}} d_1(v) = 1{,}026
$$

Ou seja, o corte não recusa nenhuma das 36 paráfrases válidas. A paráfrase mais distante foi *"O edital é enorme, por onde começo sem estudar tudo na ordem?"* (q04).

Onde ficam as 28 perguntas de teste em relação a esse corte:

| Grupo | $d_1$ | Resultado com $\tau = 1{,}026$ |
| :-- | :-- | :-- |
| 18 em escopo | máx. 0,901 | Nenhuma recusada ✓ |
| q25, q23, q19, q22, q27, q24 (fora de escopo) | 1,028 a 1,466 | **Barradas** antes do LLM ✓ |
| q21 (1,011), q28 (0,988), q20 (0,931), q26 (0,916) | abaixo do corte | **Passam** para o LLM ✗ |

O porteiro barra **6 das 10** perguntas fora de escopo. As 4 que passam são as "perigosas", próximas do tema da base (café, cadeira ergonômica, emagrecer estudando sentado). No T4, elas ainda encontram a segunda barreira, o prompt. No T2, não há segunda barreira.

> No [cap. 5](../05-resultados.md), a margem do bge-m3 era de só +0,014: existia um corte perfeito, mas colado nas perguntas válidas. Calibrando com paráfrases, o corte fica mais folgado (protege as perguntas válidas escritas de outro jeito) e, em troca, deixa passar as fora de escopo mais parecidas com a base. É a troca certa para um porteiro: **recusar uma pergunta válida é pior do que mandar uma fora de escopo para o LLM**, que ainda pode recusá-la.

## Métricas

### As que já existiam (cap. 8)

| Métrica | Pergunta | Quem mede |
| :-- | :-- | :-- |
| Recusa correta / indevida | Recusou quando devia? Recusou sem motivo? | Regra |
| Cita artigo correto | Citou pelo menos um artigo esperado pelo gabarito? | Regra |
| Fidelidade ao **contexto** | O que ele disse está nos trechos que recebeu? | Juiz |
| Completude | Cobre os pontos-chave do gabarito? | Juiz, com evidência |

### O problema: a fidelidade antiga não compara os 4 testes

A fidelidade ao contexto pergunta "o LLM disse algo que **não recebeu**?". Isso não se aplica a dois dos testes:

- **T1** não recebe contexto nenhum. Toda afirmação seria "não sustentada", inclusive as corretas.
- **T2** devolve o próprio contexto. A fidelidade é 1,00 por construção.

Precisávamos de uma régua que valesse igual para os 4.

### Métricas novas

**1. Fidelidade à base.** Fração das afirmações da resposta que têm respaldo em **algum dos 20 artigos**, e não só nos trechos recuperados.

$$
\text{Fidelidade à base}(r) = \frac{\#\,\text{afirmações sustentadas por algum artigo da base}}{\#\,\text{afirmações}}
$$

Ela responde à pergunta que interessa ao produto: **o que o StudyBuddy diz tem respaldo nos artigos científicos que escolhemos?** Um conselho "verdadeiro no mundo real", mas ausente da base (ex.: "use a Matriz de Eisenhower"), conta como não sustentado: o StudyBuddy promete responder **com base em evidências da base**, não com conhecimento genérico.

Como ela é medida em cada teste:

| Teste | Como | Por quê |
| :-- | :-- | :-- |
| T1 (sem contexto) | Juiz, **uma afirmação por chamada**, contra os 20 artigos inteiros (~4 mil tokens, `num_ctx=8192`) | É o único teste em que a fidelidade antiga não se aplica |
| T2 (trechos crus) | 1,00 por construção | A resposta é o próprio texto da base |
| T3, T4, Ref | Igual à fidelidade ao **contexto** | O contexto é um pedaço da base: o que está sustentado nele está sustentado na base. É um **piso**: uma afirmação fora do contexto, mas presente em outro artigo, conta como não sustentada |

**2. Relevância.** Numa chamada separada, que recebe só a pergunta e a resposta, o juiz quebra a resposta em afirmações e marca se cada uma **ajuda diretamente a responder o que foi perguntado** (conselho genérico que serviria para qualquer pergunta conta como não). Ela cobre o ponto cego visto na q15 do cap. 8: fidelidade 1,00, mas a resposta falava de outro assunto.

$$
\text{Relevância}(r) = \frac{\#\,\text{afirmações relevantes para a pergunta}}{\#\,\text{afirmações}}
$$

**3. Taxa de sucesso**, uma métrica de produto por pergunta: o resultado **serve ao estudante**?

| Tipo de pergunta | Sucesso se... |
| :-- | :-- |
| Fora de escopo | recusou |
| Em escopo | não recusou **e** completude ≥ 2/3 **e** fidelidade à base ≥ 0,8 **e** citou um artigo correto |

A taxa de sucesso é a fração das 28 perguntas com sucesso. Ela junta, numa regra só e explícita, as quatro coisas que o produto precisa ao mesmo tempo: **responder, responder o suficiente, não inventar e deixar o estudante conferir a fonte**. As métricas individuais continuam no relatório para explicar *por que* um teste falha.

**4. Custo:** latência p50 (de ponta a ponta, por pergunta), tokens da resposta, tokens de contexto e **número de chamadas ao LLM**.

### Calibração do juiz novo: três versões até um método utilizável

O juiz é um instrumento de medida e foi conferido à mão antes de entrar na tabela. A fidelidade à base precisou de três versões:

| Versão | Como julgava | Problema encontrado | Exemplo |
| :-- | :-- | :-- | :-- |
| v1 | Base inteira + todas as afirmações numa chamada só | **Generoso demais:** aprovava por semelhança de tema | Aprovou *"Use a técnica SMART"* e *"Matriz de Eisenhower"*. Um `grep` nos 20 artigos não acha nenhum dos dois |
| v2 | Igual à v1, mas exigindo a frase da base copiada como `evidencia`, conferida em Python | **Severo demais:** com ~15 afirmações e 4 mil tokens de base numa chamada, o modelo de 12B parava de procurar e reprovava sem evidência | Na q10, reprovou *"2 a 4 horas de trabalho profundo"* (artigo 06) e *"Pomodoro de 25 + 5 min"* (artigo 10), que estão literalmente na base. O T1 deu 0,195, número descartado |
| **v3 (usada)** | **Uma afirmação por chamada**, base no início do prompt, ainda exigindo evidência conferida em Python | **Um pouco generoso** em afirmações vagas | Na q10, aprovou *"Exemplos: escrever, programar..."* citando um trecho sobre prática intercalada, sem relação |

A v3 acerta os casos que derrubaram as versões anteriores (reprova SMART e Eisenhower, aprova o Pomodoro e as "2 a 4 horas"), mas pode superestimar um pouco a fidelidade do T1 em afirmações genéricas. Por isso, o relatório traz junto uma **conferência manual por amostragem** (ver [Resultados](02-resultados.md)).

O arquivo de cada rodada guarda, para cada afirmação, o veredito original do juiz (`veredito_juiz`), a evidência copiada e o veredito final. Assim, qualquer número pode ser auditado.

> **Lição de método:** cada correção do juiz foi provocada por um caso concreto conferido no texto. Sem essa conferência, a fidelidade do LLM sem RAG teria saído como 0,53 (v1, no teste com 2 perguntas) ou 0,195 (v2, nas 18 perguntas): dois números errados, em direções opostas.

## Como reproduzir

A partir de `studybuddy/pesquisa/`, com o Ollama aberto e `LLM_PROVIDER=ollama` / `EMBEDDING_PROVIDER=ollama` no `dev/.env`:

```bash
# corte do porteiro (calibrado nas paráfrases)
../.venv/bin/python -m metricas.comparacao.modos calibrar

# gerar as respostas de cada teste
../.venv/bin/python -m metricas.comparacao.modos llm_puro        # T1 (~30 min)
../.venv/bin/python -m metricas.comparacao.modos so_busca        # T2 (segundos)
../.venv/bin/python -m metricas.comparacao.modos rag_llm         # T3 (reaproveita a rodada k=5)
../.venv/bin/python -m metricas.comparacao.modos rag_otimizado   # T4 (~10 min)
../.venv/bin/python -m metricas.comparacao.modos rag_llm_k3      # referência k=3

# avaliar tudo com o mesmo juiz e montar a tabela comparativa
../.venv/bin/python -m metricas.comparacao.avaliar llm_puro so_busca rag_llm rag_otimizado rag_llm_k3
```

| Arquivo (em `pesquisa/metricas/relatorios/`) | Conteúdo |
| :-- | :-- |
| `comparacao_porteiro.json` | $\tau$, as distâncias de cada pergunta e quais o porteiro barra |
| `comparacao_respostas_<teste>.json` | Pergunta, trechos, contexto exato, resposta, latência e $d_1$ |
| `comparacao_<teste>.json` | Veredito por pergunta: cada afirmação (relevância, base, evidência) e cada ponto-chave |
| `comparacao.md` / `.csv` | A tabela comparativa dos 5 conjuntos |

O código fica em `pesquisa/metricas/comparacao/` (`modos.py` gera, `avaliar.py` avalia), e as tarefas novas do juiz são `Juiz.relevancia` e `Juiz.sustentada_na_base`, em `pesquisa/metricas/geracao/juiz.py`.

Os resultados estão em [Resultados](02-resultados.md) e a escolha final em [Decisão](03-decisao.md).
