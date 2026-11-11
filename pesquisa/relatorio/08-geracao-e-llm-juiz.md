# 8. Camada de geração e o LLM-juiz

A camada de busca responde se **os trechos certos chegaram ao LLM**. A camada de geração responde se **o LLM usou bem o que recebeu**: ele se manteve fiel aos trechos? cobriu o que a pergunta pedia? citou a fonte certa? recusou quando não havia resposta?

```
pergunta ──► [1] BUSCA ──► trechos ──► [2] GERAÇÃO ──► resposta ──► [3] AVALIAÇÃO
             bge-m3                    qwen3:8b                     regras (Python)
                                                                    + juiz (gemma3:12b)
```

## Configuração

| Papel | Modelo | Observação |
| :-- | :-- | :-- |
| Embedding (busca) | `bge-m3`, chunk 500, k=3 | Melhor configuração da camada de busca (ver [Resultados](05-resultados.md)) |
| **Gerador** | `qwen3:8b` (Ollama, local) | `temperature=0` e "modo pensamento" desligado (`reasoning=False`) |
| **Juiz** | `gemma3:12b` (Ollama, local) | Outra família de modelo e maior que o gerador |

O gerador usa **o mesmo prompt do notebook** (`RAG_SYSTEM_PROMPT`): responder só com o contexto, citar `[Fonte: ...]` e, se não houver informação, responder exatamente *"Não encontrei informações suficientes na base de conhecimento para responder a essa pergunta."*

## Métricas da geração

| Métrica | Pergunta que responde | Quem mede |
| :-- | :-- | :-- |
| **Recusa correta** | Nas perguntas fora de escopo, ele recusou? | Regra |
| **Recusa indevida** | Nas perguntas válidas, ele recusou sem necessidade? | Regra |
| **Tem citação** | A resposta citou alguma fonte? | Regra |
| **Precisão da citação** | As fontes citadas estavam mesmo no contexto recebido? | Regra |
| **Cita a esperada** | Citou pelo menos um artigo que o gabarito considera correto? | Regra |
| **Fidelidade** (*faithfulness*) | O que ele afirmou está apoiado no contexto, ou foi inventado? | **Juiz** |
| **Completude** | A resposta cobre os pontos-chave do gabarito? | **Juiz** |

### Fórmulas

**Recusa:** $\text{recusou}(r) = 1$ se a resposta $r$ contém a frase de recusa (ignorando acentos, maiúsculas e espaços).

$$
\text{Recusa correta} = \frac{1}{|F|}\sum_{q \in F}\text{recusou}(r_q)
\qquad
\text{Recusa indevida} = \frac{1}{|Q|}\sum_{q \in Q}\text{recusou}(r_q)
$$

**Citação:** seja $A_{\text{cit}}$ o conjunto de artigos citados em `[Fonte: ...]` e $A_{\text{rec}}$ o conjunto de artigos dos chunks entregues ao LLM.

$$
\text{Precisão da citação} = \frac{|A_{\text{cit}} \cap A_{\text{rec}}|}{|A_{\text{cit}}|}
$$

Se for menor que 1, o LLM citou um artigo que **não leu**: uma citação inventada.

**Fidelidade:** o juiz quebra a resposta em afirmações $a_1, \dots, a_n$ e marca cada uma como sustentada ($s_i = 1$) ou não ($s_i = 0$) pelo contexto.

$$
\text{Fidelidade}(r) = \frac{1}{n}\sum_{i=1}^{n} s_i
$$

**Completude:** para cada ponto-chave $p_j$ do gabarito, $c_j = 1$ se a resposta o cobre (com evidência, ver abaixo).

$$
\text{Completude}(r) = \frac{1}{|P|}\sum_{j=1}^{|P|} c_j
$$

Uma recusa numa pergunta válida tem completude 0, sem consultar o juiz.

### Fidelidade e completude medem coisas diferentes

| | Fidelidade alta | Fidelidade baixa |
| :-- | :-- | :-- |
| **Completude alta** | Ideal: respondeu tudo, sem inventar | Respondeu tudo, mas **inventou** parte (perigoso: parece ótima) |
| **Completude baixa** | Fiel, mas **incompleta**. Em geral a culpa é da **busca**, que não trouxe o trecho | Pior caso |

A completude é medida contra o **gabarito**, não contra o contexto. Se a busca não trouxe o trecho com um ponto-chave, um gerador perfeitamente fiel **não tem como** cobri-lo. Por isso, completude baixa com fidelidade alta aponta para a camada de busca, e não para o LLM.

---

## O LLM-juiz

### Para que serve

Algumas perguntas sobre a resposta se resolvem com regra: procurar a frase de recusa ou extrair `[Fonte: artigo_NN]` com uma expressão regular. Outras exigem **entender o significado**, e aí regra não basta.

**Exemplo real (q01).** O gerador escreveu:

> "Priorize a qualidade do tempo de estudo em vez da quantidade."

Isso está no contexto?

- Uma regra de **palavras em comum** diria "quase": "estudo", "tempo" e "quantidade" aparecem nos trechos.
- Mas o contexto **não diz isso**. É uma conclusão que o modelo tirou sozinho: uma alucinação leve, que soa razoável e por isso passa despercebida.

Só quem entende o texto percebe a diferença. Um avaliador humano faria isso bem, mas avaliar 28 respostas, com 5 a 10 afirmações cada, **a cada mudança de configuração** não é viável. O LLM-juiz é um **substituto escalável do avaliador humano**.

### Como ele trabalha

O juiz **não dá uma nota** do tipo "de 0 a 10". Notas abertas são subjetivas, variam entre execuções e não dizem *onde* está o problema. Em vez disso, ele faz **tarefas estreitas e verificáveis**, sempre em JSON com formato fixo:

| Tarefa | O que o juiz devolve | O que o Python faz com isso |
| :-- | :-- | :-- |
| Fidelidade | Lista de afirmações da resposta, cada uma com `sustentada: true/false` | Calcula sustentadas ÷ total |
| Completude | Para cada ponto-chave: o `trecho` da resposta que o cobre e `coberto: true/false` | Confere se o trecho existe na resposta e calcula cobertos ÷ total |

Escolhas de projeto:

1. **Saída estruturada:** o Ollama força o juiz a responder num esquema JSON definido no código. A resposta sempre pode ser lida e nunca vem "conversada".
2. **A conta fica com o Python:** o juiz só classifica itens; a média é calculada no código. Assim a nota é rastreável: dá para ver **qual** afirmação foi reprovada.
3. **`temperature=0`:** o mesmo veredito em toda execução, tanto quanto possível.
4. **Julgar só pelo contexto:** o prompt proíbe o juiz de usar conhecimento externo. Uma afirmação verdadeira no mundo real, mas ausente do contexto, conta como **não sustentada**, porque o objetivo do RAG é justamente responder *a partir da base*.

### Por que um modelo diferente do gerador

Modelos tendem a aprovar textos parecidos com os que eles mesmos escreveriam (**viés de autopreferência**, *self-preference bias*). Se o `qwen3:8b` julgasse as próprias respostas, tenderia a considerar fiéis as paráfrases que ele mesmo faz. É como um aluno corrigindo a própria prova.

Por isso:
- o juiz é de **outra família** (Gemma, do Google, contra Qwen, da Alibaba);
- o juiz é **maior** (12B contra 8B parâmetros), para ter mais capacidade de leitura que o modelo avaliado;
- o código **se recusa a rodar** se juiz e gerador forem o mesmo modelo.

### O juiz também erra: calibração

O juiz é um **instrumento de medida** e, como qualquer instrumento, precisa ser conferido antes de se confiar nos números.

**O erro encontrado.** No primeiro teste, com 3 perguntas, o juiz de completude disse que a resposta da q01 cobria *"micro-blocos matinais de 45 a 60 minutos"* e *"estudar antes do expediente"*. A resposta **não falava de nenhum dos dois**: o tema (rotina de quem trabalha) era parecido, e o juiz deu "sim" por semelhança.

**A correção, exigir evidência:**
1. O esquema JSON passou a pedir o campo `trecho` **antes** do veredito `coberto`. O juiz precisa *encontrar* a frase na resposta antes de decidir. Como o modelo gera o texto em ordem, a evidência vem antes da conclusão.
2. O Python **confere** se esse trecho existe mesmo na resposta (pelo menos 80% dos termos em comum, tolerando acento e flexão). Se o juiz "citar" algo que a resposta não diz, o ponto conta como **não coberto**, seja qual for o veredito.

**Resultado:** a completude da q01 caiu de 0,67 para **0,00**, o valor correto. O chunk recuperado tinha sido cortado antes do trecho com esses pontos, então a resposta não tinha como cobri-los.

O juiz de **fidelidade** acertou já no primeiro teste: as 3 afirmações que ele reprovou na q01 eram de fato acréscimos do gerador. Exemplos: *"Opte por um modelo mais flexível"* e *"Priorize a qualidade em vez da quantidade"*.

### Limitações

- **O juiz é um modelo de 12B rodando localmente.** Ele é mais fraco que juízes comerciais (GPT-4o, Gemini) e pode errar em afirmações sutis. Os números servem para comparar configurações e achar padrões, não como verdade absoluta.
- **A divisão em afirmações varia:** o mesmo texto pode virar 6 ou 8 afirmações, conforme o juiz o quebre. Isso muda um pouco a fidelidade sem que a resposta mude.
- **Rigor variável:** o juiz às vezes reprova paráfrases legítimas (ex.: "opte por um modelo flexível" é uma consequência razoável do contexto) e às vezes aprova extrapolações. Em média, os erros para os dois lados tendem a se compensar, mas uma pergunta isolada pode estar errada.
- **Calibração recomendada:** de tempos em tempos, um humano revisa uma amostra dos vereditos (ex.: 20 afirmações) e conta quantas vezes concorda com o juiz. Uma concordância abaixo de ~80% indica que o prompt ou o modelo do juiz precisa mudar.

## Resultados (primeira rodada completa)

Configuração: busca com `bge-m3`, chunk 500 e k=3; gerador `qwen3:8b`; juiz `gemma3:12b`. São 28 perguntas: 18 em escopo e 10 fora de escopo.

### Visão geral

| Métrica | Resultado | Leitura |
| :-- | :-- | :-- |
| **Recusa correta** (10 fora de escopo) | **100%** | Recusou todas, inclusive as "perigosas" (remédio para dormir, Bhaskara, código Python), sempre com a frase exata do prompt |
| **Recusa indevida** (18 em escopo) | **0%** | Nunca recusou uma pergunta que a base responde |
| **Tem citação** | **100%** | Toda resposta citou `[Fonte: ...]` |
| **Precisão da citação** | **100%** | Nenhuma citação inventada: só citou artigos que recebeu |
| **Cita a esperada** | **100%** | Sempre citou pelo menos um artigo correto do gabarito |
| **Fidelidade** | **0,93** | 93% das afirmações estão apoiadas no contexto |
| **Completude** | **0,81** | 81% dos pontos-chave aparecem na resposta |

### Por tipo de pergunta

| Tipo | n | Fidelidade | Completude |
| :-- | :-- | :-- | :-- |
| direta | 4 | 1,00 | 1,00 |
| parafraseada | 9 | 0,95 | 0,89 |
| multi_fonte | 5 | **0,84** | **0,53** |

### Custo

| Latência p50 | Latência p95 | Tokens por resposta |
| :-- | :-- | :-- |
| 15,8 s | 31,1 s | 192 |

As recusas levam ~4 s; as respostas completas, de 10 a 35 s. A geração é **mais de 100 vezes mais lenta** que a busca (~0,13 s): o gargalo do RAG é o LLM.

### Calibração do juiz nesta rodada

Antes de confiar nos números, uma amostra dos vereditos foi revisada à mão:

- **Fidelidade:** foram revisadas 25 afirmações (q07, q09, q12 e q17). A concordância com o juiz foi de **23 em 25 (92%)**. As 2 discordâncias foram **rigor excessivo**:
  - q09, *"Não é recomendável dormir só 5 horas"*: uma conclusão direta do contexto ("menos de 7 horas deteriora o foco"), que o juiz reprovou.
  - q07, *"A prática intercalada é mais útil para o desempenho em testes"*: o contexto diz que ela "mimetiza a realidade das provas", e o juiz reprovou.

  O juiz tende a ser **mais severo** que um humano, então a fidelidade real provavelmente é um pouco maior que 0,93.
- **Completude:** os pontos não cobertos de q01, q08 e q15 foram conferidos no texto. **Todos os vereditos estavam corretos**: o conteúdo de fato não estava na resposta.

### A descoberta principal: a incompletude vem da busca

Para cada ponto-chave **não coberto**, verificamos se ele estava no **contexto** entregue ao gerador:

| Pergunta | Ponto-chave que faltou | Estava no contexto? | Culpa de |
| :-- | :-- | :-- | :-- |
| q01 | Micro-blocos matinais de 45 a 60 min | Não | **Busca** |
| q01 | Estudar antes do expediente | Não | **Busca** |
| q01 | Fins de semana para aprofundamento | Não | **Busca** |
| q02 | Micro-blocos matinais de 45 a 60 min | Não | **Busca** |
| q02 | Evitar decisões à noite (força de vontade esgotada) | Não | **Busca** |
| q03 | Ciclo de estudos em sequência circular | Não | **Busca** |
| q08 | Memória de trabalho de 4 a 7 elementos | Não | **Busca** |
| q15 | Decompor metas em marcos semanais | Não | **Busca** |
| q16 | Eliminar a fricção do ambiente | **Sim** | **Gerador** |

**8 dos 9 pontos perdidos nunca chegaram ao gerador.** O LLM não tinha como cobri-los. Um exemplo é a q01: o chunk do artigo 19 foi cortado logo **antes** da frase com os "micro-blocos matinais de 45 a 60 minutos" e os "fins de semana".

Isso confirma, agora na resposta final, o que a camada de busca já indicava: as perguntas **multi-fonte** são o ponto fraco (completude 0,53), e a solução está na busca (**k=5** ou outro chunking), não no LLM.

> Uma lição de método: na primeira tentativa, a atribuição automática (cobertura lexical do contexto) culpou o gerador nas q01 e q02. Ao conferir à mão, o "45" encontrado no contexto era o **"Chunk ID: 45"**, não "45 minutos". Métricas automáticas precisam de revisão por amostragem, inclusive as nossas.

### Um ponto cego da fidelidade

Na **q15** ("por que eu empurro o estudo quando a prova está longe? como resolver?"), a busca não trouxe o trecho com a solução do artigo correto ("marcos semanais com feedback rápido"). O gerador então respondeu *"como resolver"* com **repetição espaçada e prática intercalada**, que vêm de outros artigos que também estavam no contexto.

- **Fidelidade = 1,00:** tudo o que ele disse estava no contexto.
- **Mas a resposta à pergunta é fraca:** são técnicas de estudo, não de combate à procrastinação.

Algo parecido aconteceu na q17 (confiança para terminar os estudos): o gerador acrescentou um parágrafo sobre "reduzir a carga extrínseca de telas poluídas", fiel ao contexto, mas fora do assunto.

**Fidelidade mede se o LLM inventou, não se ele respondeu o que foi perguntado.** Uma métrica de **relevância da resposta** (o LLM-juiz avalia se cada parte responde à pergunta) cobriria esse ponto cego e é uma boa candidata para a próxima versão.

## Comparação k=3 × k=5

A primeira rodada indicou que a completude das perguntas multi-fonte (0,53) esbarrava na busca: os pontos que faltavam nunca chegavam ao gerador. A hipótese testada: **entregar 5 trechos em vez de 3 resolve isso?** Tudo o mais foi mantido igual (bge-m3, chunk 500, qwen3:8b gerando, gemma3:12b julgando, as mesmas 28 perguntas). Só mudou `TOP_K`.

### Visão geral

| Métrica | k=3 | k=5 | Variação |
| :-- | :-- | :-- | :-- |
| Recusa correta (10 fora de escopo) | 100% | 100% | = |
| Recusa indevida (18 em escopo) | 0% | 0% | = |
| Tem citação / precisão / cita esperada | 100% / 100% / 100% | 100% / 100% / 100% | = |
| **Fidelidade** | 0,93 | **0,97** | **+0,04** |
| **Completude** | 0,81 | **0,91** | **+0,10** |

### Por tipo de pergunta

| Tipo | Fidelidade k=3 | Fidelidade k=5 | Completude k=3 | Completude k=5 |
| :-- | :-- | :-- | :-- | :-- |
| direta (4) | 1,00 | 1,00 | 1,00 | 1,00 |
| parafraseada (9) | 0,95 | 0,95 | 0,89 | 0,89 |
| **multi_fonte (5)** | 0,84 | **0,97** | **0,53** | **0,87** |

**A hipótese se confirmou:** a completude das multi-fonte subiu de 0,53 para **0,87**. As diretas não mudaram, e as parafraseadas ficaram no mesmo valor médio, mas com trocas internas (ver abaixo).

### O que mudou, pergunta a pergunta

| Pergunta | Tipo | Completude | Fidelidade | O que aconteceu |
| :-- | :-- | :-- | :-- | :-- |
| q01 | multi_fonte | 0,00 → **1,00** | 0,62 → **1,00** | O artigo 14 (força de vontade) e o trecho completo do 19 entraram no contexto |
| q02 | multi_fonte | 0,33 → **0,67** | 0,60 → **1,00** | Melhorou, mas o artigo 14 ainda ficou de fora |
| q16 | multi_fonte | 0,67 → **1,00** | 1,00 → 1,00 | "Eliminar a fricção do ambiente" passou a aparecer na resposta |
| q15 | parafraseada | 0,50 → **1,00** | 1,00 → 0,91 | Entrou o trecho com "marcos semanais": a resposta agora ataca a procrastinação de verdade |
| q12 | parafraseada | 1,00 → **0,50** | 1,00 → 1,00 | **Regressão:** a resposta perdeu "defina quando, onde e como" (ver abaixo) |
| q03 | multi_fonte | 0,67 → 0,67 | 1,00 → 0,83 | Trocou um ponto coberto por outro (ganhou o "ciclo circular", perdeu o do perfeccionismo) |

### Por que a fidelidade também subiu

Na q01 com k=3, o juiz reprovou afirmações como *"Priorize a qualidade em vez da quantidade"*: o gerador **preenchia lacunas** com conselhos próprios porque o contexto não trazia o que a pergunta pedia. Com k=5, o contexto passou a ter a resposta, e o gerador **parou de improvisar** (fidelidade 0,62 → 1,00).

**Contexto insuficiente não só deixa a resposta incompleta: ele empurra o modelo a inventar.**

### Onde estão os pontos perdidos agora

| | k=3 | k=5 |
| :-- | :-- | :-- |
| Pontos-chave não cobertos | 9 | **4** |
| … porque não vieram no contexto (**busca**) | 8 | **2** (q02, q08) |
| … porque estavam no contexto e o gerador omitiu (**gerador**) | 1 | **2** (q03, q12) |

Cada caso foi conferido à mão no texto do contexto e da resposta.

Com k=5, **o gargalo começou a migrar da busca para o gerador**. Os 2 pontos que ainda são falha da busca:
- **q02:** o artigo 14 (força de vontade) não aparece nem entre os 5 primeiros;
- **q08:** o artigo 04 vem, mas não o chunk com "4 a 7 elementos".

### O efeito colateral: diluição

A **q12** ("como sair do 'vou estudar amanhã'?") piorou:

- **k=3:** contexto com os artigos 08, 13 e 14. A resposta trouxe a instrução central do artigo 08: *"Defina com precisão quando, onde e como a ação será iniciada"*.
- **k=5:** entrou também um chunk do artigo 17 (ciclo de estudos). A resposta **trocou** a instrução central por um item novo, *"Utilize ciclos de estudo eficientes"*, menos relevante para a pergunta.

O trecho com "Quando, Onde e Como" **continuava no contexto**. O gerador simplesmente deixou de usá-lo. Com mais trechos para combinar, o modelo distribui a atenção e pode deixar de lado justamente o ponto mais importante. É um efeito conhecido em RAG: **mais contexto aumenta a cobertura, mas também a chance de diluir o essencial**.

A q03 mostra o mesmo efeito em menor escala: com k=5 ela passou a cobrir o "ciclo de estudos circular", mas deixou de fora "planejamento excessivo é adiamento disfarçado (perfeccionismo)", que estava no contexto.

Nesta rodada, o saldo foi claramente positivo (**7 pontos-chave ganhos, 2 perdidos**), mas é um alerta para não aumentar o K sem medir.

### Custo

| | k=3 | k=5 | Variação |
| :-- | :-- | :-- | :-- |
| Tokens de contexto por pergunta (média, em escopo) | 372 | 614 | **+65%** |
| Tokens da resposta (média, em escopo) | 288 | 315 | +9% |
| Latência mediana das respostas | 19,7 s | 20,8 s | +6% |
| Latência mediana das recusas | 3,8 s | 5,8 s | +53% |

- O custo **em tokens** cresce bastante (+65% de contexto). Com um LLM pago por token, isso pesa diretamente na conta.
- A **latência** quase não muda nas respostas: o tempo é dominado pela escrita da resposta, não pela leitura do contexto.
- As **recusas** ficaram mais lentas (3,8 → 5,8 s) porque, mesmo para recusar, o modelo precisa ler um contexto maior.

### Ruído de medida

Algumas variações pequenas não vêm da mudança de K, e sim da própria medida:
- **q09:** a fidelidade foi de 0,80 para 0,67 com uma resposta praticamente igual; o juiz reprovou uma conclusão a mais.
- A divisão das respostas em afirmações também varia entre rodadas.

Variações de ±0,1 a 0,2 numa pergunta isolada podem ser só o juiz. **As conclusões acima se baseiam nas mudanças grandes e explicáveis** (q01, q02, q15, q16, q12), que foram conferidas no texto.

### Veredito

**k=5 é melhor para este RAG.** A completude foi de 0,81 para 0,91, a das multi-fonte de 0,53 para 0,87, e a fidelidade subiu porque o modelo deixou de improvisar. Recusa e citações continuam perfeitas. O custo é de +65% de tokens de contexto e dois casos de diluição (q12 e q03).

Recomendações:
1. **Adotar k=5** no notebook, ou usar k dinâmico: 5 para perguntas mais longas ou compostas, 3 para as simples.
2. **Para os 2 pontos que ainda dependem da busca** (q02 e q08), testar um chunking que não separe os parágrafos-chave, em vez de aumentar ainda mais o K.
3. **Para a diluição**, reforçar no prompt que a resposta deve priorizar as instruções mais diretamente ligadas à pergunta, e medir de novo.

### Conclusões da primeira rodada (k=3)

1. **O gerador é disciplinado:** recusa quando deve, nunca recusa sem motivo, sempre cita e nunca inventa fontes.
2. **Quase não alucina:** 93% de fidelidade, com um juiz que tende a ser severo. O que ele acrescenta são conclusões e conselhos genéricos ("priorize a qualidade"), não fatos falsos.
3. **O limite da qualidade está na busca:** 8 de 9 pontos perdidos não estavam no contexto. Melhorar o RAG agora passa por **k=5 nas perguntas multi-fonte** ou por um chunking que não corte os parágrafos-chave.
4. **Próximos passos:** testar k=5 na geração (feito: ver a comparação acima), adicionar a métrica de relevância da resposta e comparar com outro gerador (ex.: um modelo via API).

## Como rodar

A partir de `studybuddy/pesquisa/`, com `LLM_PROVIDER=ollama` e `EMBEDDING_PROVIDER=ollama` no `dev/.env`:

```bash
# gera as respostas e avalia (~20 min para as 28 perguntas)
../.venv/bin/python -m metricas.geracao.runner --nome qwen3_bge

# mesmo teste com 5 trechos por pergunta em vez de 3
TOP_K=5 ../.venv/bin/python -m metricas.geracao.runner --nome qwen3_bge_k5

# teste rápido com as 3 primeiras perguntas
../.venv/bin/python -m metricas.geracao.runner --nome teste --limite 3

# reavaliar respostas já geradas (ex.: depois de mudar o juiz), sem gerar de novo
../.venv/bin/python -m metricas.geracao.runner --nome qwen3_bge --so-avaliar

# usar outro juiz
OLLAMA_JUIZ_MODEL=outro-modelo ../.venv/bin/python -m metricas.geracao.runner --nome qwen3_bge --so-avaliar
```

As respostas geradas ficam em `metricas/relatorios/geracao_respostas_<nome>.json`, com a pergunta, os trechos recuperados, o contexto exato enviado ao LLM e a resposta. A avaliação fica em `geracao_<nome>.json` e `.csv`, com o veredito do juiz para cada afirmação e cada ponto-chave.
