# 5. Resultados: bge-m3 x qwen3-embedding

Comparativo de **modelos de embedding** e **tamanhos de chunk** na etapa de recuperação do RAG do StudyBuddy. A geração (LLM) ainda não foi avaliada.

## Método

| Item | Valor |
| :-- | :-- |
| Base | 20 artigos (`dev/data/`) |
| Gabarito | 28 perguntas: 4 diretas, 9 parafraseadas, 5 multi-fonte, 10 fora de escopo. Cada pergunta em escopo tem ainda 2 paráfrases (36 no total) e artigos complementares para o nDCG |
| Perguntas nas médias | 18 (as fora de escopo entram só nas métricas de recusa) |
| Banco vetorial | ChromaDB, distância L2 |
| Chunking | `RecursiveCharacterTextSplitter`, chunk de 300 / 500 / 800 com overlap de 10% |
| K avaliado | 1, 3 e 5 (as tabelas usam k=3, que é o Top-K do notebook) |
| Execução | Local, via Ollama, num Apple M4 com 24 GB |

Modelos comparados:

| Modelo | Tamanho | Dimensão do vetor | Observação |
| :-- | :-- | :-- | :-- |
| Offline (hash de palavras) | — | 256 | Piso de comparação: só "vê" palavras iguais |
| **bge-m3** | ~570M parâmetros, 1,2 GB | 1024 | Multilíngue, forte em busca |
| **qwen3-embedding:0.6b** | ~600M parâmetros, 640 MB | 1024 | Multilíngue. Testado **sem** o prefixo de instrução recomendado (ver Limitações) |

Para reproduzir (a partir de `pesquisa/`, com `EMBEDDING_PROVIDER=ollama` no `dev/.env`):

```
python -m metricas.recuperacao.comparar --modelos bge-m3 qwen3-embedding:0.6b --chunks 300 500 800
```

## Passo 1: o salto em relação ao baseline

Configuração do notebook (chunk 500, k=3):

| Modelo | Hit@1 | Hit@3 | MRR@3 | Recall@3 |
| :-- | :-- | :-- | :-- | :-- |
| Offline (hash) | 0,39 | 0,44 | 0,42 | 0,41 |
| bge-m3 | **0,94** | **1,00** | **0,97** | **0,93** |
| qwen3-embedding:0.6b | **0,94** | 0,94 | 0,94 | 0,84 |

**Leitura:** a maior melhoria do experimento vem da troca do embedding offline por um modelo semântico. O offline acertava 3 de 9 perguntas parafraseadas; os dois modelos neurais acertam todas ou quase todas.

## Passo 2: comparativo completo

| Modelo | Chunk | Nº de chunks | Hit@1 | Hit@3 | MRR@3 | Prec@3 | Recall@3 | Recall@3 multi_fonte | Recall@5 | Margem* |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| bge-m3 | 300 | 69 | 0,83 | 1,00 | 0,92 | 0,52 | 0,93 | 0,73 | 0,98 | +0,03 |
| **bge-m3** | **500** | **49** | **0,94** | **1,00** | **0,97** | 0,48 | **0,93** | **0,73** | **0,98** | +0,03 |
| bge-m3 | 800 | 22 | 0,83 | 1,00 | 0,91 | 0,39 | 0,93 | 0,73 | 0,93 | +0,03 |
| qwen3-emb | 300 | 69 | 0,89 | 0,94 | 0,91 | **0,56** | 0,87 | 0,73 | 0,94 | +0,11 |
| qwen3-emb | 500 | 49 | 0,94 | 0,94 | 0,94 | 0,48 | 0,84 | 0,63 | 0,95 | **+0,15** |
| qwen3-emb | 800 | 22 | 0,78 | 1,00 | 0,88 | 0,39 | 0,88 | 0,57 | 0,88 | −0,02 |

\*Margem = menor distância entre as perguntas fora de escopo − maior distância entre as perguntas em escopo (sempre do 1º chunk). Se for positiva, um corte de distância consegue recusar as perguntas fora de escopo sem recusar nenhuma válida.

## Passo 3: o que cada métrica mostrou

### Hit@3 chegou ao teto

Os dois modelos ficam entre 0,94 e 1,00. Isso confirma que ambos funcionam, mas **não ajuda a escolher entre eles**. Quando uma métrica satura, a decisão passa para as métricas seguintes.

### Hit@1 e MRR: quem põe o acerto em 1º lugar

O **bge-m3 com chunk 500** lidera (MRR 0,97): o artigo certo só não vem em 1º lugar numa única pergunta. Isso importa porque o LLM tende a dar mais peso ao primeiro trecho do contexto.

### Precision e tamanho do chunk: foco x contexto

O efeito é igual nos dois modelos:

- **Chunk 300 (69 chunks):** precisão maior (0,52 a 0,56). Vários pedaços do artigo certo ocupam o top 3. Em troca, cada pedaço carrega pouco contexto para o LLM.
- **Chunk 800 (22 chunks):** praticamente um chunk por artigo. O top 3 passa a trazer outros artigos, e a precisão cai para 0,39.
- **Chunk 500:** equilíbrio entre os dois extremos.

### Recall em multi_fonte: diluição do significado

É o indicador mais sensível ao chunking:

- **bge-m3:** estável em 0,73 nos três tamanhos de chunk.
- **qwen3-embedding:** cai de 0,73 (chunk 300) para 0,57 (chunk 800).

Um chunk grande mistura vários assuntos num vetor só. O vetor vira uma "média" que não fica perto de nenhum tema específico. O bge-m3 se mostrou **mais robusto** a esse efeito.

### Margem fora de escopo: saber dizer "não sei"

> Esta subseção usa a primeira versão do gabarito, com só 2 perguntas fora de escopo. Os números com 10 perguntas fora de escopo estão no Passo 5.

| Modelo (chunk 500) | Maior distância em escopo | q19 (ações) | q20 (café) | Margem |
| :-- | :-- | :-- | :-- | :-- |
| bge-m3 | 0,90 | 1,17 | 0,93 | +0,03 |
| qwen3-embedding | 1,00 | 1,57 | 1,15 | **+0,15** |

- No **bge-m3**, a q20 ("quanto café posso tomar para me concentrar?") fica quase colada à pergunta válida mais distante. Um corte de distância seria arriscado; a recusa fica com o LLM.
- O **qwen3-embedding** separa melhor o que está fora da base. É o único ponto em que ele supera o bge-m3 com folga.

> Os valores absolutos de distância **não** se comparam entre modelos. Cada um tem sua escala. Por isso o critério é a margem dentro de cada modelo.

## Passo 4: os erros, um por um

Perguntas em que o artigo certo não veio em 1º lugar no top 3:

| Configuração | Pergunta | Esperado | Top-3 devolvido |
| :-- | :-- | :-- | :-- |
| bge-m3 c300 | q04 edital extenso e validação prática | 16, 18 | 13, **16**, **18** |
| bge-m3 c300 / c500 / c800 | q05 releio e não lembro na prova | 01 | (16 / 16 / 07), **01**, … |
| bge-m3 c300 | q12 sair do "vou estudar amanhã" | 08 | 13, **08**, 14 |
| bge-m3 c800 | q01 rotina trabalhando 8h | 14, 19 | 17, **19**, 06 |
| bge-m3 c800 | q08 muitas abas e aplicativos abertos | 04 | 03, 17, **04** |
| qwen c300 / c800 | q04 edital extenso e validação prática | 16, 18 | (17, 02, **16**) / (02, **16**, 03) |
| qwen c300 | q05 releio e não lembro na prova | 01 | 02, 12, 20 ✗ |
| qwen c500 | q15 empurro o estudo quando a prova está longe | 11 | 02, 03, 17 ✗ |
| qwen c800 | q08 / q09 / q15 | 04 / 05 / 11 | acerto só em 2º ou 3º lugar |

Os números são os dos artigos (`artigo_01` … `artigo_20`). Em negrito, os acertos; ✗ = o artigo certo não apareceu no top 3.

Padrões:

1. **A q05 é a pergunta mais difícil para os dois modelos.** A resposta (active recall, artigo 01) não compartilha nenhuma palavra com "releio e não lembro". É o tipo de pergunta que mais separa um embedding bom de um ruim.
2. **O qwen tem um "artigo-ímã".** O `artigo_02` (repetição espaçada) aparece no top 3 de muitas perguntas sem relação com ele. Esse fenômeno se chama *hubness*: um documento genérico fica "perto de todo mundo" e rouba vagas do contexto.
3. **Com o bge-m3, os erros são leves:** o artigo certo quase sempre está em 2º lugar. Com o qwen, há casos em que ele some do top 3.

## Passo 5: métricas adicionais

Depois do primeiro comparativo, a camada de busca ganhou mais métricas: nDCG, cobertura de pontos-chave, AUC-ROC, robustez e custo. A matemática de cada uma está em [Métricas da busca](07-metricas-da-busca.md). O gabarito também cresceu: de 2 para **10 perguntas fora de escopo** (4 delas "perigosas", próximas do tema da base, como remédio para dormir e fórmula de Bhaskara) e **2 paráfrases** para cada pergunta em escopo.

Para reproduzir: `python -m metricas.recuperacao.comparar` (a partir de `pesquisa/`).

### 5.1 Qualidade do contexto

| Modelo | Chunk | MRR@3 | nDCG@3 | Cobertura@3 | Cobertura@5 | Tokens@3 | Cobertura por 100 tokens |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| bge-m3 | 300 | 0,92 | 0,84 | 0,52 | 0,70 | 183 | 0,28 |
| **bge-m3** | **500** | **0,97** | **0,84** | 0,79 | 0,89 | 274 | **0,29** |
| bge-m3 | 800 | 0,91 | 0,79 | **0,91** | **0,93** | 580 | 0,16 |
| qwen3-emb | 300 | 0,91 | 0,83 | 0,54 | 0,67 | 171 | 0,31 |
| qwen3-emb | 500 | 0,94 | 0,77 | 0,72 | 0,86 | 270 | 0,27 |
| qwen3-emb | 800 | 0,88 | 0,77 | 0,89 | 0,89 | 531 | 0,17 |

**O que mudou na leitura:**

1. **Artigo certo ≠ trecho certo.** Com chunk 300, o Hit@3 é 1,00 no bge-m3, mas a cobertura é só **0,52**: o artigo certo veio, mas em pedaços que deixam metade dos pontos-chave de fora. Essa lacuna era invisível para as métricas por artigo.
2. **Chunk maior cobre mais, mas custa mais.** O chunk 800 chega a 0,91 de cobertura, porém com o **dobro de tokens** (580 contra 274). Na **cobertura por 100 tokens**, o chunk 500 empata com o 300 (0,29 contra 0,28) e rende quase o dobro do 800 (0,16). **O chunk 500 é o ponto eficiente.**
3. **Multi-fonte continua sendo o ponto fraco:** a cobertura@3 do bge-m3 com chunk 500 é 1,00 nas diretas, 0,83 nas parafraseadas e só **0,53** nas multi-fonte. Com k=5 sobe para 0,89 no geral, mais um argumento para testar K=5 na geração.
4. **O nDCG confirma o bge-m3** e cobra a repetição. Exemplo: na q10, 3 chunks do mesmo artigo dão MRR 1,0, mas nDCG 0,76, porque o complementar ficou de fora.

### 5.2 Recusa (10 perguntas fora de escopo)

| Modelo | Chunk | Margem | AUC-ROC | Recusa sem erro |
| :-- | :-- | :-- | :-- | :-- |
| bge-m3 | 300 | +0,012 | 1,00 | 100% |
| bge-m3 | 500 | +0,014 | 1,00 | 100% |
| bge-m3 | 800 | +0,003 | 1,00 | 100% |
| qwen3-emb | 300 | +0,030 | 1,00 | 100% |
| qwen3-emb | 500 | **+0,071** | 1,00 | 100% |
| qwen3-emb | 800 | −0,090 | 0,97 | 60% |

**Leitura:**

- **AUC = 1,00 em quase tudo:** nas 180 comparações (18 válidas × 10 fora de escopo), toda pergunta válida ficou mais perto da base do que toda fora de escopo. Uma surpresa positiva, inclusive para as "perigosas".
- **Mas a margem do bge-m3 é minúscula** (+0,014). A pergunta fora de escopo mais perigosa foi a q26 ("emagrecer passando o dia sentado estudando", $d_1 = 0{,}916$), contra a válida mais distante, a q18 ($d_1 = 0{,}901$). Em cosseno, a diferença é de apenas $0{,}014/2 = 0{,}007$. Um corte nesse ponto quebraria com a próxima pergunta nova.
- Este é o exemplo clássico do **limite da AUC**: ela só olha a ordem, e a margem mostra que a separação do bge-m3 é frágil.
- **O qwen3-embedding com chunk 500 tem 5 vezes mais folga** (+0,071) e continua sendo o mais indicado se a ideia for recusar pela distância.
- Com chunk 800, o qwen perde a separação: AUC 0,97 e só 60% das fora de escopo recusáveis sem erro. É de novo a diluição: chunks grandes ficam "medianamente perto de tudo".

### 5.3 Robustez

| Modelo | Chunk | Hit@3 original | Hit@3 com ruído | Hit@3 paráfrases | MRR@3 original | MRR@3 paráfrases | Jaccard |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| bge-m3 | 300 | 1,00 | **1,00** | 0,86 | 0,92 | 0,73 | 0,46 |
| bge-m3 | 500 | 1,00 | **1,00** | 0,83 | 0,97 | 0,68 | 0,44 |
| bge-m3 | 800 | 1,00 | 0,94 | 0,83 | 0,91 | 0,71 | 0,41 |
| qwen3-emb | 300 | 0,94 | 0,89 | **0,89** | 0,91 | **0,79** | 0,47 |
| qwen3-emb | 500 | 0,94 | 0,89 | 0,83 | 0,94 | 0,76 | 0,41 |
| qwen3-emb | 800 | 1,00 | 0,89 | 0,78 | 0,88 | 0,62 | 0,50 |

**Leitura:**

- **Ruído:** o bge-m3 ignorou completamente os erros de digitação e a falta de acentos (Hit@3 continua 1,00 com chunk 300 e 500). O qwen perde ~1 pergunta.
- **Paráfrases são o teste mais duro:** os dois caem para ~0,83 de Hit@3. É uma estimativa mais realista do uso real do que o 1,00 das perguntas originais, porque estudantes não escrevem como o gabarito. E, com 36 consultas, a amostra é maior.
- **Aqui o qwen leva vantagem no MRR:** 0,76 a 0,79 contra 0,68 a 0,73 do bge-m3. O bge-m3 acerta o artigo, mas o coloca mais abaixo quando a pergunta muda de forma. **É a primeira métrica em que o qwen supera o bge-m3 na qualidade**, não só na recusa.
- **Jaccard ~0,44:** em média, a original e a paráfrase compartilham menos da metade dos artigos do top 3. O artigo certo costuma continuar lá, mas o resto do contexto muda bastante conforme o jeito de perguntar.

### 5.4 Custo

| Modelo | Chunk | Nº de chunks | Latência p50 | Latência p95 | Tokens@3 | Tokens@5 | Vetores |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| bge-m3 | 300 | 69 | 126 ms | 135 ms | 183 | 302 | 276 KB |
| bge-m3 | 500 | 49 | 126 ms | 139 ms | 274 | 452 | 196 KB |
| bge-m3 | 800 | 22 | 127 ms | 139 ms | 580 | 959 | 88 KB |
| qwen3-emb | 300 | 69 | **90 ms** | **101 ms** | 171 | 290 | 276 KB |
| qwen3-emb | 500 | 49 | **91 ms** | **101 ms** | 270 | 450 | 196 KB |
| qwen3-emb | 800 | 22 | **90 ms** | **100 ms** | 531 | 887 | 88 KB |

**Leitura:**

- **O qwen é ~30% mais rápido** (90 contra 126 ms), com tamanhos parecidos. A latência não muda com o chunk: quase todo o tempo é o embedding da pergunta, e a busca entre 22 e 69 vetores é instantânea.
- Os dois ficam bem abaixo do tempo de resposta de um LLM (segundos), então **a latência da busca não é o gargalo** do RAG.
- **Os tokens dependem do chunking, não do modelo:** com chunk 500 e k=3, ~270 tokens por pergunta. Passar para k=5 custa ~65% a mais de tokens.
- O tempo de indexação (1 a 3 s) e o tamanho dos vetores (88 a 276 KB) são irrelevantes neste tamanho de base. Com milhares de documentos, o tamanho cresce linearmente: $N \times 1024 \times 4$ bytes.

## Conclusão

| Pergunta | Resposta |
| :-- | :-- |
| Qual embedding usar? | **bge-m3**, ainda: melhor MRR e nDCG nas perguntas originais, recall estável e imune a ruído. **Ressalva:** nas paráfrases, o qwen ordena melhor (MRR 0,76 contra 0,68). A vantagem do bge-m3 é menor do que parecia no primeiro comparativo |
| Qual chunking usar? | **500 / overlap 50**: melhor MRR e a melhor cobertura por token. 300 deixa metade dos pontos-chave de fora; 800 cobre mais, mas com o dobro de tokens e diluindo os assuntos |
| Qual K usar? | **3** nas perguntas simples. Nas multi-fonte, **k=5** sobe a cobertura de 0,79 para 0,89, com ~65% mais tokens. Vale testar na geração |
| O qwen3-embedding tem utilidade? | Sim: **5 vezes mais margem** para recusar pela distância, **~30% menos latência** e melhor ordenação em paráfrases |
| A busca sozinha recusa perguntas fora da base? | No gabarito atual, sim (AUC 1,0), mas com o bge-m3 a margem é frágil demais para virar regra. A recusa precisa continuar garantida e medida no LLM |

## Limitações

- **Amostra pequena:** 18 perguntas válidas; cada uma vale ~0,056. As diferenças entre chunk 300 e 500 no bge-m3 são de 2 perguntas no Hit@1. Os padrões que se repetem em várias configurações (robustez do bge-m3, margem maior do qwen) são mais confiáveis que qualquer número isolado.
- **O qwen3-embedding foi testado em desvantagem:** ele foi treinado para receber as perguntas com um prefixo de instrução (`Instruct: ...\nQuery: ...`), e o teste não usou esse prefixo.
- **O gabarito é por artigo, não por chunk:** Hit, MRR, Precision, Recall e nDCG não verificam se o chunk trazido contém o trecho que responde. A cobertura de pontos-chave reduz essa lacuna, mas é lexical e subestima paráfrases.
- **As paráfrases, os complementares e as perguntas fora de escopo novas foram escritos por julgamento**, assim como as fontes esperadas.
- **As fontes esperadas foram definidas por julgamento humano** e podem ter vieses.
- **Só a recuperação foi avaliada:** nada aqui diz se o LLM usa bem o contexto.

## Próximos passos

1. Repetir o qwen3-embedding **com o prefixo de instrução**, para uma comparação justa.
2. Testar o **qwen3-embedding:4b** (maior), para ver se o tamanho compensa.
3. Testar **K=5** com o bge-m3 nas perguntas multi-fonte.
4. Construir a **camada de geração**: citação válida, faithfulness, recusa correta (q19 e q20) e completude, com um LLM-juiz diferente do gerador.
