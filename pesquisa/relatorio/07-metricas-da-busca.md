# 7. Métricas da camada de busca: o que usamos e a matemática de cada uma

Este arquivo lista **todas as métricas implementadas na camada de recuperação** do StudyBuddy (`pesquisa/metricas/recuperacao/metrics.py` e `robustez.py`, no repositório) e explica, passo a passo, como cada uma é calculada.

## Visão geral

| Grupo | Métrica | Pergunta que responde | Faixa | Melhor |
| :-- | :-- | :-- | :-- | :-- |
| Acerto | **Hit@k** | O artigo certo apareceu? | 0 a 1 | ↑ |
| Ranking | **MRR@k** | O acerto vem em 1º lugar? | 0 a 1 | ↑ |
| Ranking | **nDCG@k** | Os artigos mais importantes vêm antes? | 0 a 1 | ↑ |
| Ruído | **Precision@k** | Quanto do contexto é útil? | 0 a 1 | ↑ |
| Cobertura | **Recall@k** | Vieram todos os artigos necessários? | 0 a 1 | ↑ |
| Cobertura | **Cobertura de pontos-chave@k** | O *conteúdo* da resposta está no contexto? | 0 a 1 | ↑ |
| Diversidade | **Distintas@k** | Quantos artigos diferentes vieram? | 1 a k | depende |
| Recusa | **Distância do 1º chunk** | Quão perto da base está a pergunta? | ≥ 0 | ↓ em escopo |
| Recusa | **Margem** | Existe folga entre válidas e fora de escopo? | qualquer | ↑ (> 0) |
| Recusa | **AUC-ROC da distância** | A distância separa válidas de fora de escopo? | 0 a 1 | ↑ |
| Recusa | **Recusa sem erro** | Quantas fora de escopo dá para recusar sem recusar nenhuma válida? | 0 a 1 | ↑ |
| Robustez | **Hit@k / MRR@k com ruído** | Erros de digitação e falta de acento quebram a busca? | 0 a 1 | ↑ |
| Robustez | **Hit@k / MRR@k em paráfrases** | A mesma pergunta, dita de outro jeito, ainda acerta? | 0 a 1 | ↑ |
| Robustez | **Jaccard entre paráfrases** | Paráfrases trazem os mesmos artigos? | 0 a 1 | ↑ |
| Custo | **Latência p50 / p95** | Quanto tempo leva cada busca? | ms | ↓ |
| Custo | **Tokens de contexto@k** | Quanto texto vai para o LLM? | tokens | ↓ (com qualidade igual) |
| Custo | **Tempo de indexação** | Quanto leva para montar o índice? | s | ↓ |
| Custo | **Tamanho dos vetores** | Quanto espaço os embeddings ocupam? | KB | ↓ |

---

## 0. Notação

Todas as fórmulas usam os mesmos símbolos:

| Símbolo | Significado |
| :-- | :-- |
| $q$ | uma pergunta do gabarito |
| $k$ | quantos chunks a busca devolve (Top-K) |
| $r_1, r_2, \dots, r_k$ | o **artigo de origem** de cada chunk devolvido, em ordem ($r_1$ é o mais próximo). Pode repetir: um artigo tem vários chunks |
| $E$ | conjunto de artigos **esperados** (principais) da pergunta |
| $C$ | conjunto de artigos **complementares** da pergunta |
| $\text{rel}(i)$ | 1 se $r_i \in E$, senão 0 (o chunk da posição $i$ é relevante?) |
| $\mathbb{1}[\cdot]$ | indicadora: vale 1 se a condição for verdadeira, 0 se for falsa |
| $Q$ | conjunto de perguntas **em escopo** (18 no gabarito atual) |
| $F$ | conjunto de perguntas **fora de escopo** (10 no gabarito atual) |

**Como as médias são feitas:** cada métrica é calculada **por pergunta** e depois tirada a média simples sobre as perguntas em escopo (*macro-média*). Todas as perguntas pesam igual:

$$
\text{Métrica} = \frac{1}{|Q|} \sum_{q \in Q} \text{métrica}(q)
$$

As perguntas fora de escopo ficam fora dessas médias, porque não existe "artigo certo" para elas. Elas só entram nas métricas de recusa (seção 4).

---

## 1. Acerto e ranking

### 1.1 Hit@k

$$
\text{Hit@}k(q) = \mathbb{1}\Big[\, \exists\, i \le k : r_i \in E \,\Big]
$$

Vale 1 se **pelo menos um** dos $k$ chunks vem de um artigo esperado, senão 0. A média sobre as perguntas é a **fração de perguntas em que a busca deu ao LLM alguma chance de acertar**.

**Exemplo:** $E = \{A\}$, devolvido $= (B, A, C)$. Hit@1 = 0 (o 1º é B) e Hit@3 = 1 (A está entre os 3).

**Limite:** não diz em que posição está o acerto, nem quanto ruído veio junto.

### 1.2 MRR@k (Mean Reciprocal Rank)

Primeiro, a posição do primeiro acerto:

$$
\text{pos}(q) = \min \{\, i \le k : r_i \in E \,\}
$$

O *reciprocal rank* é o inverso dessa posição (0 se não houver acerto):

$$
\text{RR@}k(q) = \begin{cases} \dfrac{1}{\text{pos}(q)} & \text{se houver acerto nos } k \\[2mm] 0 & \text{caso contrário} \end{cases}
\qquad
\text{MRR@}k = \frac{1}{|Q|}\sum_{q} \text{RR@}k(q)
$$

| Posição do 1º acerto | 1 | 2 | 3 | 4 | 5 | nenhum |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| RR | 1,00 | 0,50 | 0,33 | 0,25 | 0,20 | 0 |

**Por que $1/\text{pos}$:** a punição é forte logo no começo (cair de 1º para 2º custa metade) e fraca depois (de 4º para 5º custa só 0,05). Isso imita o comportamento do LLM, que dá muito mais peso ao primeiro trecho.

**Leitura útil:** $1/\text{MRR}$ é aproximadamente a posição média do primeiro acerto. MRR = 0,97 indica um acerto quase sempre em 1º lugar.

**Limite:** só olha o **primeiro** acerto. Não distingue "acertei e o resto é ruído" de "acertei e o resto também é útil".

### 1.3 nDCG@k (normalized Discounted Cumulative Gain)

Resolve os dois limites do MRR: considera **todos** os acertos e aceita **relevância graduada**.

**Passo 1, o ganho de cada artigo:**

$$
g(a) = \begin{cases} 2 & a \in E \text{ (principal)} \\ 1 & a \in C \text{ (complementar)} \\ 0 & \text{caso contrário} \end{cases}
$$

**Passo 2, o DCG:** soma os ganhos descontados pela posição. Cada artigo só conta **na primeira vez** que aparece, então chunks repetidos não somam de novo:

$$
\text{DCG@}k = \sum_{i=1}^{k} \frac{g(r_i)\cdot \mathbb{1}[r_i \text{ não apareceu antes}]}{\log_2(i+1)}
$$

O desconto $1/\log_2(i+1)$ vale:

| Posição $i$ | 1 | 2 | 3 | 4 | 5 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| $1/\log_2(i+1)$ | 1,000 | 0,631 | 0,500 | 0,431 | 0,387 |

O desconto é mais suave que o do MRR: um acerto em 3º lugar ainda vale metade.

**Passo 3, o DCG ideal (IDCG):** a melhor ordem possível, com os ganhos em ordem decrescente nas primeiras posições:

$$
\text{IDCG@}k = \sum_{j=1}^{\min(k,\,|E|+|C|)} \frac{g^*_j}{\log_2(j+1)}, \qquad g^*_1 \ge g^*_2 \ge \dots
$$

**Passo 4, a normalização:**

$$
\text{nDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k} \in [0, 1]
$$

**Exemplo real (q10 com o bge-m3, chunk 500):** $E = \{06\}$, $C = \{07\}$, devolvido $= (06, 06, 06)$.

- DCG $= \dfrac{2}{\log_2 2} + 0 + 0 = 2{,}000$ (o 06 só conta na posição 1)
- IDCG $= \dfrac{2}{\log_2 2} + \dfrac{1}{\log_2 3} = 2 + 0{,}631 = 2{,}631$
- nDCG $= 2/2{,}631 = \mathbf{0{,}760}$

O MRR dessa pergunta é 1,0, mas o nDCG é 0,76: os slots 2 e 3 repetiram o mesmo artigo quando poderiam ter trazido o complementar.

> Usamos **ganho linear** ($g$). Uma variante comum usa ganho exponencial ($2^g - 1$, ou seja, 3 e 1), que valoriza mais os principais. Com só dois níveis de relevância, a diferença é pequena.

---

## 2. Ruído e cobertura

### 2.1 Precision@k

$$
\text{Precision@}k(q) = \frac{1}{k}\sum_{i=1}^{k} \text{rel}(i)
$$

É a fração dos $k$ **slots** ocupada por chunks relevantes. Ela trabalha no nível de **chunk**: dois chunks do mesmo artigo certo contam duas vezes. O denominador é sempre $k$.

**Exemplo:** $E=\{A\}$, devolvido $=(A, A, B)$ dá $2/3 = 0{,}67$.

**Teto:** se a pergunta tem 1 artigo esperado e ele foi cortado em 2 chunks, com $k=3$ o máximo possível é $2/3$. Por isso a Precision nunca chega a 1 em muitas perguntas, mesmo com uma busca perfeita.

### 2.2 Recall@k

$$
\text{Recall@}k(q) = \frac{\big|\{r_1,\dots,r_k\} \cap E\big|}{|E|}
$$

É a fração dos **artigos** esperados que apareceram. Trabalha no nível de **artigo**: um artigo repetido conta uma vez só. Só faz diferença nas perguntas `multi_fonte` ($|E| \ge 2$); nas demais é igual ao Hit.

**Exemplo:** $E=\{A, C\}$, devolvido $=(B, C, B)$ dá $1/2 = 0{,}5$.

### 2.3 Cobertura de pontos-chave@k

As métricas anteriores olham só **de qual artigo** o chunk veio. Esta olha **o que o chunk diz**: os `pontos_chave` do gabarito aparecem no texto entregue ao LLM?

**Passo 1, normalizar um texto num conjunto de termos $T(\cdot)$:**
1. passa para minúsculas e tira os acentos ("consolidação" vira "consolidacao");
2. separa em palavras;
3. números ficam inteiros ("25", "7"), porque costumam ser a resposta;
4. palavras com 3 letras ou menos e palavras vazias ("para", "como", "mais"…) são descartadas;
5. as demais viram um **radical de 5 letras** ("consolidacao" e "consolida" viram "conso"), para tolerar flexões.

**Passo 2, decidir se cada ponto está coberto.** Seja $\text{ctx}$ o texto dos $k$ chunks juntos. O ponto $p$ está coberto se pelo menos uma fração $\theta$ dos seus termos aparece no contexto:

$$
\text{coberto}(p) = \mathbb{1}\left[\frac{|T(p) \cap T(\text{ctx})|}{|T(p)|} \ge \theta\right], \qquad \theta = 0{,}5
$$

**Passo 3, a cobertura da pergunta:**

$$
\text{Cobertura@}k(q) = \frac{1}{|P_q|}\sum_{p \in P_q} \text{coberto}(p)
$$

onde $P_q$ são os pontos-chave da pergunta.

**Exemplo:** ponto "Intervalos de 25 minutos com pausas de 5 minutos" vira $T = \{\text{inter}, 25, \text{minut}, \text{pausa}, 5\}$. Com o chunk "Use intervalos de 25 minutos e pausas curtas de 5 minutos", os 5 termos aparecem: $5/5 = 1 \ge 0{,}5$, ponto coberto.

**Calibração de $\theta$:** com o **artigo inteiro** como contexto (a melhor busca possível), a cobertura média foi de 0,98 com $\theta = 0{,}5$, de 0,91 com 0,6 e de 0,88 com 0,7. Escolhemos 0,5: o teto fica perto de 1, e as falhas restantes são pontos escritos com palavras diferentes das do artigo.

**Limite:** é uma medida **lexical**. Ela subestima quando o chunk diz a mesma coisa com outras palavras. Não usamos similaridade de embedding de propósito: o próprio modelo que está sendo avaliado acabaria julgando a si mesmo.

### 2.4 Distintas@k

$$
\text{Distintas@}k(q) = \big|\{r_1, \dots, r_k\}\big|
$$

É o número de artigos diferentes entre os $k$ chunks (de 1 a $k$). Não existe um valor "certo": numa pergunta de um assunto só, 1 pode ser ótimo; numa multi-fonte, precisa ser $\ge |E|$. Serve como diagnóstico de redundância.

---

## 3. A distância usada pela busca

Cada pergunta e cada chunk viram vetores $\mathbf{v}_q, \mathbf{v}_c \in \mathbb{R}^D$ (com $D = 1024$ no bge-m3 e no qwen3-embedding). O ChromaDB, configurado em `l2`, ordena pela **distância euclidiana ao quadrado**:

$$
d(q, c) = \|\mathbf{v}_q - \mathbf{v}_c\|_2^2 = \sum_{j=1}^{D} (v_{q,j} - v_{c,j})^2
$$

Os dois modelos devolvem vetores de **norma 1** (verificado: $\|\mathbf{v}\| = 1{,}0000$). Nesse caso, a distância e o cosseno são a mesma informação:

$$
d = \|\mathbf{v}_q\|^2 + \|\mathbf{v}_c\|^2 - 2\,\mathbf{v}_q\!\cdot\!\mathbf{v}_c = 2\,\big(1 - \cos\theta\big)
\quad\Longleftrightarrow\quad
\cos\theta = 1 - \frac{d}{2}
$$

Verificação feita no projeto: para "Como funciona a técnica Pomodoro?" contra um trecho do artigo 10, o bge-m3 deu $\cos = 0{,}6889$ e $2(1-0{,}6889) = 0{,}6221$, exatamente a distância que o ChromaDB devolveu.

| $d$ | 0 | 0,5 | 0,73 (média em escopo, bge-m3) | 1,0 | 2,0 | 4,0 |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| $\cos\theta$ | 1 (idênticos) | 0,75 | 0,64 | 0,5 | 0 (sem relação) | −1 (opostos) |

> O notebook chama essa grandeza de "Distância L2". Rigorosamente, ela é a L2 **ao quadrado**. Para ordenar os resultados dá no mesmo (elevar ao quadrado preserva a ordem), mas importa ao converter para cosseno.

> **Distâncias não se comparam entre modelos:** cada modelo distribui os vetores de um jeito no espaço. Para o qwen, 1,0 é uma pergunta válida "difícil"; para o bge-m3, 1,0 já é fora de escopo.

---

## 4. Recusa: separar "tenho resposta" de "não tenho"

Seja $d_1(q)$ a distância do **1º** chunk devolvido para a pergunta $q$. A ideia é usar $d_1$ como um classificador: se $d_1(q) > \tau$ (um corte), recusa sem chamar o LLM.

### 4.1 Margem

$$
\text{Margem} = \min_{f \in F} d_1(f) \;-\; \max_{q \in Q} d_1(q)
$$

É a fora de escopo **mais perigosa** (a mais perto da base) menos a pergunta válida **mais difícil** (a mais longe). Se for positiva, existe um corte que separa os dois grupos sem erro, e o valor diz **quanta folga** esse corte tem.

**Limite:** depende só de duas perguntas, os dois piores casos. Uma pergunta nova pode mudar o resultado.

### 4.2 AUC-ROC da distância

**A curva ROC:** para cada corte $\tau$ possível, calcula-se:
- **TPR** (taxa de verdadeiros positivos): a fração das perguntas válidas com $d_1 \le \tau$, ou seja, aceitas corretamente;
- **FPR** (taxa de falsos positivos): a fração das fora de escopo com $d_1 \le \tau$, ou seja, aceitas por engano.

Variando $\tau$ de 0 a ∞, os pontos (FPR, TPR) desenham uma curva de (0,0) a (1,1). A **AUC** é a área sob essa curva.

**O atalho usado no código (estatística de Mann-Whitney):** a AUC é igual à probabilidade de uma pergunta válida sorteada ficar **mais perto** da base que uma fora de escopo sorteada:

$$
\text{AUC} = \frac{1}{|Q|\cdot|F|}\sum_{q \in Q}\sum_{f \in F}
\Big( \mathbb{1}[d_1(q) < d_1(f)] + \tfrac{1}{2}\,\mathbb{1}[d_1(q) = d_1(f)] \Big)
$$

Com 18 válidas e 10 fora de escopo, são $18 \times 10 = 180$ pares comparados.

| AUC | Significado |
| :-- | :-- |
| 1,0 | Toda válida fica mais perto que toda fora de escopo: separação perfeita |
| 0,5 | Tanto faz: a distância não ajuda a separar |
| < 0,5 | Invertido: as fora de escopo ficam mais perto (algo está muito errado) |

**Exemplo:** válidas $\{0{,}5;\ 0{,}8\}$, fora $\{0{,}7;\ 0{,}9\}$. Os pares são (0,5 < 0,7) ✓, (0,5 < 0,9) ✓, (0,8 < 0,7) ✗ e (0,8 < 0,9) ✓, então AUC $= 3/4 = 0{,}75$.

**Limite importante:** a AUC só olha a **ordem**, não o **tamanho** da diferença. AUC = 1 com margem de 0,001 e AUC = 1 com margem de 0,5 dão o mesmo número, embora a segunda seja muito mais segura. Por isso olhamos a AUC **junto** com a margem.

### 4.3 Recusa sem erro

$$
\tau^* = \max_{q \in Q} d_1(q), \qquad
\text{Recusa sem erro} = \frac{1}{|F|}\sum_{f \in F} \mathbb{1}\big[d_1(f) > \tau^*\big]
$$

Põe o corte logo acima da pergunta válida mais distante (assim nenhuma válida é recusada) e conta quantas fora de escopo ficariam acima dele. Traduz a AUC num uso prático: **quantas perguntas fora de escopo eu barraria antes de gastar uma chamada ao LLM?**

**Exemplo:** válidas $\{0{,}5;\ 0{,}8\}$ dão $\tau^* = 0{,}8$. Fora $\{0{,}7;\ 0{,}9\}$: só 0,9 passa do corte, então recusa sem erro $= 1/2$.

---

## 5. Robustez

Repetimos a busca com versões **alteradas** das 18 perguntas em escopo e vemos o quanto as métricas caem.

### 5.1 Ruído (erros de digitação)

A versão com ruído de cada pergunta é gerada automaticamente:
1. minúsculas e sem acentos ("Técnica" vira "tecnica");
2. em cada palavra com mais de 4 letras, com probabilidade 0,3, **troca duas letras vizinhas** (nunca a 1ª letra): "estudos" vira "estuods".

Exemplo real: *"tarbalho 8 hoars por dia e sinto que nao consigo render nos estuods…"*

O gerador usa uma **semente fixa** por pergunta, então os mesmos erros se repetem em toda execução e as comparações entre modelos são justas. Calculamos Hit@3 e MRR@3 sobre as versões com ruído e a **queda**:

$$
\text{Queda} = \text{Hit@}3_{\text{original}} - \text{Hit@}3_{\text{ruído}}
$$

### 5.2 Paráfrases

Cada pergunta em escopo tem 2 `variacoes` escritas à mão no gabarito, na linguagem de um estudante. Ex.: q09 "Dá para dormir só 5 horas…?" tem as variações "Vale a pena virar a noite estudando antes da prova?" e "posso cortar horas de sono pra estudar mais?". São $18 \times 2 = 36$ consultas, e sobre elas calculamos Hit@3, MRR@3 e a queda em relação à original.

> Com 36 consultas, as métricas de paráfrase têm **o dobro de amostra** das originais. Elas tendem a ser uma estimativa mais realista do uso real.

### 5.3 Jaccard entre a original e as paráfrases

Para cada par (original, variação), compara os **conjuntos de artigos** do top-3:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

e tira a média dos 36 pares.

**Exemplo:** original traz $\{01, 16, 07\}$ e a variação traz $\{01, 02, 07\}$. Então $|A\cap B| = 2$ ($\{01, 07\}$), $|A \cup B| = 4$ e $J = 0{,}5$.

$J = 1$ significa que a busca é **estável** (o jeito de perguntar não muda o resultado). Um $J$ baixo com Hit alto quer dizer que o artigo certo continua aparecendo, mas o **resto do contexto** muda muito de uma formulação para outra.

---

## 6. Custo operacional

### 6.1 Latência p50 e p95

Para cada consulta, medimos o tempo de **gerar o embedding da pergunta + buscar no ChromaDB** (com `time.perf_counter`). Com os $n$ tempos ordenados $x_{(0)} \le \dots \le x_{(n-1)}$, o percentil $p$ por interpolação linear é:

$$
\text{pos} = \frac{(n-1)\,p}{100}, \qquad
P_p = x_{(\lfloor \text{pos} \rfloor)} + \Big(x_{(\lceil \text{pos} \rceil)} - x_{(\lfloor \text{pos} \rfloor)}\Big)\cdot\big(\text{pos} - \lfloor \text{pos} \rfloor\big)
$$

- **p50 (mediana):** metade das consultas é mais rápida que isso. É o tempo típico.
- **p95:** só 5% das consultas são mais lentas. É o pior caso comum.

**Exemplo:** tempos $(10, 20, 30, 40)$ e $p = 50$ dão $\text{pos} = 3 \times 0{,}5 = 1{,}5$, então $P_{50} = 20 + (30-20)\times 0{,}5 = 25$.

Usamos percentis e não a média porque a latência costuma ter "caudas": algumas consultas muito lentas puxariam a média para cima.

### 6.2 Tokens de contexto@k

$$
\text{Tokens@}k(q) = \sum_{i=1}^{k} \text{tokens}(\text{texto do chunk } i)
$$

São contados com o tokenizador `cl100k_base` (o mesmo do GPT-4o-mini). Os tokens são o que o LLM cobra e o que ocupa a janela de contexto, então esta métrica liga o **chunking** ao **custo da geração**.

Uma derivada útil é a **cobertura por 100 tokens**, que mostra o que se ganha de conteúdo por unidade de custo:

$$
\text{Eficiência} = \frac{\text{Cobertura@}k}{\text{Tokens@}k / 100}
$$

### 6.3 Tempo de indexação e tamanho dos vetores

- **Tempo de indexação:** segundos para carregar, cortar, gerar os embeddings e gravar no ChromaDB. Medido uma vez por configuração, então é ruidoso (o primeiro modelo carregado no Ollama paga o "aquecimento").
- **Tamanho dos vetores:** cada número é um `float32` de 4 bytes, então
  $$\text{KB} = \frac{N_{\text{chunks}} \times D \times 4}{1024}$$
  Ex.: 49 chunks × 1024 dimensões × 4 bytes ÷ 1024 = **196 KB**. É só a parte dos vetores; o ChromaDB guarda também o texto e o índice de busca.

---

## 7. Incerteza: quanto confiar em cada número

Com 18 perguntas em escopo, **uma pergunta vale $1/18 \approx 0{,}056$**. Qualquer diferença menor que isso é, literalmente, menos que uma pergunta.

Para métricas de 0 ou 1 por pergunta (como o Hit), o **erro padrão** de uma proporção $\hat p$ medida em $n$ perguntas é:

$$
\text{EP} = \sqrt{\frac{\hat p\,(1-\hat p)}{n}}
$$

| $\hat p$ | $n$ | EP | Intervalo aproximado de 95% ($\pm 2\,\text{EP}$) |
| :-- | :-- | :-- | :-- |
| 0,83 | 18 (originais) | 0,089 | 0,65 a 1,00 |
| 0,83 | 36 (paráfrases) | 0,063 | 0,71 a 0,96 |
| 0,94 | 18 | 0,056 | 0,83 a 1,00 |

Como ler: um Hit de 0,83 medido em 18 perguntas é compatível com um valor real entre ~0,65 e 1,00. Para afirmar que uma configuração é melhor que outra, a diferença precisa ser **grande** ou **se repetir** em várias métricas e configurações. Dobrar o gabarito reduz o erro padrão em cerca de 30% (fator $1/\sqrt{2}$).

> Quando $\hat p$ está perto de 0 ou 1 com $n$ pequeno, essa aproximação fica imprecisa (o intervalo "bate" no limite). Para algo mais rigoroso, use o intervalo de Wilson.
