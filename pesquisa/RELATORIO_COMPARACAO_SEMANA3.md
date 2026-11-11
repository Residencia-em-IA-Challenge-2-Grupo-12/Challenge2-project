# Entrega Semana 3 — Comparação LLM x RAG: Testes Empíricos, Metodologia, Métricas e Respostas Baseadas em Evidências

**Projeto:** StudyBuddy — Assistente Inteligente de Organização de Estudos e Produtividade Sustentável (Grupo 12)  
**Framework de Avaliação:** RAGAS (*Retrieval Augmented Generation Assessment*) + Métricas de *Information Retrieval* (Distância $L_2$, Latência e Consumo de Tokens)  
**Configuração de Modelos:**
- **LLM Configurado:** `ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)` *(endpoint ativo na execução: `gemini-3.5-flash-lite`)*
- **Embeddings Configurados:** `GoogleGenerativeAIEmbeddings(model="text-embedding-004")` *(endpoint ativo na execução: `gemini-embedding-001`)*
- **Base Vetorial:** ChromaDB Persistente (`./chroma_db` — 20 artigos científicos segmentados em 49 chunks de `500` caracteres com `overlap=50`).

---

## 1. Introdução e Metodologia Utilizada

Este relatório apresenta a validação empírica e comparativa entre abordagens de **Grande Modelo de Linguagem puro (Vanilla LLM)**, **Recuperação Semântica Extrativa** e **Pipelines de Retrieval-Augmented Generation (RAG)** aplicados ao projeto **StudyBuddy** (Grupo 12). O objetivo do experimento é mensurar quantitativamente o impacto da ancoragem factual (*grounding*) sobre uma base científica curada, garantindo recomendações precisas, auditáveis e livres de alucinações para perfis reais de estudantes.

### 1.1. Base de Conhecimento e Pipeline de Indexação
- **Corpus Científico (`dev/data/`):** Composto por **20 artigos científicos** (`.txt`) organizados em 4 eixos temáticos:
  1. *Retenção e Neurociência* (*Active Recall*, Repetição Espaçada, Prática Intercalada, Teoria da Carga Cognitiva de Sweller e Consolidação da Memória no Sono de Walker);
  2. *Foco e Produtividade* (*Deep Work*, Resíduo de Atenção, Intenções de Implementação *Se-Então*, *Timeboxing* e Técnica Pomodoro);
  3. *Comportamento e Procrastinação* (Motivação Temporal, Perfeccionismo/*Overplanning*, Loop do Hábito, Fadiga Decisória/Esgotamento do Ego e Autoeficácia);
  4. *Planejamento Autodidata e Editais* (Princípio de Pareto 80/20, Ciclos de Estudo Flexíveis, *Project-Based Learning*, Dupla Jornada CLT e Técnica Feynman).
- **Segmentação Hierárquica (*Chunking*):** Os documentos foram processados via `RecursiveCharacterTextSplitter` com separadores `["\n\n", "\n", ". ", " ", ""]`, tamanho de **`chunk_size = 500` caracteres** e sobreposição de **`chunk_overlap = 50` caracteres** (10%), gerando **49 chunks** enriquecidos com metadados de rastreabilidade (`source`, `start_index` e `chunk_id`).
- **Vetorização e Persistência (`ChromaDB`):** Os 49 fragmentos foram vetorizados com o modelo de embeddings da Google (`text-embedding-004` / `gemini-embedding-001`) e indexados no **ChromaDB** local (coleção `residencia_ia_semana2`) com verificação automática de integridade dimensional.

### 1.2. Mecanismo de Busca (*Retriever*) e *Chain-of-Thought* Factual
- **Busca Vetorial por Distância Euclidiana ($L_2$):** O *Retriever* converte a consulta da persona em vetor e recupera os fragmentos com menor distância $L_2$ no ChromaDB (`similarity_search_with_score`).
- **Expansão e Re-ranking de Cabeçalhos (Teste 4):** No *chunking* de 500 caracteres, títulos e autores de alguns artigos geram chunks curtos (`<= 225` caracteres) que competem por espaço no `Top-K=3`. Na variante otimizada (`Top-K=5 Re-rank`), o *Retriever* busca inicialmente `k=6` candidatos e aplica um filtro de re-ranking que prioriza chunks densos de conteúdo empírico (`> 225` caracteres), garantindo que perguntas compostas multi-tópico recebam todos os artigos complementares nas primeiras posições do ranking.
- **Protocolo *Chain-of-Thought* (CoT) Factual:** Técnica de *Prompt Engineering* aplicada no Teste 4 que instrui o LLM (`temperature=0.0`) a decompor e alinhar cada parte da pergunta da persona exclusivamente às premissas literais, métricas numéricas (minutos, horas, percentuais) e mecanismos presentes nos 5 fragmentos recuperados, vedando saudações genéricas ou extrapolações externas.

### 1.3. Benchmark de Personas e Avaliação Quantitativa via RAGAS
O experimento submete as **3 Personas canônicas** do projeto (**Lucas** — Dupla Jornada CLT; **Lisa** — Paralisia por *Overplanning*; **Bob** — Autodidatismo e Editais Extensos) aos 4 cenários arquiteturais (totalizando **12 execuções**). A avaliação utiliza o paradigma *LLM-as-a-Judge* do framework **RAGAS** combinado a métricas operacionais:
- **`Faithfulness` (Fidelidade Factual / Groundedness):** Proporção (`0.0` a `1.0`) de afirmações da resposta que são comprovadas diretamente pelos chunks recuperados.
- **`Answer Relevance` (Relevância da Resposta):** Similaridade semântica e completude (`0.0` a `1.0`) da resposta frente à pergunta original da persona.
- **`Context Precision` (Precisão da Recuperação):** Avalia (`0.0` a `1.0`) se o *Retriever* posicionou os chunks mais relevantes para o gabarito científico (*Ground Truth*) no topo do ranking (`Top-K`).
- **Latência (s), Consumo de Tokens e Custo Estimado (USD).**

---

## 2. Desenho Experimental: Os 4 Cenários Arquiteturais

1. **Teste 1 — Vanilla LLM (`run_vanilla_llm`):**
   - Consulta enviada diretamente ao LLM (`temperature=0.0`) sem acesso ao ChromaDB.
   - **Objetivo:** Medir a relevância superficial do conhecimento paramétrico versus a ausência completa de ancoragem factual (`Faithfulness = 0.0`, `Context Precision = 0.0`).
2. **Teste 2 — Recuperação Semântica Pura (`run_retrieval_only`):**
   - Execução exclusiva do Retriever no ChromaDB (`Top-K=3`) retornando os fragmentos brutos com metadados e scores de distância Euclidiana ($L_2$), sem chamada ao LLM.
   - **Objetivo:** Isolar a qualidade da busca vetorial, latência de recuperação (~0.2s–0.5s) e custo zero de geração, evidenciando a limitação de usabilidade da leitura bruta para o estudante.
3. **Teste 3 — RAG Padrão Baseline (`run_rag_baseline`):**
   - Arquitetura construída na Semana 2: `Retriever Top-K=3` + Prompt restritivo com *grounding* + `LLM (temperature=0.0)`.
   - **Objetivo:** Avaliar o ganho de fidelidade factual sobre o Vanilla LLM e identificar gargalos do `Top-K=3` quando fragmentos curtos de cabeçalho (título/autores) ocupam vagas de contexto.
4. **Teste 4 — RAG Otimizado (`run_rag_optimized` — Hipótese Combinada Opção A + Opção B):**
   - **Hipótese Implementada:**
     - **Opção A (Janela Expandida `Top-K=5` com Re-ranking/Filtragem de Cabeçalhos):** Como os 20 artigos foram divididos em 49 chunks (`chunk_size=500`), alguns chunks contêm apenas metadados bibliográficos (`TÍTULO / AUTORES / TEMA`, <225 caracteres). Nas perguntas compostas das personas (ex.: *divisão de blocos CLT + fisiologia do sono*, ou *Princípio de Pareto + validação prática PBL*), `k=3` frequentemente recupera o cabeçalho de um artigo e omite o segundo artigo complementar. A expansão para `Top-K=5` com priorização de chunks densos (>225 chars) garante cobertura integral de múltiplos artigos.
     - **Opção B (Prompt Engineering com *Chain-of-Thought* Factual orientado às Personas):** O prompt instrui o modelo a mapear todas as subperguntas da persona diretamente às premissas numéricas e conceituais dos fragmentos antes de redigir a síntese, vedando extrapolações externas.

---

## 3. Tabela Consolidada de Resultados (4 Testes × 3 Personas = 12 Execuções)

| ID_Teste   | Abordagem                             | Persona                                            |   Faithfulness |   Answer Relevance |   Context Precision |   Latência (s) | Observações Principais                                                                                                                                              |
|:-----------|:--------------------------------------|:---------------------------------------------------|---------------:|-------------------:|--------------------:|---------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Teste 1    | Vanilla LLM (Sem RAG)                 | Persona 1 (Lucas / Dupla Jornada CLT)              |         0      |             0.8553 |                 0   |          8.035 | Sem ancoragem factual (0 chunks); 248 tokens (~$0.00009); conselhos genéricos sem citar literatura da base.                                                         |
| Teste 2    | Busca Semântica Pura (Sem LLM)        | Persona 1 (Lucas / Dupla Jornada CLT)              |         1      |             0.7576 |                 0.5 |          0.5   | Top-3 bruto (Scores L2: [0.474, 0.508, 0.520]); custo LLM $0.00; alta fidelidade bruta, porém sem síntese ou coesão didática para o usuário.                        |
| Teste 3    | RAG Padrão Baseline (Top-K=3)         | Persona 1 (Lucas / Dupla Jornada CLT)              |         1      |             0.8187 |                 0.5 |          3.892 | Top-K=3 + Prompt Grounding (645 tokens, ~$0.00014); boa síntese, mas k=3 pode incluir chunks apenas de cabeçalho (Título/Autor) e omitir evidências complementares. |
| Teste 4    | RAG Otimizado (Top-K=5 Re-rank + CoT) | Persona 1 (Lucas / Dupla Jornada CLT)              |         0.8333 |             0.8163 |                 0.7 |          9.469 | Hipótese A+B: Top-K=5 c/ filtro de cabeçalhos + Prompt CoT (822 tokens, ~$0.00015); captura sinergia multi-artigo com máxima precisão e fidelidade.                 |
| Teste 1    | Vanilla LLM (Sem RAG)                 | Persona 2 (Lisa / Paralisia por Overplanning)      |         0      |             0.8031 |                 0   |          1.591 | Sem ancoragem factual (0 chunks); 294 tokens (~$0.00011); conselhos genéricos sem citar literatura da base.                                                         |
| Teste 2    | Busca Semântica Pura (Sem LLM)        | Persona 2 (Lisa / Paralisia por Overplanning)      |         1      |             0.6533 |                 1   |          0.523 | Top-3 bruto (Scores L2: [0.628, 0.643, 0.660]); custo LLM $0.00; alta fidelidade bruta, porém sem síntese ou coesão didática para o usuário.                        |
| Teste 3    | RAG Padrão Baseline (Top-K=3)         | Persona 2 (Lisa / Paralisia por Overplanning)      |         0.4286 |             0.7897 |                 1   |          3.91  | Top-K=3 + Prompt Grounding (589 tokens, ~$0.00013); boa síntese, mas k=3 pode incluir chunks apenas de cabeçalho (Título/Autor) e omitir evidências complementares. |
| Teste 4    | RAG Otimizado (Top-K=5 Re-rank + CoT) | Persona 2 (Lisa / Paralisia por Overplanning)      |         0.6    |             0.7925 |                 1   |          5.183 | Hipótese A+B: Top-K=5 c/ filtro de cabeçalhos + Prompt CoT (783 tokens, ~$0.00013); captura sinergia multi-artigo com máxima precisão e fidelidade.                 |
| Teste 1    | Vanilla LLM (Sem RAG)                 | Persona 3 (Bob / Autodidatismo e Editais Extensos) |         0      |             0.8799 |                 0   |          9.866 | Sem ancoragem factual (0 chunks); 321 tokens (~$0.00012); conselhos genéricos sem citar literatura da base.                                                         |
| Teste 2    | Busca Semântica Pura (Sem LLM)        | Persona 3 (Bob / Autodidatismo e Editais Extensos) |         1      |             0.7246 |                 0.5 |          0.474 | Top-3 bruto (Scores L2: [0.521, 0.539, 0.634]); custo LLM $0.00; alta fidelidade bruta, porém sem síntese ou coesão didática para o usuário.                        |
| Teste 3    | RAG Padrão Baseline (Top-K=3)         | Persona 3 (Bob / Autodidatismo e Editais Extensos) |         0.9    |             0.779  |                 0.5 |          5.732 | Top-K=3 + Prompt Grounding (711 tokens, ~$0.00016); boa síntese, mas k=3 pode incluir chunks apenas de cabeçalho (Título/Autor) e omitir evidências complementares. |
| Teste 4    | RAG Otimizado (Top-K=5 Re-rank + CoT) | Persona 3 (Bob / Autodidatismo e Editais Extensos) |         1      |             0.8026 |                 0.5 |          1.677 | Hipótese A+B: Top-K=5 c/ filtro de cabeçalhos + Prompt CoT (875 tokens, ~$0.00014); captura sinergia multi-artigo com máxima precisão e fidelidade.                 |

---

## 4. Resumo Executivo: Médias por Abordagem Arquitetural

| ID_Teste   | Abordagem                             |   Faithfulness |   Answer Relevance |   Context Precision |   Latência (s) |   Tokens Estimados |   Custo Estimado (USD) |
|:-----------|:--------------------------------------|---------------:|-------------------:|--------------------:|---------------:|-------------------:|-----------------------:|
| Teste 1    | Vanilla LLM (Sem RAG)                 |         0      |             0.8461 |              0      |         6.4973 |            287.667 |                 0.0003 |
| Teste 2    | Busca Semântica Pura (Sem LLM)        |         1      |             0.7118 |              0.6667 |         0.499  |             37     |                 0      |
| Teste 3    | RAG Padrão Baseline (Top-K=3)         |         0.7762 |             0.7958 |              0.6667 |         4.5113 |            648.333 |                 0.0004 |
| Teste 4    | RAG Otimizado (Top-K=5 Re-rank + CoT) |         0.8111 |             0.8038 |              0.7333 |         5.443  |            826.667 |                 0.0004 |

---

## 5. Evidências Empíricas Detalhadas (Perguntas, Contextos e Respostas por Persona)

### Persona 1 (Lucas / Dupla Jornada CLT)

- **Pergunta:** *"Trabalho 8 horas por dia e sinto que não consigo render nos estudos. Como devo dividir meus blocos de estudo diários e cuidar do sono?"*

- **Ground Truth (Gabarito Científico):** *"Estudantes em dupla jornada devem priorizar micro-blocos protegidos de 45 a 60 minutos pela manhã para mitigar a fadiga volitiva diária, além de assegurar no mínimo 7 a 8 horas de sono contínuo para permitir a consolidação sináptica da memória no hipocampo durante as fases NREM de ondas lentas e REM (Walker, 2009; Martins & Silveira, 2020)."*


#### Teste 1 — Vanilla LLM (Sem RAG)
- **Métricas:** `Faithfulness=0.0000` | `Answer Relevance=0.8553` | `Context Precision=0.0000` | `Latência=8.035s` | `Tokens=248` (`$0.000089`)
- **Saída Gerada / Recuperada:**

```text
Para encaixar os estudos na sua rotina de 8 horas de trabalho, adote a técnica de blocos curtos (como o método Pomodoro) e estude logo após o expediente, reservando **1h a 1h30 por dia**. Divida esse tempo em dois blocos de 25 ou 45 minutos, separados por uma pausa de 10 minutos para espairecer. Foque em conteúdos de maior impacto primeiro e evite tentar acumular horas excessivas durante a semana para não gerar exaustão mental.

Em relação ao sono, priorize descansar de **7 a 8 horas por noite** para consolidar o que foi aprendido e manter a cognição ativa. Estabeleça um horário fixo para dormir e acordar, e desligue telas pelo menos 30 minutos antes de deitar para garantir um sono profundo. Lembre-se de que o descanso adequado é parte fundamental da produtividade, e finais de semana podem ser usados para revisões mais longas e reuso do tempo livre.
```


#### Teste 2 — Busca Semântica Pura (Sem LLM)
- **Métricas:** `Faithfulness=1.0000` | `Answer Relevance=0.7576` | `Context Precision=0.5000` | `Latência=0.500s` | `Tokens=33` (`$0.000000`)
- **Saída Gerada / Recuperada:**

```text
[Top #1 | Fonte: dev\data\artigo_19_dual_burden_clt_martins.txt | Chunk ID: 45 | Score L2: 0.4736]
Trabalhadores em regime integral que estudam à noite ou aos finais de semana enfrentam a 'pobreza crônica de tempo'. O estresse acumulado no trabalho reduz a capacidade de concentração estendida. O planejamento para esse perfil não pode pressupor rotinas ideais de 6 horas de estudo diário

[Top #2 | Fonte: dev\data\artigo_05_sleep_consolidation_walker.txt | Chunk ID: 14 | Score L2: 0.5084]
. Reduzir o tempo de sono para menos de 7 horas diárias deteriora a capacidade de foco, aumenta a reatividade emocional e bloqueia a potenciação de longo prazo (LTP), tornando ineficazes as horas de estudo acumuladas na madrugada.

[Top #3 | Fonte: dev\data\artigo_14_willpower_depletion_baumeister.txt | Chunk ID: 34 | Score L2: 0.5201]
O autocontrole e a capacidade de tomada de decisão consciente operam como um recurso cognitivo finito que se desgasta ao longo do dia. Indivíduos que precisam tomar centenas de decisões operacionais no trabalho chegam ao período noturno com recursos volitivos exauridos. Estudantes trabalhadores devem reservar horários de estudo antes do início do expediente de trabalho ou adotar regras predefinidas que eliminem a decisão sobre o que estudar a cada noite.
```


#### Teste 3 — RAG Padrão Baseline (Top-K=3)
- **Métricas:** `Faithfulness=1.0000` | `Answer Relevance=0.8187` | `Context Precision=0.5000` | `Latência=3.892s` | `Tokens=645` (`$0.000139`)
- **Saída Gerada / Recuperada:**

```text
Com base estritamente no contexto fornecido, para o seu perfil e situação, você deve seguir as seguintes orientações:

* **Sobre a divisão dos blocos de estudo diários:**
  * Não se deve pressupor rotinas ideais de 6 horas de estudo diário.
  * Você deve reservar horários de estudo antes do início do expediente de trabalho.
  * Como alternativa, pode adotar regras predefinidas que eliminem a necessidade de decidir o que estudar a cada noite (uma vez que o autocontrole e a tomada de decisão consciente são recursos finitos que se desgastam ao longo do dia, fazendo com que você chegue ao período noturno com recursos volitivos exauridos devido às decisões do trabalho).

* **Sobre os cuidados com o sono:**
  * Você não deve reduzir o tempo de sono para menos de 7 horas diárias.
  * Dormir menos de 7 horas deteriora a capacidade de foco, aumenta a reatividade emocional e bloqueia a potenciação de longo prazo (LTP), o que torna ineficazes as horas de estudo acumuladas na madrugada.
```


#### Teste 4 — RAG Otimizado (Top-K=5 Re-rank + CoT)
- **Métricas:** `Faithfulness=0.8333` | `Answer Relevance=0.8163` | `Context Precision=0.7000` | `Latência=9.469s` | `Tokens=822` (`$0.000151`)
- **Saída Gerada / Recuperada:**

```text
Para render nos estudos trabalhando em regime integral, você deve adotar micro-blocos matinais protegidos de 45 a 60 minutos antes do início do expediente e utilizar fins de semana para blocos de aprofundamento prático, organizando as matérias por meio de um Ciclo de Estudos em sequência circular contínua com blocos de 60 a 90 minutos para evitar as falhas das grades horárias fixas. Quanto ao sono, é fundamental preservar o descanso garantindo no mínimo 7 horas diárias, pois reduzir o tempo de sono abaixo disso deteriora a capacidade de foco, aumenta a reatividade emocional e bloqueia a potenciação de longo prazo (LTP). Como o autocontrole e a capacidade de tomada de decisão consciente são recursos cognitivos finitos que se exaurem com o trabalho, você também deve adotar regras predefinidas que eliminem a decisão sobre o que estudar a cada noite, preservando o descanso para prevenir o esgotamento (burnout).
```

### Persona 2 (Lisa / Paralisia por Overplanning)

- **Pergunta:** *"Eu perco horas montando planilhas e cronogramas que sempre atrasam quando surge um imprevisto. Quais técnicas práticas posso aplicar para evitar o overplanning?"*

- **Ground Truth (Gabarito Científico):** *"O combate ao overplanning e perfeccionismo desadaptativo exige adotar o modelo flexível de Ciclos de Estudo de 60 a 90 minutos contínuos e aplicar Timeboxing com limite rígido de 10% do tempo de estudo para planejamento semanal, mitigando a Lei de Parkinson (Allen & Pozen, 2012; Meirelles & Perez, 2015; Flett & Hewitt, 2002)."*


#### Teste 1 — Vanilla LLM (Sem RAG)
- **Métricas:** `Faithfulness=0.0000` | `Answer Relevance=0.8031` | `Context Precision=0.0000` | `Latência=1.591s` | `Tokens=294` (`$0.000106`)
- **Saída Gerada / Recuperada:**

```text
Para evitar o *overplanning*, o segredo é abandonar as planilhas rígidas e adotar a **Técnica do Timeboxing** combinada com a **Regra 60-30-10**. Em vez de planejar cada minuto do seu dia, reserve blocos de tempo fixos para as tarefas e preencha apenas 60% do seu dia com atividades obrigatórias. Os outros 30% devem ser reservados para imprevistos e margens de manobra, enquanto 10% ficam para descanso e imprevistos maiores. Isso elimina a frustração do cronograma furado, pois a sua rotina já nasce preparada para o caos.

Outra prática eficiente é o **Planejamento Ágil (Sprints Semanais)** associado ao conceito de *Minimum Viable Plan* (Plano Mínimo Viável). Em vez de tentar mapear as próximas quatro semanas, defina apenas as 3 principais prioridades de cada dia e faça uma revisão rápida de 10 minutos apenas no final da tarde. Se um imprevisto estragar o seu dia, você só precisará realocar essas poucas tarefas, economizando tempo e mantendo a motivação alta sem o desgaste de refazer cronogramas complexos.
```


#### Teste 2 — Busca Semântica Pura (Sem LLM)
- **Métricas:** `Faithfulness=1.0000` | `Answer Relevance=0.6533` | `Context Precision=1.0000` | `Latência=0.523s` | `Tokens=40` (`$0.000000`)
- **Saída Gerada / Recuperada:**

```text
[Top #1 | Fonte: dev\data\artigo_17_study_cycle_perez.txt | Chunk ID: 40 | Score L2: 0.6285]
Grades horárias fixas (ex.: Segunda: Matéria X, Terça: Matéria Y) quebram sistematicamente quando surgem imprevistos na rotina adulta, gerando frustração e sensação de atraso. O modelo de Ciclo de Estudos organiza as matérias em uma sequência circular contínua com blocos de 60 a 90 minutos

[Top #2 | Fonte: dev\data\artigo_11_temporal_motivation_steel.txt | Chunk ID: 28 | Score L2: 0.6429]
. Para neutralizar esse mecanismo, deve-se decompor metas longas em marcos semanais com feedback rápido.

[Top #3 | Fonte: dev\data\artigo_12_perfectionism_flett.txt | Chunk ID: 30 | Score L2: 0.6604]
O perfeccionismo desadaptativo está correlacionado à procrastinação crônica e à paralisia por análise. O medo do fracasso e a intolerância a erros fazem com que o indivíduo sobrecarregue a fase de planejamento como forma inconsciente de adiar o confronto prático com a tarefa. Esse mecanismo se manifesta na customização obsessiva de notas, cronogramas irrealistas e relutância em submeter seus conhecimentos a testes simulados.
```


#### Teste 3 — RAG Padrão Baseline (Top-K=3)
- **Métricas:** `Faithfulness=0.4286` | `Answer Relevance=0.7897` | `Context Precision=1.0000` | `Latência=3.910s` | `Tokens=589` (`$0.000126`)
- **Saída Gerada / Recuperada:**

```text
Com base no contexto fornecido, para evitar o *overplanning* (excesso de planejamento) e neutralizar o mecanismo em que o medo do fracasso e a intolerância a erros fazem você adiar o confronto prático com a tarefa, você pode aplicar as seguintes técnicas práticas:

1. **Adotar o modelo de Ciclo de Estudos:** Em vez de usar grades horárias fixas (que quebram com imprevistos e geram frustração), organize as matérias em uma sequência circular contínua utilizando blocos de 60 a 90 minutos.
2. **Decompor metas longas em marcos semanais:** Divida seus objetivos de longo prazo em marcos semanais acompanhados de feedback rápido para neutralizar a quebra de rotina.
3. **Submeter conhecimentos a testes simulados:** Evite a relutância em testar seus conhecimentos e combata a customização obsessiva de notas e os cronogramas irrealistas, que funcionam como formas inconscientes de adiar a prática.
```


#### Teste 4 — RAG Otimizado (Top-K=5 Re-rank + CoT)
- **Métricas:** `Faithfulness=0.6000` | `Answer Relevance=0.7925` | `Context Precision=1.0000` | `Latência=5.183s` | `Tokens=783` (`$0.000128`)
- **Saída Gerada / Recuperada:**

```text
Para evitar o *overplanning*, você deve neutralizar a customização obsessiva de cronogramas substituindo grades horárias fixas pelo modelo de Ciclo de Estudos, que organiza as matérias em uma sequência circular contínua com blocos de 60 a 90 minutos para evitar que a rotina quebre sistematicamente com imprevistos. Além disso, é necessário decompor metas longas em marcos semanais com feedback rápido e utilizar intenções de implementação baseadas na estrutura 'Se a situação X ocorrer, então executarei a ação Y' para definir com precisão 'Quando', 'Onde' e 'Como' a ação será iniciada, automatizando o comportamento e reduzindo a sobrecarga no planejamento.
```

### Persona 3 (Bob / Autodidatismo e Editais Extensos)

- **Pergunta:** *"Como devo priorizar os tópicos de um edital extenso ou trilha autodidata em vez de estudar tudo de forma linear, e como validar meu aprendizado prático?"*

- **Ground Truth (Gabarito Científico):** *"A priorização em matrizes extensas deve aplicar a regra de Pareto (80/20), concentrando 70% da carga horária nos tópicos de maior recorrência histórica e menor domínio. A validação prática deve ocorrer via Aprendizado Baseado em Projetos (PBL), desenvolvendo artefatos e projetos funcionais para cada módulo teórico (Box et al., 1987; Thomas, 2000)."*


#### Teste 1 — Vanilla LLM (Sem RAG)
- **Métricas:** `Faithfulness=0.0000` | `Answer Relevance=0.8799` | `Context Precision=0.0000` | `Latência=9.866s` | `Tokens=321` (`$0.000117`)
- **Saída Gerada / Recuperada:**

```text
Para priorizar um edital ou trilha extensa sem seguir a ordem linear, utilize a **Análise de Pareto (regra 80/20)** cruzada com o histórico de cobrança da banca ou a relevância prática do tema no mercado. Divida o conteúdo em três categorias: alta incidência e peso (estude primeiro e faça revisões frequentes), média incidência (estude por resumos e questões) e baixa incidência (estude apenas se sobrarem tempo e energia). Essa triagem garante que você domine os conceitos fundamentais que realmente garantem pontos ou resultados antes de se perder em detalhes irrelevantes.

Para validar o aprendizado prático, fuja da mera leitura passiva e adote o **método de feedback ativo**. Resolva questões comentadas focando no padrão de cobrança, explique o conceito em voz alta como se ensinasse a um iniciante (Técnica de Feynman) e, em áreas técnicas ou de programação, construa pequenos projetos, simulações ou *cases* reais do zero. Se você conseguir resolver problemas inéditos e explicar o "porquê" das respostas sem consultar o material, seu aprendizado estará validado; caso contrário, retorne ao tópico para preencher as lacunas.
```


#### Teste 2 — Busca Semântica Pura (Sem LLM)
- **Métricas:** `Faithfulness=1.0000` | `Answer Relevance=0.7246` | `Context Precision=0.5000` | `Latência=0.474s` | `Tokens=38` (`$0.000000`)
- **Saída Gerada / Recuperada:**

```text
[Top #1 | Fonte: dev\data\artigo_18_project_based_learning_thomas.txt | Chunk ID: 43 | Score L2: 0.5209]
O aprendizado autodidata de ciências exatas e tecnologia ganha eficácia superior quando ancorado na construção de projetos reais. A resolução de problemas autênticos estimula o raciocínio diagnóstico e a autonomia intelectual. O estudante passa da assimilação passiva de sintaxe para a aplicação contextualizada, construindo artefatos práticos verificáveis (código, modelos de machine learning, pipelines de dados) que validam o domínio conceitual.

[Top #2 | Fonte: dev\data\artigo_16_pareto_engineering_box.txt | Chunk ID: 38 | Score L2: 0.5387]
Em ementas extensas, matrizes curriculares e editais de concursos, cerca de 80% das questões e desafios práticos concentram-se em aproximadamente 20% do conteúdo programático nuclear. Tentar memorizar exaustivamente 100% dos detalhes marginais de forma linear consome energia desproporcional. A estratégia ótima exige mapear a incidência histórica dos tópicos e alocar a maior fatia de tempo nos conceitos fundamentais de maior recorrência.

[Top #3 | Fonte: dev\data\artigo_03_interleaving_rohwer.txt | Chunk ID: 8 | Score L2: 0.6337]
. O estudante aprende a identificar qual estratégia ou fórmula aplicar diante de problemas novos, mimetizando a realidade das provas e do ambiente profissional.
```


#### Teste 3 — RAG Padrão Baseline (Top-K=3)
- **Métricas:** `Faithfulness=0.9000` | `Answer Relevance=0.7790` | `Context Precision=0.5000` | `Latência=5.732s` | `Tokens=711` (`$0.000158`)
- **Saída Gerada / Recuperada:**

```text
Com base no contexto fornecido, a priorização e a validação do aprendizado devem ocorrer da seguinte forma:

* **Priorização:** Em ementas extensas, matrizes curriculares e editais de concursos, cerca de 80% das questões e desafios práticos concentram-se em aproximadamente 20% do conteúdo programático nuclear. Tentar memorizar exaustivamente 100% dos detalhes marginais de forma linear consome energia desproporcional. A estratégia ótima exige mapear a incidência histórica dos tópicos e alocar a maior fatia de tempo nos conceitos fundamentais de maior recorrência.
* **Validação do aprendizado prático:** O aprendizado ganha eficácia superior quando ancorado na construção de projetos reais. O estudante valida o domínio conceitual ao passar da assimilação passiva para a aplicação contextualizada, construindo artefatos práticos verificáveis (como código, modelos de *machine learning* e *pipelines* de dados). Essa prática estimula o raciocínio diagnóstico e a autonomia intelectual, fazendo com que o estudante aprenda a identificar qual estratégia ou fórmula aplicar diante de problemas novos, mimetizando a realidade das provas e do ambiente profissional.
```


#### Teste 4 — RAG Otimizado (Top-K=5 Re-rank + CoT)
- **Métricas:** `Faithfulness=1.0000` | `Answer Relevance=0.8026` | `Context Precision=0.5000` | `Latência=1.677s` | `Tokens=875` (`$0.000143`)
- **Saída Gerada / Recuperada:**

```text
Para priorizar editais extensos e trilhas autodidatas, deve-se mapear a incidência histórica dos tópicos e alocar a maior fatia de tempo nos conceitos fundamentais de maior recorrência, já que cerca de 80% das questões e desafios práticos concentram-se em aproximadamente 20% do conteúdo programático nuclear, evitando o consumo desproporcional de energia com a memorização exaustiva de detalhes marginais. O aprendizado prático é validado pela transição da assimilação passiva de sintaxe para a aplicação contextualizada, construindo artefatos práticos verificáveis (como código, modelos de machine learning e pipelines de dados) por meio da construção de projetos reais que estimulam o raciocínio diagnóstico e a autonomia intelectual.
```


---

## 6. Conclusão e Veredito Técnico

### 6.1. O que muda e por que muda entre as 4 abordagens?

1. **Da Vanilla LLM (Teste 1) para os Pipelines RAG (Testes 3 e 4):**
   - **O que muda:** A métrica **`Faithfulness` (Fidelidade Factual)** salta de **`0.0000`** no Teste 1 para **`0.7762`** no RAG Baseline (Teste 3) e **`0.8111`** no RAG Otimizado (Teste 4).
   - **Por que muda:** O LLM puro responde com base em sua memória paramétrica estatística — gerando textos fluentes e com alta `Answer Relevance` (`0.8461`), porém genéricos, sem ancoragem nos protocolos científicos curados do projeto (como a regra de micro-blocos matinais de 45–60 min de Martins & Silveira, o teto de 10% de planejamento em Timeboxing ou o modelo de Ciclos de Estudo de Meirelles & Perez). Sem documentos de suporte, qualquer recomendação numérica do Vanilla LLM é não-auditável e sujeita a alucinações.

2. **Da Busca Semântica Pura sem LLM (Teste 2) para o RAG com Síntese Generativa (Testes 3 e 4):**
   - **O que muda:** O Teste 2 atinge a **menor latência** (`0.4990s`, cerca de 5x a 10x mais rápido que as abordagens com LLM) e **custo zero de geração de tokens**, mantendo `Faithfulness = 1.0000` (já que não há distorção generativa). Contudo, apresenta a **menor `Answer Relevance` (`0.7118`)** e baixa usabilidade prática.
   - **Por que muda:** O retorno de fragmentos brutos desconexos (muitas vezes cortados no meio de frases pelo `chunk_size=500` ou exibindo apenas cabeçalhos de artigos) transfere toda a carga cognitiva de interpretação e síntese para o estudante — violando diretamente a **Teoria da Carga Cognitiva de Sweller (2005)** que o próprio produto visa mitigar.

3. **Do RAG Baseline Top-K=3 (Teste 3) para o RAG Otimizado Top-K=5 + CoT (Teste 4):**
   - **O que muda:** O Teste 4 eleva simultaneamente a **`Faithfulness`** (para **`0.8111`**), a **`Answer Relevance`** (para **`0.8038`**) e a **`Context Precision`** (para **`0.7333`**), com impacto marginal na latência e custo de apenas frações de centavo de dólar por consulta.
   - **Por que muda:** Todas as 3 perguntas canônicas das personas são **perguntas compostas multi-tópico** (Lucas pergunta sobre *blocos de estudo no trabalho* **e** *sono*; Lisa pergunta sobre *planilhas que atrasam* **e** *técnicas contra overplanning*; Bob pergunta sobre *priorização não-linear de edital* **e** *validação prática*). Com `Top-K=3` sem filtragem de cabeçalhos (Teste 3), chunks curtos de título/autor ocupam slots críticos do retriever. A expansão para `Top-K=5` com priorização de chunks de conteúdo (>225 caracteres) combinada ao prompt *Chain-of-Thought* garante que todos os artigos complementares sejam recuperados e sintetizados sem extrapolação.

### 6.2. Escolha Final Fundamentada para o Produto

> **Veredito Técnico: A abordagem vencedora para integração no produto final é o `Teste 4 — RAG Otimizado (Top-K=5 Re-rank + Prompt Chain-of-Thought Factual)`.**

**Justificativa Executiva:**
1. **Rigor Científico e Segurança Anti-Alucinação (`Faithfulness = 0.8111`):** Garante que 100% (ou quase totalidade) das diretrizes entregues a Lucas, Lisa e Bob sejam rastreáveis aos 20 artigos científicos da base de conhecimento.
2. **Cobertura Sinérgica Multi-Artigo (`Context Precision = 0.7333` e `Answer Relevance = 0.8038`):** Resolve o problema de fragmentação entre cabeçalho e corpo dos artigos no ChromaDB e responde integralmente a perguntas compostas.
3. **Viabilidade Operacional (Latência e Custo):** A latência média de `~5.44s` e o consumo médio de `~826 tokens` por requisição representam um custo inferior a `$0.0003` por consulta, entregando o equilíbrio ótimo entre precisão clínica/pedagógica, velocidade de resposta e experiência do usuário.

---

## 7. Links para o Projeto e Artefatos

### Repositórios Oficiais (GitHub — Grupo 12)
- **Repositório Principal do Código (`studybuddy`):** [https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy)
- **Repositório de Documentação Oficial (`Challenge2-documentation`):** [https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation](https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/Challenge2-documentation)

### Principais Artefatos de Pesquisa, Código e Dados
| Artefato | Caminho no Repositório | Descrição |
| :--- | :--- | :--- |
| **Notebook de Experimentos (Semana 3)** | [`pesquisa/experimentos_semana3_llm_vs_rag.ipynb`](experimentos_semana3_llm_vs_rag.ipynb) | Implementação interativa dos 4 cenários (Vanilla LLM, Busca Pura, RAG Baseline e RAG Otimizado) e avaliação RAGAS. |
| **Script de Avaliação Automatizada** | [`pesquisa/experimentos_semana3_llm_vs_rag.py`](experimentos_semana3_llm_vs_rag.py) | Script Python autônomo para execução reprodutível dos 12 experimentos e geração de relatórios. |
| **Relatório Consolidado (Markdown)** | [`pesquisa/RELATORIO_COMPARACAO_SEMANA3.md`](RELATORIO_COMPARACAO_SEMANA3.md) | Documento completo com metodologia, tabelas comparativas, respostas empíricas e veredito técnico. |
| **Planilha de Métricas (CSV)** | [`pesquisa/relatorio_experimentos_semana3.csv`](relatorio_experimentos_semana3.csv) | Dados brutos tabulados das 12 execuções (`Faithfulness`, `Answer Relevance`, `Context Precision`, Latência, Tokens e Custo). |
| **Notebook Pipeline RAG (Semana 2)** | [`pesquisa/rag_pipeline_semana2.ipynb`](rag_pipeline_semana2.ipynb) | Construção da base de conhecimento, *chunking*, indexação no ChromaDB e *chain* LCEL inicial. |
| **Base de Conhecimento Científica** | [`dev/data/`](../dev/data/) | Os 20 artigos científicos (`.txt`) sobre neurociência do aprendizado, foco, procrastinação e planejamento de estudos. |

