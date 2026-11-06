# Descritivo Técnico: Estratégias de Chunking, Embeddings e Banco Vetorial

**Projeto:** Residência em IA — Semana 2: Base de Conhecimento e Recuperação  
**Foco:** Qualidade da recuperação de informação, fundamentação das respostas e rastreabilidade das fontes.

---

## 1. Descrição da Estratégia de Chunking

### 1.1. Motivação e Desafios
No processamento de documentos para pipelines RAG, submeter textos inteiros a modelos de embedding ou à janela de contexto de LLMs apresenta limitações severas:
- **Diluição Semântica:** Textos longos agregam temas múltiplos, fazendo com que o vetor de embedding represente uma média dispersa do conteúdo e perca relevância em buscas específicas.
- **Janela de Contexto e Custo:** LLMs possuem limites de tokens e custos proporcionais ao tamanho do contexto submetido.
- **Rastreabilidade Fina:** É essencial indicar ao usuário o trecho e parágrafo exato que fundamentou a resposta.

### 1.2. Algoritmo Escolhido: `RecursiveCharacterTextSplitter`
Optou-se pelo fatiador recursivo de caracteres do LangChain com a seguinte hierarquia de separadores:
```python
separators = ["\n\n", "\n", ". ", " ", ""]
```
- **Prioridade 1 (`\n\n`):** Tenta manter parágrafos inteiros agrupados.
- **Prioridade 2 (`\n`):** Quebra em linhas estruturadas caso o parágrafo exceda o limite.
- **Prioridade 3 (`. `):** Quebra em sentenças completas, evitando fragmentação no meio de frases.
- **Prioridade 4 (` ` e `""`):** Último recurso para palavras longas.

### 1.3. Parâmetros Selecionados e Justificativa

| Parâmetro | Valor | Justificativa Técnica |
| :--- | :---: | :--- |
| `chunk_size` | **500 caracteres** (~80 a 120 palavras em português) | Mantém cada bloco focado em **um único conceito atômico** (ex.: uma técnica de estudo, uma regra prática ou uma diretriz de rotina). Garante alta precisão no cálculo de similaridade de cosseno. |
| `chunk_overlap` | **50 caracteres** (~10% do tamanho do chunk) | Previne a perda de contexto semântico nas bordas de corte. Frases que iniciam ou encerram em um bloco são parcialmente replicadas no próximo, evitando que pronomes e orações subordinadas fiquem soltos. |

### 1.4. Preservação de Metadados
Ao realizar o split, cada novo fragmento preserva os metadados do documento-pai (`source`, `topic`, `title`) e ganha metadados posicionais (`chunk_id`, `start_index`), permitindo identificar exatamente sua localização no documento de origem.

---

## 2. Descrição da Estratégia de Embeddings

### 2.1. Modelos Suportados
O sistema foi concebido para ser agnóstico a provedores, suportando:

1. **Google Gemini (`models/text-embedding-004`):**
   - **Dimensionalidade:** 768 dimensões.
   - **Vantagens:** Otimizado para tarefas semânticas multilíngues, excelente compreensão da sintaxe e semântica do português brasileiro e baixa latência de inferência.
2. **OpenAI (`text-embedding-3-small`):**
   - **Dimensionalidade:** 1536 dimensões.
   - **Vantagens:** Estado da arte para tarefas de *information retrieval* (MTEB benchmark), com capacidade de compressão vetorial e alta discriminação em buscas densas.
3. **Fallback Local Determinístico (`LocalKeywordHashEmbeddings`):**
   - **Dimensionalidade:** 256 dimensões.
   - **Finalidade:** Mecanismo leve offline baseado em hash de n-grams para viabilizar testes, pipelines de CI/CD e execuções em ambientes sem conectividade externa ou sem chaves de API provisionadas.

### 2.2. Métrica de Similaridade e Espaço Vetorial
- **Métrica Principal:** Similaridade de Cosseno (*Cosine Similarity*) / Distância L2 (*Euclidean Distance*).
- **Normalização:** Todos os vetores são normalizados (norma unitária L2), o que torna a distância euclidiana proporcional à similaridade de cosseno:
  $$\text{Cosine Distance} = 1 - \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$
- **Top-K Retrieval:** O parâmetro padrão de recuperação foi calibrado para **Top-K = 3**, equilibrando diversidade de contexto e concisão para a janela do prompt.

---

## 3. Implementação do Banco Vetorial (ChromaDB)

### 3.1. Motivação da Escolha
O **ChromaDB** foi selecionado por:
- **Persistência Local Integrada:** Salva índices e metadados no diretório `./chroma_db`, eliminando a necessidade de infraestrutura pesada ou serviços gerenciados externos na fase de desenvolvimento.
- **Indexação Vetorial HNSW:** Utiliza grafos navegáveis hierárquicos (*Hierarchical Navigable Small World*) para busca rápida de vizinhos mais próximos com complexidade sublinear $O(\log N)$.
- **Filtragem por Metadados:** Suporta filtros estruturados (*payload filtering*) como `{"topic": "anti_overplanning"}` ou `{"source": "..."}`.

### 3.2. Estrutura dos Registros no ChromaDB
Cada registro inserido na coleção contém quatro atributos fundamentais:
```json
{
  "id": "doc_01_chunk_0",
  "embedding": [0.0124, -0.0451, ..., 0.0892],
  "document": "Texto integral do fragmento de 500 caracteres...",
  "metadata": {
    "source": "doc_01_pomodoro.txt",
    "chunk_id": "0",
    "topic": "gestao_tempo",
    "total_chunks": 3
  }
}
```

---

## 4. Geração Fundamentada (*Grounding*) e Rastreabilidade

### 4.1. Engenharia de Prompt e Restrição ao Contexto
Para evitar alucinações e respostas especulativas, a chain utiliza um prompt com cláusula de contenção de domínio:
- O modelo é instruído expressamente a responder **apenas** com base nas afirmações contidas nos fragmentos recuperados.
- Caso os fragmentos não contenham a resposta para a dúvida, o modelo é obrigado a emitir a frase padronizada:
  > *"Não encontrei informações suficientes na base de conhecimento para responder a essa pergunta."*
- Toda resposta gerada deve citar a fonte de cada alegação com o formato `[Fonte: <origem> | Chunk ID: <id>]`.

### 4.2. Evidência e Auditoria da Resposta
Durante a execução de qualquer consulta, o sistema retorna um objeto estruturado contendo:
1. A pergunta submetida;
2. A lista completa dos chunks recuperados com seus metadados (`source`, `chunk_id`, `topic`) e pontuações de similaridade (*scores*);
3. O texto consolidado da resposta gerada pelo LLM.

