# Residência em IA — Semana 2: Base de Conhecimento e Protótipo RAG

Este repositório contém a implementação do primeiro mecanismo de **Retrieval-Augmented Generation (RAG)** da equipe, desenvolvido para fundamentar e orientar rotinas de estudo, produtividade e preparação técnica com base em evidências científicas.

---

## 1. Arquitetura da Solução

O pipeline segue a arquitetura canónica de RAG, garantindo rastreabilidade desde a ingestão até à geração final da resposta:

```mermaid
flowchart TD
    subgraph Ingestao [1. Ingestão e Indexação]
        A[20 Artigos Científicos em ./data] --> B[TextLoader / PyPDFLoader]
        B --> C[RecursiveCharacterTextSplitter<br>chunk_size=500 | chunk_overlap=50]
        C --> D[Modelo de Embeddings<br>text-embedding-004]
        D --> E[(ChromaDB: Coleção Local)]
    end

    subgraph Recuperacao [2. Recuperação e Geração Fundamentada]
        F[Pergunta do Utilizador / Persona] --> G[Retriever Top-K=3]
        E -.->|Busca por Proximidade L2| G
        G --> H[Chunks Relevantes + Metadados]
        H --> I[Prompt Estruturado com Regras Estritas]
        F --> I
        I --> J[LLM: Gemini / GPT-4o-mini<br>temperature=0.0]
        J --> K[Resposta Fundamentada com Citação de Fontes]
    end