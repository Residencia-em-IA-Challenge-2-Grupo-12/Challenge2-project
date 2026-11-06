# Documentação Técnica — Entrega Semana 2 (Pipeline RAG)

Este diretório reúne os artefatos de documentação exigidos para a entrega da **Semana 2**, focados na qualidade da recuperação de informação, fundamentação das respostas e rastreabilidade das fontes.

---

## 📂 Arquivos Disponíveis

1. **[Diagrama de Arquitetura](diagrama_arquitetura.md)**  
   - Diagrama conceitual e de fluxo em **Mermaid**.
   - Separação clara entre o fluxo de ingestão (*offline/batch*) e o fluxo de inferência e resposta (*online/real-time*).
   - Detalhamento de cada componente: carregadores, chunking, embeddings, banco vetorial (ChromaDB), chain RAG e auditoria.

2. **[Pipeline de Ingestão de Dados](pipeline_ingestao.md)**  
   - Diagrama de sequência ponta a ponta em **Mermaid**.
   - Descrição detalhada de cada etapa: descoberta, carga tolerante a falhas, sanitização, particionamento semântico, enriquecimento de metadados e indexação.
   - Políticas de resiliência, idempotência e tratamento de exceções.

3. **[Descritivo de Estratégias (Chunking, Embeddings e Banco Vetorial)](descritivo_estrategias.md)**  
   - **Estratégia de Chunking:** Algoritmo `RecursiveCharacterTextSplitter`, justificativa dos parâmetros (`chunk_size=500`, `chunk_overlap=50`) e hierarquia de separadores.
   - **Estratégia de Embeddings:** Modelos suportados (OpenAI `text-embedding-3-small`, Gemini `text-embedding-004`, fallback determinístico), dimensionalidades e métrica de similaridade (distância L2 / cosseno).
   - **Banco Vetorial:** Implementação com ChromaDB, persistência local em `./chroma_db` e estrutura dos registros indexados.
   - **Fundamentação e Rastreabilidade:** Prompt com diretrizes estritas de *grounding*, recusa a alucinar e padrão de citação de fontes.

---

## 📚 Base de Conhecimento

A base de conhecimento curada encontra-se no diretório [`../data/`](../data/), contendo **documentos-fonte estruturados em texto**, abordando rotinas sustentáveis, combate ao overplanning, metodologias ativas de estudo e preparação estratégica por editais.

