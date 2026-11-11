# 1. O que é um RAG

## O problema

Um LLM sabe muita coisa, mas:

- **não conhece os seus documentos** (os 20 artigos do StudyBuddy, por exemplo);
- **pode inventar** quando não sabe (a chamada *alucinação*), e faz isso com a mesma confiança de quando acerta;
- **não diz de onde tirou** a informação.

## A ideia

**RAG** significa *Retrieval-Augmented Generation*, ou "geração aumentada por recuperação". Em vez de perguntar direto ao LLM, primeiro **buscamos** os trechos relevantes nos nossos documentos e entregamos esses trechos junto com a pergunta. O LLM passa a responder **com base no que recebeu**, como numa prova com consulta.

```
pergunta ──► [1] RECUPERAÇÃO ──► trechos ──► [2] GERAÇÃO (LLM) ──► resposta + fonte
             busca no banco                  o LLM lê os trechos
             vetorial                        e escreve a resposta
```

## As peças

### Preparação (feita uma vez, antes das perguntas)

1. **Ingestão:** carregar os documentos (aqui, 20 artigos em `dev/data/`).
2. **Chunking:** cortar cada documento em pedaços menores (ex.: 500 caracteres com 50 de sobreposição).
   - Pedaço pequeno demais perde o contexto.
   - Pedaço grande demais mistura assuntos e "dilui" o significado.
3. **Embedding:** transformar cada chunk num vetor de números que representa seu significado.
4. **Indexação:** guardar os vetores num banco vetorial (ChromaDB).

### A cada pergunta

5. **Recuperação:** a pergunta também vira um vetor. O banco devolve os **K chunks mais próximos** (Top-K).
6. **Geração:** um prompt junta "pergunta + chunks recuperados + regras" (ex.: "use só o contexto; se não souber, diga que não sabe; cite a fonte"). O LLM escreve a resposta.

## Analogia

É como um estudante numa prova com consulta:

- o **bibliotecário** (recuperação) escolhe 3 páginas dos livros;
- o **estudante** (LLM) escreve a resposta usando só essas páginas.

Se o bibliotecário traz as páginas erradas, o estudante não tem como acertar, por melhor que seja. Por isso **avaliamos as duas etapas separadamente**, e a recuperação vem primeiro.

## Onde um RAG pode falhar

| Etapa | Falha típica | Exemplo |
| :-- | :-- | :-- |
| Chunking | O trecho que responde foi cortado ao meio | A regra "25 min + 5 de pausa" ficou dividida entre dois chunks |
| Embedding | O modelo não liga a linguagem do dia a dia ao termo técnico | "deixo tudo pra depois" não encontra "procrastinação" |
| Recuperação | K pequeno demais, ou o artigo certo vem só em 4º lugar | A pergunta precisa de 2 artigos e só 1 coube no top 3 |
| Geração | O LLM inventa, ignora o contexto ou cita a fonte errada | Cita `[Fonte: artigo_10]`, mas esse artigo nem foi recuperado |
