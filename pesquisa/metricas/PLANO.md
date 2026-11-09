# Plano: métricas do RAG

Branch: `metricas-experimento` · Pasta: `pesquisa/metricas/`

## Objetivo

Medir a qualidade do RAG do notebook `rag_pipeline_semana2.ipynb` com números, em vez de avaliar só olhando as respostas. Hoje a "validação" são 3 perguntas das personas lidas a olho.

Um RAG falha em dois lugares, então medimos cada um separadamente:

```
pergunta ──► [1] RECUPERAÇÃO ──► chunks ──► [2] GERAÇÃO (LLM) ──► resposta
             busca no ChromaDB              LLM lê os chunks
```

- **Recuperação:** a busca trouxe os trechos certos? (sem LLM, sem custo de API)
- **Geração:** o LLM usou bem o que recebeu? Inventou algo? Citou a fonte certa?

Se a busca erra, o LLM não tem como acertar. Por isso a recuperação vem primeiro.

## O que já foi feito

| Item | Onde | Estado |
| :-- | :-- | :-- |
| Gabarito de 20 perguntas (4 do notebook, 14 novas, 2 fora de escopo), em português coloquial | `dataset/gabarito.json` | Pronto, validado |
| Carregador e validação do gabarito (confere se as fontes existem em `dev/data/`) | `dataset/loader.py` | Pronto |
| Fórmulas: Hit@k, MRR, Precision@k, Recall@k, Distintas@k | `recuperacao/metrics.py` | Pronto, 6 testes passando |
| Runner: busca os chunks, compara com o gabarito, salva JSON/CSV | `recuperacao/runner.py` | Escrito, **ainda não rodou de verdade** |
| Ponte com o notebook: executa as células de índice e usa o `vectorstore` dele | `recuperacao/notebook_loader.py` | Escrito, **ainda não rodou de verdade** |
| Camada de geração | `geracao/` | Vazia |

## O que falta fazer

1. **Rodar a recuperação de verdade** e olhar os números (ver "Como rodar").
2. **Interpretar o baseline:** comparar `direta` x `parafraseada`, ver Hit@3 x Hit@5, e as distâncias das perguntas fora de escopo.
3. **Experimentos de recuperação** com o mesmo gabarito: `chunk_size` (300 / 500 / 800), embedding (offline x OpenAI x Gemini) e K.
4. **Construir a camada de geração:**
   - Citação válida (sem LLM): o `[Fonte: ...]` citado está entre os chunks recuperados?
   - Faithfulness (LLM-juiz): cada afirmação da resposta está sustentada no contexto?
   - Recusa correta: nas perguntas q19 e q20 (fora de escopo), o RAG recusa em vez de inventar?
   - Completude: a resposta cobre os `pontos_chave` do gabarito?
   - O juiz deve ser um modelo diferente do que gera a resposta.
5. **Decidir o que vai para a `main`** (a ideia é que `metricas/` seja o definitivo).

## Como rodar

A partir de `pesquisa/`:

```powershell
python -m metricas.dataset.loader                       # valida o gabarito (só stdlib)
python -m metricas.recuperacao.test_metrics             # testa as fórmulas (só stdlib)
python -m pip install langchain langchain-core langchain-community langchain-text-splitters langchain-chroma langchain-openai langchain-google-genai chromadb pypdf tiktoken python-dotenv
python -m metricas.recuperacao.runner --nome baseline   # roda a recuperação de verdade
```

Atenção: o runner **reindexa** `dev/chroma_db` (apaga e recria a coleção), porque executa a célula do índice do notebook. Resultados saem em `metricas/relatorios/`.

## Bloqueio atual

Nesta máquina, o Controle de Aplicativo do Windows bloqueia DLLs das bibliotecas do RAG (`_pydantic_core`, `cygrpc`), então o runner não roda aqui. Saídas: pedir liberação ao TI, ou rodar em outro ambiente (Colab, WSL, outra máquina). O código é o mesmo em qualquer um.

## Como ler as métricas

| Métrica | Pergunta que responde |
| :-- | :-- |
| Hit@k | O artigo certo apareceu nos k primeiros? |
| MRR | O acerto vem no topo ou escondido? |
| Precision@k | Quanto do contexto entregue ao LLM é útil? |
| Recall@k | Trouxe todos os artigos que a pergunta exigia? |
| Distintas@k | Os k chunks são de artigos diferentes ou redundantes? |

| Padrão no relatório | Diagnóstico provável |
| :-- | :-- |
| `direta` boa, `parafraseada` ruim | O embedding não liga a linguagem do dia a dia ao jargão do artigo |
| Hit alto, Precision baixa | Acerta, mas traz muito ruído junto |
| Hit@5 muito maior que Hit@3 | K=3 é pequeno para essa base |
| Recall baixo em `multi_fonte` | Acha um tema, mas não cobre pergunta com mais de um assunto |
| Distância parecida entre perguntas em escopo e fora de escopo | A busca não reconhece que não há resposta; a recusa fica só com o LLM |

## Limitações a ter em mente

- **O gabarito é por artigo, não por chunk.** Serve para comparar configurações de chunking, mas a métrica não vê se o chunk trazido contém de fato o trecho que responde.
- **Amostra pequena:** 20 perguntas e 49 chunks. Uma diferença de 0,05 no Hit@3 equivale a uma pergunta, ou seja, é ruído. Os números servem para achar padrões grandes de falha, não para cravar diferenças pequenas.
- **As `fontes_esperadas` foram definidas por julgamento.** Vale revisar, porque o gabarito é a régua de todas as métricas.
- **Modo offline (hash de palavras)** só serve como piso de comparação; não mede qualidade semântica real.
