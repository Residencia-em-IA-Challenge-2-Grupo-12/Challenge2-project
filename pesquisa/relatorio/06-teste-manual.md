# 6. Como testar na mão

As métricas mostram **quanto** cada configuração acerta. O teste manual mostra **como** e **por quê**: você vê os trechos que cada modelo trouxe e entende a lógica por trás de um número.

## A ferramenta

O script `pesquisa/metricas/recuperacao/perguntar.py`, no repositório, faz uma pergunta e mostra, lado a lado, o que cada embedding recupera:

- usa a mesma ingestão e o mesmo chunking do notebook;
- monta um índice **em memória** para cada modelo, então não altera o `dev/chroma_db` nem os relatórios;
- os modelos rodam localmente, via Ollama.

### Pré-requisitos

- O Ollama aberto, com os modelos `bge-m3` e `qwen3-embedding:0.6b` baixados (`ollama list` para conferir).
- O ambiente virtual do projeto (`studybuddy/.venv`) com o `langchain-ollama` instalado.

## Comandos

A partir de `studybuddy/pesquisa/`:

```bash
# modo interativo: digite perguntas livres ou um id do gabarito (q01..q20); 'sair' para terminar
../.venv/bin/python -m metricas.recuperacao.perguntar

# uma pergunta do gabarito: mostra o esperado e marca com ✓ os acertos
../.venv/bin/python -m metricas.recuperacao.perguntar --id q05

# uma pergunta livre: mostra os artigos e trechos, sem ✓ (não há resposta "certa" conhecida)
../.venv/bin/python -m metricas.recuperacao.perguntar "como parar de procrastinar?"

# variando o chunk, o K ou os modelos
../.venv/bin/python -m metricas.recuperacao.perguntar --chunk 800 --k 5
../.venv/bin/python -m metricas.recuperacao.perguntar --modelos bge-m3
```

| Opção | Padrão | O que faz |
| :-- | :-- | :-- |
| `"texto"` | — | Pergunta livre |
| `--id` | — | Usa uma pergunta do gabarito e marca os acertos |
| `--modelos` | `bge-m3 qwen3-embedding:0.6b` | Quais embeddings comparar |
| `--chunk` | `500` | Tamanho do chunk (o overlap é 10% disso) |
| `--k` | `3` | Quantos trechos mostrar por modelo |

## Como ler a saída

Exemplo real com a pergunta mais difícil do gabarito (chunk 500, k=3):

```
PERGUNTA: Por que eu releio o material várias vezes e mesmo assim não consigo lembrar na hora da prova?
ESPERADO: artigo_01_active_recall_karpicke.txt

  [bge-m3]
   1.   d=0.782  artigo_16_pareto_engineering_box.txt
        "Em ementas extensas, matrizes curriculares e editais de concursos, cerca de 80% das questões..."
   2. ✓ d=0.824  artigo_01_active_recall_karpicke.txt
        "O estudo demonstra que a recuperação ativa de informações da memória produz efeitos..."
   3.   d=0.895  artigo_07_attention_residue_leroy.txt
        "Quando uma pessoa muda o foco de uma Tarefa A para uma Tarefa B, a atenção não migra..."

  [qwen3-embedding:0.6b]
   1. ✓ d=0.848  artigo_01_active_recall_karpicke.txt
        "O estudo demonstra que a recuperação ativa de informações da memória produz efeitos..."
   2. ✓ d=0.917  artigo_01_active_recall_karpicke.txt
        ". A prática de testes práticos, resolução de questões sem consulta imediata e a autoexplicação..."
   3.   d=0.921  artigo_02_spaced_repetition_cepeda.txt
        "A prática distribuída (repetição espaçada) supera consistentemente a prática concentrada..."
```

| Elemento | Significado |
| :-- | :-- |
| `1.`, `2.`, `3.` | Posição no ranking: o 1º é o trecho que o modelo considera mais próximo |
| `✓` | O trecho vem de um artigo esperado pelo gabarito (só aparece com `--id`) |
| `d=` | Distância L2 até a pergunta: menor = mais parecido. **Só compare dentro do mesmo modelo** |
| Nome do arquivo | O artigo de onde o trecho veio |
| Texto entre aspas | O começo do trecho, ou seja, o que o LLM receberia como contexto |

**Leitura deste exemplo, à luz das métricas:**

- **bge-m3:** Hit@3 = 1, mas RR = 0,5, porque o acerto está em 2º lugar. Precision@3 = 1/3, com dois trechos de ruído.
- **qwen3-embedding:** RR = 1, com o acerto em 1º lugar. Precision@3 = 2/3, porque os dois primeiros trechos vêm do artigo certo.
- **Na média, porém, o bge-m3 foi melhor** (ver [Resultados](05-resultados.md)). Uma pergunta isolada não decide qual modelo é melhor: ela ajuda a entender o comportamento de cada um.

## Roteiro sugerido

| # | Teste | Comando | O que observar |
| :-- | :-- | :-- | :-- |
| 1 | Pergunta direta | `--id q10` ou `--id q14` | Os dois modelos devem acertar em 1º lugar. É o "está funcionando?" |
| 2 | Pergunta parafraseada | `--id q05` e `--id q15` | Nenhuma palavra da pergunta está no artigo. Qual modelo põe o ✓ em 1º? |
| 3 | Multi-fonte | `--id q01` e `--id q04` | Os 2 artigos esperados aparecem no top 3? |
| 4 | Efeito do chunk | `--id q04 --chunk 300` e depois `--chunk 800` | Com chunk 800, a cobertura dos dois assuntos tende a cair (diluição) |
| 5 | Efeito do K | `--id q01 --k 5` | Um K maior traz o artigo que faltava? |
| 6 | Fora de escopo | `--id q19` e `--id q20` | Nenhum ✓ deveria aparecer. A distância fica maior que nas perguntas válidas? |
| 7 | Artigo-ímã | Várias perguntas seguidas | Veja se o `artigo_02` aparece no qwen mesmo sem relação com a pergunta |
| 8 | Linguagem real | Perguntas livres | Escreva como um estudante falaria: gírias, frases incompletas, erros de digitação |
| 9 | Fora de escopo "perigoso" | `"qual o melhor remédio pra dormir?"` | Pergunta fora do tema, mas parecida com a base (sono). O modelo a confunde com uma pergunta válida? |

## Do teste manual para o gabarito

Uma pergunta livre não tem ✓, porque o script não sabe qual seria o artigo certo; quem julga é você. Se uma pergunta der um resultado interessante (um erro, um empate entre os modelos, uma confusão com fora de escopo), **adicione-a ao `pesquisa/metricas/dataset/gabarito.json`** com as fontes esperadas. Assim, o que você descobriu na mão passa a ser medido automaticamente em todos os testes seguintes.

Depois de editar o gabarito, valide e rode as métricas de novo:

```bash
../.venv/bin/python -m metricas.dataset.loader
../.venv/bin/python -m metricas.recuperacao.comparar
```

## Alternativa: pelo notebook

1. No `dev/.env`, defina `OLLAMA_EMBEDDING_MODEL=qwen3-embedding:0.6b`, ou deixe a variável de fora para usar o `bge-m3`.
2. No `rag_pipeline_semana2.ipynb`, troque o texto de `sample_query` na seção 4.
3. Rode as células até a seção 4.

Esse caminho mostra **um modelo por vez** e **reindexa o `dev/chroma_db`**. Para comparar modelos lado a lado, o script é mais prático.
