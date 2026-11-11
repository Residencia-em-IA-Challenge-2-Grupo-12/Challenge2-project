# 2. O que são métricas

## Medir em vez de achar

Uma **métrica** é uma regra fixa que transforma um comportamento num número. Ler 3 respostas e achar que "ficou bom" é uma impressão. Uma métrica, ao contrário:

- é **repetível**: a mesma configuração dá o mesmo número;
- é **comparável**: dá para colocar chunk 300 e chunk 500 lado a lado;
- é **específica**: cada métrica responde a uma pergunta diferente.

## O gabarito: a régua

Para medir acerto, precisamos saber de antemão o que é "certo". O **gabarito** é uma lista de perguntas, cada uma com:

- as **fontes esperadas**: quais artigos contêm a resposta;
- os **pontos-chave**: o que uma boa resposta precisa dizer;
- o **tipo**: direta, parafraseada, multi-fonte ou fora de escopo.

No projeto, o gabarito tem 20 perguntas (`pesquisa/metricas/dataset/gabarito.json`). **Toda métrica é tão boa quanto o gabarito**: se a fonte esperada estiver errada, o número também estará.

### Os tipos de pergunta e o que cada um testa

| Tipo | Exemplo | O que testa |
| :-- | :-- | :-- |
| `direta` | "O que é a técnica Pomodoro?" | O caso fácil: a pergunta usa as mesmas palavras do artigo |
| `parafraseada` | "Por que eu releio e não lembro na prova?" | Se a busca entende o sentido, mesmo sem palavras em comum |
| `multi_fonte` | "Como revisar e ainda dormir bem?" | Se a busca traz **todos** os artigos necessários |
| `fora_de_escopo` | "Como investir em ações?" | Se o sistema reconhece que não tem a resposta |

## Métricas de recuperação

Avaliam a busca e **não precisam de LLM**, então rodam rápido e sem custo. Todas olham os K primeiros resultados ("@k").

| Métrica | Pergunta que responde | Como calcula (por pergunta) |
| :-- | :-- | :-- |
| **Hit@k** | O artigo certo apareceu? | 1 se algum dos k chunks vem de uma fonte esperada, senão 0 |
| **MRR@k** | O acerto vem no topo? | 1 dividido pela posição do primeiro acerto (1º = 1, 2º = 0,5, 3º = 0,33; nenhum = 0) |
| **Precision@k** | Quanto do contexto é útil? | Chunks relevantes ÷ k |
| **Recall@k** | Trouxe **todos** os artigos necessários? | Artigos esperados encontrados ÷ total de artigos esperados |
| **Distintas@k** | Os chunks são variados ou repetidos? | Número de artigos diferentes entre os k chunks |

Depois, tiramos a **média** sobre todas as perguntas (exceto as fora de escopo, em que não existe "acerto").

Estas são as métricas básicas. A camada de busca também usa nDCG, cobertura de pontos-chave, AUC-ROC, robustez e custo; a lista completa e a matemática de cada uma estão em [Métricas da busca](07-metricas-da-busca.md).

### Diagnóstico extra: distância

A busca também devolve a **distância** entre a pergunta e cada chunk (menor = mais parecido). Se as perguntas fora de escopo ficam claramente mais longe que as válidas, dá para recusar só pela distância, antes de chamar o LLM.

> Distâncias só se comparam **dentro do mesmo modelo de embedding**. Cada modelo tem sua própria escala.

## Métricas de geração

Avaliam a resposta do LLM. Algumas são automáticas e outras usam **outro LLM como juiz** (*LLM-as-a-judge*), que deve ser um modelo diferente do que gerou a resposta, para não "corrigir a própria prova".

| Métrica | Pergunta que responde | Como mede |
| :-- | :-- | :-- |
| **Citação válida** | A fonte citada foi mesmo recuperada? | Regra automática: o `[Fonte: ...]` está entre os chunks entregues? |
| **Faithfulness** (fidelidade) | Cada afirmação está apoiada no contexto? | LLM-juiz verifica afirmação por afirmação |
| **Recusa correta** | Nas perguntas fora de escopo, ele recusou? | Compara com a frase de recusa definida no prompt |
| **Completude** | A resposta cobre os pontos-chave? | LLM-juiz compara com os `pontos_chave` do gabarito |
