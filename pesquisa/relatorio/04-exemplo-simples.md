# 4. Exemplo simples: calculando as métricas à mão

Um RAG de brinquedo, pequeno o bastante para fazer todas as contas no papel.

## A base

| Artigo | Assunto |
| :-- | :-- |
| **A** | Sono e memória |
| **B** | Técnica Pomodoro |
| **C** | Repetição espaçada |
| **D** | Procrastinação |

Cada artigo foi cortado em alguns chunks. A busca devolve **3 chunks por pergunta (k=3)**, e cada chunk é identificado aqui pelo artigo de onde veio. O mesmo artigo pode aparecer mais de uma vez.

## O gabarito e o que a busca devolveu

| # | Pergunta | Esperado | Top-3 devolvido (em ordem) |
| :-- | :-- | :-- | :-- |
| P1 | "Dormir pouco atrapalha a memória?" | {A} | **A**, C, B |
| P2 | "Como parar de deixar tudo pra depois?" | {D} | B, **D**, **D** |
| P3 | "Como revisar e ainda dormir bem antes da prova?" | {A, C} | B, **C**, B |
| P4 | "Estudar 25 minutos e pausar funciona?" | {B} | D, A, C |
| P5 | "Qual a melhor ação para investir?" | fora de escopo | C, D, A |

Em negrito, os chunks relevantes (cuja fonte está no esperado).

## Passo a passo, pergunta por pergunta

**P1:** o acerto está em 1º lugar.
- Hit@3 = 1 (A apareceu)
- RR = 1/1 = **1**
- Precision@3 = 1 relevante ÷ 3 = **0,33**
- Recall@3 = 1 artigo encontrado ÷ 1 esperado = **1**

**P2:** o acerto está em 2º lugar e ocupa 2 posições.
- Hit@3 = 1
- RR = 1/2 = **0,5** (o primeiro acerto está na posição 2)
- Precision@3 = 2 ÷ 3 = **0,67** (dois chunks de D)
- Recall@3 = 1 ÷ 1 = **1** (o recall conta *artigos*, então D repetido conta uma vez só)

**P3:** pergunta multi-fonte, mas só um dos dois artigos veio.
- Hit@3 = 1 (C apareceu)
- RR = 1/2 = **0,5**
- Precision@3 = 1 ÷ 3 = **0,33**
- Recall@3 = 1 encontrado (C) ÷ 2 esperados (A, C) = **0,5**: faltou o artigo do sono

**P4:** erro total.
- Hit@3 = 0, RR = 0, Precision@3 = 0, Recall@3 = 0

**P5:** fora de escopo. Não existe artigo certo, então ela **fica fora das médias**. O que importa aqui é a distância (ver abaixo) e, depois, se o LLM recusa responder.

## As médias (P1 a P4)

| Métrica | Conta | Resultado |
| :-- | :-- | :-- |
| Hit@3 | (1 + 1 + 1 + 0) ÷ 4 | **0,75** |
| MRR@3 | (1 + 0,5 + 0,5 + 0) ÷ 4 | **0,50** |
| Precision@3 | (0,33 + 0,67 + 0,33 + 0) ÷ 4 | **0,33** |
| Recall@3 | (1 + 1 + 0,5 + 0) ÷ 4 | **0,625** |
| Hit@1 | só P1 acertou em 1º ÷ 4 | **0,25** |

## Como interpretar

- **Hit@3 = 0,75:** em 3 de 4 perguntas, o LLM recebeu pelo menos um trecho útil. Parece bom...
- **...mas Hit@1 = 0,25 e MRR = 0,50:** quase sempre o acerto vem em 2º lugar. Se diminuíssemos para k=1, o sistema despencaria.
- **Precision@3 = 0,33:** dois terços do contexto entregue ao LLM é ruído. O artigo B (Pomodoro) aparece em P2 e P3 sem ter nada a ver. Um artigo que aparece "em todo lugar" é um sinal de alerta.
- **Recall@3 = 0,625, puxado por P3:** a busca não cobre bem perguntas com dois assuntos. Uma resposta para P3 falaria de revisão, mas nada de sono.
- **P4 é o caso mais grave:** "25 minutos e pausar" é exatamente o Pomodoro, mas a pergunta não usa a palavra "Pomodoro". É um sinal de que o embedding depende de palavras iguais e não entende o sentido. A correção provável é **trocar o modelo de embedding**, não mexer no LLM.

### Diagnóstico pela distância

Suponha as distâncias do 1º chunk de cada pergunta:

| | P1 | P2 | P3 | P4 | P5 (fora) |
| :-- | :-- | :-- | :-- | :-- | :-- |
| Distância | 0,60 | 0,72 | 0,80 | 0,95 | 0,85 |

P5 (0,85) está **mais perto** da base que P4 (0,95), uma pergunta válida. Não existe um corte de distância que recuse P5 sem recusar P4. Nesse cenário, quem tem de reconhecer o "não sei" é o LLM, e isso precisa ser medido na etapa de geração.

## E na geração?

Imagine a resposta do LLM para P3:

> "Revise o conteúdo em intervalos crescentes [Fonte: C] e durma pelo menos 8 horas [Fonte: A]."

- **Citação válida:** a fonte C foi recuperada ✓. A fonte **A não foi** (o top-3 era B, C, B) ✗. O LLM citou algo que não recebeu.
- **Faithfulness:** "durma pelo menos 8 horas" não está em nenhum chunk entregue. É conhecimento externo, ou seja, alucinação, mesmo que a frase pareça correta.
- **Completude:** se os pontos-chave do gabarito fossem "revisão espaçada" e "sono consolida a memória", a resposta cobre o primeiro e só "acerta" o segundo por inventar.

Repare que a causa raiz desse erro de geração está na **recuperação** (Recall@3 = 0,5 em P3). Esse é o motivo de medir as duas etapas separadamente.

## O mesmo cálculo em código

```python
def hit(recuperadas, esperadas, k):
    return 1.0 if any(f in esperadas for f in recuperadas[:k]) else 0.0

def rr(recuperadas, esperadas, k):
    for pos, f in enumerate(recuperadas[:k], start=1):
        if f in esperadas:
            return 1 / pos
    return 0.0

def precision(recuperadas, esperadas, k):
    return sum(f in esperadas for f in recuperadas[:k]) / k

def recall(recuperadas, esperadas, k):
    return len(set(recuperadas[:k]) & esperadas) / len(esperadas)

gabarito = [
    (["A", "C", "B"], {"A"}),
    (["B", "D", "D"], {"D"}),
    (["B", "C", "B"], {"A", "C"}),
    (["D", "A", "C"], {"B"}),
]

for nome, f in [("Hit@3", hit), ("MRR@3", rr), ("Precision@3", precision), ("Recall@3", recall)]:
    media = sum(f(r, e, 3) for r, e in gabarito) / len(gabarito)
    print(f"{nome}: {media:.3f}")
# Hit@3: 0.750 | MRR@3: 0.500 | Precision@3: 0.333 | Recall@3: 0.625
```

São as mesmas fórmulas de `pesquisa/metricas/recuperacao/metrics.py`, no repositório.
