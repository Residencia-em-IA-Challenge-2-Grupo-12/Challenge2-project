# 3. Por que usar métricas para validar um RAG

## O limite de "olhar as respostas"

Antes das métricas, a validação do StudyBuddy eram 3 perguntas das personas lidas a olho. Isso tem quatro problemas:

1. **Amostra pequena:** 3 perguntas não mostram em quais tipos de pergunta o sistema falha.
2. **Resposta bonita não é resposta certa:** um LLM escreve bem mesmo quando recebeu os trechos errados ou quando inventa.
3. **Não diz onde está o erro:** uma resposta ruim pode vir da busca ou do LLM, e o conserto é diferente para cada um.
4. **Não permite comparar:** "a resposta com chunk 300 parece melhor" não é uma conclusão confiável.

## O que as métricas trazem

### 1. Localizam a falha

Como medimos recuperação e geração separadamente:

| Recuperação | Geração | Diagnóstico | Onde mexer |
| :-- | :-- | :-- | :-- |
| ruim | ruim | O LLM nem recebeu a informação | Chunking, embedding, K |
| boa | ruim | O LLM recebeu, mas usou mal | Prompt, modelo de LLM |
| boa | boa | Funcionando | — |

### 2. Permitem escolher configurações com dados

Ao mudar uma coisa por vez (chunk 300 / 500 / 800, ou bge-m3 x qwen3-embedding) e rodar o mesmo gabarito, a diferença nos números mostra o efeito daquela mudança.

### 3. Revelam padrões de falha

| Padrão nos números | O que significa |
| :-- | :-- |
| `direta` boa, `parafraseada` ruim | O embedding depende de palavras iguais e não entende o sentido |
| Hit alto, Precision baixa | Acerta, mas entrega muito ruído ao LLM |
| Hit@5 muito maior que Hit@3 | K=3 é pequeno para esta base |
| Recall baixo em `multi_fonte` | A busca acha um assunto, mas não cobre perguntas com dois |
| Fora de escopo tão perto quanto as válidas | A busca não sabe dizer "não sei"; a recusa fica toda com o LLM |

### 4. Protegem contra regressões

Ao mudar o prompt, o modelo ou a base, basta rodar de novo: se um número cair, algo quebrou, mesmo que as 3 respostas de sempre continuem parecendo boas.

## Evidência do próprio projeto

Na mesma base e no mesmo gabarito, trocar só o modelo de embedding mudou tudo:

| Configuração (chunk 500, k=3) | Hit@3 | Hit@3 em `parafraseada` |
| :-- | :-- | :-- |
| Embedding offline (hash de palavras) | 0,44 | 0,33 |
| bge-m3 (Ollama, local) | 1,00 | 1,00 |

Lendo as respostas a olho, os dois casos produziriam textos fluentes. Só os números mostram que, no primeiro, a busca trazia o artigo errado em mais da metade das perguntas.

## Cuidados ao ler os números

- **Amostra pequena:** com 18 perguntas válidas, cada uma vale ~0,056. Diferenças desse tamanho podem ser ruído. Confie em padrões que se repetem.
- **Teto:** quando uma métrica chega a 1,00 em todas as opções, ela para de diferenciar. Passe a olhar MRR, precisão e recall.
- **O gabarito é a régua:** se ele estiver errado, os números também estão.
- **Olhe os erros um a um:** a média esconde *qual* pergunta falhou e *por quê*.
