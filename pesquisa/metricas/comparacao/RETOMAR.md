# Retomar a avaliação em outra máquina (Windows)

> **Atualização (12/11):** T3 e T2 foram concluídos numa máquina Linux. T1 a T4 estão completos. Só a variante `rag_llm_k3` (Ref) não foi reavaliada; este guia serve para rodá-la, se for preciso.

A avaliação grava o progresso a cada pergunta (`relatorios/comparacao_<modo>.parcial.json`) e pula os modos já prontos. Para continuar em outra máquina, basta ter o repositório atualizado e o mesmo juiz.

A geração das respostas **já foi feita** (`relatorios/comparacao_respostas_*.json`). A outra máquina só roda o juiz: **não precisa** de `qwen3:8b`, `bge-m3`, ChromaDB nem do notebook.

## 1. Uma vez só (PowerShell)

**Pré-requisitos:**

| O quê | Versão | Como conferir |
| :-- | :-- | :-- |
| Git | qualquer recente | `git --version` |
| Python | 3.10 ou mais novo (no Mac foi usado o 3.14) | `python --version` |
| Ollama | 0.30 ou mais novo (no Mac: 0.32.5) | `ollama --version` |
| Espaço em disco | ~9 GB para o modelo do juiz | |

Instale o Ollama por <https://ollama.com/download> e baixe o juiz, que é o **mesmo modelo** da rodada no Mac:

```powershell
ollama pull gemma3:12b
ollama list            # deve aparecer gemma3:12b (~8 GB)
```

Pegue o repositório na branch do experimento. **Se ainda não tem o repositório** no Book4 (por HTTPS, que não exige chave SSH):

```powershell
git clone https://github.com/Residencia-em-IA-Challenge-2-Grupo-12/studybuddy.git
cd studybuddy
git checkout metricas-experimento
```

**Se já tem o repositório** (por exemplo, na `main`):

```powershell
cd studybuddy
git fetch origin
git checkout metricas-experimento
```

Crie o ambiente Python. Para o juiz, bastam duas bibliotecas:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install langchain-ollama tiktoken
```

> Se o PowerShell bloquear scripts, não é preciso ativar a venv: os comandos deste guia chamam `.venv\Scripts\python` diretamente.

## 2. Retomar

Antes de rodar, **o Ollama precisa estar aberto** (ícone na bandeja do Windows; se não estiver, abra o app ou rode `ollama serve` num terminal separado).

```powershell
git pull
cd pesquisa
..\.venv\Scripts\python -m metricas.comparacao.avaliar
```

Sem argumentos, ele avalia só o que falta, na ordem T1 → T2 → T3 → T4 → k=3, e retoma o modo interrompido a partir da última pergunta salva. A primeira linha mostra o que já está pronto e o que falta.

**Prioridade (estado em 11/11, 17h35):** T1 e T4 estão prontos. O **T3** está em andamento: 4 de 18 perguntas válidas avaliadas (q01 a q04, em `comparacao_rag_llm.parcial.json`), e ele retoma da q05. Rode o T3 primeiro, porque é ele que confirma a escolha do relatório (T3 + porteiro), e depois o resto:

```powershell
..\.venv\Scripts\python -m metricas.comparacao.avaliar rag_llm so_busca rag_llm_k3
```

Tempo estimado no Mac M4: ~30 min para o resto do T3, ~25 min para o T2 e ~30 min para a k=3.

Para parar, use `Ctrl+C` a qualquer momento. Nada do que já foi avaliado se perde.

## 3. Atualizar o relatório

O relatório fica em `pesquisa/relatorio/final/`. A tabela gerada pelo avaliador (`pesquisa/metricas/relatorios/comparacao.md`) traz os números para preencher as células ⏳ de `02-resultados.md`, e a confirmação (ou não) da escolha vai em `03-decisao.md`.

## 4. Ao terminar (ou ao parar)

```powershell
cd ..
git add pesquisa/metricas/relatorios/comparacao* pesquisa/relatorio
git commit -m "data(metricas): avaliacao dos 4 modos"
git push
```

## Observações

- **Mesmo juiz, máquina diferente:** `gemma3:12b` com `temperature=0` tende a dar os mesmos vereditos, mas GPU NVIDIA e Apple podem diferir em casos limítrofes. Registre no relatório quais modos foram avaliados em cada máquina.
- **Memória de vídeo:** o `gemma3:12b` ocupa ~8 GB. Numa RTX 4050/4070 de notebook (6 a 8 GB), parte dele roda na CPU, então pode ser mais lento que no Mac. Deixe o notebook na tomada e desligue a suspensão.
- **Controle de Aplicativo do Windows:** em máquinas gerenciadas, ele já bloqueou DLLs do Python (ver `PLANO.md`). Num notebook pessoal, isso não deve acontecer.
