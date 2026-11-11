# Retomar a avaliação em outra máquina (Windows)

A avaliação grava o progresso a cada pergunta (`relatorios/comparacao_<modo>.parcial.json`) e pula os modos já prontos. Para continuar em outra máquina, basta ter o repositório atualizado e o mesmo juiz.

A geração das respostas **já foi feita** (`relatorios/comparacao_respostas_*.json`). A outra máquina só roda o juiz: **não precisa** de `qwen3:8b`, `bge-m3`, ChromaDB nem do notebook.

## 1. Uma vez só (PowerShell)

```powershell
git clone git@github.com:Residencia-em-IA-Challenge-2-Grupo-12/studybuddy.git
cd studybuddy
git checkout metricas-experimento

python -m venv .venv
.\.venv\Scripts\python -m pip install langchain-ollama tiktoken
```

Instale o Ollama (<https://ollama.com/download>) e baixe o juiz, que é o **mesmo modelo** da rodada no Mac:

```powershell
ollama pull gemma3:12b
```

## 2. Retomar

```powershell
git pull
cd pesquisa
..\.venv\Scripts\python -m metricas.comparacao.avaliar
```

Sem argumentos, ele avalia só o que falta, na ordem T1 → T2 → T3 → T4 → k=3, e retoma o modo interrompido a partir da última pergunta salva. A primeira linha mostra o que já está pronto e o que falta.

Para parar, use `Ctrl+C` a qualquer momento. Nada do que já foi avaliado se perde.

## 3. Ao terminar (ou ao parar)

```powershell
cd ..
git add pesquisa/metricas/relatorios/comparacao*
git commit -m "data(metricas): avaliacao dos 4 modos"
git push
```

## Observações

- **Mesmo juiz, máquina diferente:** `gemma3:12b` com `temperature=0` tende a dar os mesmos vereditos, mas GPU NVIDIA e Apple podem diferir em casos limítrofes. Registre no relatório quais modos foram avaliados em cada máquina.
- **Memória de vídeo:** o `gemma3:12b` ocupa ~8 GB. Numa RTX 4050/4070 de notebook (6 a 8 GB), parte dele roda na CPU, então pode ser mais lento que no Mac. Deixe o notebook na tomada e desligue a suspensão.
- **Controle de Aplicativo do Windows:** em máquinas gerenciadas, ele já bloqueou DLLs do Python (ver `PLANO.md`). Num notebook pessoal, isso não deve acontecer.
