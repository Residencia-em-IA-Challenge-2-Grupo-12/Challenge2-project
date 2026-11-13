"""Gera o PDF didático do experimento de métricas do StudyBuddy (LLM x RAG, testes T1 a T4).

Uso (a partir de pesquisa/):
    python gerar_pdf_metricas.py [saida.pdf] [pasta_relatorios]

Padrões: saída Relatorio_Experimento_Metricas_StudyBuddy.pdf na pasta atual; relatórios em
metricas/relatorios/. Tabelas e gráficos são lidos de comparacao_<modo>.json e comparacao_porteiro.json;
os números citados no texto corrido são fixos e precisam ser revistos se os resultados mudarem.

Requer reportlab (pip install reportlab) e as fontes Arial do macOS.
"""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, Polygon
from reportlab.platypus import (BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, KeepTogether, CondPageBreak)

SAIDA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("Relatorio_Experimento_Metricas_StudyBuddy.pdf")
REL = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent / "metricas" / "relatorios"

# ---------------------------------------------------------------- fontes
F = "/System/Library/Fonts/Supplemental/"
pdfmetrics.registerFont(TTFont("Arial", F + "Arial.ttf"))
pdfmetrics.registerFont(TTFont("Arial-B", F + "Arial Bold.ttf"))
pdfmetrics.registerFont(TTFont("Arial-I", F + "Arial Italic.ttf"))
pdfmetrics.registerFont(TTFont("Arial-BI", F + "Arial Bold Italic.ttf"))
pdfmetrics.registerFont(TTFont("Uni", F + "Arial Unicode.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("Arial", normal="Arial", bold="Arial-B", italic="Arial-I", boldItalic="Arial-BI")

# ---------------------------------------------------------------- cores
INK = colors.HexColor("#0b0b0b")
INK2 = colors.HexColor("#52514e")
MUTED = colors.HexColor("#8a8984")
GRID = colors.HexColor("#e4e3de")
SURF = colors.HexColor("#f4f3ef")
S1 = colors.HexColor("#2a78d6")   # azul
S2 = colors.HexColor("#eb6834")   # laranja
S3 = colors.HexColor("#1baf7a")   # verde-água
OK = colors.HexColor("#008300")
OK_BG = colors.HexColor("#e3f2e3")
BAD = colors.HexColor("#c0392b")
BAD_BG = colors.HexColor("#fbe5e3")
WARN = colors.HexColor("#a86a00")
WARN_BG = colors.HexColor("#fdf1d8")
ACC_BG = colors.HexColor("#e8f0fb")

# ---------------------------------------------------------------- estilos
st = {}
st["body"] = ParagraphStyle("body", fontName="Arial", fontSize=10.2, leading=15, textColor=INK, spaceAfter=7)
st["small"] = ParagraphStyle("small", parent=st["body"], fontSize=8.6, leading=12, textColor=INK2, spaceAfter=4)
st["cell"] = ParagraphStyle("cell", parent=st["body"], fontSize=8.8, leading=12, spaceAfter=0)
st["cellb"] = ParagraphStyle("cellb", parent=st["cell"], fontName="Arial-B")
st["cellc"] = ParagraphStyle("cellc", parent=st["cell"], alignment=TA_CENTER)
st["cellbc"] = ParagraphStyle("cellbc", parent=st["cellb"], alignment=TA_CENTER)
st["h1"] = ParagraphStyle("h1", fontName="Arial-B", fontSize=19, leading=24, textColor=INK, spaceBefore=4, spaceAfter=10)
st["h2"] = ParagraphStyle("h2", fontName="Arial-B", fontSize=13, leading=17, textColor=INK, spaceBefore=12, spaceAfter=6)
st["h3"] = ParagraphStyle("h3", fontName="Arial-B", fontSize=10.8, leading=14, textColor=S1, spaceBefore=8, spaceAfter=4)
st["kicker"] = ParagraphStyle("kicker", fontName="Arial-B", fontSize=8.5, leading=11, textColor=S1, spaceAfter=2)
st["quote"] = ParagraphStyle("quote", parent=st["body"], fontName="Arial-I", fontSize=9.4, leading=13.5,
                             textColor=INK2, leftIndent=10, spaceAfter=6)
st["bullet"] = ParagraphStyle("bullet", parent=st["body"], leftIndent=14, bulletIndent=3, spaceAfter=4)
st["box"] = ParagraphStyle("box", parent=st["body"], spaceAfter=3)
st["cap"] = ParagraphStyle("cap", parent=st["small"], alignment=TA_LEFT, spaceBefore=2, spaceAfter=10)

W = A4[0] - 4.2 * cm  # largura útil


def P(t, s="body"):
    return Paragraph(t, st[s])


def bullets(itens, s="bullet"):
    return [Paragraph(i, st[s], bulletText="•") for i in itens]


def caixa(conteudo, fundo=SURF, borda=None, pad=10):
    if isinstance(conteudo, str):
        conteudo = [P(conteudo, "box")]
    t = Table([[conteudo]], colWidths=[W])
    estilo = [("BACKGROUND", (0, 0), (-1, -1), fundo), ("LEFTPADDING", (0, 0), (-1, -1), pad),
              ("RIGHTPADDING", (0, 0), (-1, -1), pad), ("TOPPADDING", (0, 0), (-1, -1), pad - 2),
              ("BOTTOMPADDING", (0, 0), (-1, -1), pad - 2)]
    if borda:
        estilo.append(("LINEBEFORE", (0, 0), (0, -1), 3, borda))
    t.setStyle(TableStyle(estilo))
    return t


def tabela(linhas, larguras, cab=True, destaque_col=None, destaque_lin=None, zebra=True, alinhar_num=False):
    dados = []
    for i, l in enumerate(linhas):
        linha = []
        for j, c in enumerate(l):
            if isinstance(c, str):
                estilo = "cellb" if (cab and i == 0) else ("cellc" if alinhar_num and j > 0 else "cell")
                if alinhar_num == "all":
                    estilo = "cellbc" if (cab and i == 0) else "cellc"
                c = Paragraph(c, st[estilo])
            linha.append(c)
        dados.append(linha)
    t = Table(dados, colWidths=[w * W for w in larguras], repeatRows=1 if cab else 0)
    s = [("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 5),
         ("RIGHTPADDING", (0, 0), (-1, -1), 5), ("TOPPADDING", (0, 0), (-1, -1), 4),
         ("BOTTOMPADDING", (0, 0), (-1, -1), 4), ("LINEBELOW", (0, 0), (-1, -1), 0.4, GRID)]
    if cab:
        s += [("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecebe6")), ("LINEBELOW", (0, 0), (-1, 0), 0.8, INK2)]
    if zebra:
        for i in range(1 if cab else 0, len(dados)):
            if i % 2 == 0:
                s.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#faf9f6")))
    if destaque_col is not None:
        s.append(("BACKGROUND", (destaque_col, 1 if cab else 0), (destaque_col, -1), ACC_BG))
    if destaque_lin is not None:
        s.append(("BACKGROUND", (0, destaque_lin), (-1, destaque_lin), ACC_BG))
    t.setStyle(TableStyle(s))
    return t


def selo(texto, tipo):
    cor, fundo, icone = {"ok": (OK, OK_BG, "✓"), "bad": (BAD, BAD_BG, "✗"), "warn": (WARN, WARN_BG, "!")}[tipo]
    p = Paragraph(f'<font name="Uni" color="{cor.hexval()}">{icone}</font>&nbsp; <b>{texto}</b>',
                  ParagraphStyle("selo", parent=st["body"], fontSize=10.5, textColor=INK, spaceAfter=0))
    t = Table([[p]], colWidths=[W])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), fundo), ("LINEBEFORE", (0, 0), (0, -1), 3, cor),
                           ("LEFTPADDING", (0, 0), (-1, -1), 10), ("TOPPADDING", (0, 0), (-1, -1), 7),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    return t


def ck(v):
    return f'<font name="Uni" color="{OK.hexval()}">✓</font>' if v else f'<font name="Uni" color="{BAD.hexval()}">✗</font>'


def br(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


# ---------------------------------------------------------------- dados
MODOS = ["llm_puro", "so_busca", "rag_llm", "rag_otimizado"]
NOMES = ["T1 · Só LLM", "T2 · Só busca", "T3 · RAG + LLM", "T4 · RAG otimizado"]
D = {m: json.loads((REL / f"comparacao_{m}.json").read_text(encoding="utf-8")) for m in MODOS}
R = {m: D[m]["resumo"] for m in MODOS}
PQ = {m: {r["id"]: r for r in D[m]["perguntas"]} for m in MODOS}
PORT = json.loads((REL / "comparacao_porteiro.json").read_text(encoding="utf-8"))
TAU = PORT["tau"]


# ---------------------------------------------------------------- desenhos
def caixa_d(d, x, y, w, h, titulo, sub, cor, ativo=True):
    fundo = colors.white if ativo else colors.HexColor("#f1f0ec")
    d.add(Rect(x, y, w, h, rx=6, ry=6, fillColor=fundo, strokeColor=cor if ativo else GRID, strokeWidth=1.4))
    d.add(String(x + w / 2, y + h - 15, titulo, fontName="Arial-B", fontSize=9.2,
                 fillColor=INK if ativo else MUTED, textAnchor="middle"))
    for i, linha in enumerate(sub):
        d.add(String(x + w / 2, y + h - 28 - 10.5 * i, linha, fontName="Arial", fontSize=7.6,
                     fillColor=INK2 if ativo else MUTED, textAnchor="middle"))


def seta(d, x1, y, x2, cor=INK2):
    d.add(Line(x1, y, x2 - 5, y, strokeColor=cor, strokeWidth=1.2))
    d.add(Polygon([x2, y, x2 - 6, y + 3.5, x2 - 6, y - 3.5], fillColor=cor, strokeColor=cor))


def diagrama_rag():
    w, h = W, 96
    d = Drawing(w, h)
    bw, bh, gap = 82, 56, (w - 5 * 82) / 4
    y = 22
    itens = [("Pergunta", ["do estudante"], INK2),
             ("Porteiro", ["a pergunta está", "perto da base?"], S2),
             ("Busca", ["acha os 5 trechos", "mais parecidos"], S1),
             ("LLM", ["lê os trechos e", "escreve a resposta"], S3),
             ("Resposta", ["com a fonte", "[artigo_NN]"], INK2)]
    for i, (t, s, c) in enumerate(itens):
        x = i * (bw + gap)
        caixa_d(d, x, y, bw, bh, t, s, c)
        if i < 4:
            seta(d, x + bw + 2, y + bh / 2, x + bw + gap - 2)
    d.add(String(1 * (bw + gap) + bw / 2, 8, "se estiver longe: recusa na hora", fontName="Arial-I",
                 fontSize=7.2, fillColor=INK2, textAnchor="middle"))
    d.add(String(2 * (bw + gap) + bw / 2, 8, "20 artigos científicos", fontName="Arial-I", fontSize=7.2,
                 fillColor=INK2, textAnchor="middle"))
    d.add(String(0, h - 8, "Como o StudyBuddy responde (configuração escolhida: T3 + porteiro)",
                 fontName="Arial-B", fontSize=8.5, fillColor=INK2))
    return d


def diagrama_testes():
    linhas = [("T1 · Só LLM", (False, False, True), "o LLM responde de memória"),
              ("T2 · Só busca", (True, True, False), "o estudante recebe 5 trechos crus"),
              ("T3 · RAG + LLM", (False, True, True), "busca + LLM (prompt original)"),
              ("T4 · RAG otimizado", (True, True, True), "porteiro + busca + LLM (prompt novo)")]
    lh, w = 46, W
    d = Drawing(w, lh * len(linhas) + 22)
    cols = [("Porteiro", S2), ("Busca", S1), ("LLM", S3)]
    x0, bw, bh = 112, 58, 28
    for j, (n, c) in enumerate(cols):
        d.add(String(x0 + j * (bw + 22) + bw / 2, lh * len(linhas) + 8, n, fontName="Arial-B", fontSize=8.5,
                     fillColor=INK2, textAnchor="middle"))
    for i, (nome, ativos, desc) in enumerate(linhas):
        y = lh * (len(linhas) - 1 - i) + 6
        d.add(String(0, y + bh / 2 - 3, nome, fontName="Arial-B", fontSize=9.2, fillColor=INK))
        for j, (n, c) in enumerate(cols):
            x = x0 + j * (bw + 22)
            a = ativos[j]
            d.add(Rect(x, y, bw, bh, rx=5, ry=5, fillColor=colors.white if a else colors.HexColor("#f1f0ec"),
                       strokeColor=c if a else GRID, strokeWidth=1.3, strokeDashArray=None if a else [3, 2]))
            d.add(String(x + bw / 2, y + bh / 2 - 3.5, "usa" if a else "não usa", fontName="Arial-B" if a else "Arial",
                         fontSize=8.5, fillColor=INK if a else MUTED, textAnchor="middle"))
            if j < 2:
                seta(d, x + bw + 2, y + bh / 2, x + bw + 20, GRID if not (a and ativos[j + 1]) else INK2)
        d.add(String(x0 + 3 * (bw + 22) - 6, y + bh / 2 - 3, desc, fontName="Arial", fontSize=8.3, fillColor=INK2))
    return d


def grafico_barras_h(valores, rotulos, titulo, maxv=1.0, fmt=lambda v: br(v), cor=S1, destaque=None):
    lh, w = 30, W
    x0, x1 = 120, w - 46
    d = Drawing(w, lh * len(valores) + 34)
    d.add(String(0, lh * len(valores) + 22, titulo, fontName="Arial-B", fontSize=9, fillColor=INK2))
    for k in range(0, 11, 2):
        x = x0 + (x1 - x0) * k / 10
        d.add(Line(x, 12, x, lh * len(valores) + 6, strokeColor=GRID, strokeWidth=0.5))
        d.add(String(x, 1, br(maxv * k / 10, 1), fontName="Arial", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    for i, (v, r) in enumerate(zip(valores, rotulos)):
        y = lh * (len(valores) - 1 - i) + 16
        d.add(String(0, y + 3, r, fontName="Arial-B" if i == destaque else "Arial", fontSize=8.8, fillColor=INK))
        largura = (x1 - x0) * v / maxv
        d.add(Rect(x0, y - 2, max(largura, 1.5), 14, rx=3, ry=3, fillColor=cor if (destaque is None or i == destaque) else
                   colors.HexColor("#9cc0ec"), strokeColor=None))
        d.add(String(x0 + largura + 5, y + 2, fmt(v), fontName="Arial-B", fontSize=8.5, fillColor=INK))
    return d


def grafico_agrupado():
    series = [("Completude", "completude", S1), ("Fidelidade à base", "fidelidade_base", S2),
              ("Relevância", "relevancia", S3)]
    w, h = W, 205
    d = Drawing(w, h)
    x0, y0, y1 = 30, 26, h - 40
    gw = (w - x0) / 4
    bw = 26
    for k in range(0, 6):
        y = y0 + (y1 - y0) * k / 5
        d.add(Line(x0, y, w, y, strokeColor=GRID, strokeWidth=0.5))
        d.add(String(x0 - 5, y - 2.5, br(k / 5, 1), fontName="Arial", fontSize=7, fillColor=MUTED, textAnchor="end"))
    for i, m in enumerate(MODOS):
        gx = x0 + gw * i + (gw - 3 * bw - 4) / 2
        for j, (_, ch, c) in enumerate(series):
            v = R[m][ch]
            x = gx + j * (bw + 2)
            hh = (y1 - y0) * v
            d.add(Rect(x, y0, bw, hh, fillColor=c, strokeColor=None))
            d.add(String(x + bw / 2, y0 + hh + 3, br(v), fontName="Arial", fontSize=7, fillColor=INK2, textAnchor="middle"))
        d.add(String(x0 + gw * i + gw / 2, y0 - 14, NOMES[i], fontName="Arial-B", fontSize=8.4, fillColor=INK,
                     textAnchor="middle"))
    lx = x0
    for nome, _, c in series:
        d.add(Rect(lx, h - 14, 9, 9, rx=2, ry=2, fillColor=c, strokeColor=None))
        d.add(String(lx + 13, h - 12.5, nome, fontName="Arial", fontSize=8.3, fillColor=INK))
        lx += 13 + pdfmetrics.stringWidth(nome, "Arial", 8.3) + 18
    d.add(String(w, h - 12.5, "* T2: fidelidade = 1,00 por construção", fontName="Arial-I", fontSize=7.2,
                 fillColor=INK2, textAnchor="end"))
    return d


def grafico_porteiro():
    w, h = W, 175
    d = Drawing(w, h)
    x0, x1 = 128, w - 10
    lo, hi = 0.45, 1.5
    X = lambda v: x0 + (x1 - x0) * (v - lo) / (hi - lo)
    linhas = [("Paráfrases (calibração)", PORT["d1_variacoes"], S1, "36 perguntas válidas reescritas"),
              ("Perguntas válidas (teste)", list(PORT["d1_em_escopo"].values()), S1, "18 perguntas"),
              ("Fora de escopo (teste)", list(PORT["d1_fora"].values()), S2, "10 perguntas")]
    ys = [128, 88, 48]
    for k in range(5, 16):
        v = k / 10
        d.add(Line(X(v), 28, X(v), 146, strokeColor=GRID, strokeWidth=0.5))
        d.add(String(X(v), 17, br(v, 1), fontName="Arial", fontSize=7, fillColor=MUTED, textAnchor="middle"))
    d.add(String((x0 + x1) / 2, 4, "distância até o trecho mais parecido da base  (menor = mais perto do assunto)",
                 fontName="Arial", fontSize=7.4, fillColor=INK2, textAnchor="middle"))
    for (nome, vals, c, sub), y in zip(linhas, ys):
        d.add(String(0, y + 1, nome, fontName="Arial-B", fontSize=8.4, fillColor=INK))
        d.add(String(0, y - 10, sub, fontName="Arial", fontSize=7.3, fillColor=INK2))
        for v in vals:
            d.add(Circle(X(v), y + 3, 4.2, fillColor=c, strokeColor=colors.white, strokeWidth=1.2, fillOpacity=0.85))
    xt = X(TAU)
    d.add(Line(xt, 26, xt, 152, strokeColor=INK, strokeWidth=1.3, strokeDashArray=[4, 3]))
    d.add(String(xt + 4, 156, f"corte do porteiro = {br(TAU, 3)}", fontName="Arial-B", fontSize=8, fillColor=INK))
    d.add(String(xt + 4, 147, "à direita: recusa sem chamar o LLM", fontName="Arial", fontSize=7.3, fillColor=INK2))
    d.add(String(xt - 4, 147, "à esquerda: segue para a busca", fontName="Arial", fontSize=7.3, fillColor=INK2,
                 textAnchor="end"))
    a, b = X(0.9155), X(1.0112)
    d.add(Line(a, 37, a, 34, strokeColor=INK2, strokeWidth=0.8))
    d.add(Line(b, 37, b, 34, strokeColor=INK2, strokeWidth=0.8))
    d.add(Line(a, 34, b, 34, strokeColor=INK2, strokeWidth=0.8))
    d.add(String(a - 4, 30, "passam: emagrecer, café, cadeira, remédio", fontName="Arial-I", fontSize=7,
                 fillColor=INK2, textAnchor="end"))
    return d


# ---------------------------------------------------------------- página
def cabecalho(canvas, doc):
    canvas.saveState()
    if doc.page > 1:
        canvas.setFont("Arial", 7.8)
        canvas.setFillColor(MUTED)
        canvas.drawString(2.1 * cm, A4[1] - 1.25 * cm, "StudyBuddy · Experimento de métricas (LLM x RAG)")
        canvas.drawRightString(A4[0] - 2.1 * cm, 1.2 * cm, f"{doc.page}")
        canvas.setStrokeColor(GRID)
        canvas.setLineWidth(0.5)
        canvas.line(2.1 * cm, A4[1] - 1.4 * cm, A4[0] - 2.1 * cm, A4[1] - 1.4 * cm)
    canvas.restoreState()


def capa(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(colors.HexColor("#14325c"))
    canvas.rect(0, A4[1] - 11 * cm, A4[0], 11 * cm, fill=1, stroke=0)
    canvas.setFillColor(colors.white)
    canvas.setFont("Arial-B", 9.5)
    canvas.drawString(2.1 * cm, A4[1] - 3 * cm, "RESIDÊNCIA EM IA · CHALLENGE 2 · GRUPO 12 · SEMANA 3")
    canvas.setFont("Arial-B", 30)
    canvas.drawString(2.1 * cm, A4[1] - 5.1 * cm, "Experimento de métricas")
    canvas.drawString(2.1 * cm, A4[1] - 6.4 * cm, "do StudyBuddy")
    canvas.setFont("Arial", 13.5)
    canvas.setFillColor(colors.HexColor("#cfe0f6"))
    canvas.drawString(2.1 * cm, A4[1] - 7.9 * cm, "LLM sozinho, busca sozinha ou os dois juntos?")
    canvas.drawString(2.1 * cm, A4[1] - 8.6 * cm, "Como medimos, o que descobrimos e o que escolhemos.")
    canvas.restoreState()


doc = BaseDocTemplate(str(SAIDA), pagesize=A4, leftMargin=2.1 * cm, rightMargin=2.1 * cm,
                      topMargin=2 * cm, bottomMargin=1.9 * cm, title="Experimento de métricas do StudyBuddy",
                      author="Grupo 12", subject="LLM x RAG: metodologia, métricas e resultados")
fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
fr_capa = Frame(doc.leftMargin, doc.bottomMargin, doc.width, A4[1] - 11 * cm - 2.6 * cm, id="fc")
doc.addPageTemplates([PageTemplate("capa", [fr_capa], onPage=capa),
                      PageTemplate("normal", [fr], onPage=cabecalho)])

S = []
from reportlab.platypus import NextPageTemplate

# ================================================================= CAPA
S.append(NextPageTemplate("normal"))
S.append(P("Em uma frase", "kicker"))
S.append(caixa([P("A melhor forma de o StudyBuddy responder é <b>buscar os trechos certos nos 20 artigos e "
                  "deixar o LLM escrever a resposta a partir deles</b> (teste T3), com um <b>“porteiro”</b> "
                  "que recusa na hora as perguntas que não têm nada a ver com a base, sem gastar o LLM.", "box")],
               fundo=ACC_BG, borda=S1, pad=12))
S.append(Spacer(1, 14))
S.append(P("O que você vai encontrar aqui", "h2"))
S += bullets([
    "<b>1.</b> Resumo em uma página, para quem tem dois minutos",
    "<b>2.</b> O que é o StudyBuddy e o que é RAG, sem jargão",
    "<b>3.</b> As peças do sistema: cada modelo e o papel dele",
    "<b>4.</b> Como chegamos até aqui (o que as semanas anteriores já tinham decidido)",
    "<b>5.</b> A “prova”: as 28 perguntas usadas para medir",
    "<b>6.</b> As métricas, uma por uma, com exemplos",
    "<b>7.</b> O juiz: quem corrige as respostas, e como conferimos o próprio juiz",
    "<b>8.</b> Os 4 testes e o porteiro",
    "<b>9.</b> A tabela final e os resultados de cada teste (T1 a T4)",
    "<b>10.</b> Tudo junto: a decisão, os cuidados ao ler os números e os próximos passos",
    "<b>Anexos:</b> pergunta a pergunta, glossário e onde estão os arquivos",
])
S.append(Spacer(1, 12))
S.append(P("Configuração medida: tudo rodando localmente (Apple M4, 24 GB), com o gerador <b>qwen3:8b</b>, o juiz "
           "<b>gemma3:12b</b> e a busca com <b>bge-m3</b>. Dados da branch <font name='Arial-I'>metricas-experimento</font>, "
           "avaliação final de 12/11/2026.", "small"))
S.append(PageBreak())

# ================================================================= 1. RESUMO
S.append(P("1 · Resumo para quem tem dois minutos", "h1"))
S.append(P("<b>A pergunta.</b> O StudyBuddy é um mentor de estudos que responde com base em 20 artigos científicos. "
           "Ele junta duas peças: uma <b>busca</b>, que encontra os trechos certos dos artigos, e um <b>LLM</b> "
           "(um modelo de linguagem, como o ChatGPT), que escreve a resposta. Queríamos saber: <b>precisamos mesmo "
           "das duas peças?</b> E qual é a melhor forma de combiná-las?"))
S.append(P("<b>O que fizemos.</b> Montamos 4 versões do sistema, tirando ou trocando uma peça por vez, e aplicamos "
           "a mesma “prova” de 28 perguntas a todas, corrigida pelo mesmo “professor” automático."))
S.append(tabela([
    ["Teste", "O que é", "Nota final*", "Veredito"],
    ["T1 · Só LLM", "O LLM responde sozinho, de memória, sem os artigos", "0,32", f"{ck(0)} Reprovado"],
    ["T2 · Só busca", "O estudante recebe 5 trechos dos artigos, sem ninguém escrever a resposta", "0,75",
     f"{ck(0)} Reprovado como produto (serve como plano B)"],
    ["T3 · RAG + LLM", "A busca acha os trechos e o LLM escreve a resposta a partir deles", "<b>0,93</b>",
     f"{ck(1)} <b>Aprovado: o melhor</b>"],
    ["T4 · RAG otimizado", "T3 + porteiro + instruções novas para o LLM", "0,75",
     f"{ck(0)} Reprovado como pacote (o porteiro fica)"],
], [0.2, 0.42, 0.13, 0.25], destaque_lin=3))
S.append(P("* Nota final = <b>taxa de sucesso</b>: em quantas das 28 perguntas o resultado realmente serviu ao "
           "estudante (explicada na seção 6).", "cap"))
S.append(P("<b>O que aprendemos.</b>"))
S += bullets([
    "<b>Sem os artigos (T1), o LLM vira um chatbot genérico.</b> Escreve três vezes mais, cobre menos do que os "
    "artigos recomendam e nunca diz de onde tirou a informação. Às vezes inventa a fonte, como “[Fonte: Atomic Habits]”.",
    "<b>Sem o LLM (T2), ninguém sabe dizer “não sei”.</b> É instantâneo e 100% fiel aos artigos, mas entregou "
    "trechos sobre sono para quem perguntou sobre <i>remédio</i> para dormir.",
    "<b>Juntos (T3), as duas peças se completam:</b> a busca traz a evidência, e o LLM lê, filtra, organiza e "
    "recusa o que não deve responder.",
    "<b>No T4, uma mudança deu certo e a outra não.</b> O porteiro barrou 6 das 10 perguntas fora do assunto em "
    "0,14 segundo, sem chamar o LLM. As instruções novas, porém, pioraram as respostas em 8 perguntas e melhoraram "
    "só em 2.",
])
S.append(Spacer(1, 4))
S.append(caixa([P("<b>Escolha final: T3 + porteiro.</b> Fica a qualidade das respostas do T3 (nota 0,93) e a "
                  "economia do porteiro do T4 (6 de 28 chamadas ao LLM a menos e recusas em 0,14 s em vez de 5,8 s).",
                  "box")], fundo=OK_BG, borda=OK, pad=12))
S.append(PageBreak())

# ================================================================= 2. O QUE É RAG
S.append(P("2 · O StudyBuddy e o RAG, sem jargão", "h1"))
S.append(P("Pense numa <b>prova com consulta</b>. Um aluno que só confia na memória (o LLM sozinho) escreve bem, mas "
           "pode misturar coisas, esquecer o que o professor pediu e não consegue mostrar de onde tirou cada ideia. "
           "Um aluno que só copia trechos do livro (a busca sozinha) nunca erra o texto, mas entrega recortes soltos "
           "e às vezes copia a página errada. O melhor aluno <b>consulta o livro e depois escreve com as próprias "
           "palavras, citando a página</b>. Isso é o <b>RAG</b>."))
S.append(P("RAG vem do inglês <i>Retrieval-Augmented Generation</i>, ou “geração apoiada por busca”. Na prática, "
           "são duas etapas:"))
S += bullets([
    "<b>Buscar (o bibliotecário):</b> diante da pergunta, encontrar nos 20 artigos os trechos que mais têm a ver com ela.",
    "<b>Gerar (o redator):</b> entregar esses trechos ao LLM com uma instrução clara: “responda <i>só</i> com base "
    "nisto, cite a fonte e, se a resposta não estiver aqui, diga que não encontrou”.",
])
S.append(Spacer(1, 6))
S.append(diagrama_rag())
S.append(P("Figura 1 · O caminho de uma pergunta na configuração escolhida. O porteiro é a novidade desta semana.", "cap"))
S.append(P("Por que isso importa para o StudyBuddy", "h2"))
S.append(P("O StudyBuddy promete <b>orientação baseada em evidência</b>. Um conselho inventado “com cara de "
           "científico” é o pior erro possível, porque o estudante confia nele. Por isso, não basta a resposta "
           "<i>parecer</i> boa: ela precisa vir dos artigos, mostrar de qual artigo veio e saber recusar o que não "
           "está na base. As métricas deste experimento existem para medir exatamente isso."))

# ================================================================= 3. MODELOS
S.append(CondPageBreak(9 * cm))
S.append(P("3 · As peças do sistema (os modelos)", "h1"))
S.append(P("Todos os modelos rodam <b>no próprio computador</b> (um Mac M4 com 24 GB), pelo programa Ollama. "
           "Não há custo de API nem envio de dados para fora. Os resultados valem para esta configuração."))
S.append(tabela([
    ["Peça", "Modelo", "O que faz, em palavras simples"],
    ["Base de conhecimento", "20 artigos (~14 mil caracteres)", "Os textos científicos sobre estudo e "
     "produtividade: repetição espaçada, sono, Pomodoro, hábitos etc."],
    ["Divisão em pedaços", "Pedaços de 500 caracteres, com 50 de sobreposição", "Os artigos são cortados em 49 "
     "pedaços. A sobreposição evita que uma ideia se perca bem no corte."],
    ["Embedding (a “impressão digital” do texto)", "bge-m3", "Transforma cada pedaço, e cada pergunta, numa lista de "
     "1.024 números que representa o <i>sentido</i> do texto. Textos com sentido parecido ficam com números "
     "parecidos, mesmo usando palavras diferentes."],
    ["Banco vetorial", "ChromaDB", "Guarda essas impressões digitais e encontra rapidamente as mais próximas da "
     "pergunta. É a “estante” do bibliotecário."],
    ["Quantos trechos entram (k)", "k = 5", "A busca entrega os 5 pedaços mais próximos ao LLM."],
    ["Gerador (o redator)", "qwen3:8b", "O LLM que escreve a resposta. Tem 8 bilhões de parâmetros, roda com "
     "“temperatura 0” (sempre a mesma resposta para a mesma entrada) e sem o modo de “pensar em voz alta”."],
    ["Juiz (o corretor)", "gemma3:12b", "Outro LLM, de <b>outra empresa e maior</b> que o gerador, que corrige as "
     "respostas. Ser diferente evita que o modelo “corrija a própria prova”."],
    ["Porteiro", "Um corte de distância (1,026)", "Uma regra simples, sem IA: se nem o trecho mais parecido está "
     "perto o bastante da pergunta, o sistema recusa antes de chamar o LLM."],
], [0.22, 0.24, 0.54]))

# ================================================================= 4. HISTÓRICO
S.append(CondPageBreak(8 * cm))
S.append(P("4 · Como chegamos até aqui", "h1"))
S.append(P("Este experimento é a terceira etapa. As duas anteriores já tinham escolhido, com medições, as peças da "
           "busca. Vale entender de onde elas vieram:"))
S.append(tabela([
    ["Etapa", "O que comparamos", "O que ganhou", "Por quê"],
    ["1 · Como buscar", "Busca por palavras iguais × busca por sentido (bge-m3 e qwen3-embedding)",
     "<b>bge-m3</b>", "Acerto do artigo certo em 1º lugar foi de 0,39 (palavras) para 0,94. A busca por palavras "
     "falhava em 6 de 9 perguntas escritas “com outras palavras”."],
    ["2 · Tamanho do pedaço", "300, 500 e 800 caracteres", "<b>500</b>", "Pedaços pequenos cortam a ideia ao meio; "
     "grandes misturam assuntos. O 500 teve o melhor resultado em pôr o artigo certo no topo (acerto em 1º lugar de 0,94)."],
    ["3 · Quantos trechos (k)", "k = 3 × k = 5", "<b>k = 5</b>", "Com mais contexto, a completude subiu de 0,81 para "
     "0,91, e nas perguntas que dependem de vários artigos, de 0,53 para 0,87."],
    ["<b>4 · Este experimento</b>", "Precisamos do LLM? Precisamos da busca? Qual combinação é melhor?",
     "<b>T3 + porteiro</b>", "É o que este documento explica."],
], [0.17, 0.3, 0.14, 0.39], destaque_lin=4))
S.append(Spacer(1, 6))
S.append(P("Até aqui, todas as medições usavam <b>busca + LLM</b>. Nunca tínhamos medido o que acontece sem uma das "
           "duas peças. Esse é o objetivo desta etapa: provar com números que cada peça se paga."))

# ================================================================= 5. GABARITO
S.append(PageBreak())
S.append(P("5 · A “prova”: 28 perguntas com gabarito", "h1"))
S.append(P("Para comparar versões do sistema, todas fazem <b>a mesma prova</b>. Escrevemos 28 perguntas no tom de um "
           "estudante real e, para cada uma, anotamos o <b>gabarito</b>: quais artigos respondem e quais são os "
           "<b>pontos-chave</b> que uma boa resposta precisa ter."))
S.append(tabela([
    ["Tipo", "Qtd.", "O que testa", "Exemplo"],
    ["Direta", "4", "A pergunta usa as mesmas palavras do artigo", "“Como funciona a técnica Pomodoro? Quanto tempo dura "
     "cada intervalo?”"],
    ["Parafraseada", "9", "A pergunta diz a mesma coisa com outras palavras", "“Por que eu releio o material várias "
     "vezes e mesmo assim não consigo lembrar na hora da prova?”"],
    ["Multi-fonte", "5", "A resposta precisa juntar dois ou mais artigos", "“Como organizar minha rotina de estudos "
     "trabalhando 8 horas por dia?”"],
    ["Fora de escopo", "10", "O sistema precisa <b>recusar</b>. Quatro são “perigosas”, porque parecem do tema",
     "“Qual o melhor remédio para dormir quando estou ansioso antes da prova?”"],
], [0.16, 0.07, 0.35, 0.42]))
S.append(Spacer(1, 6))
S.append(P("Exemplo de gabarito", "h3"))
S.append(caixa([P("<b>Pergunta q12:</b> “Como faço para sair do ‘vou estudar amanhã’ e realmente começar?”", "box"),
                P("<b>Artigo esperado:</b> artigo_08 (intenções de implementação)", "box"),
                P("<b>Pontos-chave:</b> (1) usar a estrutura “Se a situação X ocorrer, então farei Y”; "
                  "(2) definir com precisão <i>quando</i>, <i>onde</i> e <i>como</i> o estudo vai começar.", "box")]))
S.append(Spacer(1, 6))
S.append(P("Além das 28, cada pergunta válida ganhou <b>2 versões reescritas</b> (36 no total). Elas <b>não entram na "
           "nota</b>: servem só para ajustar o porteiro sem “colar” da prova (seção 8)."))
S.append(caixa("<b>Por que só 28 perguntas?</b> Cada resposta precisa ser corrigida pelo juiz, várias vezes, por "
               "métricas diferentes, num computador local. É um tamanho que permite rodar tudo em algumas horas e "
               "conferir os casos à mão. A contrapartida, discutida na seção 10, é que diferenças pequenas entre os "
               "testes podem ser só acaso.", fundo=WARN_BG, borda=WARN))

# ================================================================= 6. MÉTRICAS
S.append(PageBreak())
S.append(P("6 · As métricas, uma por uma", "h1"))
S.append(P("Uma métrica é só <b>uma pergunta objetiva com uma resposta em número</b>. Em vez de dizer “a resposta "
           "parece boa”, dizemos “cobriu 2 de 3 pontos-chave”. Organizamos as métricas pelas cinco coisas que o "
           "StudyBuddy precisa fazer, em ordem de importância."))
S.append(P("Algumas são medidas por <b>regra</b> (o computador confere sozinho, por exemplo, se apareceu "
           "“[Fonte: artigo_08]”). Outras precisam de interpretação e são medidas pelo <b>juiz</b> (seção 7)."))


def bloco_metricas(titulo, linhas):
    out = [P(titulo, "h2")]
    out.append(tabela([["Métrica", "A pergunta que ela responde", "Como é calculada", "Quem mede"], *linhas],
                      [0.19, 0.34, 0.35, 0.12]))
    return out


S += bloco_metricas("1º · Não inventar", [
    ["<b>Fidelidade à base</b> <font color='#2a78d6'>(nova)</font>", "O que a resposta afirma tem respaldo em "
     "algum dos 20 artigos?", "A resposta é quebrada em afirmações curtas. Fidelidade = afirmações com respaldo ÷ "
     "total de afirmações.", "Juiz"],
    ["Fidelidade ao contexto", "O que a resposta afirma está nos 5 trechos que o LLM recebeu?",
     "Igual, mas comparando só com os trechos recebidos.", "Juiz"],
])
S.append(P("<b>Por que criamos a fidelidade à base?</b> A fidelidade antiga não serve para comparar os 4 testes. No T1, "
           "o LLM não recebe trecho nenhum, então <i>tudo</i> seria “inventado”, até o que é verdade. No T2, a resposta "
           "<i>é</i> o próprio trecho, então a nota seria sempre 1,00. A fidelidade à base faz a mesma pergunta para "
           "todos: “isso está nos nossos artigos?”. Um conselho verdadeiro no mundo real, mas ausente dos artigos "
           "(como a “Matriz de Eisenhower”), conta como <b>sem respaldo</b>, porque o StudyBuddy promete responder com "
           "base <i>nos artigos</i>.", "small"))
S += bloco_metricas("2º · Deixar o estudante conferir", [
    ["Cita a fonte", "A resposta indica de qual artigo veio, no formato [Fonte: artigo_NN]?", "Sim ou não, por "
     "resposta.", "Regra"],
    ["Cita um artigo correto", "Pelo menos um dos artigos citados é um dos que o gabarito espera?", "Sim ou não.",
     "Regra"],
    ["Precisão da citação", "Os artigos citados estavam mesmo entre os trechos recebidos? (ou o LLM citou algo que "
     "não leu?)", "Citados que foram recebidos ÷ citados.", "Regra"],
])
S += bloco_metricas("3º · Saber dizer “não sei”", [
    ["Recusa correta", "Nas 10 perguntas fora do assunto, o sistema recusou?", "Recusas ÷ 10.", "Regra"],
    ["Recusa indevida", "Nas 18 perguntas válidas, ele recusou sem motivo?", "Recusas ÷ 18. Quanto menor, melhor.",
     "Regra"],
])
S += bloco_metricas("4º · Responder o que importa", [
    ["Completude", "A resposta cobre os pontos-chave do gabarito?", "Pontos cobertos ÷ pontos do gabarito. O juiz "
     "precisa copiar a frase da resposta que cobre cada ponto.", "Juiz"],
    ["<b>Relevância</b> <font color='#2a78d6'>(nova)</font>", "Cada afirmação ajuda a responder <i>esta</i> pergunta, "
     "ou é conselho genérico e fora do assunto?", "Afirmações relevantes ÷ total de afirmações.", "Juiz"],
])
S += bloco_metricas("5º · Custo", [
    ["Latência", "Quanto tempo o estudante espera?", "Mediana (p50) do tempo de ponta a ponta: metade das perguntas "
     "foi mais rápida que isso, metade mais lenta.", "Relógio"],
    ["Tokens", "Quanto texto foi lido e escrito?", "Tokens são pedaços de palavra (~4 letras). Mais tokens = mais "
     "lento e, numa API, mais caro.", "Contagem"],
    ["Chamadas ao LLM", "Quantas vezes o LLM precisou ser acionado?", "Contagem, de 28.", "Contagem"],
])
S.append(Spacer(1, 4))
S.append(P("A nota final: taxa de sucesso <font color='#2a78d6'>(nova)</font>", "h2"))
S.append(P("Cada métrica olha um pedaço. Para o produto, interessa uma pergunta mais direta: <b>a resposta serviu ao "
           "estudante?</b> A taxa de sucesso responde isso, pergunta por pergunta, com uma regra explícita:"))
S.append(tabela([
    ["Tipo de pergunta", "Conta como sucesso se..."],
    ["Fora do assunto (10)", "o sistema <b>recusou</b>"],
    ["Válida (18)", "<b>não recusou</b> E cobriu <b>pelo menos 2/3</b> dos pontos-chave E <b>pelo menos 80%</b> das "
     "afirmações têm respaldo nos artigos E <b>citou um artigo correto</b>"],
], [0.28, 0.72]))
S.append(Spacer(1, 4))
S.append(P("A taxa de sucesso é a fração das 28 perguntas que passaram. Ela junta as quatro exigências do produto "
           "numa só: <b>responder, responder o suficiente, não inventar e deixar conferir</b>. As outras métricas "
           "continuam no relatório para explicar <i>por que</i> um teste falhou.", "body"))
S.append(caixa("<b>Um cuidado:</b> a taxa de sucesso não inclui a relevância. Isso favorece o T2 (só busca): os "
               "trechos crus têm fidelidade e citação perfeitas por construção, mesmo quando metade do texto não "
               "tem a ver com a pergunta. Por isso olhamos a relevância separadamente.", fundo=WARN_BG, borda=WARN))

# ================================================================= 7. JUIZ
S.append(PageBreak())
S.append(P("7 · O juiz: quem corrige a prova", "h1"))
S.append(P("Algumas perguntas não dá para conferir com uma regra simples. “Essa frase diz a mesma coisa que o "
           "artigo, com outras palavras?” exige ler e interpretar. Corrigir 28 respostas × 4 testes × várias métricas "
           "à mão levaria dias. Por isso usamos um <b>LLM-juiz</b>: outro modelo de linguagem, com instruções de "
           "corretor rigoroso."))
S.append(P("Três cuidados para o juiz ser confiável", "h2"))
S += bullets([
    "<b>Juiz diferente do gerador.</b> O gerador é o qwen3:8b; o juiz é o gemma3:12b, de outra empresa e maior. "
    "Um modelo tende a achar boas as próprias respostas.",
    "<b>Juiz não dá nota, só vereditos.</b> Ele diz, para cada afirmação, “tem respaldo: sim/não”. A conta "
    "(sim ÷ total) é feita pelo nosso código. Isso evita notas “no chute”.",
    "<b>Juiz precisa mostrar a prova.</b> Para dizer que um ponto foi coberto, ou que uma afirmação está nos artigos, "
    "ele precisa <b>copiar a frase</b>. O nosso código confere se essa frase existe mesmo. Se não existe, o “sim” vira "
    "“não”.",
])
S.append(P("Conferimos o juiz antes de confiar nele", "h2"))
S.append(P("O juiz é um instrumento de medida, como uma balança. Antes de usar os números, testamos a balança com "
           "casos que sabíamos a resposta. A fidelidade à base precisou de <b>três versões</b>:"))
S.append(tabela([
    ["Versão", "Como julgava", "O que deu errado", "Caso concreto"],
    ["v1", "Todos os artigos + todas as afirmações de uma vez", f"{ck(0)} <b>Generoso demais</b>: aprovava pelo tema "
     "parecido", "Aprovou “use a técnica SMART” e “Matriz de Eisenhower”. Uma busca no texto dos 20 artigos não "
     "encontra nenhum dos dois."],
    ["v2", "Igual à v1, mas exigindo a frase copiada como prova", f"{ck(0)} <b>Severo demais</b>: com ~15 afirmações "
     "de uma vez, o juiz “cansava” e reprovava sem procurar", "Reprovou “Pomodoro de 25 + 5 min” e “2 a 4 horas de "
     "trabalho profundo”, que estão <b>literalmente</b> nos artigos."],
    ["<b>v3 (usada)</b>", "<b>Uma afirmação por vez</b>, ainda exigindo a frase copiada", f"{ck(1)} Acerta os casos "
     "acima. Ainda é <b>um pouco generoso</b> com afirmações vagas", "Aprovou “exemplos: escrever, programar...” citando "
     "um trecho sem relação."],
], [0.12, 0.27, 0.3, 0.31], destaque_lin=3))
S.append(Spacer(1, 6))
S.append(caixa("<b>Lição de método:</b> sem conferir o juiz, a fidelidade do LLM sem RAG (T1) teria saído como 0,53 "
               "(v1) ou 0,195 (v2), dois números errados em direções opostas. Com a v3, ela foi 0,81. Como a v3 ainda "
               "tende a ser generosa, esse 0,81 deve ser lido como um <b>teto</b>: o valor real provavelmente é um "
               "pouco menor. Uma conferência manual por amostragem está prevista, mas <b>ainda não foi feita</b>.",
               fundo=ACC_BG, borda=S1))

# ================================================================= 8. OS 4 TESTES
S.append(PageBreak())
S.append(P("8 · Os 4 testes", "h1"))
S.append(P("A regra de ouro de um experimento justo: <b>mudar uma coisa por vez</b>. Todos os testes usam os mesmos "
           "20 artigos, as mesmas 28 perguntas, o mesmo gerador, o mesmo juiz e o mesmo computador. Só muda qual "
           "peça está ligada."))
S.append(diagrama_testes())
S.append(P("Figura 2 · Quais peças cada teste usa. Caixas tracejadas = peça desligada.", "cap"))
S.append(tabela([
    ["Teste", "O que muda em relação ao T3", "A pergunta que ele responde"],
    ["<b>T1 · Só LLM</b>", "Tira a busca. O LLM recebe só a pergunta, com o mesmo papel (“Mentor de Estudos”), o "
     "mesmo formato e a mesma chance de recusar.", "Os artigos fazem diferença, ou um LLM bom já basta?"],
    ["<b>T2 · Só busca</b>", "Tira o LLM. O estudante recebe os 5 trechos crus, cada um com a fonte. Quem recusa é o "
     "porteiro.", "O LLM faz diferença, ou basta mostrar os trechos?"],
    ["<b>T3 · RAG + LLM</b>", "Nada: é a melhor configuração da semana passada (k = 5, instruções do notebook).",
     "É a referência."],
    ["<b>T4 · RAG otimizado</b>", "Duas mudanças: (1) porteiro antes do LLM; (2) instruções novas para atacar um "
     "problema visto na q12.", "Dá para melhorar o T3?"],
], [0.19, 0.48, 0.33]))
S.append(Spacer(1, 6))
S.append(P("As instruções novas do T4", "h3"))
S.append(P("No T3, ao receber 5 trechos, o LLM às vezes “se perdia” num trecho fora do assunto e esquecia o ponto "
           "principal (na q12, deixou de fora “defina quando, onde e como”). Chamamos isso de <b>diluição</b>. Para "
           "corrigir, o T4 manteve as regras de não inventar e de recusar e acrescentou:"))
S.append(P("3. Comece com 1 ou 2 frases que respondem diretamente ao que foi perguntado. · 4. Liste as "
           "recomendações MAIS ligadas à pergunta, priorizando instruções concretas. · 5. Ignore os fragmentos que "
           "não tratam do que foi perguntado. · 6. Termine cada tópico com a fonte.", "quote"))

S.append(CondPageBreak(10 * cm))
S.append(P("O porteiro: recusar sem gastar o LLM", "h2"))
S.append(P("Toda busca devolve uma <b>distância</b>: o quanto o trecho mais parecido está longe da pergunta. "
           "“Capital da Austrália” fica muito longe de qualquer artigo sobre estudo; “Pomodoro” fica muito perto. "
           "O porteiro usa um <b>corte</b>: se a distância passa dele, o sistema recusa na hora."))
S.append(P("O cuidado é <b>onde pôr o corte</b>. Se escolhêssemos olhando as 28 perguntas da prova, o corte ficaria "
           "perfeito na prova e falharia com perguntas novas, como decorar o gabarito. Por isso usamos as <b>36 "
           "versões reescritas</b>, que não entram na nota: o corte é a maior distância entre elas (1,026). Assim, "
           "nenhuma pergunta válida escrita de outro jeito é recusada."))
S.append(grafico_porteiro())
S.append(P("Figura 3 · Cada ponto é uma pergunta. Todas as válidas ficam à esquerda do corte. Das 10 fora do assunto, "
           "6 ficam à direita e são barradas; as 4 que passam (café, emagrecer, cadeira, remédio) são as "
           "“perigosas”, próximas do tema.", "cap"))
S.append(P("É uma troca consciente: <b>recusar uma pergunta válida é pior do que deixar passar uma fora do assunto</b>, "
           "porque, com o LLM ligado, ainda há uma segunda barreira (a instrução “se não estiver no contexto, "
           "recuse”). No T2, sem LLM, não há segunda barreira."))

# ================================================================= 9. RESULTADOS
S.append(PageBreak())
S.append(P("9 · Resultados", "h1"))
S.append(P("A tabela final", "h2"))


def v(m, ch, n=2):
    x = R[m].get(ch)
    return "—" if x is None else br(x, n)


lin = [["Métrica", *NOMES],
       ["<b>Taxa de sucesso (28)</b>", *[f"<b>{v(m, 'taxa_sucesso')}</b>" for m in MODOS]],
       ["Completude", *[v(m, "completude") for m in MODOS]],
       ["  · diretas (4)", *[br(R[m]["por_tipo"]["direta"]["completude"]) for m in MODOS]],
       ["  · parafraseadas (9)", *[br(R[m]["por_tipo"]["parafraseada"]["completude"]) for m in MODOS]],
       ["  · multi-fonte (5)", *[br(R[m]["por_tipo"]["multi_fonte"]["completude"]) for m in MODOS]],
       ["Fidelidade à base", "0,81 ‡", "1,00 §", "0,98", "0,86"],
       ["Relevância", *[v(m, "relevancia") for m in MODOS]],
       ["Cita um artigo correto", *[v(m, "cita_esperada") for m in MODOS]],
       ["Recusa correta (de 10)", "9", "6", "10", "10"],
       ["Recusa indevida (de 18)", "0", "0", "0", "0"],
       ["Latência p50 das respostas", "54,7 s", "0,15 s", "20,8 s", "29,3 s ¶"],
       ["Latência p50 das recusas", "2,1 s", "0,15 s", "5,8 s", "0,14 s"],
       ["Tokens por resposta", *[str(R[m]["tokens_resposta"]) for m in MODOS]],
       ["Tokens de contexto", *[str(R[m]["tokens_contexto"]) for m in MODOS]],
       ["Chamadas ao LLM (de 28)", *[str(R[m]["chamadas_llm"]) for m in MODOS]]]
S.append(tabela(lin, [0.32, 0.17, 0.17, 0.17, 0.17], destaque_col=3, alinhar_num=True))
S.append(P("‡ Medida com o juiz v3, que tende a ser generoso: é um teto. § Por construção: a resposta é o próprio "
           "texto dos artigos. Nos T3 e T4, a fidelidade à base é um piso (o que está nos trechos está nos artigos). "
           "¶ T3 e T4 foram gerados em dias diferentes; essa diferença de tempo é inconclusiva (seção 10). "
           "Coluna em azul: o teste vencedor.", "cap"))
S.append(grafico_barras_h([R[m]["taxa_sucesso"] for m in MODOS], NOMES,
                          "Taxa de sucesso: em quantas das 28 perguntas o resultado serviu ao estudante", destaque=2))
S.append(P("Figura 4 · O T3 é o único acima de 0,9. T2 e T4 empatam em 0,75, mas por motivos opostos: o T2 erra nas "
           "recusas, e o T4 erra na cobertura.", "cap"))
S.append(CondPageBreak(9 * cm))
S.append(grafico_agrupado())
S.append(P("Figura 5 · Qualidade das respostas em escopo (18 perguntas). O T3 é o único que fica alto nas três ao mesmo "
           "tempo. O T2 cobre bem, mas entrega muito texto fora do assunto (relevância 0,57).", "cap"))
S.append(P("Como ler os números", "h3"))
S += bullets([
    "Todas as notas vão de 0 a 1. Pense como porcentagem: 0,91 = 91%.",
    "Com 18 perguntas válidas, cada pergunta vale cerca de 0,056. Diferenças menores que ~0,1 podem ser acaso.",
    "Tempo e tokens: quanto menor, melhor. O resto: quanto maior, melhor (menos a recusa indevida).",
])

# ================================================================= RESULTADO POR TESTE
S.append(CondPageBreak(14 * cm))
S.append(P("Resultado de cada etapa", "h1"))

# T1
S.append(P("T1 · Só LLM (sem os artigos)", "h2"))
S.append(selo("Reprovado. Nota 0,32: nenhuma das 18 perguntas válidas passou.", "bad"))
S.append(Spacer(1, 6))
S.append(tabela([["Sucesso", "Completude", "Fidelidade à base", "Cita artigo", "Recusa correta", "Tempo"],
                 ["0,32", "0,59", "0,81 (teto)", "0,00", "9 de 10", "54,7 s"]],
                [1 / 6] * 6, alinhar_num="all", zebra=False))
S.append(P("O que mudou e por quê", "h3"))
S += bullets([
    "<b>Cobre menos do que importa (0,59 contra 0,91 do T3).</b> Escreve quase 3 vezes mais (844 tokens contra 315), "
    "mas com conselhos genéricos de produtividade. <i>Por quê:</i> o modelo responde com uma “média da internet” e não "
    "tem como saber o que os nossos 20 artigos recomendam.",
    "<b>Nunca cita um artigo.</b> Quando tenta citar, inventa a fonte: “[Fonte: <i>Atomic Habits</i> de James Clear]”, "
    "“[Fonte: Técnicas de Gestão de Tempo e Metas SMART]”. O estudante não tem como conferir.",
    "<b>Recusa bem, mas pelo motivo errado.</b> Recusou 9 das 10 perguntas pelo <i>tema</i> (“não é sobre estudos”), e "
    "não por falta de evidência. Por isso respondeu a q20 (“quanto café posso tomar?”): é do tema, mas não está nos "
    "artigos. Só um sistema com base sabe fazer essa recusa.",
    "<b>É o mais lento</b> (54,7 s), justamente porque escreve muito mais.",
])
S.append(P("Exemplo real · q01, “Como organizar minha rotina trabalhando 8 horas por dia?”", "h3"))
S.append(P("“Use a técnica SMART... Priorize tarefas com base em importância e urgência (Matriz de Eisenhower)... "
           "Trabalhe em blocos de 90 minutos (técnica Pomodoro: 25 minutos de estudo + 5 de pausa).”", "quote"))
S.append(P("Nenhuma das duas técnicas está nos artigos, e “blocos de 90 minutos (Pomodoro: 25 + 5)” se contradiz. "
           "A recomendação central do artigo 19 (<b>blocos de 45 a 60 min antes do expediente</b>) não aparece."))
S.append(caixa("<b>O veredito não depende só da regra de citação.</b> Mesmo ignorando a exigência de citar um artigo, "
               "só 5 das 18 respostas passariam (cobertura ≥ 2/3 e fidelidade ≥ 0,8). Nas perguntas multi-fonte, a "
               "fidelidade cai para 0,63: quanto mais específica a pergunta, mais o LLM preenche com conhecimento "
               "genérico.", fundo=SURF))

# T2
S.append(CondPageBreak(12 * cm))
S.append(P("T2 · Só busca (sem o LLM)", "h2"))
S.append(selo("Reprovado como produto principal. Útil como plano B, quando o LLM estiver fora do ar.", "bad"))
S.append(Spacer(1, 6))
S.append(tabela([["Sucesso", "Completude", "Relevância", "Cita artigo", "Recusa correta", "Tempo"],
                 ["0,75", "0,87", "0,57", "1,00", "6 de 10", "0,15 s"]],
                [1 / 6] * 6, alinhar_num="all", zebra=False))
S.append(P("O que mudou e por quê", "h3"))
S += bullets([
    "<b>Instantâneo e sempre rastreável.</b> 0,15 s, mais de 100 vezes mais rápido que os testes com LLM. Cada trecho "
    "vem com a fonte e nada é inventado, porque o texto <i>é</i> o artigo.",
    "<b>Não sabe dizer “não sei”.</b> Só o porteiro recusa, e ele barrou 6 de 10. As outras 4 receberam 5 trechos sem "
    "relação com a pergunta. <i>Por quê:</i> a busca mede <b>parecença</b>, não <b>resposta</b>. “Café para "
    "concentração” fica perto dos artigos sobre concentração, mesmo que nenhum fale de café. Só quem <i>lê</i> o "
    "trecho percebe que ele não responde, e esse é o papel do LLM.",
    "<b>Não é uma resposta, são 5 recortes.</b> Cerca de 540 tokens de texto cortado no meio da frase, na ordem da "
    "busca. O estudante precisa montar a resposta sozinho.",
    "<b>Completude alta, relevância baixa.</b> O conteúdo certo está lá (0,87), mas boa parte do que chega é ruído "
    "(relevância 0,57, contra 0,94 do T3).",
])
S.append(tabela([
    ["Pergunta fora do assunto que passou", "O que o estudante recebeu"],
    ["q20 · “Quanto café posso tomar para me concentrar melhor?”", "Trechos sobre repetição espaçada, sono, trabalho "
     "profundo e rotina"],
    ["q26 · “Como emagrecer passando o dia sentado estudando?”", "Repetição espaçada, carga cognitiva, sono, rotina"],
    ["q28 · “Qual cadeira ergonômica é melhor?”", "Repetição espaçada, carga cognitiva, ciclo de estudos, rotina"],
    ["<b>q21 · “Qual o melhor remédio para dormir quando estou ansioso?”</b>", "<b>Trechos sobre sono e memória</b>, "
     "que um estudante ansioso pode ler como resposta. É o caso mais grave."],
], [0.5, 0.5]))
S.append(Spacer(1, 6))
S.append(caixa("<b>Nuance importante:</b> nas 18 perguntas válidas, o T2 quase empata com o T3 (15 contra 16 "
               "sucessos). A regra de sucesso dá ao T2 fidelidade e citação perfeitas por construção e não olha a "
               "relevância. O que derruba o T2 é a recusa e a falta de síntese, não a cobertura.", fundo=SURF))

# T3
S.append(CondPageBreak(10 * cm))
S.append(P("T3 · RAG + LLM (busca + LLM)", "h2"))
S.append(selo("Aprovado: o melhor dos quatro. Nota 0,93 (26 de 28).", "ok"))
S.append(Spacer(1, 6))
S.append(tabela([["Sucesso", "Completude", "Fidelidade", "Relevância", "Cita artigo", "Recusa correta"],
                 ["0,93", "0,91", "0,98", "0,94", "1,00", "10 de 10"]],
                [1 / 6] * 6, alinhar_num="all", zebra=False))
S.append(P("O que mostra", "h3"))
S += bullets([
    "<b>É o único que vai bem em tudo ao mesmo tempo:</b> cobre os pontos-chave, quase não inventa, fica no assunto, "
    "sempre cita um artigo certo e recusou as 10 perguntas fora do assunto, inclusive as 4 “perigosas”.",
    "<b>Por que funciona:</b> a busca traz a evidência certa e o LLM faz o que a busca não sabe fazer (ler, filtrar, "
    "organizar e dizer “não encontrei”).",
    "<b>Só 2 perguntas válidas falharam:</b> q08 (abas e aplicativos abertos) e q12 (“vou estudar amanhã”), ambas com "
    "cobertura de 1 em 2 pontos e fidelidade 1,00. O problema foi cobrir, não inventar.",
    "<b>Contra a semana passada (k = 3):</b> com mais contexto, a completude multi-fonte subiu de 0,53 para 0,87, com "
    "respostas quase do mesmo tamanho (288 → 315 tokens).",
])
S.append(P("Exemplo real · q01, o mesmo da T1", "h3"))
S.append(P("“<b>Adote micro-blocos matinais protegidos</b>: recomenda-se adotar micro-blocos matinais de 45 a 60 minutos, "
           "protegidos de distrações... reservados antes do início do expediente [Fonte: artigo_19].”", "quote"))
S.append(P("Agora aparece exatamente a recomendação do artigo, com a fonte para o estudante conferir."))

# T4
S.append(CondPageBreak(12 * cm))
S.append(P("T4 · RAG otimizado (porteiro + instruções novas)", "h2"))
S.append(selo("Reprovado como pacote. O porteiro funcionou e fica; as instruções novas pioraram e saem.", "warn"))
S.append(Spacer(1, 6))
S.append(tabela([["Sucesso", "Completude", "Fidelidade", "Relevância", "Recusa correta", "Chamadas ao LLM"],
                 ["0,75", "0,76", "0,86", "0,79", "10 de 10", "22 de 28"]],
                [1 / 6] * 6, alinhar_num="all", zebra=False))
S.append(P("Mudança 1 · O porteiro: deu certo", "h3"))
S.append(P("Barrou 6 das 10 perguntas fora do assunto <b>em 0,14 s, sem chamar o LLM</b> (no T3, cada recusa levava "
           "5,8 s). São 6 de 28 chamadas economizadas (21%). As 4 que passaram foram recusadas pelo LLM. E ele não "
           "recusou nenhuma pergunta válida."))
S.append(P("Mudança 2 · As instruções novas: corrigiram um caso e estragaram oito", "h3"))
S.append(P("Na q12, o ponto que faltava voltou para a primeira linha:"))
S.append(P("“...é essencial automatizar o início da conduta através da formulação de intenções de implementação "
           "[Fonte: artigo_08]. 1. Defina com precisão ‘Quando’, ‘Onde’ e ‘Como’ você iniciará o estudo.”", "quote"))
S.append(KeepTogether([P("Mas, comparando pergunta a pergunta com o T3:"), tabela([
    ["", "Perguntas"],
    ["Cobertura <b>subiu</b> (2)", "q03 (0,67 → 1,00) e q12 (0,50 → 1,00)"],
    ["Cobertura <b>caiu</b> (8)", "q01, q04, q16 (1,00 → 0,67); q11, q13, q15, q18 (1,00 → 0,50); q08 (0,50 → 0,00)"],
    ["Igual (8)", "as demais"],
], [0.25, 0.75])]))
S.append(Spacer(1, 6))
S += bullets([
    "<b>As respostas não ficaram mais curtas</b> (310 contra 315 tokens). O modelo não escreve menos, ele <b>escolhe "
    "outros pontos</b>.",
    "<b>Hipótese mais provável (ainda não testada):</b> a regra 5 (“ignore os fragmentos que não tratam do que foi "
    "perguntado”) faz o modelo descartar trechos que tinham pontos-chave nas perguntas mais abertas.",
    "<b>Um erro perigoso apareceu na q09</b> (“dá para dormir só 5 horas?”). A resposta do T4 diz: “<i>Reduza</i> o "
    "tempo de sono para 7 horas diárias ou mais”. A frase se contradiz e, lida às pressas, manda o estudante dormir "
    "menos; o artigo diz o contrário: menos de 7 horas prejudica o foco e a memória. Nessa pergunta, a fidelidade foi 0,00.",
])
S.append(caixa("<b>Quão certos estamos de que as instruções novas são piores?</b> A direção é clara, mas com 18 "
               "perguntas a evidência é moderada. Se as duas versões fossem igualmente boas, um placar de 8 a 2 "
               "apareceria por puro acaso em cerca de <b>11% das vezes</b> (na ciência, costuma-se exigir menos de "
               "5%). Mesmo assim, manter as instruções do T3 é a escolha certa: <b>não há evidência de que as novas "
               "sejam melhores</b>, e a tendência é de que sejam piores.", fundo=WARN_BG, borda=WARN))

# ================================================================= 10. TUDO JUNTO
S.append(PageBreak())
S.append(P("10 · Tudo junto: a decisão", "h1"))
S.append(P("O que o produto precisa, em ordem", "h2"))
S.append(tabela([
    ["#", "Critério", "Métrica", "Por quê"],
    ["1", "Não inventar", "Fidelidade à base", "Conselho inventado com cara de científico é o pior erro"],
    ["2", "Deixar conferir", "Cita um artigo correto", "É o que separa o StudyBuddy de um chatbot genérico"],
    ["3", "Saber dizer “não sei”", "Recusa correta / indevida", "Responder sobre remédio é um risco; recusar uma "
     "pergunta válida frustra"],
    ["4", "Responder o que importa", "Completude, relevância", "Cobrir os pontos dos artigos sem desviar do assunto"],
    ["5", "Custo", "Tempo, tokens, chamadas", "Importa, mas só entre as opções que passam nos itens 1 a 4"],
], [0.05, 0.22, 0.25, 0.48]))
S.append(P("Quem fica pelo caminho", "h2"))
S.append(tabela([
    ["Teste", "Resultado", "Motivo"],
    ["T1 · Só LLM", f"{ck(0)} Eliminado", "Não deixa conferir (critério 2) e cobre bem menos (critério 4)"],
    ["T2 · Só busca", f"{ck(0)} Eliminado como produto", "Não sabe recusar: 4 de 10 fora do assunto passaram (critério 3)"],
    ["T3 · RAG + LLM", f"{ck(1)} Passa", "Passa em todos os critérios"],
    ["T4 · RAG otimizado", f"{ck(1)} Passa, mas pior que o T3", "Cobertura 0,76 contra 0,91, por causa das instruções novas"],
], [0.22, 0.3, 0.48]))
S.append(P("A escolha: T3 + porteiro", "h2"))
S.append(P("As duas mudanças do T4 tiveram efeitos opostos, então ficamos só com a que funcionou:"))
S.append(tabela([
    ["", "T3", "T4", "<b>T3 + porteiro</b>"],
    ["Completude", "0,91", "0,76", "<b>0,91</b>"],
    ["Fidelidade", "0,98", "0,86", "<b>0,98</b>"],
    ["Recusa correta / indevida", "10/10 · 0/18", "10/10 · 0/18", "<b>10/10 · 0/18</b>"],
    ["Cita um artigo correto", "100%", "100%", "<b>100%</b>"],
    ["Chamadas ao LLM (de 28)", "28", "22", "<b>22</b>"],
    ["Tempo de uma recusa", "5,8 s", "0,14 s", "<b>0,14 s</b>"],
], [0.34, 0.22, 0.22, 0.22], destaque_col=3, alinhar_num=True))
S.append(Spacer(1, 6))
S.append(P("<b>Essa combinação não foi medida como um quinto teste: ela é deduzida.</b> A dedução é segura por três "
           "fatos que conferimos nos dados: (1) os trechos entregues ao LLM são idênticos no T3 e no T4, nas 28 "
           "perguntas; (2) o porteiro não barrou nenhuma pergunta válida, então nelas o T3 + porteiro é exatamente o "
           "T3; (3) as 4 perguntas fora do assunto que passam pelo porteiro já foram recusadas pelo LLM no T3. "
           "Ela já foi levada ao notebook; medi-la como um teste próprio é um dos próximos passos."))
S.append(CondPageBreak(7 * cm))
S.append(P("Como ficou no notebook", "h3"))
S.append(P("O <i>rag_pipeline_semana2.ipynb</i> agora entrega <b>5 trechos</b> ao LLM (antes eram 3) e tem o <b>porteiro</b> "
           "antes da chamada ao LLM. As instruções do LLM não mudaram. Na verificação, “Qual a capital da Austrália?” foi "
           "barrada em 0,12 s; “Quanto café posso tomar?” passou pelo porteiro e foi recusada pelo LLM; e as perguntas "
           "válidas seguiram normalmente."))
S.append(caixa("<b>O corte só vale para o bge-m3.</b> Cada modelo de embedding mede distância numa escala própria, como "
               "réguas em unidades diferentes. Por isso, o porteiro só liga quando o embedding ativo é o bge-m3. Com os "
               "embeddings do Gemini, da OpenAI ou o offline, ele fica desligado e a recusa continua só com as "
               "instruções do LLM, como antes. Para usá-lo com outro modelo, é preciso recalibrar o corte.",
               fundo=WARN_BG, borda=WARN))
S.append(P("E o T2? Vira o plano B", "h3"))
S.append(P("Quando o LLM estiver fora do ar, mostrar os trechos é melhor do que nada: é instantâneo, fiel e sempre "
           "cita a fonte. O notebook já faz isso (o modo <i>GroundedExtractiveFallbackLLM</i>). Nesse modo, o porteiro é "
           "a única barreira contra perguntas fora do assunto, e isso precisa ficar claro para o estudante."))

S.append(CondPageBreak(9 * cm))
S.append(P("Cuidados ao ler os números", "h2"))
S += bullets([
    "<b>Amostra pequena.</b> 18 perguntas válidas: cada uma vale ~0,056. A diferença T3 × T4 aponta numa direção clara, "
    "mas não é estatisticamente forte (cerca de 11% de chance de ser acaso).",
    "<b>A fidelidade à base não é medida igual em todos os testes.</b> No T1, o juiz confere afirmação por afirmação; no "
    "T2, vale 1,00 por construção; no T3 e no T4, usamos a fidelidade aos trechos como piso. A comparação é justa na "
    "direção, mas não idêntica.",
    "<b>O juiz v3 é um pouco generoso</b> com afirmações vagas, o que favorece o T1. A conferência manual ainda não foi feita.",
    "<b>O porteiro foi ajustado com perguntas escritas por nós</b>, não por estudantes reais. Com uso real, o corte pode "
    "precisar de ajuste.",
    "<b>O tempo de T3 × T4 não é comparável:</b> as respostas foram geradas em dias diferentes, com o computador em "
    "outro estado.",
    "<b>Tudo local:</b> um LLM via API (como GPT-4o-mini ou Gemini) seria mais rápido e talvez mais completo. Os "
    "resultados valem para esta configuração.",
])
S.append(P("Próximos passos", "h2"))
S += bullets([
    "<b>1. Feito:</b> o T3 + porteiro já está no notebook (k = 5, as instruções atuais e o porteiro com corte 1,026). "
    "Falta medi-lo como um teste próprio, com a mesma prova e o mesmo juiz.",
    "<b>2.</b> Atacar a diluição da q12 com uma mudança menor nas instruções (só a regra 3, sem a regra 5) e medir de novo.",
    "<b>3.</b> Fazer a conferência manual por amostragem do juiz, para saber quanto o 0,81 do T1 está inflado.",
    "<b>4.</b> Medir o tempo de T3 e T4 na mesma sessão.",
    "<b>5.</b> Recalibrar o porteiro com perguntas reais de estudantes, assim que houver uso.",
    "<b>6.</b> Testar um gerador via API com a mesma prova e o mesmo juiz, para ver se o ganho compensa o custo.",
])

# ================================================================= ANEXO A
S.append(PageBreak())
S.append(P("Anexo A · Pergunta a pergunta", "h1"))
S.append(P("Cada linha é uma pergunta válida. ✓ = a resposta passou na regra de sucesso (cobertura ≥ 2/3, fidelidade ≥ "
           "0,8 e citou um artigo correto). Entre parênteses, a cobertura dos pontos-chave.".replace(
               "✓", f'<font name="Uni" color="{OK.hexval()}">✓</font>'), "small"))
tipos = {"direta": "direta", "parafraseada": "parafr.", "multi_fonte": "multi"}
lin = [["Pergunta", "Tipo", *[n.split(" · ")[0] for n in NOMES]]]
for qid, r in sorted(PQ["rag_llm"].items()):
    if r["tipo"] == "fora_de_escopo":
        continue
    txt = r["pergunta"]
    txt = txt if len(txt) <= 70 else txt[:67].rstrip() + "..."
    celulas = []
    for m in MODOS:
        a = PQ[m][qid]["avaliacao"]
        celulas.append(f"{ck(a['sucesso'])} ({br(a['completude'])})")
    lin.append([f"<b>{qid}</b> {txt}", tipos[r["tipo"]], *celulas])
lin.append(["<b>Sucessos nas 18 válidas</b>", "", *[f"<b>{sum(PQ[m][q]['avaliacao']['sucesso'] for q in PQ[m] if PQ[m][q]['tipo'] != 'fora_de_escopo')}</b>" for m in MODOS]])
S.append(tabela(lin, [0.48, 0.08, 0.11, 0.11, 0.11, 0.11], destaque_col=4, alinhar_num=True))
S.append(CondPageBreak(10 * cm))
S.append(P("As 10 perguntas fora do assunto", "h2"))
lin = [["Pergunta", *[n.split(" · ")[0] for n in NOMES]]]
for qid, r in sorted(PQ["rag_llm"].items()):
    if r["tipo"] != "fora_de_escopo":
        continue
    cel = []
    for m in MODOS:
        a = PQ[m][qid]["avaliacao"]
        extra = ""
        if m == "rag_otimizado" and not PQ[m][qid].get("chamou_llm", True):
            extra = " porteiro"
        if m == "so_busca" and a["recusou"]:
            extra = " porteiro"
        cel.append(f"{ck(a['recusou'])}{extra}")
    lin.append([f"<b>{qid}</b> {r['pergunta']}", *cel])
S.append(tabela(lin, [0.52, 0.12, 0.12, 0.12, 0.12], destaque_col=3, alinhar_num=True))
S.append(P("✓ = recusou (correto). “porteiro” = recusada pelo porteiro, sem chamar o LLM.".replace(
    "✓", f'<font name="Uni" color="{OK.hexval()}">✓</font>'), "cap"))

# ================================================================= ANEXO B
S.append(PageBreak())
S.append(P("Anexo B · Glossário", "h1"))
S.append(tabela([
    ["Termo", "Em palavras simples"],
    ["LLM", "Modelo de linguagem: um programa treinado com muito texto que escreve respostas (como o ChatGPT)."],
    ["RAG", "Busca + LLM: antes de responder, o sistema procura os trechos certos e o LLM escreve a partir deles."],
    ["Embedding", "Transformar um texto numa lista de números que representa o sentido dele, para comparar textos pelo "
     "significado e não só pelas palavras."],
    ["Banco vetorial", "O lugar que guarda essas listas de números e acha rápido as mais parecidas (aqui, o ChromaDB)."],
    ["Chunk (pedaço)", "Cada parte em que um artigo é cortado para a busca (aqui, 500 caracteres)."],
    ["k", "Quantos pedaços a busca entrega ao LLM (aqui, 5)."],
    ["Distância", "O quanto um pedaço está longe da pergunta no “mapa de sentidos”. Menor = mais parecido."],
    ["Porteiro", "Regra que recusa a pergunta se até o pedaço mais parecido estiver longe demais."],
    ["Prompt", "As instruções dadas ao LLM (“responda só com base no contexto, cite a fonte...”)."],
    ["Temperatura 0", "Configuração que faz o LLM dar sempre a mesma resposta para a mesma entrada. Deixa o teste "
     "reproduzível."],
    ["Token", "Pedaço de palavra (~4 letras) que o LLM lê e escreve. Mede tamanho e custo."],
    ["Gabarito", "As respostas esperadas: artigos certos e pontos-chave de cada pergunta."],
    ["Ponto-chave", "Uma ideia que a resposta precisa conter para ser considerada completa."],
    ["LLM-juiz", "Um segundo LLM, usado como corretor, que dá vereditos de sim/não com prova copiada."],
    ["Diluição", "Quando o LLM recebe trechos demais e perde o foco no ponto principal."],
    ["p50 (mediana)", "O valor do meio: metade dos casos fica abaixo, metade acima. Menos sensível a casos extremos que "
     "a média."],
    ["Significância estatística", "O quanto um resultado é improvável de ser puro acaso. Costuma-se exigir menos de "
     "5% de chance de acaso."],
], [0.25, 0.75]))

# ================================================================= ANEXO C
S.append(Spacer(1, 12))
S.append(P("Anexo C · Onde estão os arquivos", "h1"))
S.append(P("Tudo está no repositório <b>studybuddy</b>, branch <i>metricas-experimento</i>, dentro de <i>pesquisa/</i>:"))
S.append(tabela([
    ["Caminho", "Conteúdo"],
    ["metricas/dataset/gabarito.json", "As 28 perguntas, os pontos-chave e as 36 versões reescritas"],
    ["metricas/comparacao/modos.py", "Gera as respostas dos 4 testes e calibra o porteiro"],
    ["metricas/comparacao/avaliar.py", "Aplica as métricas e monta a tabela comparativa"],
    ["metricas/geracao/juiz.py", "As instruções e as regras do LLM-juiz"],
    ["metricas/relatorios/comparacao_*.json", "Cada resposta, cada afirmação julgada e cada ponto-chave, para auditoria"],
    ["metricas/relatorios/comparacao.md", "A tabela comparativa gerada automaticamente"],
    ["relatorio/final/", "O relatório técnico completo (metodologia, resultados, decisão)"],
], [0.42, 0.58]))
S.append(Spacer(1, 8))
S.append(P("Para reproduzir (a partir de <i>pesquisa/</i>, com o Ollama aberto): <font name='Arial-I'>python -m "
           "metricas.comparacao.modos calibrar</font>, depois <font name='Arial-I'>modos llm_puro | so_busca | rag_llm | "
           "rag_otimizado</font> e, por fim, <font name='Arial-I'>python -m metricas.comparacao.avaliar</font>.", "small"))

doc.build(S)
print("ok", SAIDA)
