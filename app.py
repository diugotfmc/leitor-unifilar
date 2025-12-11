
import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from collections import defaultdict

st.set_page_config(page_title="Leitor de Unifilar – Tabela com cabeçalho embaixo", layout="wide")
st.title("📐 Leitor de Unifilar (DW) – Tabela com cabeçalho embaixo")
st.caption("Extrai a tabela do unifilar deduzindo as colunas pela posição do cabeçalho (localizado na parte inferior).")

# =========================
# Upload
# =========================
dw_file = st.file_uploader("Envie o PDF do unifilar/desenho", type=["pdf"])

# =========================
# Parâmetros (ajustáveis)
# =========================
with st.expander("⚙️ Opções avançadas"):
    y_tol   = st.slider("Tolerância vertical para agrupar palavras em linhas (px)", 1, 12, 5)
    x_tol   = st.slider("Tolerância horizontal para ordenar palavras (px)", 1, 12, 3)
    win_h   = st.slider("Altura da janela acima do cabeçalho para capturar linhas (px)", 100, 900, 450)
    header_required = st.number_input("Mínimo de títulos do cabeçalho a encontrar na linha (>=)", 3, 10, 6)
    mostrar_pre = st.checkbox("Mostrar linhas reconstruídas / debug (opcional)", value=False)

# =========================
# Lista de títulos de coluna esperados (ajustáveis)
# =========================
COLUMN_TITLES = [
    "CBS",
    "PCS",
    "Quantidade para PCS",
    "Item",
    "Qtd mont na linha",
    "Qtd mont a bordo",
    "Qtd avulsa",
    "Desenho nº",
    "Revisão",
    "Descrição Comercial",
]

# Tokens (palavras) de cada título para localizar blocos no cabeçalho
def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r'\s+', ' ', s)
    s = s.replace("º", "o")  # normalização simples
    s = s.strip()
    return s

TITLE_TOKENS = {t: normalize_text(t).split() for t in COLUMN_TITLES}

# =========================
# Auxiliares para reconstrução de linhas e localização de cabeçalho
# =========================
def extract_words(page, xt=x_tol, yt=y_tol):
    return page.extract_words(x_tolerance=xt, y_tolerance=yt, keep_blank_chars=False, use_text_flow=True)

def rebuild_lines_from_words(words, yt=y_tol):
    """
    Agrupa palavras em linhas por proximidade no eixo Y.
    Retorna lista de dicionários: [{'top':y, 'text':str, 'words':[...]}]
    """
    buckets = defaultdict(list)
    for w in words:
        key = round(w["top"] / yt)
        buckets[key].append(w)
    lines = []
    for key in sorted(buckets.keys()):
        ws = sorted(buckets[key], key=lambda w: w["x0"])
        txt = " ".join(w["text"] for w in ws)
        txt = re.sub(r'\s{2,}', ' ', txt).strip()
        if txt:
            # top médio (ou do primeiro)
            top = sum(w["top"] for w in ws) / len(ws)
            lines.append({"top": top, "text": txt, "words": ws})
    return lines

def find_header_line(lines):
    """
    Procura a linha do cabeçalho: a que contiver o MAIOR número de títulos conhecidos,
    priorizando linhas mais embaixo (maior 'top').
    """
    best = None
    best_score = (-1, -1.0)  # (qtd_titulos, top)
    for ln in lines:
        ltxt = normalize_text(ln["text"])
        hits = 0
        for title in COLUMN_TITLES:
            if all(tok in ltxt.split() for tok in TITLE_TOKENS[title]):
                hits += 1
        score = (hits, ln["top"])
        if score > best_score:
            best = ln
            best_score = score
    if best and best_score[0] >= header_required:
        return best
    return None

def locate_title_x_positions(header_line):
    """
    Para cada título, tenta descobrir o x0 aproximado do "início" do título na linha do cabeçalho,
    casando os tokens na sequência de palavras da linha.
    Retorna dict: {titulo: x0_médio (float)}
    """
    out = {}
    words = header_line["words"]
    texts = [normalize_text(w["text"]) for w in words]

    # função para buscar sequência de tokens no vetor de palavras
    def find_seq(tokens):
        n = len(tokens)
        for i in range(0, len(texts) - n + 1):
            if texts[i:i+n] == tokens:
                # x0 do primeiro token da sequência
                xs = [words[i + k]["x0"] for k in range(n)]
                return sum(xs)/len(xs)
        return None

    for title, tokens in TITLE_TOKENS.items():
        x_found = find_seq(tokens)
        if x_found is None:
            # fallback: buscar por inclusão parcial do primeiro token
            first = tokens[0]
            for i, t in enumerate(texts):
                if t == first:
                    x_found = words[i]["x0"]
                    break
        if x_found is not None:
            out[title] = x_found
    return out

def build_column_boundaries(title_x_map, page_width):
    """
    Gera os limites (x_left, x_right) de cada coluna a partir dos x-centers dos títulos.
    Ordena pelos x e cria fronteiras na meia-distância entre vizinhos.
    Retorna lista ordenada de tuplas: [(col_name, x_left, x_right), ...]
    """
    # ordenar por x
    items = sorted(title_x_map.items(), key=lambda kv: kv[1])
    if not items:
        return []
    centers = [x for _, x in items]
    names   = [n for n, _ in items]

    bounds = []
    # limites à esquerda e direita
    left_edges = [max(0, centers[0] - 40)]  # margem esquerda
    right_edges = []
    for i in range(len(centers)-1):
        mid = (centers[i] + centers[i+1]) / 2.0
        left_edges.append(mid)
        right_edges.append(mid)
    right_edges.append(page_width)  # margem direita

    # montar
    cols = []
    for i, name in enumerate(names):
        x_left = left_edges[i]
        x_right = right_edges[i]
        cols.append((name, x_left, x_right))
    return cols

def assign_words_to_columns(words, columns):
    """
    Atribui cada palavra à coluna cujo intervalo [x_left, x_right] contenha x_center da palavra.
    Retorna dict: {col_name: "texto concatenado"}
    """
    by_col = {name: [] for name,_,_ in columns}
    for w in words:
        x_center = (w["x0"] + w["x1"]) / 2.0
        for name, xl, xr in columns:
            if xl <= x_center <= xr:
                by_col[name].append(w["text"])
                break
    # juntar tokens por coluna
    return {name: " ".join(tokens).strip() for name, tokens in by_col.items()}

# =========================
# Execução
# =========================
if not dw_file:
    st.info("Envie o PDF do unifilar para extrair a tabela.")
    st.stop()

all_rows = []

with pdfplumber.open(dw_file) as pdf:
    for pi, page in enumerate(pdf.pages):
        words = extract_words(page, xt=x_tol, yt=y_tol)
        lines = rebuild_lines_from_words(words, yt=y_tol)

        if mostrar_pre:
            st.markdown(f"**Página {pi+1} – linhas reconstruídas (até 60):**")
            st.code("\n".join([l['text'] for l in lines[:60]]))

