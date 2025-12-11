
import streamlit as st
import pdfplumber
import pandas as pd
import re
import io
from collections import defaultdict

# =========================
# Setup
# =========================
st.set_page_config(page_title="Leitor de Unifilar – Cabeçalho embaixo", layout="wide")
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
    y_tol   = st.slider("Tolerância vertical para agrupar palavras (px)", 1, 12, 5)
    x_tol   = st.slider("Tolerância horizontal para ordenar palavras (px)", 1, 12, 5)
    win_h   = st.slider("Altura da janela acima do cabeçalho (px)", 100, 900, 500)
    header_required = st.number_input("Mínimo de títulos do cabeçalho a encontrar (>=)", 3, 10, 6)
    mostrar_pre = st.checkbox("Mostrar linhas reconstruídas / debug", value=False)
    mostrar_colunas = st.checkbox("Mostrar limites de colunas (x_left/x_right) – debug", value=False)
    # Agora força números ou traços nas 4 primeiras colunas
    forcar_quatro_campos = st.checkbox("Forçar 4 primeiros campos (número ou traço) em CBS/PCS/Qtd p/ PCS/Item", value=True)

# =========================
# Colunas canônicas + aliases
# =========================
CANONICAL_TITLES = [
    "CBS",
    "PCS",
    "Quantidade para PCS",
    "Item",
    "Qtd mont na linha",
    "Qtd mont a bordo",
    "Qtd avulsa",
    "Desenho nº",
    "Revisão",
    "Descrição",
]

# Aceitar variações do desenho e normalizar para os nomes acima
COLUMN_ALIASES = {
    "CBS": ["cbs"],
    "PCS": ["pcs"],
    "Quantidade para PCS": ["quantidade para pcs", "qtd para pcs", "qtd p/ pcs"],
    "Item": ["item", "pos.", "posicao", "posição"],
    "Qtd mont na linha": ["qtd mont na linha", "qtd. mont na linha", "qtd mont. na linha"],
    "Qtd mont a bordo": ["qtd mont a bordo", "qtd. mont a bordo", "qtd mont. a bordo"],
    "Qtd avulsa": ["qtd avulsa", "qtd. avulsa", "quantidade avulsa"],
    "Desenho nº": ["desenho nº", "desenho n°", "desenho no", "desenho n.", "desenho num"],
    "Revisão": ["revisao", "revisão", "rev", "rev."],
    "Descrição": ["descrição", "descricao", "descrição comercial", "descricao comercial"],
}

def normalize_text(s: str) -> str:
    s = s.lower()
    s = s.replace("º", "o").replace("°", "o")  # nº/n°
    s = s.replace("ª", "a")
    s = re.sub(r'[“”„"´`’]', '', s)           # aspas/acentos tipográficos
    s = re.sub(r'\s+', ' ', s)
    return s.strip()

# Converter aliases para tokens
ALIAS_TOKENS = {canon: [normalize_text(a).split() for a in aliases]
                for canon, aliases in COLUMN_ALIASES.items()}

# =========================
# Auxiliares de extração e reconstrução
# =========================
def extract_words(page, xt, yt):
    return page.extract_words(
        x_tolerance=xt,
        y_tolerance=yt,
        keep_blank_chars=False,
        use_text_flow=True
    )

def rebuild_lines_from_words(words, yt):
    """
    Agrupa palavras em linhas por proximidade no eixo Y.
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
            top = sum(w["top"] for w in ws) / len(ws)
            bottom = max(w.get("bottom", w["top"]) for w in ws)
            lines.append({"top": top, "bottom": bottom, "text": txt, "words": ws})
    return lines

def line_hits_for_canonical(line_text_norm: str):
    """
    Retorna (hits_total, hits_por_canon) indicando quantos títulos canônicos (via aliases) aparecem na linha.
    """
    words = line_text_norm.split()
    hits_by = {canon: 0 for canon in CANONICAL_TITLES}

    # busca de sequência exata
    def has_seq(seq):
        n = len(seq)
        for i in range(len(words) - n + 1):
            if words[i:i+n] == seq:
                return True
        return False

    total = 0
    for canon, list_alias_tokens in ALIAS_TOKENS.items():
        found = any(has_seq(toks) for toks in list_alias_tokens)
        if found:
            hits_by[canon] = 1
            total += 1
    return total, hits_by

def find_header_line(lines, header_required):
    """
    Cabeçalho: linha com maior número de títulos canônicos encontrados (via aliases),
    desempate por maior 'top' (mais embaixo).
    """
    best = None
    best_score = (-1, -1.0)
    for ln in lines:
        ltxt = normalize_text(ln["text"])
        hits_total, _ = line_hits_for_canonical(ltxt)
        score = (hits_total, ln["top"])
        if score > best_score:
            best = ln
            best_score = score
    if best and best_score[0] >= header_required:
        return best
    return None

def locate_title_x_positions(header_line):
    """
    Para cada título canônico, obtém x0 aproximado do início através dos aliases.
    """
    out = {}
    words = header_line["words"]
    texts = [normalize_text(w["text"]) for w in words]

    def find_seq(tokens):
        n = len(tokens)
        for i in range(0, len(texts) - n + 1):
            if texts[i:i+n] == tokens:
                xs = [words[i + k]["x0"] for k in range(n)]
                return sum(xs) / len(xs)
        return None

    for canon, alias_tokens_list in ALIAS_TOKENS.items():
        x_found = None
        # tenta cada variação até encontrar a primeira que casa
        for tokens in alias_tokens_list:
            x_found = find_seq(tokens)
            if x_found is not None:
                break
        # fallback: se não encontrar, tenta o primeiro token da primeira variação
        if x_found is None and alias_tokens_list:
            first_tok = alias_tokens_list[0][0]
            for i, t in enumerate(texts):
                if t == first_tok:
                    x_found = words[i]["x0"]
                    break
        if x_found is not None:
            out[canon] = x_found
    return out

def build_column_boundaries(title_x_map, page_width):
    """
    Limites laterais por meia distância entre centros dos títulos.
    """
    items = sorted(title_x_map.items(), key=lambda kv: kv[1])
    if not items:
        return []
    centers = [x for _, x in items]
    names   = [n for n, _ in items]

    left_edges = [0.0]
    right_edges = []
    for i in range(len(centers)-1):
        mid = (centers[i] + centers[i+1]) / 2.0
        left_edges.append(mid)
        right_edges.append(mid)
    right_edges.append(page_width)

    cols = []
    for i, name in enumerate(names):
        x_left = min(left_edges[i], right_edges[i])
        x_right = max(left_edges[i], right_edges[i])
        cols.append((name, x_left, x_right))
    return cols

def assign_words_to_columns(words, columns):
    """
    Atribui tokens à coluna cujo x_center está dentro dos limites.
    Mantém tudo como texto (sem conversões).
    """
    by_col = {name: [] for name, _, _ in columns}
    for w in words:
        x_center = (w["x0"] + w["x1"]) / 2.0
        for name, xl, xr in columns:
            if xl <= x_center <= xr:
                by_col[name].append(w["text"])
                break
    return {name: " ".join(tokens).strip() for name, tokens in by_col.items()}

# =========================
# Regra para 4 primeiros "primitivos" (número ou traço)
# =========================
# Aceitar inteiros/decimais com ponto/vírgula (ex.: 104.1, 1,0000, 200) e traços -, –, —
NUM_RE   = re.compile(r'^[+-]?\d+(?:[.,]\d+)?$')
DASH_RE  = re.compile(r'^[-–—]$')
PRIMITIVE_RE = re.compile(r'^(?:[+-]?\d+(?:[.,]\d+)?|[-–—])$')

def normalize_dash(s: str) -> str:
    # unifica traços tipográficos para '-'
    s = s.replace('–', '-').replace('—', '-')
    return s

def is_primitive_val(val: str) -> bool:
    v = normalize_dash(str(val).strip())
    return bool(PRIMITIVE_RE.match(v))

def fill_first_four_primitives(row: dict, line_words: list) -> dict:
    """
    Força os quatro primeiros 'primitivos' da linha (número ou traço) a preencher:
    CBS, PCS, Quantidade para PCS, Item (nessa ordem).
    Mantém tudo como texto.
    Ex.: '- - - 33' -> CBS='-', PCS='-', Quantidade para PCS='-', Item='33'
    """
    first_four_cols = ["CBS", "PCS", "Quantidade para PCS", "Item"]

    # Aplicar se algum dos quatro estiver vazio ou não for 'primitivo'
    need_fix = False
    for c in first_four_cols:
        val = str(row.get(c, "")).strip()
        if not val or not is_primitive_val(val):
            need_fix = True
            break

    # Coletar tokens (esquerda→direita) que sejam 'primitivos'
    if need_fix:
        ordered = sorted(line_words, key=lambda w: w["x0"])
        prims = []
        for w in ordered:
            t = normalize_dash(w["text"].strip())
            if PRIMITIVE_RE.match(t):
                prims.append(t)
            # pare cedo se já coletou 4
            if len(prims) >= 4:
                break

        # Se encontramos ao menos 4, preencher na ordem
        if len(prims) >= 4:
            row["CBS"] = prims[0]
            row["PCS"] = prims[1]
            row["Quantidade para PCS"] = prims[2]
            row["Item"] = prims[3]

    return row

# =========================
# Execução
# =========================
if not dw_file:
    st.info("Envie o PDF do unifilar para extrair a tabela.")
    st.stop()

all_rows = []
header_debug = []

with pdfplumber.open(dw_file) as pdf:
    for pi, page in enumerate(pdf.pages):
        page_width = page.width
        words = extract_words(page, xt=x_tol, yt=y_tol)
        lines = rebuild_lines_from_words(words, yt=y_tol)

        if mostrar_pre:
            st.markdown(f"**Página {pi+1} – linhas reconstruídas (até 60):**")
            st.code("\n".join([l['text'] for l in lines[:60]]))

        # 1) localizar cabeçalho (assumido embaixo)
        header_line = find_header_line(lines, header_required=header_required)
        if not header_line:
            st.warning(f"Página {pi+1}: não foi possível localizar um cabeçalho com >= {header_required} títulos.")
            continue

        header_top = header_line["top"]
        y_min = max(0, header_top - win_h)
        y_max = header_top

        # 2) posições X dos títulos (canônicos, via aliases)
        title_x_map = locate_title_x_positions(header_line)
        if not title_x_map:
            st.warning(f"Página {pi+1}: não foi possível inferir posições dos títulos.")
            continue

        columns = build_column_boundaries(title_x_map, page_width=page_width)
        if not columns:
            st.warning(f"Página {pi+1}: não foi possível construir limites de coluna.")
            continue

        if mostrar_colunas:
            st.markdown(f"**Página {pi+1} – limites das colunas (x_left → x_right):**")
            for name, xl, xr in columns:
                st.write(f"- {name}: {xl:.1f} → {xr:.1f}")

        # 3) pegar apenas as linhas na janela acima do cabeçalho
        window_lines = [ln for ln in lines if (y_min <= ln["top"] <= y_max)]
        window_lines = [ln for ln in window_lines if ln["text"].strip()]

        # 4) montar linhas
        for ln in window_lines:
            row = assign_words_to_columns(ln["words"], columns)

            # garantir todas as colunas canônicas presentes (mesmo vazias)
            for c in CANONICAL_TITLES:
                row.setdefault(c, "")

            # >>> Patch para garantir CBS/PCS/Quantidade para PCS/Item com números ou traços
            if forcar_quatro_campos:
                row = fill_first_four_primitives(row, ln["words"])

            # ignorar linha totalmente vazia (em colunas canônicas)
            if not any(str(row.get(c, "")).strip() for c in CANONICAL_TITLES):
                continue

            row["_page"] = pi + 1
            row["_y"] = ln["top"]
            all_rows.append(row)

        header_debug.append({
            "page": pi + 1,
            "header_text": header_line["text"],
            "y_top": header_top,
            "y_min": y_min,
            "y_max": y_max,
            "title_x_map": title_x_map,
        })

# =========================
# Resultado
# =========================
if not all_rows:
    st.error("Nenhuma linha de tabela foi extraída. Ajuste as tolerâncias ou a altura da janela e tente novamente.")
    st.stop()

df = pd.DataFrame(all_rows)

# ORDER: colunas canônicas primeiro, depois extras, e metadados no fim
ordered_cols = [c for c in CANONICAL_TITLES]
extras = [c for c in df.columns if c not in ordered_cols + ["_page", "_y"]]
df = df[ordered_cols + extras + ["_page", "_y"]]

# ordena por página e Y (de cima para baixo)
df = df.sort_values(by=["_page", "_y"]).reset_index(drop=True)

st.success("Tabela extraída com sucesso!")
st.dataframe(df, use_container_width=True)

# =========================
# Download
# =========================
csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="⬇️ Baixar CSV",
    data=csv_bytes,
    file_name="unifilar_tabela.csv",
    mime="text/csv",
)

xlsx_buffer = io.BytesIO()
with pd.ExcelWriter(xlsx_buffer, engine="openpyxl") as writer:
    df.to_excel(writer, index=False, sheet_name="Tabela")
st.download_button(
    label="⬇️ Baixar XLSX",
    data=xlsx_buffer.getvalue(),
    file_name="unifilar_tabela.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# =========================
# Debug do cabeçalho
# =========================
if mostrar_pre and header_debug:
    st.markdown("### 🔎 Debug do Cabeçalho Encontrado")
    for hd in header_debug:
        st.write(f"**Página {hd['page']}** | y_top={hd['y_top']:.1f} | janela: [{hd['y_min']:.1f}, {hd['y_max']:.1f}]")
        st.code(hd["header_text"])
        for k, v in sorted(hd["title_x_map"].items(), key=lambda kv: kv[1]):
            st.write(f"- {k}: x≈{v:.1f}")
