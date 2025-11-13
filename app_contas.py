import json
import re
import hashlib
from pathlib import Path

import gspread
import pandas as pd
import streamlit as st
from google.oauth2.service_account import Credentials

# ----------------- Autenticação simples (via secrets) ----------------- #

def require_login():
    """Login com vários utilizadores definidos no secrets.toml."""

    if "auth" not in st.secrets or "users" not in st.secrets["auth"]:
        st.error("Faltam utilizadores configurados no secrets.toml (secção [auth]).")
        st.stop()

    users_list = st.secrets["auth"]["users"]

    # Convert to dict: {username: password}
    valid_users = {u["username"]: u["password"] for u in users_list}

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        with st.sidebar:
            if st.button("🔒 Terminar sessão"):
                st.session_state.logged_in = False
                st.rerun()
        return

    # Formulário de login
    st.title("🔐 Contas de Guardanapo - Login")

    with st.form("login_form"):
        username = st.text_input("Utilizador")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Entrar")

    if submitted:
        if username in valid_users and password == valid_users[username]:
            st.session_state.logged_in = True
            st.session_state.user = username  # opcional
            st.success(f"Sessão iniciada com sucesso, {username}.")
            st.rerun()
        else:
            st.error("Credenciais inválidas.")

    st.stop()

# ----------------- Configuração base ----------------- #

st.set_page_config(page_title="Analizador de Contas Pessoais", layout="wide")

MAPPING_FILE = "merchant_map.json"

GSHEET_SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

DEFAULT_CATEGORIES = [
    "Casa",
    "Supermercado",
    "Restauração & Bares",
    "Transportes & Combustível",
    "Carro & Manutenção",
    "Educação & Crianças",
    "Saúde",
    "Seguros",
    "Subscrições & Apps",
    "Lazer & Entretenimento",
    "Vestuário & Acessórios",
    "Viagens & Férias",
    "Comissões & Impostos",
    "Rendimentos",
    "Transferências internas",
    "Outros / Por classificar",
]

# ----------------- Funções utilitárias ----------------- #


def load_mapping():
    """Carrega o dicionário fornecedor → categoria."""
    if Path(MAPPING_FILE).exists():
        with open(MAPPING_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_mapping(mapping: dict):
    """Grava o dicionário fornecedor → categoria."""
    with open(MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def clean_text(s):
    if pd.isna(s):
        return ""
    return str(s).upper()


def guess_merchant(description: str) -> str:
    """
    Extrai uma chave 'normalizada' para o fornecedor a partir da descrição:
    - converte para maiúsculas
    - remove números e pontuação
    - remove palavras genéricas
    - usa as 2–3 primeiras palavras relevantes como chave
    """
    text = clean_text(description)

    # Fica só com letras (incluindo acentos) e espaços
    text = re.sub(r"[^A-ZÀ-Ü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    generic_tokens = {
        "COMPRA",
        "PAGAMENTO",
        "PAGTO",
        "PAG",
        "DEBITO",
        "DÉBITO",
        "CREDITO",
        "CRÉDITO",
        "LOJA",
        "ONLINE",
        "REF",
        "REFERENCIA",
        "REFERÊNCIA",
        "ENTIDADE",
        "ATM",
        "MBWAY",
        "MB",
        "LX",
        "LISBOA",
        "PORTO",
        "PT",
        "SA",
        "LDA",
        "SPA",
        "SUPERMERCADO",
        "HIPERMERCADO",
        "SERVICO",
        "SERVIÇO",
    }

    tokens = [t for t in text.split(" ") if t and t not in generic_tokens]

    if tokens:
        key = " ".join(tokens[:3])
    else:
        key = text

    return key


def auto_categorize_row(description: str, amount: float, mapping: dict) -> str:
    """
    Sugere uma categoria com base:
      1) no mapping aprendido
      2) em regras simples de texto
      3) no sinal do valor (fallback)
    """
    desc_clean = clean_text(description)

    # 1) Regras aprendidas (merchant_map.json)
    for key, cat in mapping.items():
        if key in desc_clean:
            return cat

    # 2) Regras simples
    if any(x in desc_clean for x in ["PINGO DOCE", "CONTINENTE", "LIDL", "ALDI", "MERCADONA"]):
        return "Supermercado"
    if any(x in desc_clean for x in ["UBER", "BOLT", "CABIFY", "CP ", "METRO", "CARRIS", "VIA VERDE"]):
        return "Transportes & Combustível"
    if any(x in desc_clean for x in ["GALP", "BP ", "REPSOL", "CEPSA"]):
        return "Transportes & Combustível"
    if any(x in desc_clean for x in ["NETFLIX", "SPOTIFY", "DISNEY", "HBO", "YOUTUBE PREMIUM"]):
        return "Subscrições & Apps"
    if any(x in desc_clean for x in ["EDP", "ENDESA", "GÁS", "ELETRICIDADE", "ÁGUA", "EPAL"]):
        return "Casa"
    if any(x in desc_clean for x in ["SEGURO", "TRIGÉSIMA", "PREMIO SEGURO"]):
        return "Seguros"
    if any(x in desc_clean for x in ["GINÁSIO", "FITNESS", "GYM"]):
        return "Lazer & Entretenimento"

    if amount > 0:
        return "Rendimentos"

    return "Outros / Por classificar"


# ----------------- Google Sheets (histórico) ----------------- #


def history_enabled() -> bool:
    return "gcp_service_account" in st.secrets and "gsheet" in st.secrets


def get_gspread_client():
    """Cria cliente gspread a partir do service account em st.secrets."""
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=GSHEET_SCOPE)
    return gspread.authorize(creds)


def get_history_worksheet():
    """Abre (ou cria) a worksheet de movimentos no Google Sheet."""
    client = get_gspread_client()

    spreadsheet_name = st.secrets["gsheet"]["spreadsheet_name"]
    worksheet_name = st.secrets["gsheet"]["worksheet_name"]

    sh = client.open(spreadsheet_name)
    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=1000, cols=10)
        ws.append_row(
            ["tx_id", "date", "description", "amount", "category", "year", "month"]
        )
    return ws


def build_tx_id(row: pd.Series) -> str:
    """Gera um ID único para cada movimento (para evitar duplicados no histórico)."""
    merchant_key = guess_merchant(row["description"])
    key = f"{row['date'].date()}|{row['amount']}|{merchant_key}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def load_history_df() -> pd.DataFrame:
    """Lê todo o histórico do Google Sheets para um DataFrame."""
    ws = get_history_worksheet()
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame(
            columns=["tx_id", "date", "description", "amount", "category", "year", "month"]
        )
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["year"].astype(int)
    return df

# --------------- Categorias dinâmicas (Google Sheets) --------------- #

CATEGORIES_WORKSHEET_NAME = "categorias"


def default_categories_df() -> pd.DataFrame:
    """Tabela de categorias por defeito (local, sem Google Sheets)."""
    return pd.DataFrame(
        {
            "category": DEFAULT_CATEGORIES,
            "subcategory": ["" for _ in DEFAULT_CATEGORIES],
            "description": ["" for _ in DEFAULT_CATEGORIES],
            "active": [True for _ in DEFAULT_CATEGORIES],
        }
    )


def get_categories_worksheet(create_if_missing: bool = True):
    """
    Tenta obter a worksheet de categorias no Google Sheets.
    Se não existir e create_if_missing=True, cria-a com cabeçalho base.
    Se algo falhar, devolve None.
    """
    try:
        client = get_gspread_client()
        spreadsheet_name = st.secrets["gsheet"]["spreadsheet_name"]
        sh = client.open(spreadsheet_name)
    except Exception as e:
        st.warning(
            f"Não foi possível aceder ao Google Sheets para as categorias "
            f"({e}). Vou usar apenas as categorias por defeito."
        )
        return None

    # Tentar obter a worksheet existente
    try:
        return sh.worksheet(CATEGORIES_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        if not create_if_missing:
            return None

        # Criar worksheet nova
        ws = sh.add_worksheet(
            title=CATEGORIES_WORKSHEET_NAME,
            rows=200,
            cols=4,
        )
        ws.append_row(["category", "subcategory", "description", "active"])
        return ws
    except Exception as e:
        st.warning(
            f"Erro ao obter a folha de categorias no Google Sheets ({e}). "
            "Vou usar apenas as categorias por defeito."
        )
        return None


def load_categories_df() -> pd.DataFrame:
    """
    Lê a tabela de categorias do Google Sheets.
    Se não conseguir, usa as categorias por defeito em memória.
    """
    # Se nem sequer houver configuração de Sheets, usar logo o default
    if not history_enabled():
        return default_categories_df()

    ws = get_categories_worksheet(create_if_missing=True)
    if ws is None:
        return default_categories_df()

    try:
        rows = ws.get_all_records()
    except Exception as e:
        st.warning(
            f"Não foi possível ler as categorias no Google Sheets "
            f"({e}). Vou usar apenas as categorias por defeito."
        )
        return default_categories_df()

    if not rows:
        # Folha existe mas está vazia → inicializar com default
        return default_categories_df()

    df = pd.DataFrame(rows)

    # Garantir que todas as colunas existem
    for col in ["category", "subcategory", "description", "active"]:
        if col not in df.columns:
            if col == "active":
                df[col] = True
            else:
                df[col] = ""

    df["subcategory"] = df["subcategory"].fillna("")
    df["description"] = df["description"].fillna("")
    df["active"] = df["active"].fillna(True).astype(bool)

    # Ordenar/selecionar colunas
    return df[["category", "subcategory", "description", "active"]]


def save_categories_df(df: pd.DataFrame):
    """
    Grava a tabela de categorias no Google Sheets.
    Se não houver Sheets configurado, não faz nada.
    """
    if not history_enabled():
        return

    ws = get_categories_worksheet(create_if_missing=True)
    if ws is None:
        return

    df_to_save = df.copy()
    df_to_save["active"] = df_to_save["active"].astype(bool)

    # Limpar e escrever cabeçalho + dados
    ws.clear()
    ws.append_row(list(df_to_save.columns))
    if not df_to_save.empty:
        ws.append_rows(df_to_save.values.tolist())

# ----------------- Leitura e normalização do extracto ----------------- #


def load_statement(uploaded_file) -> pd.DataFrame:
    """
    Lê extractos em:
      - .csv (separador ;)
      - .xls / .xlsx (formato Santander 'Saldos e movimentos')
    e devolve um DataFrame com cabeçalhos corretos.
    """
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8", engine="python")
        return df

    df0 = pd.read_excel(uploaded_file, header=None)

    col0 = df0.iloc[:, 0].astype(str).str.strip().str.upper()
    mask = col0.eq("DATA OPERAÇÃO")

    if mask.any():
        header_idx = mask[mask].index[0]
        header = df0.loc[header_idx]
        df = df0.loc[header_idx + 1 :].copy()
        df.columns = header
        df = df.reset_index(drop=True)
        return df

    return pd.read_excel(uploaded_file)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia colunas do banco para colunas internas:
      date, description, amount, year, month
    """
    col_map_possiveis = {
        "date": ["Data Operação", "Data operação", "Data valor", "Data"],
        "description": ["Descrição", "Descritivo", "Designação"],
        "amount": ["Montante( EUR )", "Montante (EUR)", "Montante", "Valor"],
    }

    norm = {}
    for target, candidates in col_map_possiveis.items():
        for c in candidates:
            if c in df.columns:
                norm[target] = c
                break

    missing = [k for k in ["date", "description", "amount"] if k not in norm]
    if missing:
        raise ValueError(f"Faltam colunas obrigatórias no ficheiro: {missing}")

    df_norm = pd.DataFrame()

    df_norm["date"] = pd.to_datetime(
        df[norm["date"]],
        errors="coerce",
        dayfirst=True,
    )

    df_norm["description"] = df[norm["description"]].astype(str)

    df_norm["amount"] = pd.to_numeric(
        df[norm["amount"]].astype(str).str.replace(",", "."),
        errors="coerce",
    )

    df_norm = df_norm.dropna(subset=["date", "amount"])
    df_norm["year"] = df_norm["date"].dt.year
    df_norm["month"] = df_norm["date"].dt.to_period("M").astype(str)

    return df_norm


def add_auto_categories(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    df = df.copy()
    df["suggested_category"] = df.apply(
        lambda r: auto_categorize_row(r["description"], r["amount"], mapping),
        axis=1,
    )
    df["category"] = df["suggested_category"]
    return df


# ----------------- Resumos e previsão ----------------- #


def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Resumo de despesas (amount < 0) por ano, mês, categoria."""
    d = df[df["amount"] < 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["year", "month", "category", "total", "total_abs"])

    summary = (
        d.groupby(["year", "month", "category"])["amount"]
        .sum()
        .reset_index()
        .rename(columns={"amount": "total"})
    )
    summary["total_abs"] = summary["total"].abs()
    return summary


def forecast_next_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Previsão muito simples:
    - média mensal histórica por categoria (despesas)
    """
    d = df[df["amount"] < 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["category", "monthly_avg", "forecast_next_month"])

    d["month_period"] = d["date"].dt.to_period("M")
    grouped = (
        d.groupby(["category", "month_period"])["amount"]
        .sum()
        .reset_index()
    )

    avg = (
        grouped.groupby("category")["amount"]
        .mean()
        .reset_index()
        .rename(columns={"amount": "monthly_avg"})
    )
    avg["forecast_next_month"] = avg["monthly_avg"]
    avg["monthly_avg"] = avg["monthly_avg"].abs().round(2)
    avg["forecast_next_month"] = avg["forecast_next_month"].abs().round(2)
    return avg.sort_values("forecast_next_month", ascending=False)


# ----------------- UI Streamlit ----------------- #
require_login()
st.title("🔍 Contas de Guardanapo")

st.markdown(
    """
Carrega o extracto (Santander ou outro, em **.xls / .xlsx / .csv**),
deixa a aplicação propor categorias e ajusta onde for preciso.  
As tuas correcções vão sendo **aprendidas** para futuros meses.
"""
)

# 0. Mapping aprendido (merchant → categoria)
mapping = load_mapping()

# 0.1. Carregar categorias dinâmicas (ou fallback para DEFAULT_CATEGORIES)
if history_enabled():
    categories_df = load_categories_df()
else:
    # Fallback local se não houver Google Sheets configurado
    categories_df = pd.DataFrame(
        {
            "category": DEFAULT_CATEGORIES,
            "subcategory": ["" for _ in DEFAULT_CATEGORIES],
            "descricao": ["" for _ in DEFAULT_CATEGORIES],
            "active": [True for _ in DEFAULT_CATEGORIES],
        }
    )

# Lista de categorias activas para usar nas opções
category_options = sorted(
    categories_df.loc[categories_df["active"], "category"].dropna().unique()
)

# Lista de subcategorias (opcional)
if "subcategory" in categories_df.columns:
    subcat_options = (
        categories_df["subcategory"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    subcat_options = sorted([s for s in subcat_options.unique() if s])
else:
    subcat_options = []

# 1. Upload do ficheiro
uploaded_file = st.file_uploader(
    "Carregar extracto bancário (.xls / .xlsx / .csv)",
    type=["xls", "xlsx", "csv"],
)

if uploaded_file is not None:
    # 1. Ler e pré-visualizar ficheiro original
    try:
        raw_df = load_statement(uploaded_file)
    except Exception as e:
        st.error(f"Erro a ler o ficheiro: {e}")
        st.stop()

    st.subheader("Pré-visualização do ficheiro original")
    st.dataframe(raw_df.head())

    # 2. Normalizar para formato interno
    try:
        df = normalize_df(raw_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # 3. Categorizar automaticamente
    df = add_auto_categories(df, mapping)

    # 3.1. Gerar tx_id para histórico
    df["tx_id"] = df.apply(build_tx_id, axis=1)

    st.subheader("Movimentos com categorias (editável)")
    st.markdown(
        """
Podes alterar a coluna **category**.  
Quando mudares a categoria de uma descrição que estava em *Outros / Por classificar*,
essa escolha passa a ser aplicada a **todos os movimentos com a mesma descrição**
(e também a futuros uploads).
"""
    )

    # 3.2 Editor com categorias dinâmicas (e subcategoria opcional)
    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "category": st.column_config.SelectboxColumn(
                "category",
                options=category_options,
            ),
            "subcategory": st.column_config.SelectboxColumn(
                "subcategory",
                options=subcat_options,
                required=False,
            ),
        },
        hide_index=True,
    )

    final_df = edited_df.copy()

    # 4. Aprendizagem a partir das correcções
    if st.button("💾 Guardar correcções e actualizar 'inteligência'"):
        new_mapping = mapping.copy()

        for _, row in edited_df.iterrows():
            merchant_key = guess_merchant(row["description"])
            cat = row["category"]
            if cat and cat != "Outros / Por classificar":
                new_mapping[merchant_key] = cat

        save_mapping(new_mapping)
        mapping = new_mapping

        # Reaplicar categorias a todas as linhas, se quiseres ser mais “estrito”
        final_df["category"] = final_df.apply(
            lambda r: auto_categorize_row(r["description"], r["amount"], mapping),
            axis=1,
        )

        st.success(
            "Correcções guardadas. Todas as linhas com a mesma descrição foram actualizadas "
            "e o sistema aprendeu estes novos mapeamentos."
        )
    else:
        final_df["category"] = final_df["category"].fillna(final_df["suggested_category"])

    # 4.1 Filtros sobre os movimentos carregados
    st.subheader("🎛️ Filtros dos movimentos carregados")

    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    anos_disponiveis = sorted(final_df["year"].unique())
    anos_sel = col_f1.multiselect(
        "Ano",
        options=anos_disponiveis,
        default=anos_disponiveis,
    )

    meses_disponiveis = sorted(final_df["month"].unique())
    meses_sel = col_f2.multiselect(
        "Mês",
        options=meses_disponiveis,
        default=meses_disponiveis,
    )

    categorias_disponiveis = sorted(final_df["category"].unique())
    categorias_sel = col_f3.multiselect(
        "Categoria",
        options=categorias_disponiveis,
        default=categorias_disponiveis,
    )

    min_val = float(final_df["amount"].min())
    max_val = float(final_df["amount"].max())
    valor_min, valor_max = col_f4.slider(
        "Montante (intervalo)",
        min_value=min_val,
        max_value=max_val,
        value=(min_val, max_val),
        step=0.01,
    )

    mask = (
        final_df["year"].isin(anos_sel)
        & final_df["month"].isin(meses_sel)
        & final_df["category"].isin(categorias_sel)
        & final_df["amount"].between(valor_min, valor_max)
    )

    df_filtrado = final_df[mask].copy()

    # 5. Resumo mensal e gráficos
    st.subheader("📊 Resumo mensal por categoria (despesas)")
    summary = compute_monthly_summary(df_filtrado)
    if not summary.empty:
        tabela = (
            summary.pivot_table(
                index=["year", "month"],
                columns="category",
                values="total_abs",
                aggfunc="sum",
            )
            .fillna(0)
            .round(2)
        )
        st.dataframe(tabela)

        st.markdown("### Despesas totais por mês")
        monthly_totals = (
            df_filtrado[df_filtrado["amount"] < 0]
            .groupby("month")["amount"]
            .sum()
            .abs()
            .reset_index()
            .rename(columns={"amount": "total_despesas"})
        )
        st.bar_chart(monthly_totals, x="month", y="total_despesas")
    else:
        st.info("Sem despesas (valores negativos) para resumir.")

    # 6. Previsão simples
    st.subheader("🔮 Previsão de despesas por categoria (média mensal histórica)")
    forecast_df = forecast_next_month(df_filtrado)
    if not forecast_df.empty:
        st.dataframe(forecast_df)
    else:
        st.info("Ainda não há dados suficientes para previsão.")

    # 7. Tabela de categorias / descrições
    st.subheader("📋 Categorias e descrições associadas")
    with st.expander("Ver tabela de categorias / descrições", expanded=False):
        cat_desc = (
            df_filtrado.groupby(["category", "description"])
            .size()
            .reset_index(name="num_movimentos")
            .sort_values(["category", "num_movimentos"], ascending=[True, False])
        )
        st.dataframe(cat_desc, use_container_width=True)

    # 8. Guardar histórico no Google Sheets
    if history_enabled():
        st.subheader("💾 Histórico (Google Sheets)")
        if st.button("➡️ Guardar estes movimentos no histórico"):
            try:
                ws = get_history_worksheet()
                history_df = load_history_df()
                existing_ids = set(history_df["tx_id"]) if not history_df.empty else set()

                to_save = final_df[~final_df["tx_id"].isin(existing_ids)].copy()

                if to_save.empty:
                    st.info("Nenhum movimento novo para guardar (já estão todos no histórico).")
                else:
                    to_save["date"] = to_save["date"].dt.strftime("%Y-%m-%d")
                    to_save["year"] = to_save["year"].astype(int)

                    cols = ["tx_id", "date", "description", "amount", "category", "year", "month"]
                    rows = to_save[cols].values.tolist()
                    ws.append_rows(rows)

                    st.success(f"{len(rows)} movimentos adicionados ao histórico.")
            except Exception as e:
                st.error(f"Erro ao guardar no histórico: {e}")
    else:
        st.info("Histórico em Google Sheets não configurado (faltam secrets).")
# ---------------------------------------------------------
# RAMO SEM FICHEIRO CARREGADO – CONSULTA DO HISTÓRICO
# ---------------------------------------------------------
else:
    st.info(
        "Carrega um ficheiro de extracto para começar ou consulta o histórico consolidado (se existir)."
    )

    if history_enabled():
        try:
            history_df = load_history_df()
        except Exception as e:
            history_df = pd.DataFrame()
            st.error(f"Não foi possível carregar o histórico: {e}")

        if not history_df.empty:
            # Filtros do histórico
            st.subheader("🎛️ Filtros do histórico")

            col_h1, col_h2, col_h3 = st.columns(3)

            anos_hist = sorted(history_df["year"].unique())
            anos_sel_h = col_h1.multiselect(
                "Ano",
                options=anos_hist,
                default=anos_hist,
            )

            meses_hist = sorted(history_df["month"].unique())
            meses_sel_h = col_h2.multiselect(
                "Mês",
                options=meses_hist,
                default=meses_hist,
            )

            cats_hist = sorted(history_df["category"].unique())
            cats_sel_h = col_h3.multiselect(
                "Categoria",
                options=cats_hist,
                default=cats_hist,
            )

            mask_hist = (
                history_df["year"].isin(anos_sel_h)
                & history_df["month"].isin(meses_sel_h)
                & history_df["category"].isin(cats_sel_h)
            )
            hist_filtrado = history_df[mask_hist].copy()

            st.subheader("📚 Histórico consolidado (Google Sheets)")
            st.markdown("Pré-visualização dos últimos movimentos filtrados:")
            st.dataframe(
                hist_filtrado.sort_values("date", ascending=False).head(50),
                use_container_width=True,
            )

            st.subheader("📊 Resumo histórico por mês e categoria (despesas)")
            summary_hist = compute_monthly_summary(hist_filtrado)
            if not summary_hist.empty:
                tabela_hist = (
                    summary_hist.pivot_table(
                        index=["year", "month"],
                        columns="category",
                        values="total_abs",
                        aggfunc="sum",
                    )
                    .fillna(0)
                    .round(2)
                )
                st.dataframe(tabela_hist)
            else:
                st.info("Ainda não existem despesas registadas no histórico.")
        else:
            st.info(
                "Ainda não há histórico guardado. Carrega um extracto e usa o botão de guardar."
            )
    else:
        st.info("Histórico em Google Sheets não configurado (faltam secrets).")


# =========================================================
# 9. Gestão de categorias dinâmicas (fora do if/else anterior)
# =========================================================
st.subheader("🗂️ Gestão de categorias")

if history_enabled():

    st.markdown(
        """
Aqui podes gerir:

- **categoria**: nível principal (ex.: Supermercado, Casa, Saúde)  
- **subcategoria**: nível secundário opcional  
- **descrição**: texto livre  
- **active**: se FALSE, deixa de aparecer nas opções mas mantém o histórico existente.
"""
    )

    categories_df = load_categories_df()

    edited_cats_df = st.data_editor(
        categories_df,
        num_rows="dynamic",
        hide_index=True,
    )

    if st.button("💾 Guardar categorias", key="save_categorias"):
        save_categories_df(edited_cats_df)
        st.success("Categorias actualizadas. Faz refresh à página para aplicar.")
else:
    st.info(
        "Gestão de categorias requer configuração do Google Sheets "
        "(secção [gsheet] em secrets.toml)."
    )






