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
    valid_users = {u["username"]: u["password"] for u in users_list}

    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        with st.sidebar:
            if st.button("🔒 Terminar sessão"):
                st.session_state.logged_in = False
                st.rerun()
        return

    st.title("🔐 Contas de Guardanapo - Login")

    with st.form("login_form"):
        username = st.text_input("Utilizador")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Entrar")

    if submitted:
        if username in valid_users and password == valid_users[username]:
            st.session_state.logged_in = True
            st.session_state.user = username
            st.success(f"Sessão iniciada com sucesso, {username}.")
            st.rerun()
        else:
            st.error("Credenciais inválidas.")

    st.stop()

# ----------------- Configuração base ----------------- #

st.set_page_config(page_title="Contas de Guardanapo", layout="wide")

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

CATEGORIES_WORKSHEET_NAME = "categorias"
RULES_WORKSHEET_NAME = "regras"

# ----------------- Funções utilitárias ----------------- #

def clean_text(s) -> str:
    if pd.isna(s):
        return ""
    return str(s).upper()

def guess_merchant(description: str) -> str:
    """
    Extrai uma chave normalizada para o fornecedor a partir da descrição:
    - maiúsculas
    - remove números/pontuação
    - remove tokens genéricos
    - usa 2–3 primeiras palavras relevantes
    """
    text = clean_text(description)
    text = re.sub(r"[^A-ZÀ-Ü\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    generic_tokens = {
        "COMPRA", "PAGAMENTO", "PAGTO", "PAG",
        "DEBITO", "DÉBITO", "CREDITO", "CRÉDITO",
        "LOJA", "ONLINE",
        "REF", "REFERENCIA", "REFERÊNCIA",
        "ENTIDADE", "ATM",
        "MBWAY", "MB",
        "LX", "LISBOA", "PORTO", "PT",
        "SA", "LDA", "SPA",
        "SUPERMERCADO", "HIPERMERCADO",
        "SERVICO", "SERVIÇO",
    }

    tokens = [t for t in text.split(" ") if t and t not in generic_tokens]
    if tokens:
        return " ".join(tokens[:3])
    return text

# ----------------- Google Sheets (base) ----------------- #

def history_enabled() -> bool:
    return "gcp_service_account" in st.secrets and "gsheet" in st.secrets

def get_gspread_client():
    creds_info = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(creds_info, scopes=GSHEET_SCOPE)
    return gspread.authorize(creds)

def get_spreadsheet():
    client = get_gspread_client()
    spreadsheet_name = st.secrets["gsheet"]["spreadsheet_name"]
    return client.open(spreadsheet_name)

# ----------------- Google Sheets (histórico movimentos) ----------------- #

def get_history_worksheet():
    sh = get_spreadsheet()
    worksheet_name = st.secrets["gsheet"]["worksheet_name"]

    try:
        ws = sh.worksheet(worksheet_name)
    except gspread.WorksheetNotFound:
        ws = sh.add_worksheet(title=worksheet_name, rows=2000, cols=10)
        ws.append_row(["tx_id", "date", "description", "amount", "category", "subcategory", "year", "month"])
    return ws

def build_tx_id(row: pd.Series) -> str:
    merchant_key = guess_merchant(row["description"])
    key = f"{row['date'].date()}|{row['amount']}|{merchant_key}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

def load_history_df() -> pd.DataFrame:
    ws = get_history_worksheet()
    rows = ws.get_all_records()
    if not rows:
        return pd.DataFrame(columns=["tx_id", "date", "description", "amount", "category", "subcategory", "year", "month"])

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    df["subcategory"] = df.get("subcategory", "").fillna("")
    return df

# ----------------- Categorias dinâmicas (Google Sheets) ----------------- #

def default_categories_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "category": DEFAULT_CATEGORIES,
            "subcategory": ["" for _ in DEFAULT_CATEGORIES],
            "description": ["" for _ in DEFAULT_CATEGORIES],
            "active": [True for _ in DEFAULT_CATEGORIES],
        }
    )

def get_categories_worksheet(create_if_missing: bool = True):
    try:
        sh = get_spreadsheet()
    except Exception as e:
        st.warning(f"Não foi possível aceder ao Google Sheets (categorias): {e}")
        return None

    try:
        return sh.worksheet(CATEGORIES_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        if not create_if_missing:
            return None
        ws = sh.add_worksheet(title=CATEGORIES_WORKSHEET_NAME, rows=300, cols=4)
        ws.append_row(["category", "subcategory", "description", "active"])
        return ws

def load_categories_df() -> pd.DataFrame:
    if not history_enabled():
        return default_categories_df()

    ws = get_categories_worksheet(create_if_missing=True)
    if ws is None:
        return default_categories_df()

    try:
        rows = ws.get_all_records()
    except Exception as e:
        st.warning(f"Não foi possível ler categorias: {e}")
        return default_categories_df()

    if not rows:
        return default_categories_df()

    df = pd.DataFrame(rows)
    for col in ["category", "subcategory", "description", "active"]:
        if col not in df.columns:
            df[col] = True if col == "active" else ""

    df["subcategory"] = df["subcategory"].fillna("").astype(str)
    df["description"] = df["description"].fillna("").astype(str)
    df["active"] = df["active"].fillna(True).astype(bool)

    df = df[["category", "subcategory", "description", "active"]]
    df["category"] = df["category"].fillna("").astype(str).str.strip()
    df["subcategory"] = df["subcategory"].fillna("").astype(str).str.strip()
    return df

def save_categories_df(df: pd.DataFrame):
    if not history_enabled():
        return

    ws = get_categories_worksheet(create_if_missing=True)
    if ws is None:
        return

    df_to_save = df.copy()
    for col in ["category", "subcategory", "description"]:
        if col not in df_to_save.columns:
            df_to_save[col] = ""
    if "active" not in df_to_save.columns:
        df_to_save["active"] = True

    df_to_save["active"] = df_to_save["active"].fillna(True).astype(bool)
    df_to_save = df_to_save[["category", "subcategory", "description", "active"]]

    ws.clear()
    ws.append_row(list(df_to_save.columns))
    if not df_to_save.empty:
        ws.append_rows(df_to_save.values.tolist())

# ----------------- Regras de categorização (Google Sheets) ----------------- #

def get_rules_worksheet(create_if_missing=True):
    sh = get_spreadsheet()
    try:
        return sh.worksheet(RULES_WORKSHEET_NAME)
    except gspread.WorksheetNotFound:
        if not create_if_missing:
            raise
        ws = sh.add_worksheet(title=RULES_WORKSHEET_NAME, rows=1000, cols=4)
        ws.append_row(["merchant_key", "category", "subcategory", "raw_description"])
        return ws

def load_rules_df() -> pd.DataFrame:
    if not history_enabled():
        return pd.DataFrame(columns=["merchant_key", "category", "subcategory", "raw_description"])

    try:
        ws = get_rules_worksheet(create_if_missing=True)
        rows = ws.get_all_records()
    except Exception:
        return pd.DataFrame(columns=["merchant_key", "category", "subcategory", "raw_description"])

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["merchant_key", "category", "subcategory", "raw_description"])

    for col in ["merchant_key", "category", "subcategory", "raw_description"]:
        if col not in df.columns:
            df[col] = ""

    df["subcategory"] = df["subcategory"].fillna("")
    df["raw_description"] = df["raw_description"].fillna("")
    df["merchant_key"] = df["merchant_key"].fillna("").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["subcategory"] = df["subcategory"].fillna("").astype(str)
    return df[["merchant_key", "category", "subcategory", "raw_description"]]

def save_rules_df(df: pd.DataFrame):
    ws = get_rules_worksheet(create_if_missing=True)
    ws.clear()
    ws.append_row(["merchant_key", "category", "subcategory", "raw_description"])
    if not df.empty:
        df_to_save = df[["merchant_key", "category", "subcategory", "raw_description"]].copy()
        ws.append_rows(df_to_save.values.tolist())

def build_rules_map(rules_df: pd.DataFrame) -> dict:
    rules_map = {}
    if rules_df is None or rules_df.empty:
        return rules_map

    for _, row in rules_df.iterrows():
        mk = str(row.get("merchant_key", "")).strip()
        if not mk:
            continue
        rules_map[mk] = {
            "category": str(row.get("category", "")).strip(),
            "subcategory": str(row.get("subcategory", "")).strip(),
        }
    return rules_map

# ----------------- Categorização ----------------- #

def auto_categorize_row(description: str, amount: float, rules_map: dict) -> tuple[str, str]:
    """
    Devolve (category, subcategory)
    1) regras aprendidas (sheet 'regras')
    2) heurísticas simples
    3) fallback por sinal do valor
    """
    desc_clean = clean_text(description)
    merchant_key = guess_merchant(description)

    # 1) Regras aprendidas
    if merchant_key in rules_map:
        rule = rules_map[merchant_key]
        cat = rule.get("category", "").strip() or "Outros / Por classificar"
        sub = rule.get("subcategory", "").strip()
        return cat, sub

    # 2) Heurísticas simples (sem subcategoria)
    if any(x in desc_clean for x in ["PINGO DOCE", "CONTINENTE", "LIDL", "ALDI", "MERCADONA"]):
        return "Supermercado", ""
    if any(x in desc_clean for x in ["UBER", "BOLT", "CABIFY", " CP", "METRO", "CARRIS", "VIA VERDE"]):
        return "Transportes & Combustível", ""
    if any(x in desc_clean for x in ["GALP", " BP", "REPSOL", "CEPSA"]):
        return "Transportes & Combustível", ""
    if any(x in desc_clean for x in ["NETFLIX", "SPOTIFY", "DISNEY", "HBO", "YOUTUBE PREMIUM"]):
        return "Subscrições & Apps", ""
    if any(x in desc_clean for x in ["EDP", "ENDESA", "GÁS", "ELETRICIDADE", "ÁGUA", "EPAL"]):
        return "Casa", ""
    if any(x in desc_clean for x in ["SEGURO", "TRIGÉSIMA", "PRÉMIO SEGURO", "PREMIO SEGURO"]):
        return "Seguros", ""
    if any(x in desc_clean for x in ["GINÁSIO", "FITNESS", "GYM"]):
        return "Lazer & Entretenimento", ""

    # 3) Fallback
    if amount > 0:
        return "Rendimentos", ""
    return "Outros / Por classificar", ""

def add_auto_categories(df: pd.DataFrame, rules_map: dict) -> pd.DataFrame:
    df = df.copy()

    cats = df.apply(lambda r: auto_categorize_row(r["description"], r["amount"], rules_map), axis=1)
    df["suggested_category"] = [c[0] for c in cats]
    df["suggested_subcategory"] = [c[1] for c in cats]

    df["category"] = df["suggested_category"]
    df["subcategory"] = df["suggested_subcategory"]
    return df

# ----------------- Leitura e normalização do extracto ----------------- #

def load_statement(uploaded_file) -> pd.DataFrame:
    """
    Lê extractos em:
      - .csv (separador ;)
      - .xlsx (Excel moderno – recomendado)

    NOTA:
    Ficheiros .xls (Excel antigo) não são suportados.
    """
    name = uploaded_file.name.lower()

    # CSV
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file, sep=";", encoding="utf-8", engine="python")

    # Excel moderno
    if name.endswith(".xlsx"):
        df0 = pd.read_excel(uploaded_file, engine="openpyxl", header=None)

        # Tentar detectar o formato Santander
        col0 = df0.iloc[:, 0].astype(str).str.strip().str.upper()
        mask = col0.eq("DATA OPERAÇÃO")

        if mask.any():
            header_idx = mask[mask].index[0]
            header = df0.loc[header_idx]
            df = df0.loc[header_idx + 1 :].copy()
            df.columns = header
            return df.reset_index(drop=True)

        # fallback: assume que já vem com cabeçalho
        return pd.read_excel(uploaded_file, engine="openpyxl")

    # Excel antigo (bloqueado)
    if name.endswith(".xls"):
        st.error(
            "Ficheiros .xls (Excel antigo) não são suportados.\n\n"
            "Abre o ficheiro no Excel e guarda como **.xlsx**."
        )
        st.stop()

    st.error("Formato de ficheiro não suportado.")
    st.stop()


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
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
    df_norm["date"] = pd.to_datetime(df[norm["date"]], errors="coerce", dayfirst=True)
    df_norm["description"] = df[norm["description"]].astype(str)
    df_norm["amount"] = pd.to_numeric(df[norm["amount"]].astype(str).str.replace(",", "."), errors="coerce")

    df_norm = df_norm.dropna(subset=["date", "amount"])
    df_norm["year"] = df_norm["date"].dt.year
    df_norm["month"] = df_norm["date"].dt.to_period("M").astype(str)
    return df_norm

# ----------------- Resumos e previsão ----------------- #

def compute_monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
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
    d = df[df["amount"] < 0].copy()
    if d.empty:
        return pd.DataFrame(columns=["category", "monthly_avg", "forecast_next_month"])

    d["month_period"] = d["date"].dt.to_period("M")
    grouped = d.groupby(["category", "month_period"])["amount"].sum().reset_index()

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

# 0) Carregar regras e categorias
rules_df = load_rules_df()
rules_map = build_rules_map(rules_df)

categories_df = load_categories_df()

category_options = sorted(
    categories_df.loc[categories_df["active"], "category"].dropna().astype(str).str.strip().unique()
)

subcat_options = (
    categories_df.loc[categories_df["active"], "subcategory"]
    .dropna()
    .astype(str)
    .str.strip()
)
subcat_options = sorted([s for s in subcat_options.unique() if s])

uploaded_file = st.file_uploader(
    "Carregar extracto bancário (.xls / .xlsx / .csv)",
    type=["xls", "xlsx", "csv"],
)

if uploaded_file is not None:
    try:
        raw_df = load_statement(uploaded_file)
    except Exception as e:
        st.error(f"Erro a ler o ficheiro: {e}")
        st.stop()

    st.subheader("Pré-visualização do ficheiro original")
    st.dataframe(raw_df.head())

    try:
        df = normalize_df(raw_df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # 3) Categorizar automaticamente via regras + heurísticas
    df = add_auto_categories(df, rules_map)

    # 3.1) tx_id
    df["tx_id"] = df.apply(build_tx_id, axis=1)

    st.subheader("Movimentos com categorias (editável)")
    st.markdown(
        """
Podes alterar as colunas **category** e **subcategory**.  
Quando guardares, essa escolha passa a ser aplicada a **todos os movimentos futuros**
com a mesma descrição (via chave `merchant_key`).
"""
    )

    edited_df = st.data_editor(
        df,
        num_rows="dynamic",
        column_config={
            "category": st.column_config.SelectboxColumn("category", options=category_options),
            "subcategory": st.column_config.SelectboxColumn("subcategory", options=subcat_options, required=False),
        },
        hide_index=True,
    )

    final_df = edited_df.copy()

    # 4) Guardar regras aprendidas
    if st.button("💾 Guardar correcções e actualizar 'inteligência'"):
        new_rules = rules_df.copy()

        # garantir colunas
        if new_rules.empty:
            new_rules = pd.DataFrame(columns=["merchant_key", "category", "subcategory", "raw_description"])

        for _, row in edited_df.iterrows():
            merchant_key = guess_merchant(row["description"])
            cat = str(row.get("category", "")).strip()
            sub = str(row.get("subcategory", "")).strip()

            if cat and cat != "Outros / Por classificar":
                # remover regra antiga (se existir)
                if not new_rules.empty:
                    new_rules = new_rules[new_rules["merchant_key"] != merchant_key]

                # adicionar regra
                new_rules.loc[len(new_rules)] = [merchant_key, cat, sub, row["description"]]

        try:
            save_rules_df(new_rules)
            rules_df = new_rules.copy()
            rules_map = build_rules_map(rules_df)

            # reaplicar categorizações com as novas regras
            final_df = add_auto_categories(final_df.drop(columns=[c for c in ["suggested_category", "suggested_subcategory"] if c in final_df.columns]), rules_map)

            st.success("Regras actualizadas no Google Sheets e reaplicadas aos movimentos.")
        except Exception as e:
            st.error(f"Erro ao guardar regras no Google Sheets: {e}")

    # 4.1) filtros
    st.subheader("🎛️ Filtros dos movimentos carregados")
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)

    anos_disponiveis = sorted(final_df["year"].unique())
    anos_sel = col_f1.multiselect("Ano", options=anos_disponiveis, default=anos_disponiveis)

    meses_disponiveis = sorted(final_df["month"].unique())
    meses_sel = col_f2.multiselect("Mês", options=meses_disponiveis, default=meses_disponiveis)

    categorias_disponiveis = sorted(final_df["category"].unique())
    categorias_sel = col_f3.multiselect("Categoria", options=categorias_disponiveis, default=categorias_disponiveis)

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

    # 5) resumo
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

    # 6) previsão
    st.subheader("🔮 Previsão de despesas por categoria (média mensal histórica)")
    forecast_df = forecast_next_month(df_filtrado)
    if not forecast_df.empty:
        st.dataframe(forecast_df)
    else:
        st.info("Ainda não há dados suficientes para previsão.")

    # 7) tabela categorias/descrições
    st.subheader("📋 Categorias e descrições associadas")
    with st.expander("Ver tabela de categorias / descrições", expanded=False):
        cat_desc = (
            df_filtrado.groupby(["category", "description"])
            .size()
            .reset_index(name="num_movimentos")
            .sort_values(["category", "num_movimentos"], ascending=[True, False])
        )
        st.dataframe(cat_desc, use_container_width=True)

    # 8) guardar histórico
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
                    to_save["date"] = pd.to_datetime(to_save["date"]).dt.strftime("%Y-%m-%d")
                    to_save["year"] = pd.to_numeric(to_save["year"], errors="coerce").fillna(0).astype(int)

                    cols = ["tx_id", "date", "description", "amount", "category", "subcategory", "year", "month"]
                    rows = to_save[cols].values.tolist()
                    ws.append_rows(rows)
                    st.success(f"{len(rows)} movimentos adicionados ao histórico.")
            except Exception as e:
                st.error(f"Erro ao guardar no histórico: {e}")
    else:
        st.info("Histórico em Google Sheets não configurado (faltam secrets).")

else:
    st.info("Carrega um ficheiro de extracto para começar ou consulta o histórico consolidado (se existir).")

    if history_enabled():
        try:
            history_df = load_history_df()
        except Exception as e:
            history_df = pd.DataFrame()
            st.error(f"Não foi possível carregar o histórico: {e}")

        if not history_df.empty:
            st.subheader("🎛️ Filtros do histórico")
            col_h1, col_h2, col_h3 = st.columns(3)

            anos_hist = sorted(history_df["year"].unique())
            anos_sel_h = col_h1.multiselect("Ano", options=anos_hist, default=anos_hist)

            meses_hist = sorted(history_df["month"].unique())
            meses_sel_h = col_h2.multiselect("Mês", options=meses_hist, default=meses_hist)

            cats_hist = sorted(history_df["category"].unique())
            cats_sel_h = col_h3.multiselect("Categoria", options=cats_hist, default=cats_hist)

            mask_hist = (
                history_df["year"].isin(anos_sel_h)
                & history_df["month"].isin(meses_sel_h)
                & history_df["category"].isin(cats_sel_h)
            )
            hist_filtrado = history_df[mask_hist].copy()

            st.subheader("📚 Histórico consolidado (Google Sheets)")
            st.markdown("Pré-visualização dos últimos movimentos filtrados:")
            st.dataframe(hist_filtrado.sort_values("date", ascending=False).head(50), use_container_width=True)

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
            st.info("Ainda não há histórico guardado. Carrega um extracto e usa o botão de guardar.")
    else:
        st.info("Histórico em Google Sheets não configurado (faltam secrets).")

# ----------------- Gestão de categorias ----------------- #

st.subheader("🗂️ Gestão de categorias")

if history_enabled():
    st.markdown(
        """
Aqui podes gerir:

- **category**: nível principal (ex.: Supermercado, Casa, Saúde)  
- **subcategory**: nível secundário opcional  
- **description**: texto livre  
- **active**: se FALSE, deixa de aparecer nas opções mas mantém o histórico existente.
"""
    )

    categories_df = load_categories_df()
    edited_cats_df = st.data_editor(categories_df, num_rows="dynamic", hide_index=True)

    if st.button("💾 Guardar categorias", key="save_categorias"):
        try:
            save_categories_df(edited_cats_df)
            st.success("Categorias actualizadas. Faz refresh à página para aplicar.")
        except Exception as e:
            st.error(f"Erro ao guardar categorias: {e}")
else:
    st.info(
        "Gestão de categorias requer configuração do Google Sheets "
        "(secção [gsheet] em secrets.toml)."
    )

