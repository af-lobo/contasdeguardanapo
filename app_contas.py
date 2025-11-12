import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ----------------- Configuração base ----------------- #

st.set_page_config(page_title="Analizador de Contas Pessoais", layout="wide")

MAPPING_FILE = "merchant_map.json"

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
    Extrai uma chave para o 'fornecedor'.
    Para já usamos a própria descrição limpa;
    mais tarde podemos refinar (ex.: primeiros tokens, remover números, etc.).
    """
    return clean_text(description)


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

    # 2) Regras simples (ajusta à tua realidade com o tempo)
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

    # 3) Fallback pelo sinal
    if amount > 0:
        return "Rendimentos"

    return "Outros / Por classificar"


# ----------------- Leitura e normalização do extracto ----------------- #


def load_statement(uploaded_file) -> pd.DataFrame:
    """
    Lê extractos em:
      - .csv (separador ;)
      - .xls / .xlsx (formato Santander 'Saldos e movimentos')
    e devolve um DataFrame com cabeçalhos corretos.
    """
    name = uploaded_file.name.lower()

    # Caso CSV
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file, sep=";", encoding="utf-8", engine="python")
        return df

    # Caso Excel (Santander)
    # Nota: para .xls poderás precisar de 'pip install xlrd==1.2.0'
    df0 = pd.read_excel(uploaded_file, header=None)

    # Procurar linha onde começa o cabeçalho "Data Operação"
    col0 = df0.iloc[:, 0].astype(str).str.strip().str.upper()
    mask = col0.eq("DATA OPERAÇÃO")

    if mask.any():
        header_idx = mask[mask].index[0]
        header = df0.loc[header_idx]
        df = df0.loc[header_idx + 1 :].copy()
        df.columns = header
        df = df.reset_index(drop=True)
        return df

    # Se não encontrarmos esse padrão, tentamos ler com o cabeçalho da primeira linha
    return pd.read_excel(uploaded_file)


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mapeia colunas do banco para colunas internas:
      date, description, amount, year, month
    """
    col_map_possiveis = {
        "date": [
            "Data Operação",
            "Data operação",
            "Data valor",
            "Data",
        ],
        "description": [
            "Descrição",
            "Descritivo",
            "Designação",
        ],
        "amount": [
            "Montante( EUR )",
            "Montante (EUR)",
            "Montante",
            "Valor",
        ],
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

    # Datas dd-mm-aaaa / dd/mm/aaaa
    df_norm["date"] = pd.to_datetime(
        df[norm["date"]],
        errors="coerce",
        dayfirst=True,
    )

    df_norm["description"] = df[norm["description"]].astype(str)

    # Valor, tratando vírgulas
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

st.subheader("📋 Categorias e descrições associadas")

with st.expander("Ver tabela de categorias / descrições", expanded=False):
    # Agrupar por categoria e descrição, contando nº de movimentos
    cat_desc = (
        final_df.groupby(["category", "description"])
        .size()
        .reset_index(name="num_movimentos")
        .sort_values(["category", "num_movimentos"], ascending=[True, False])
    )
    st.dataframe(cat_desc, use_container_width=True)

    st.markdown(
        """
Dica:  
- Ordena pela coluna **category** para ver tudo o que está em *Outros / Por classificar*.  
- Se mudares uma dessas linhas na tabela principal e carregares em **Guardar correcções**, 
  na próxima vez que abrires a app essa descrição já virá com a categoria correcta.
"""
    )

# ----------------- UI Streamlit ----------------- #

st.title("🔍 Analizador de Contas Pessoais")

st.markdown(
    """
Carrega o extracto (Santander ou outro, em **.xls / .xlsx / .csv**),
deixa a aplicação propor categorias e ajusta onde for preciso.  
As tuas correcções vão sendo **aprendidas** para futuros meses.
"""
)

mapping = load_mapping()

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

    st.subheader("Movimentos com categorias (editável)")
st.markdown(
    """
Podes alterar a coluna **category**.  
Quando mudares a categoria de uma descrição que estava em *Outros / Por classificar*,
essa escolha passa a ser aplicada a **todos os movimentos com a mesma descrição**
(e também a futuros uploads).
"""
)

edited_df = st.data_editor(
    df,
    num_rows="dynamic",
    column_config={
        "category": st.column_config.SelectboxColumn(
            "category",
            options=DEFAULT_CATEGORIES,
        )
    },
    hide_index=True,
)

# Este será o DataFrame "final" usado nos resumos e previsões
final_df = edited_df.copy()

# 4. Aprendizagem a partir das correcções
if st.button("💾 Guardar correcções e actualizar 'inteligência'"):
    new_mapping = mapping.copy()

    # 4.1. Actualizar mapping com base nas linhas editadas
    for _, row in edited_df.iterrows():
        merchant_key = guess_merchant(row["description"])
        cat = row["category"]
        if cat and cat != "Outros / Por classificar":
            new_mapping[merchant_key] = cat

    save_mapping(new_mapping)
    mapping = new_mapping

    # 4.2. Reaplicar categorias a TODAS as linhas, com base no novo mapping
    final_df["category"] = final_df.apply(
        lambda r: auto_categorize_row(r["description"], r["amount"], mapping),
        axis=1,
    )

    st.success(
        "Correcções guardadas. Todas as linhas com a mesma descrição foram actualizadas "
        "e o sistema aprendeu estes novos mapeamentos."
    )
else:
    # Se ainda não carregaste no botão, usamos as categorias tal como estão editadas
    final_df["category"] = final_df["category"].fillna(final_df["suggested_category"])


    # 5. Resumo mensal e gráficos
    st.subheader("📊 Resumo mensal por categoria (despesas)")
    summary = compute_monthly_summary(final_df)
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
            final_df[final_df["amount"] < 0]
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
    forecast_df = forecast_next_month(final_df)
    if not forecast_df.empty:
        st.dataframe(forecast_df)
    else:
        st.info("Ainda não há dados suficientes para previsão.")
else:
    st.info("Carrega um ficheiro de extracto para começar.")



