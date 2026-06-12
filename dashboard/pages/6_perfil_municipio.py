"""
pages/6_🔍_Município.py
Ficha de um município específico com todos os indicadores.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import carregar_dados_zonas, carregar_dados_diamante, ZONA_CORES, nome_amigavel
from utils.charts import grafico_evolucao_temporal, grafico_perfil_municipio, grafico_comparacao_municipios
from utils.theme import apply_theme
from utils.components import (
    cabecalho_pagina, secao_com_tooltip, caixa_explicacao,
    separador, badge_zona, metrica_inline
)

st.set_page_config(
    page_title="Perfil do Município | Saneamento ES",
    page_icon="🔍",
    layout="wide",
)

apply_theme(show_toggle=True)

df_zonas = carregar_dados_zonas()
df_diamante = carregar_dados_diamante()

cabecalho_pagina(
    titulo="Perfil do Município",
    descricao=(
        "Explore o perfil completo de qualquer município do Espírito Santo: "
        "indicadores de saneamento, saúde, zona de risco e evolução histórica."
    ),
    icone="🔍",
)

# ─── Seleção de Município ────────────────────────────────────────────────────
municipios = sorted(df_zonas["nome_municipio"].dropna().unique())
col_m1, col_m2 = st.columns([3, 2])
with col_m1:
    municipio_sel = st.selectbox("🏙️ Selecione o Município", municipios)
with col_m2:
    ano_max = int(df_zonas["ano"].max())
    ano_sel = st.selectbox("📅 Ano de Referência", sorted(df_zonas["ano"].dropna().unique().astype(int), reverse=True))

# Filtrar dados do município
df_mun = df_zonas[df_zonas["nome_municipio"] == municipio_sel].sort_values("ano")
df_mun_ano = df_mun[df_mun["ano"] == ano_sel]

if df_mun_ano.empty:
    st.warning(f"Dados não encontrados para {municipio_sel} no ano {ano_sel}.")
    st.stop()

row = df_mun_ano.iloc[0]

separador()

# ─── Cabeçalho do Município ───────────────────────────────────────────────────
col_h1, col_h2 = st.columns([2, 3])
with col_h1:
    fig_gauge = grafico_perfil_municipio(df_zonas, municipio_sel)
    st.plotly_chart(fig_gauge, use_container_width=True)
with col_h2:
    st.markdown(f"### 🏙️ {municipio_sel}")
    st.markdown(f"**Ano:** {ano_sel}")

    zona = str(row.get("zona_vulnerabilidade", ""))
    badge_zona(zona)

    st.markdown("")

    # Grid de métricas
    m1, m2 = st.columns(2)
    with m1:
        agua = row.get("indice_atendimento_total_agua")
        metrica_inline("Cobertura de Água", f"{agua:.1f}%" if agua is not None and str(agua) != "nan" else "N/D", "#38bdf8")
        st.markdown("")
        esgoto = row.get("indice_atendimento_esgoto_agua")
        metrica_inline("Cobertura de Esgoto", f"{esgoto:.1f}%" if esgoto is not None and str(esgoto) != "nan" else "N/D", "#fb923c")
    with m2:
        trat = row.get("indice_tratamento_esgoto")
        metrica_inline("Tratamento de Esgoto", f"{trat:.1f}%" if trat is not None and str(trat) != "nan" else "N/D", "#a78bfa")
        st.markdown("")
        morb = row.get("Taxa_Morbidade_100k_Hab")
        metrica_inline("Morbidade (por 100k)", f"{morb:.1f}" if morb is not None and str(morb) != "nan" else "N/D", "#f472b6")

    pop = row.get("populacao_ref")
    invest = row.get("investimento_total_consolidado")
    if pop and str(pop) != "nan":
        st.caption(f"👥 **População:** {int(pop):,} habitantes")
    if invest and str(invest) != "nan":
        st.caption(f"💰 **Investimento Total:** R$ {invest:,.0f}")

separador()

# ─── Evolução Histórica ───────────────────────────────────────────────────────
secao_com_tooltip(
    "Evolução Histórica do Risco Social",
    tooltip=(
        "O que mostra: Como o Índice de Risco Social do município evoluiu ao longo dos anos. "
        "Como interpretar: Linha descendo = melhora. Linha subindo = piora. "
        "As faixas coloridas de fundo indicam os níveis de risco."
    ),
)

fig_evo = grafico_evolucao_temporal(df_zonas, municipios=[municipio_sel])
st.plotly_chart(fig_evo, use_container_width=True)

separador()

# ─── Série histórica de indicadores ──────────────────────────────────────────
secao_com_tooltip(
    "Série Histórica dos Indicadores de Saneamento",
    tooltip=(
        "O que mostra: Evolução de múltiplos indicadores ao longo do tempo para este município. "
        "Como usar: Selecione as métricas que deseja acompanhar no seletor abaixo."
    ),
)

cols_historico = [
    "indice_atendimento_total_agua", "indice_atendimento_esgoto_agua",
    "indice_tratamento_esgoto", "RISCO_SOCIAL_FINAL", "Taxa_Morbidade_100k_Hab",
    "vazio_sanitario",
]
cols_hist_existentes = [c for c in cols_historico if c in df_mun.columns]

cols_sel = st.multiselect(
    "📊 Indicadores para visualizar",
    cols_hist_existentes,
    default=cols_hist_existentes[:3],
    format_func=nome_amigavel,
)

if cols_sel:
    df_serie = df_mun[["ano"] + cols_sel].dropna(subset=["ano"])
    df_melted = df_serie.melt(id_vars="ano", value_vars=cols_sel, var_name="indicador", value_name="valor")
    df_melted["indicador"] = df_melted["indicador"].map(nome_amigavel)

    fig_serie = px.line(
        df_melted,
        x="ano",
        y="valor",
        color="indicador",
        markers=True,
        color_discrete_sequence=["#60a5fa", "#34d399", "#f472b6", "#fb923c", "#a78bfa", "#fbbf24"],
        labels={"ano": "Ano", "valor": "Valor", "indicador": "Indicador"},
    )
    fig_serie.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(148,163,184,0.28)", borderwidth=1),
    )
    st.plotly_chart(fig_serie, use_container_width=True)

separador()

# ─── Comparação com outros municípios ────────────────────────────────────────
secao_com_tooltip(
    "Comparação com Outros Municípios",
    tooltip=(
        "O que mostra: Gráfico de teia comparando o perfil de saneamento do município "
        "selecionado com até outros 3 municípios. "
        "Como usar: Adicione municípios no seletor abaixo. "
        "Como interpretar: Área maior = melhores indicadores."
    ),
)

outros_municipios = st.multiselect(
    "🏙️ Adicionar municípios para comparar",
    [m for m in municipios if m != municipio_sel],
    max_selections=3,
    default=[],
)

todos_comparar = [municipio_sel] + outros_municipios
if len(todos_comparar) >= 1:
    fig_comp = grafico_comparacao_municipios(df_zonas, todos_comparar)
    st.plotly_chart(fig_comp, use_container_width=True)

separador()

# ─── Tabela de dados do município ────────────────────────────────────────────
with st.expander("📋 Ver todos os dados históricos deste município"):
    cols_tabela = [
        "ano", "zona_vulnerabilidade", "RISCO_SOCIAL_FINAL",
        "indice_atendimento_total_agua", "indice_atendimento_esgoto_agua",
        "indice_tratamento_esgoto", "Taxa_Morbidade_100k_Hab",
        "populacao_ref", "investimento_total_consolidado",
    ]
    cols_tab_exist = [c for c in cols_tabela if c in df_mun.columns]
    df_display = df_mun[cols_tab_exist].copy()
    df_display.columns = [nome_amigavel(c) for c in cols_tab_exist]

    for col in df_display.select_dtypes(include="float").columns:
        df_display[col] = df_display[col].round(2)

    st.dataframe(df_display, use_container_width=True, hide_index=True)
