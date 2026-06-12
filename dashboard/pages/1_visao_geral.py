"""
pages/1_🏠_Visão_Geral.py
Página principal do dashboard: KPIs e distribuição de risco.
"""

import sys
from pathlib import Path

# Garante que o diretório pai (dashboard/) esteja no path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import carregar_dados_zonas, carregar_dados_diamante
from utils.charts import grafico_distribuicao_risco, grafico_ranking_municipios
from utils.theme import apply_theme
from utils.components import (
    kpi_card, cabecalho_pagina, secao_com_tooltip,
    caixa_explicacao, separador
)

# ─── Configuração da Página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Visão Geral | Saneamento e Saúde no ES",
    page_icon="📊",
    layout="wide",
)

apply_theme(show_toggle=True)

# ─── Carregar dados ───────────────────────────────────────────────────────────
df_zonas = carregar_dados_zonas()
df_diamante = carregar_dados_diamante()

# ─── Cabeçalho ───────────────────────────────────────────────────────────────
cabecalho_pagina(
    titulo="Visão Geral — Risco Social no Espírito Santo",
    descricao=(
        "Este painel apresenta um panorama geral do saneamento básico nos "
        "municípios do Espírito Santo e sua relação com a saúde da população. "
        "O Índice de Risco Social combina déficits de água, esgoto e taxas de internação hospitalar."
    ),
    icone="📊",
)

# ─── Filtros ─────────────────────────────────────────────────────────────────
anos_disponiveis = sorted(df_zonas["ano"].dropna().unique().astype(int), reverse=True)
col_f1, col_f2 = st.columns([2, 5])
with col_f1:
    ano_sel = st.selectbox("📅 Selecionar Ano", anos_disponiveis, index=0)

df_ano = df_zonas[df_zonas["ano"] == ano_sel].copy()

separador()

# ─── KPIs ─────────────────────────────────────────────────────────────────────
st.markdown("### 📊 Indicadores Principais")

total_mun = df_ano["id_municipio"].nunique()
risco_medio = df_ano["RISCO_SOCIAL_FINAL"].mean()
risco_max = df_ano["RISCO_SOCIAL_FINAL"].max()

criticos = df_ano[df_ano["zona_vulnerabilidade"].str.contains("Vermelha", na=False)]
pct_criticos = len(criticos) / total_mun * 100 if total_mun > 0 else 0

verdes = df_ano[df_ano["zona_vulnerabilidade"].str.contains("Verde", na=False)]
pct_verdes = len(verdes) / total_mun * 100 if total_mun > 0 else 0

agua_media = df_ano["indice_atendimento_total_agua"].mean() if "indice_atendimento_total_agua" in df_ano.columns else 0
esgoto_medio = df_ano["indice_atendimento_esgoto_agua"].mean() if "indice_atendimento_esgoto_agua" in df_ano.columns else 0

c1, c2, c3, c4, c5, c6 = st.columns(6)

with c1:
    kpi_card("Total de Municípios", str(total_mun), f"Ano {ano_sel}", "#60a5fa", "🏙️")
with c2:
    kpi_card("Risco Social Médio", f"{risco_medio:.1f}", "Escala 0–100", "#a78bfa", "⚠️")
with c3:
    kpi_card("Em Risco Crítico", f"{pct_criticos:.1f}%", f"{len(criticos)} municípios", "#ef4444", "🔴")
with c4:
    kpi_card("Baixo Risco", f"{pct_verdes:.1f}%", f"{len(verdes)} municípios", "#22c55e", "🟢")
with c5:
    kpi_card("Cobertura Água", f"{agua_media:.1f}%", "Média estadual", "#38bdf8", "💧")
with c6:
    kpi_card("Cobertura Esgoto", f"{esgoto_medio:.1f}%", "Média estadual", "#fb923c", "🚿")

separador()

# ─── Distribuição do Risco ────────────────────────────────────────────────────
secao_com_tooltip(
    "Distribuição do Índice de Risco Social",
    tooltip=(
        "O que mostra: Como os municípios estão distribuídos ao longo do Índice de Risco Social (0 a 100). "
        "Por que importa: Ajuda a entender se a maioria está em situação boa ou ruim. "
        "Como interpretar: Barras à esquerda (valores baixos) = municípios seguros; barras à direita (valores altos) = municípios críticos. "
        "As linhas tracejadas indicam os quartis da distribuição."
    ),
)
caixa_explicacao(
    "O Índice de Risco Social varia de 0 (sem risco) a 100 (risco máximo). "
    "Ele combina a falta de saneamento (água e esgoto) com as internações hospitalares por doenças relacionadas à falta de saneamento.",
    tipo="info",
)

fig_dist = grafico_distribuicao_risco(df_ano)
st.plotly_chart(fig_dist, use_container_width=True)

separador()

# ─── Ranking + Distribuição por Zona ─────────────────────────────────────────
col_rank, col_zona = st.columns([3, 2])

with col_rank:
    secao_com_tooltip(
        "Municípios Mais Críticos",
        tooltip=(
            "O que mostra: Os 10 municípios com maior Índice de Risco Social. "
            "Como interpretar: Quanto maior a barra, pior é a situação do município. "
            "As cores seguem o código de risco: vermelho = crítico, laranja = elevado."
        ),
    )
    fig_rank = grafico_ranking_municipios(df_ano, top_n=10, modo="piores")
    st.plotly_chart(fig_rank, use_container_width=True)

with col_zona:
    secao_com_tooltip(
        "Municípios por Zona de Risco",
        tooltip=(
            "O que mostra: Quantos municípios estão em cada zona de vulnerabilidade. "
            "Como interpretar: Verde = seguro, Amarelo = atenção, Laranja = preocupante, Vermelho = emergência."
        ),
    )
    caixa_explicacao(
        "As zonas são definidas pela clusterização K-Means, que agrupa municípios "
        "com perfis similares de saneamento e saúde.",
        tipo="info",
    )

    # Gráfico de pizza das zonas
    if "zona_vulnerabilidade" in df_ano.columns:
        contagem_zonas = df_ano["zona_vulnerabilidade"].value_counts().reset_index()
        contagem_zonas.columns = ["zona", "quantidade"]

        from utils.data_loader import ZONA_CORES
        fig_pizza = px.pie(
            contagem_zonas,
            values="quantidade",
            names="zona",
            color="zona",
            color_discrete_map=ZONA_CORES,
            hole=0.45,
        )
        fig_pizza.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#0f172a"),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(148,163,184,0.28)", borderwidth=1, font=dict(size=11)),
            margin=dict(t=20, b=20),
        )
        fig_pizza.update_traces(
            textinfo="percent+label",
            textfont=dict(size=11, color="#0f172a"),
            hovertemplate="<b>%{label}</b><br>%{value} municípios (%{percent})<extra></extra>",
        )
        st.plotly_chart(fig_pizza, use_container_width=True)

separador()

# ─── Tabela de Resumo ─────────────────────────────────────────────────────────
secao_com_tooltip(
    "Resumo por Zona de Vulnerabilidade",
    tooltip=(
        "O que mostra: Médias dos principais indicadores para cada zona de risco. "
        "Como interpretar: Compare as colunas de saneamento entre as zonas — "
        "a Zona Vermelha deve ter os piores valores de água e esgoto."
    ),
)

cols_resumo = [
    "zona_vulnerabilidade", "RISCO_SOCIAL_FINAL",
    "indice_atendimento_total_agua", "indice_atendimento_esgoto_agua",
    "indice_tratamento_esgoto", "Taxa_Morbidade_100k_Hab"
]
cols_existentes = [c for c in cols_resumo if c in df_ano.columns]

if len(cols_existentes) > 1:
    resumo = (
        df_ano[cols_existentes]
        .groupby("zona_vulnerabilidade")
        .mean()
        .round(2)
        .reset_index()
    )
    resumo.columns = [
        "Zona", "Risco Social", "Cobertura Água (%)",
        "Cobertura Esgoto (%)", "Tratamento Esgoto (%)", "Morbidade (por 100k)"
    ][:len(resumo.columns)]

    st.dataframe(
        resumo,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Zona": st.column_config.TextColumn("Zona", width=240),
            "Risco Social": st.column_config.ProgressColumn(
                "Risco Social", min_value=0, max_value=100, format="%.1f"
            ),
        },
    )
