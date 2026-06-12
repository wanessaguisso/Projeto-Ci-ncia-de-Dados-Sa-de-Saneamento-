"""
pages/5_🤖_Clusterização.py
Visualização dos clusters K-Means e zonas de vulnerabilidade.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.data_loader import carregar_dados_zonas, ZONA_CORES, nome_amigavel
from utils.charts import grafico_scatter_zonas
from utils.theme import apply_theme
from utils.components import cabecalho_pagina, secao_com_tooltip, caixa_explicacao, separador, kpi_card, badge_zona

st.set_page_config(
    page_title="Clusterização | Saneamento e Saúde no ES",
    page_icon="🤖",
    layout="wide",
)

apply_theme(show_toggle=True)

df = carregar_dados_zonas()

cabecalho_pagina(
    titulo="Clusterização K-Means — Zonas de Vulnerabilidade",
    descricao=(
        "O algoritmo K-Means agrupou automaticamente os municípios em zonas de risco "
        "semelhantes, baseando-se em indicadores de saneamento e saúde. "
        "Cada zona representa um perfil distinto de vulnerabilidade social."
    ),
    icone="🤖",
)

caixa_explicacao(
    "K-Means é um algoritmo de Machine Learning que encontra grupos naturais nos dados. "
    "Ele agrupou os municípios sem saber previamente as 'respostas certas' — apenas com base nos números. "
    "O resultado revelou 4 perfis distintos de vulnerabilidade, do verde (seguro) ao vermelho (crítico).",
    tipo="info",
)

separador()

# ─── Filtros ──────────────────────────────────────────────────────────────────
anos = sorted(df["ano"].dropna().unique().astype(int), reverse=True)
col_f1, col_f2, col_f3 = st.columns([2, 2, 3])
with col_f1:
    ano_sel = st.selectbox("📅 Ano de Referência", anos, index=0)
with col_f2:
    x_col = st.selectbox(
        "Eixo X",
        [c for c in ["vazio_sanitario", "indice_atendimento_total_agua", "def_agua"] if c in df.columns],
        format_func=nome_amigavel,
    )
with col_f3:
    y_col = st.selectbox(
        "Eixo Y",
        [c for c in ["Taxa_Morbidade_100k_Hab", "RISCO_SOCIAL_FINAL", "indice_tratamento_esgoto"] if c in df.columns],
        format_func=nome_amigavel,
    )

df_ano = df[df["ano"] == ano_sel].copy()

separador()

# ─── Scatter principal ────────────────────────────────────────────────────────
secao_com_tooltip(
    "Mapa de Dispersão por Zona de Vulnerabilidade",
    tooltip=(
        "O que mostra: Cada ponto é um município posicionado pelo nível de saneamento (X) "
        "e saúde (Y). As cores indicam a zona de risco. "
        "Como interpretar: Municípios no canto superior esquerdo (muito saneamento, alta morbidade) "
        "são casos atípicos. O padrão esperado é: mais saneamento = menos morbidade (nuvem inclinada)."
    ),
)

fig_scatter = grafico_scatter_zonas(df_ano, x_col=x_col, y_col=y_col)
st.plotly_chart(fig_scatter, use_container_width=True)

separador()

# ─── Perfil dos Clusters ──────────────────────────────────────────────────────
secao_com_tooltip(
    "Perfil Médio por Zona de Vulnerabilidade",
    tooltip=(
        "O que mostra: A média de cada indicador para os municípios de cada zona. "
        "Como usar: Compare as linhas — a Zona Vermelha deve ter os piores indicadores "
        "e a Zona Verde os melhores. "
        "Isso confirma que o agrupamento automático capturou grupos realmente distintos."
    ),
)

cols_perfil = [
    "zona_vulnerabilidade",
    "RISCO_SOCIAL_FINAL",
    "vazio_sanitario",
    "indice_atendimento_total_agua",
    "indice_atendimento_esgoto_agua",
    "indice_tratamento_esgoto",
    "Taxa_Morbidade_100k_Hab",
    "investimento_total_consolidado",
]
cols_existentes_perfil = [c for c in cols_perfil if c in df_ano.columns]

if len(cols_existentes_perfil) > 1:
    perfil = (
        df_ano[cols_existentes_perfil]
        .groupby("zona_vulnerabilidade")
        .agg(["mean", "count"])
    )

    # Simplificar para display
    perfil_simple = df_ano[cols_existentes_perfil].groupby("zona_vulnerabilidade").mean().round(2)
    perfil_simple["Nº Municípios"] = df_ano.groupby("zona_vulnerabilidade")["id_municipio"].nunique()
    perfil_simple = perfil_simple.reset_index()

    cols_rename = {
        "zona_vulnerabilidade": "Zona",
        "RISCO_SOCIAL_FINAL": "Risco Social",
        "vazio_sanitario": "Vazio Sanitário (%)",
        "indice_atendimento_total_agua": "Cobertura Água (%)",
        "indice_atendimento_esgoto_agua": "Cobertura Esgoto (%)",
        "indice_tratamento_esgoto": "Tratamento Esgoto (%)",
        "Taxa_Morbidade_100k_Hab": "Morbidade (100k)",
        "investimento_total_consolidado": "Investimento (R$)",
    }
    perfil_simple = perfil_simple.rename(columns={k: v for k, v in cols_rename.items() if k in perfil_simple.columns})

    st.dataframe(perfil_simple, use_container_width=True, hide_index=True)

separador()

# ─── Badges e KPIs por zona ───────────────────────────────────────────────────
st.markdown("### Resumo por Zona")

zonas_ordenadas = [
    "Zona Verde - Baixo Risco",
    "Zona Amarela - Risco Moderado",
    "Zona Laranja - Risco Elevado",
    "Zona Vermelha - Risco Critico",
]
descricoes_zona = {
    "Zona Verde - Baixo Risco": (
        "Municípios com boa cobertura de água e esgoto, baixas taxas de internação. "
        "São referências de boas práticas em saneamento."
    ),
    "Zona Amarela - Risco Moderado": (
        "Saneamento razoável, mas com pontos de atenção. "
        "Investimentos focados podem evitar que esses municípios piorem."
    ),
    "Zona Laranja - Risco Elevado": (
        "Déficits significativos de saneamento, com impacto crescente na saúde. "
        "Ação prioritária é necessária."
    ),
    "Zona Vermelha - Risco Critico": (
        "Situação de emergência social. Falta grave de saneamento combinada com "
        "altas taxas de internação hospitalar por doenças evitáveis."
    ),
}

cols_zona = st.columns(4)
for col, zona in zip(cols_zona, zonas_ordenadas):
    with col:
        subset_z = df_ano[df_ano["zona_vulnerabilidade"] == zona]
        n_mun = len(subset_z)
        risco_med = subset_z["RISCO_SOCIAL_FINAL"].mean() if not subset_z.empty else 0

        badge_zona(zona)
        st.markdown(f"**{n_mun} municípios** · Risco médio: **{risco_med:.1f}**")
        st.caption(descricoes_zona.get(zona, ""))

separador()

# ─── Gráfico Radar por Zona ───────────────────────────────────────────────────
secao_com_tooltip(
    "Comparação em Radar — Perfil de Cada Zona",
    tooltip=(
        "O que mostra: Um gráfico de teia/aranha comparando as zonas em múltiplas dimensões. "
        "Como interpretar: Uma área maior = melhores indicadores. "
        "A Zona Verde deve ter a maior área; a Zona Vermelha, a menor."
    ),
)

metricas_radar = {
    "Cobertura Água": "indice_atendimento_total_agua",
    "Cobertura Esgoto": "indice_atendimento_esgoto_agua",
    "Tratamento Esgoto": "indice_tratamento_esgoto",
    "Baixo Risco\n(invertido)": "RISCO_SOCIAL_FINAL",
}
metricas_existentes = {k: v for k, v in metricas_radar.items() if v in df_ano.columns}

if metricas_existentes:
    fig_radar = go.Figure()
    cors_radar = {
        "Zona Verde - Baixo Risco": "#22c55e",
        "Zona Amarela - Risco Moderado": "#eab308",
        "Zona Laranja - Risco Elevado": "#f97316",
        "Zona Vermelha - Risco Crítico": "#ef4444",
    }
    categories = list(metricas_existentes.keys())

    for zona in zonas_ordenadas:
        subset_z = df_ano[df_ano["zona_vulnerabilidade"] == zona]
        if subset_z.empty:
            continue

        vals = []
        for label, col in metricas_existentes.items():
            v = float(subset_z[col].mean() or 0)
            if "invertido" in label or col == "RISCO_SOCIAL_FINAL":
                v = max(0, 100 - v)
            vals.append(min(100, max(0, v)))

        cor = cors_radar.get(zona, "#94a3b8")
        cor_fill = "rgba(148,163,184,0.18)"
        if isinstance(cor, str) and cor.startswith("#") and len(cor) == 7:
            r = int(cor[1:3], 16)
            g = int(cor[3:5], 16)
            b = int(cor[5:7], 16)
            cor_fill = f"rgba({r},{g},{b},0.18)"
        fig_radar.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=cor_fill,
            line=dict(color=cor, width=2),
            name=zona,
        ))

    fig_radar.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        polar=dict(
            bgcolor="rgba(255,255,255,0.72)",
            radialaxis=dict(range=[0, 100], gridcolor="rgba(148,163,184,0.15)",
                           tickfont=dict(color="#334155")),
            angularaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        ),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(148,163,184,0.28)", borderwidth=1),
        title=dict(text="Perfil Médio por Zona de Vulnerabilidade",
                  font=dict(color="#0f172a", size=15), x=0.02),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
