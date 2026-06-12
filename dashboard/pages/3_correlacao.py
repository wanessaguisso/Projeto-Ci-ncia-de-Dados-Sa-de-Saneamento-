"""
pages/3_📈_Correlação.py
Análise de correlação de Spearman entre saneamento e saúde.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import carregar_dados_diamante, nome_amigavel
from utils.charts import grafico_heatmap_correlacao
from utils.theme import apply_theme
from utils.components import cabecalho_pagina, secao_com_tooltip, caixa_explicacao, separador

st.set_page_config(
    page_title="Correlação | Saneamento e Saúde no ES",
    page_icon="📈",
    layout="wide",
)

apply_theme(show_toggle=True)

df = carregar_dados_diamante()

cabecalho_pagina(
    titulo="Correlação entre Saneamento e Saúde",
    descricao=(
        "A correlação de Spearman mede a força da relação entre duas variáveis. "
        "Aqui investigamos como indicadores de saneamento básico se relacionam com "
        "internações hospitalares e o índice de risco social."
    ),
    icone="📈",
)

# ─── Explicação do método ────────────────────────────────────────────────────
caixa_explicacao(
    "A Correlação de Spearman é um método estatístico que mede se duas variáveis "
    "tendem a aumentar ou diminuir juntas. Valores próximos de +1 indicam que as duas "
    "variáveis crescem juntas (correlação positiva). Valores próximos de -1 indicam "
    "relação inversa (quando uma sobe, a outra desce). Valores próximos de 0 = sem relação.",
    tipo="info",
)

separador()

# ─── Seleção de variáveis ────────────────────────────────────────────────────
colunas_saneamento = [
    "indice_atendimento_total_agua",
    "indice_atendimento_esgoto_agua",
    "indice_tratamento_esgoto",
    "indice_perda_distribuicao_agua",
    "vazio_sanitario",
    "def_agua",
    "def_esgoto",
]
colunas_saude = [
    "Taxa_Morbidade_100k_Hab",
    "internacoes_agua",
    "internacoes_esgoto",
    "RISCO_SOCIAL_FINAL",
]
colunas_financeiras = [
    "investimento_total_consolidado",
    "eficiencia_arrecadacao",
]

todas_possiveis = colunas_saneamento + colunas_saude + colunas_financeiras
existentes = [c for c in todas_possiveis if c in df.columns]

col_s1, col_s2 = st.columns([2, 3])
with col_s1:
    cols_sel = st.multiselect(
        "📌 Selecione as variáveis para correlação",
        existentes,
        default=existentes[:min(8, len(existentes))],
        format_func=nome_amigavel,
    )

if len(cols_sel) < 2:
    st.warning("⚠️ Selecione ao menos 2 variáveis para calcular a correlação.")
    st.stop()

separador()

# ─── Heatmap principal ───────────────────────────────────────────────────────
secao_com_tooltip(
    "Heatmap de Correlação de Spearman",
    tooltip=(
        "O que mostra: A força da relação entre cada par de variáveis. "
        "Cor verde = correlação positiva (variáveis crescem juntas). "
        "Cor vermelha = correlação negativa (uma sobe, a outra desce). "
        "Como interpretar: Foque nas células que conectam saneamento (linhas) com saúde (colunas)."
    ),
)

fig_hm = grafico_heatmap_correlacao(df, cols_sel)
st.plotly_chart(fig_hm, use_container_width=True)

separador()

# ─── Scatter: Saneamento vs Morbidade ────────────────────────────────────────
secao_com_tooltip(
    "Relação entre Saneamento e Internações Hospitalares",
    tooltip=(
        "O que mostra: Cada ponto é um par (município, ano). O eixo X é o saneamento, "
        "o eixo Y são as internações por 100 mil habitantes. "
        "Por que importa: Uma nuvem de pontos inclinada de cima para baixo (da esquerda para a direita) "
        "indica que municípios com mais saneamento têm menos internações. "
        "Como interpretar: Linha de tendência descendente confirma a hipótese de que "
        "melhor saneamento = menos doenças."
    ),
)

col_x, col_y = st.columns(2)
with col_x:
    x_col = st.selectbox(
        "Eixo X (Saneamento)",
        [c for c in existentes if c in df.columns],
        format_func=nome_amigavel,
        index=0,
    )
with col_y:
    y_col = st.selectbox(
        "Eixo Y (Saúde/Risco)",
        [c for c in existentes if c in df.columns],
        format_func=nome_amigavel,
        index=min(len(existentes)-1, len([c for c in colunas_saneamento if c in existentes])),
    )

dados_scatter = df.dropna(subset=[x_col, y_col]).copy()

if not dados_scatter.empty:
    nome_col = "nome_municipio" if "nome_municipio" in dados_scatter.columns else "id_municipio"

    fig_scatter = px.scatter(
        dados_scatter,
        x=x_col,
        y=y_col,
        color="zona_vulnerabilidade" if "zona_vulnerabilidade" in dados_scatter.columns else None,
        hover_name=nome_col if nome_col in dados_scatter.columns else None,
        hover_data={"ano": True} if "ano" in dados_scatter.columns else {},
        trendline="ols",
        color_discrete_map={
            "Zona Verde - Baixo Risco": "#22c55e",
            "Zona Amarela - Risco Moderado": "#eab308",
            "Zona Laranja - Risco Elevado": "#f97316",
            "Zona Vermelha - Risco Crítico": "#ef4444",
        },
        labels={
            x_col: nome_amigavel(x_col),
            y_col: nome_amigavel(y_col),
            "zona_vulnerabilidade": "Zona",
        },
        opacity=0.7,
    )
    fig_scatter.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#0f172a"),
        legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(148,163,184,0.28)", borderwidth=1),
        xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Calcular correlação pontual
    import scipy.stats as stats
    cor_val, p_val = stats.spearmanr(
        dados_scatter[x_col].dropna(),
        dados_scatter[y_col].dropna(),
    )

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Correlação de Spearman (ρ)", f"{cor_val:.3f}")
    with col_r2:
        st.metric("P-valor", f"{p_val:.4f}")
    with col_r3:
        sig = "✅ Significativa (p < 0.05)" if p_val < 0.05 else "❌ Não Significativa (p ≥ 0.05)"
        st.metric("Significância Estatística", sig)

    if p_val < 0.05:
        interpretacao = (
            f"A correlação entre **{nome_amigavel(x_col)}** e **{nome_amigavel(y_col)}** é "
            f"**{'positiva' if cor_val > 0 else 'negativa'}** e **estatisticamente significativa** "
            f"(ρ = {cor_val:.3f}, p = {p_val:.4f}). "
        )
        if cor_val < 0:
            interpretacao += "Isso indica que, à medida que o saneamento aumenta, as internações tendem a **diminuir**."
        else:
            interpretacao += "Isso indica que as duas variáveis tendem a crescer juntas."
        caixa_explicacao(interpretacao, tipo="sucesso")
    else:
        caixa_explicacao(
            f"A correlação não é estatisticamente significativa (p = {p_val:.4f}). "
            "Pode ser necessário usar outros indicadores ou filtrar por ano específico.",
            tipo="aviso",
        )
