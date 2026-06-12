"""
pages/4_🔬_Análise_Estatística.py
Teste de Kruskal-Wallis e boxplots de morbidade por nível de saneamento.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import scipy.stats as stats

from utils.data_loader import carregar_dados_diamante, nome_amigavel
from utils.charts import grafico_boxplot_morbidade
from utils.theme import apply_theme
from utils.components import cabecalho_pagina, secao_com_tooltip, caixa_explicacao, separador, kpi_card

st.set_page_config(
    page_title="Análise Estatística | Saneamento ES",
    page_icon="🔬",
    layout="wide",
)

apply_theme(show_toggle=True)

df = carregar_dados_diamante()

cabecalho_pagina(
    titulo="Análise Estatística — Saneamento e Saúde",
    descricao=(
        "Investigamos estatisticamente se o nível de saneamento influencia "
        "a taxa de internações hospitalares. Usamos testes não-paramétricos "
        "(adequados para dados reais, que raramente seguem distribuição normal)."
    ),
    icone="🔬",
)

# ─── Normalidade ─────────────────────────────────────────────────────────────
secao_com_tooltip(
    "Teste de Normalidade (Shapiro-Wilk)",
    tooltip=(
        "O que mostra: Se os dados seguem uma distribuição 'normal' (curva em sino). "
        "Por que importa: Muitos testes estatísticos assumem normalidade. "
        "Se os dados não são normais, usamos testes alternativos (como Kruskal-Wallis). "
        "Como interpretar: P-valor < 0.05 significa que os dados NÃO são normais."
    ),
)

caixa_explicacao(
    "A maioria dos dados socioeconômicos e de saúde NÃO segue distribuição normal. "
    "Por isso, usamos o teste de Shapiro-Wilk para confirmar isso e justificar o uso "
    "do teste de Kruskal-Wallis (equivalente não-paramétrico da ANOVA).",
    tipo="info",
)

cols_normalidade = [
    "RISCO_SOCIAL_FINAL", "Taxa_Morbidade_100k_Hab",
    "indice_atendimento_total_agua", "indice_atendimento_esgoto_agua",
    "vazio_sanitario",
]
cols_existentes = [c for c in cols_normalidade if c in df.columns]

if cols_existentes:
    resultados_shapiro = []
    for col in cols_existentes:
        serie = df[col].dropna()
        if len(serie) < 3:
            continue
        amostra = serie.sample(min(5000, len(serie)), random_state=42)
        _, p_val = stats.shapiro(amostra)
        resultados_shapiro.append({
            "Variável": nome_amigavel(col),
            "N (amostras)": len(amostra),
            "P-valor": round(p_val, 6),
            "Distribuição": "🟡 Normal" if p_val > 0.05 else "🔴 Não-Normal",
            "Nota": "Usar teste paramétrico" if p_val > 0.05 else "Usar Kruskal-Wallis",
        })

    df_shapiro = pd.DataFrame(resultados_shapiro)
    st.dataframe(df_shapiro, use_container_width=True, hide_index=True)

separador()

# ─── Kruskal-Wallis ───────────────────────────────────────────────────────────
secao_com_tooltip(
    "Teste de Kruskal-Wallis — Morbidade por Nível de Saneamento",
    tooltip=(
        "O que mostra: Se a taxa de internações é diferente entre municípios com "
        "baixo, médio e alto saneamento. "
        "Por que importa: Confirma estatisticamente que saneamento afeta a saúde. "
        "Como interpretar: P-valor < 0.05 confirma que há diferença real entre os grupos "
        "(não é coincidência). Statística H maior = diferença mais forte."
    ),
)

col_san = st.selectbox(
    "Indicador de Saneamento para o Teste",
    [c for c in ["indice_atendimento_total_agua", "indice_atendimento_esgoto_agua", "vazio_sanitario"] if c in df.columns],
    format_func=nome_amigavel,
)
col_sau = st.selectbox(
    "Indicador de Saúde",
    [c for c in ["Taxa_Morbidade_100k_Hab", "RISCO_SOCIAL_FINAL", "internacoes_agua"] if c in df.columns],
    format_func=nome_amigavel,
)

df_kw = df[[col_san, col_sau]].dropna().copy()

if len(df_kw) >= 30:
    try:
        df_kw["grupo_saneamento"] = pd.qcut(
            df_kw[col_san], q=3,
            labels=["Baixo Saneamento", "Saneamento Médio", "Alto Saneamento"],
            duplicates="drop",
        )
        grupos = [
            df_kw[df_kw["grupo_saneamento"] == g][col_sau]
            for g in df_kw["grupo_saneamento"].cat.categories
        ]
        grupos = [g for g in grupos if len(g) > 1]

        if len(grupos) >= 2:
            stat_kw, p_kw = stats.kruskal(*grupos)

            # KPIs do teste
            col_k1, col_k2, col_k3 = st.columns(3)
            with col_k1:
                kpi_card("Estatística H", f"{stat_kw:.3f}", "Kruskal-Wallis", "#a78bfa", "📊")
            with col_k2:
                kpi_card("P-valor", f"{p_kw:.6f}", "", "#60a5fa", "🔢")
            with col_k3:
                resultado = "✅ Diferença Confirmada" if p_kw < 0.05 else "❌ Sem Diferença Significativa"
                kpi_card("Resultado", resultado, "α = 0.05", "#22c55e" if p_kw < 0.05 else "#94a3b8", "🎯")

            if p_kw < 0.05:
                caixa_explicacao(
                    f"✅ O teste confirma que os grupos de saneamento têm taxas de morbidade "
                    f"**estatisticamente diferentes** (p = {p_kw:.6f} < 0.05). "
                    "Isso significa que o saneamento realmente impacta a saúde da população.",
                    tipo="sucesso",
                )
            else:
                caixa_explicacao(
                    f"Os grupos não apresentam diferença estatisticamente significativa (p = {p_kw:.4f}). "
                    "Tente outros indicadores ou verifique o tamanho da amostra.",
                    tipo="aviso",
                )
    except Exception as e:
        caixa_explicacao(f"Erro ao executar o teste: {e}", tipo="critico")

separador()

# ─── Boxplot ──────────────────────────────────────────────────────────────────
secao_com_tooltip(
    "Boxplot — Distribuição da Morbidade por Grupo de Saneamento",
    tooltip=(
        "O que mostra: A distribuição das internações hospitalares em cada grupo de saneamento. "
        "Como interpretar: A linha no meio da caixa é a mediana. A caixa contém 50% dos municípios. "
        "Pontos externos = municípios atípicos (outliers). "
        "Espera-se que o grupo de ALTO saneamento tenha caixas mais baixas (menos internações)."
    ),
)

fig_box = grafico_boxplot_morbidade(df)
st.plotly_chart(fig_box, use_container_width=True)

separador()

# ─── Histogramas de distribuição ─────────────────────────────────────────────
secao_com_tooltip(
    "Distribuição das Variáveis Principais",
    tooltip=(
        "O que mostra: A forma da distribuição de cada variável selecionada. "
        "Como interpretar: Histogramas com cauda à direita (assimétricos) confirmam "
        "que os dados não são normais — justificando o uso de testes não-paramétricos."
    ),
)

col_hist = st.selectbox(
    "Variável para visualizar",
    cols_existentes,
    format_func=nome_amigavel,
)

dados_hist = df[col_hist].dropna()
fig_hist = px.histogram(
    dados_hist,
    x=col_hist,
    nbins=30,
    marginal="box",
    color_discrete_sequence=["#60a5fa"],
    labels={col_hist: nome_amigavel(col_hist)},
    title=f"Distribuição: {nome_amigavel(col_hist)}",
)
fig_hist.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#0f172a"),
    xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
    showlegend=False,
)
st.plotly_chart(fig_hist, use_container_width=True)

# Estatísticas descritivas
st.markdown("##### Estatísticas Descritivas")
desc = dados_hist.describe().round(3)
st.dataframe(desc.to_frame().T, use_container_width=True)
