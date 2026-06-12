"""
pages/7_📊_Análises_Avançadas.py
Análises avançadas: ranking, evolução temporal, investimento vs risco, top 10.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.data_loader import carregar_dados_zonas, carregar_dados_diamante, nome_amigavel
from utils.charts import (
    grafico_ranking_municipios,
    grafico_evolucao_temporal,
    grafico_investimento_vs_risco,
    grafico_top10_saneamento,
)
from utils.theme import apply_theme
from utils.components import cabecalho_pagina, secao_com_tooltip, caixa_explicacao, separador, kpi_card

st.set_page_config(
    page_title="Análises Avançadas | Saneamento ES",
    page_icon="📊",
    layout="wide",
)

apply_theme(show_toggle=True)

df_zonas = carregar_dados_zonas()
df_diamante = carregar_dados_diamante()

cabecalho_pagina(
    titulo="Análises Avançadas",
    descricao=(
        "Ranking de municípios, evolução temporal, eficiência de investimentos e comparações. "
        "Descubra tendências e identifique oportunidades de melhoria no saneamento do ES."
    ),
    icone="📊",
)

# ─── Tabs de análises ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Ranking",
    "📅 Linha do Tempo",
    "💰 Investimento × Risco",
    "🏅 Top 10 Saneamento",
    "📉 Tendências",
])

# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    secao_com_tooltip(
        "Ranking dos Municípios por Risco Social",
        tooltip=(
            "O que mostra: Os municípios ordenados pelo Índice de Risco Social. "
            "Como interpretar: Os primeiros (cor vermelha) são os que mais necessitam de atenção. "
            "Os últimos (cor verde) são os melhores exemplos de saneamento."
        ),
    )

    col_r1, col_r2, col_r3 = st.columns([2, 2, 3])
    with col_r1:
        modo = st.radio("📊 Modo", ["Piores (Mais Críticos)", "Melhores (Melhor Saneamento)"])
    with col_r2:
        top_n = st.slider("🔢 Quantidade", 5, 30, 15)

    modo_key = "piores" if "Piores" in modo else "melhores"
    fig_rank = grafico_ranking_municipios(df_zonas, top_n=top_n, modo=modo_key)
    st.plotly_chart(fig_rank, use_container_width=True)

    if modo_key == "piores":
        caixa_explicacao(
            "Estes municípios necessitam de intervenção prioritária em saneamento básico. "
            "Investimentos aqui têm o maior potencial de reduzir internações hospitalares.",
            tipo="critico",
        )
    else:
        caixa_explicacao(
            "Estes municípios são referências em saneamento. "
            "Suas práticas e políticas podem ser replicadas nos municípios críticos.",
            tipo="sucesso",
        )

# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    secao_com_tooltip(
        "Evolução Temporal do Risco Social",
        tooltip=(
            "O que mostra: Como o risco social evoluiu ao longo dos anos para os municípios selecionados. "
            "Como interpretar: Linhas descendo = melhora. Linhas subindo = piora. "
            "Compare a trajetória entre municípios para identificar diferenças de política."
        ),
    )

    municipios = sorted(df_zonas["nome_municipio"].dropna().unique())

    # Sugerir os 5 mais críticos como padrão
    df_latest = df_zonas[df_zonas["ano"] == int(df_zonas["ano"].max())]
    top5_criticos = df_latest.nlargest(5, "RISCO_SOCIAL_FINAL")["nome_municipio"].tolist()

    municipios_sel = st.multiselect(
        "🏙️ Selecione os municípios (máx. 10)",
        municipios,
        default=top5_criticos[:3],
        max_selections=10,
    )

    if municipios_sel:
        fig_evo = grafico_evolucao_temporal(df_zonas, municipios=municipios_sel)
        st.plotly_chart(fig_evo, use_container_width=True)

        # Estatística de tendência
        st.markdown("##### Tendência de cada município")
        tendencias = []
        for m in municipios_sel:
            serie = (
                df_zonas[df_zonas["nome_municipio"] == m]
                .dropna(subset=["RISCO_SOCIAL_FINAL"])
                .sort_values("ano")
            )
            if len(serie) >= 2:
                primeiro = float(serie.iloc[0]["RISCO_SOCIAL_FINAL"])
                ultimo = float(serie.iloc[-1]["RISCO_SOCIAL_FINAL"])
                variacao = ultimo - primeiro
                tendencias.append({
                    "Município": m,
                    "Risco (início)": round(primeiro, 2),
                    "Risco (atual)": round(ultimo, 2),
                    "Variação": round(variacao, 2),
                    "Tendência": "📈 Piorou" if variacao > 0 else "📉 Melhorou" if variacao < 0 else "→ Estável",
                })

        if tendencias:
            st.dataframe(pd.DataFrame(tendencias), use_container_width=True, hide_index=True)
    else:
        caixa_explicacao("Selecione ao menos um município para visualizar a evolução.", tipo="aviso")

# ──────────────────────────────────────────────────────────────────────────────
with tab3:
    secao_com_tooltip(
        "Eficiência do Investimento — Investimento per Capita × Risco Social",
        tooltip=(
            "O que mostra: Cada bolha é um município. Eixo X = investimento per capita em saneamento. "
            "Eixo Y = nível de risco social. O tamanho da bolha = população. "
            "Como interpretar: Municípios no canto inferior direito (alto investimento, baixo risco) "
            "são os mais eficientes. Municípios no canto superior esquerdo são ineficientes."
        ),
    )

    caixa_explicacao(
        "Esta análise mostra se os municípios que investem mais em saneamento conseguem "
        "reduzir o risco social. Correlação negativa (investimento alto = risco baixo) "
        "confirma que o investimento está funcionando.",
        tipo="info",
    )

    fig_inv = grafico_investimento_vs_risco(df_zonas)
    st.plotly_chart(fig_inv, use_container_width=True)

    # Correlação investimento x risco
    df_inv_calc = df_zonas.dropna(subset=["investimento_total_consolidado", "RISCO_SOCIAL_FINAL", "populacao_ref"]).copy()
    if "ano" in df_inv_calc.columns:
        idx = df_inv_calc.groupby("nome_municipio")["ano"].idxmax()
        df_inv_calc = df_inv_calc.loc[idx]

    df_inv_calc = df_inv_calc[df_inv_calc["populacao_ref"] > 0].copy()
    df_inv_calc["invest_pc"] = df_inv_calc["investimento_total_consolidado"] / df_inv_calc["populacao_ref"]

    if len(df_inv_calc) >= 5:
        import scipy.stats as stats
        r, p = stats.spearmanr(df_inv_calc["invest_pc"], df_inv_calc["RISCO_SOCIAL_FINAL"])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Correlação (Spearman ρ)", f"{r:.3f}")
        with col_c2:
            st.metric("P-valor", f"{p:.4f}")

        if p < 0.05 and r < 0:
            caixa_explicacao(
                "✅ O investimento per capita está negativamente correlacionado com o risco social: "
                "municípios que mais investem tendem a ter menor risco. O investimento está surtindo efeito!",
                tipo="sucesso",
            )
        elif p < 0.05 and r > 0:
            caixa_explicacao(
                "⚠️ Municípios com maior investimento ainda apresentam alto risco. "
                "Isso pode indicar que o investimento está sendo direcionado para os municípios mais críticos, "
                "que ainda estão em processo de melhoria.",
                tipo="aviso",
            )

# ──────────────────────────────────────────────────────────────────────────────
with tab4:
    secao_com_tooltip(
        "Top 10 — Melhores e Piores em Saneamento",
        tooltip=(
            "O que mostra: Os 10 municípios com maior e menor cobertura de água potável. "
            "Como interpretar: Barras verdes = referências positivas; barras vermelhas = casos críticos. "
            "Use para identificar disparidades regionais."
        ),
    )

    fig_top10 = grafico_top10_saneamento(df_zonas)
    st.plotly_chart(fig_top10, use_container_width=True)

    separador()

    # Análise por dimensão
    secao_com_tooltip(
        "Comparação de Indicadores: Melhores × Piores",
        tooltip=(
            "O que mostra: A diferença entre os 20% melhores e 20% piores municípios "
            "em cada indicador de saneamento. "
            "Como interpretar: Diferença grande = há muito a melhorar. Diferença pequena = mais homogêneo."
        ),
    )

    cols_comp = [
        "indice_atendimento_total_agua",
        "indice_atendimento_esgoto_agua",
        "indice_tratamento_esgoto",
        "RISCO_SOCIAL_FINAL",
    ]
    cols_comp_exist = [c for c in cols_comp if c in df_zonas.columns]

    if cols_comp_exist:
        df_latest2 = df_zonas[df_zonas["ano"] == int(df_zonas["ano"].max())].copy()

        comparacao = []
        for col in cols_comp_exist:
            q20 = df_latest2[col].quantile(0.20)
            q80 = df_latest2[col].quantile(0.80)
            comparacao.append({
                "Indicador": nome_amigavel(col),
                "Piores 20% (média)": round(df_latest2[df_latest2[col] <= q20][col].mean(), 2),
                "Melhores 20% (média)": round(df_latest2[df_latest2[col] >= q80][col].mean(), 2),
                "Diferença (gap)": round(abs(
                    df_latest2[df_latest2[col] >= q80][col].mean() -
                    df_latest2[df_latest2[col] <= q20][col].mean()
                ), 2),
            })
        st.dataframe(pd.DataFrame(comparacao), use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab5:
    secao_com_tooltip(
        "Tendências Estaduais — Evolução dos Indicadores",
        tooltip=(
            "O que mostra: A evolução da média estadual dos indicadores ao longo do tempo. "
            "Como interpretar: Linhas crescentes em cobertura de água = o estado está melhorando. "
            "Linhas decrescentes em risco = a situação está melhorando."
        ),
    )

    caixa_explicacao(
        "Esta análise mostra a evolução histórica da média estadual. "
        "É uma visão macro do progresso do saneamento no Espírito Santo.",
        tipo="info",
    )

    cols_tendencia = [
        "indice_atendimento_total_agua",
        "indice_atendimento_esgoto_agua",
        "indice_tratamento_esgoto",
        "RISCO_SOCIAL_FINAL",
        "Taxa_Morbidade_100k_Hab",
    ]
    cols_tend_exist = [c for c in cols_tendencia if c in df_diamante.columns]

    cols_tend_sel = st.multiselect(
        "📊 Selecione os indicadores",
        cols_tend_exist,
        default=cols_tend_exist[:3],
        format_func=nome_amigavel,
    )

    if cols_tend_sel:
        serie_estadual = (
            df_diamante.groupby("ano")[cols_tend_sel]
            .mean()
            .reset_index()
        )
        serie_melted = serie_estadual.melt(
            id_vars="ano", value_vars=cols_tend_sel,
            var_name="indicador", value_name="média"
        )
        serie_melted["indicador"] = serie_melted["indicador"].map(nome_amigavel)

        fig_tend = px.line(
            serie_melted,
            x="ano",
            y="média",
            color="indicador",
            markers=True,
            color_discrete_sequence=["#60a5fa", "#34d399", "#f472b6", "#fb923c", "#a78bfa"],
            labels={"ano": "Ano", "média": "Valor Médio Estadual", "indicador": "Indicador"},
            title="Tendência Histórica — Médias Estaduais",
        )
        fig_tend.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="#0f172a"),
            xaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            yaxis=dict(gridcolor="rgba(148,163,184,0.1)"),
            legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(148,163,184,0.28)", borderwidth=1),
            title_font_color="#0f172a",
        )
        st.plotly_chart(fig_tend, use_container_width=True)
