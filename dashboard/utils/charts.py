"""
utils/charts.py
Funções de criação de gráficos Plotly reutilizáveis.
Todos os gráficos seguem o design system do dashboard com suporte a modo claro/escuro.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from utils.data_loader import ZONA_CORES, nome_amigavel
from utils.theme import get_theme_mode, THEME_LIGHT, THEME_DARK

# ─── Design System ────────────────────────────────────────────────────────────

def _get_tokens():
    """Retorna os tokens do tema ativo."""
    mode = get_theme_mode()
    return THEME_LIGHT if mode == "Claro" else THEME_DARK


def _get_layout_base():
    """Retorna o layout base adaptado ao tema ativo."""
    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=tokens['title'], size=13),
        margin=dict(t=60, b=50, l=60, r=40),
        legend=dict(
            bgcolor=tokens['surface'],
            bordercolor=tokens['surface_border'],
            borderwidth=1,
            font=dict(size=12, color=tokens['text']),
        ),
        coloraxis_colorbar=dict(
            tickfont=dict(color=tokens['text']),
            outlinecolor=tokens['surface_border'],
            outlinewidth=1,
        ),
        hoverlabel=dict(
            bgcolor=tokens['surface'],
            font=dict(family="Inter, sans-serif", color=tokens['title'], size=12),
            bordercolor=tokens['surface_border'],
        ),
    )


def _get_grid_style():
    """Retorna o estilo de grid adaptado ao tema."""
    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    grid_color = "rgba(148,163,184,0.08)" if is_dark else "rgba(148,163,184,0.15)"
    zeroline_color = "rgba(148,163,184,0.15)" if is_dark else "rgba(148,163,184,0.25)"

    return dict(
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=zeroline_color,
            color=tokens['text'],
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=zeroline_color,
            color=tokens['text'],
        ),
    )


# Paleta de cores para risco (funciona em ambos os modos)
SEQUENCIAL_RISK = [
    [0.0, "#22C55E"],   # Verde
    [0.33, "#EAB308"],  # Amarelo
    [0.66, "#F97316"],  # Laranja
    [1.0, "#EF4444"],   # Vermelho
]


def _hex_to_rgba(hex_color: str, alpha: float = 0.15) -> str:
    """Converte cor hexadecimal (#RRGGBB) para string rgba(...)."""
    color = hex_color.lstrip("#")
    if len(color) != 6:
        return f"rgba(148,163,184,{alpha})"
    r = int(color[0:2], 16)
    g = int(color[2:4], 16)
    b = int(color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _aplicar_layout(fig: go.Figure, titulo: str = "", **kwargs) -> go.Figure:
    """Aplica layout padrão ao figure com suporte ao tema ativo."""
    tokens = _get_tokens()
    layout = {**_get_layout_base(), **_get_grid_style()}

    if titulo:
        layout["title"] = dict(
            text=titulo,
            font=dict(size=17, color=tokens['title'], family="Inter, sans-serif"),
            x=0.02,
            xanchor="left",
        )

    layout.update(kwargs)
    fig.update_layout(**layout)
    return fig


# ─── 1. Distribuição do Risco Social ─────────────────────────────────────────

def grafico_distribuicao_risco(df: pd.DataFrame, ano: int | None = None) -> go.Figure:
    """
    Histograma + KDE da distribuição do Índice de Risco Social.
    Mostra quão concentrados estão os municípios em cada faixa de risco.
    """
    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    dados = df.copy()
    if ano:
        dados = dados[dados["ano"] == ano]
    dados = dados.dropna(subset=["RISCO_SOCIAL_FINAL"])

    fig = go.Figure()

    # Histograma
    fig.add_trace(go.Histogram(
        x=dados["RISCO_SOCIAL_FINAL"],
        nbinsx=25,
        name="Municípios",
        marker=dict(
            color=dados["RISCO_SOCIAL_FINAL"],
            colorscale=SEQUENCIAL_RISK,
            line=dict(
                color="rgba(255,255,255,0.2)" if is_dark else "rgba(15,23,42,0.1)",
                width=0.8
            ),
        ),
        opacity=0.9,
        hovertemplate="<b>Risco:</b> %{x:.1f}<br><b>Municípios:</b> %{y}<extra></extra>",
    ))

    # Linhas de quartis
    q25, q50, q75 = dados["RISCO_SOCIAL_FINAL"].quantile([0.25, 0.50, 0.75]).values
    for val, label, color in [
        (q25, "Q1 (25%)", "#22C55E"),
        (q50, "Mediana", "#EAB308"),
        (q75, "Q3 (75%)", "#EF4444"),
    ]:
        fig.add_vline(
            x=val,
            line_dash="dash",
            line_color=color,
            line_width=2,
            annotation=dict(
                text=f"{label}: {val:.1f}",
                font=dict(color=color, size=11, family="Inter, sans-serif"),
                bgcolor="rgba(0,0,0,0.05)" if not is_dark else "rgba(255,255,255,0.05)",
                bordercolor=color,
                borderwidth=1,
                borderpad=4,
            ),
            annotation_position="top right",
        )

    _aplicar_layout(
        fig,
        titulo="Distribuição do Índice de Risco Social nos Municípios do ES",
        xaxis_title="Índice de Risco Social (0–100)",
        yaxis_title="Número de Municípios",
        showlegend=False,
        xaxis=dict(range=[0, 100]),
    )
    return fig


# ─── 2. Heatmap de Correlação de Spearman ────────────────────────────────────

def grafico_heatmap_correlacao(df: pd.DataFrame, colunas: list[str]) -> go.Figure:
    """
    Heatmap interativo da correlação de Spearman entre variáveis de saneamento e saúde.
    """
    import scipy.stats as stats

    df_corr = df[colunas].dropna()
    if len(df_corr) < 10:
        fig = go.Figure()
        fig.add_annotation(text="Dados insuficientes para correlação.", showarrow=False)
        return fig

    corr_matrix, _ = stats.spearmanr(df_corr)
    if np.ndim(corr_matrix) == 0:
        corr_matrix = np.array([[1.0]])
    corr_df = pd.DataFrame(corr_matrix, index=df_corr.columns, columns=df_corr.columns)

    labels = [nome_amigavel(c) for c in colunas]

    # Máscara triangular superior
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_masked = np.where(mask, np.nan, corr_matrix)

    fig = go.Figure(go.Heatmap(
        z=corr_masked,
        x=labels,
        y=labels,
        colorscale=[
            [0.0, "#ef4444"],
            [0.5, "#1e293b"],
            [1.0, "#22c55e"],
        ],
        zmin=-1,
        zmax=1,
        text=np.round(corr_masked, 2),
        texttemplate="%{text}",
        textfont=dict(size=11, color="#f1f5f9"),
        hovertemplate="<b>%{x}</b> vs <b>%{y}</b><br>Correlação: %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(
            title="Correlação",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1 (inversa forte)", "-0.5", "0 (sem relação)", "0.5", "+1 (direta forte)"],
        ),
    ))

    _aplicar_layout(
        fig,
        titulo="Correlação de Spearman — Saneamento × Saúde",
        xaxis=dict(tickangle=-35, gridcolor="rgba(0,0,0,0)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0)"),
        height=520,
    )
    return fig


# ─── 3. Boxplot Morbidade por Nível de Saneamento ────────────────────────────

def grafico_boxplot_morbidade(df: pd.DataFrame) -> go.Figure:
    """
    Boxplot mostrando como a taxa de morbidade varia conforme o nível de saneamento.
    Cada caixa representa um terço (tercil) dos municípios ordenados por saneamento.
    """
    cols = ["indice_atendimento_total_agua", "Taxa_Morbidade_100k_Hab"]
    dados = df[cols].dropna().copy()
    if dados.empty:
        fig = go.Figure()
        fig.add_annotation(text="Dados insuficientes.", showarrow=False)
        return fig

    dados["grupo_saneamento"] = pd.qcut(
        dados["indice_atendimento_total_agua"],
        q=3,
        labels=["Saneamento Baixo\n(0–33%)", "Saneamento Médio\n(33–66%)", "Saneamento Alto\n(66–100%)"],
        duplicates="drop",
    )

    cores = {"Saneamento Baixo\n(0–33%)": "#ef4444",
             "Saneamento Médio\n(33–66%)": "#eab308",
             "Saneamento Alto\n(66–100%)": "#22c55e"}

    fig = go.Figure()
    for grupo, color in cores.items():
        subset = dados[dados["grupo_saneamento"] == grupo]["Taxa_Morbidade_100k_Hab"]
        if subset.empty:
            continue
        fig.add_trace(go.Box(
            y=subset,
            name=grupo,
            marker_color=color,
            boxmean=True,
            hovertemplate=(
                f"<b>{grupo}</b><br>"
                "Mediana: %{median:.1f}<br>"
                "Q1: %{q1:.1f} | Q3: %{q3:.1f}<extra></extra>"
            ),
        ))

    _aplicar_layout(
        fig,
        titulo="Morbidade por Nível de Saneamento",
        xaxis_title="Grupo de Saneamento",
        yaxis_title="Taxa de Morbidade (por 100 mil hab.)",
    )
    return fig


# ─── 4. Scatter de Clusterização ─────────────────────────────────────────────

def grafico_scatter_zonas(
    df: pd.DataFrame,
    x_col: str = "vazio_sanitario",
    y_col: str = "Taxa_Morbidade_100k_Hab",
    tamanho_col: str | None = None,
) -> go.Figure:
    """
    Scatter interativo dos municípios coloridos por zona de vulnerabilidade.
    Cada ponto representa um município; o eixo X é o vazio sanitário e
    o eixo Y é a taxa de morbidade.
    """
    cols_req = [x_col, y_col, "zona_vulnerabilidade", "nome_municipio"]
    dados = df.dropna(subset=[x_col, y_col]).copy()

    if "nome_municipio" not in dados.columns:
        dados["nome_municipio"] = dados.get("id_municipio", "?")

    fig = go.Figure()
    zonas_ordem = [
        "Zona Verde - Baixo Risco",
        "Zona Amarela - Risco Moderado",
        "Zona Laranja - Risco Elevado",
        "Zona Vermelha - Risco Critico",
    ]

    for zona in zonas_ordem:
        subset = dados[dados["zona_vulnerabilidade"] == zona]
        if subset.empty:
            continue
        cor = ZONA_CORES.get(zona, "#94a3b8")
        tamanho = (
            (subset[tamanho_col] / subset[tamanho_col].max() * 20 + 8).clip(8, 28)
            if tamanho_col and tamanho_col in subset.columns
            else 10
        )
        fig.add_trace(go.Scatter(
            x=subset[x_col],
            y=subset[y_col],
            mode="markers",
            name=zona,
            marker=dict(
                color=cor,
                size=tamanho,
                opacity=0.82,
                line=dict(color="rgba(255,255,255,0.2)", width=1),
            ),
            text=subset["nome_municipio"],
            customdata=subset[["RISCO_SOCIAL_FINAL"]].values if "RISCO_SOCIAL_FINAL" in subset.columns else None,
            hovertemplate=(
                "<b>%{text}</b><br>"
                f"{nome_amigavel(x_col)}: %{{x:.1f}}<br>"
                f"{nome_amigavel(y_col)}: %{{y:.1f}}<br>"
                "Risco Social: %{customdata[0]:.2f}<extra></extra>"
            ),
        ))

    _aplicar_layout(
        fig,
        titulo="Zonas de Vulnerabilidade — Saneamento × Saúde",
        xaxis_title=nome_amigavel(x_col),
        yaxis_title=nome_amigavel(y_col),
        legend=dict(
            title="Zona de Risco",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="rgba(148,163,184,0.3)",
            borderwidth=1,
        ),
    )
    return fig


# ─── 5. Evolução Temporal do Risco ───────────────────────────────────────────

def grafico_evolucao_temporal(df: pd.DataFrame, municipios: list[str] | None = None) -> go.Figure:
    """
    Linha do tempo do índice de risco social por município.
    Permite comparar a evolução de diferentes municípios ao longo dos anos.
    """
    if "nome_municipio" not in df.columns:
        df = df.copy()
        df["nome_municipio"] = df.get("id_municipio", "?")

    dados = df.dropna(subset=["RISCO_SOCIAL_FINAL", "ano"]).copy()
    dados["ano"] = dados["ano"].astype(int)

    if municipios:
        dados = dados[dados["nome_municipio"].isin(municipios)]

    # Agregar por município+ano (média)
    serie = dados.groupby(["nome_municipio", "ano"])["RISCO_SOCIAL_FINAL"].mean().reset_index()

    if serie.empty:
        fig = go.Figure()
        fig.add_annotation(text="Selecione ao menos um município.", showarrow=False, font_color="#334155")
        return _aplicar_layout(fig)

    fig = px.line(
        serie,
        x="ano",
        y="RISCO_SOCIAL_FINAL",
        color="nome_municipio",
        markers=True,
        labels={"ano": "Ano", "RISCO_SOCIAL_FINAL": "Índice de Risco Social", "nome_municipio": "Município"},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(line_width=2.5, marker_size=6)
    _aplicar_layout(
        fig,
        titulo="Evolução Temporal do Índice de Risco Social",
        xaxis_title="Ano",
        yaxis_title="Índice de Risco Social (0–100)",
    )
    # Faixa de risco de fundo
    for min_v, max_v, cor, label in [
        (0, 25, "rgba(34,197,94,0.07)", "Baixo"),
        (25, 50, "rgba(234,179,8,0.07)", "Moderado"),
        (50, 75, "rgba(249,115,22,0.07)", "Elevado"),
        (75, 100, "rgba(239,68,68,0.07)", "Crítico"),
    ]:
        fig.add_hrect(y0=min_v, y1=max_v, fillcolor=cor, line_width=0,
                      annotation_text=label, annotation_position="right",
                      annotation_font_color="#94a3b8", annotation_font_size=10)
    return fig


# ─── 6. Ranking dos Municípios ───────────────────────────────────────────────

def grafico_ranking_municipios(df: pd.DataFrame, top_n: int = 15, modo: str = "piores") -> go.Figure:
    """
    Gráfico de barras horizontal com ranking dos municípios por risco social.
    'piores' mostra os mais críticos; 'melhores' os com menor risco.
    """
    cols = ["nome_municipio", "RISCO_SOCIAL_FINAL", "zona_vulnerabilidade"]
    dados = df.dropna(subset=["RISCO_SOCIAL_FINAL"]).copy()

    # Pegar o ano mais recente por município
    if "ano" in dados.columns:
        idx = dados.groupby("nome_municipio")["ano"].idxmax()
        dados = dados.loc[idx]

    dados = dados[cols].dropna()

    if modo == "piores":
        dados = dados.nlargest(top_n, "RISCO_SOCIAL_FINAL")
        titulo = f"🔴 Top {top_n} Municípios em Situação Mais Crítica"
    else:
        dados = dados.nsmallest(top_n, "RISCO_SOCIAL_FINAL")
        titulo = f"🟢 Top {top_n} Municípios com Melhor Saneamento"
        dados = dados.sort_values("RISCO_SOCIAL_FINAL", ascending=False)

    cores = dados["zona_vulnerabilidade"].map(ZONA_CORES).fillna("#94a3b8")

    fig = go.Figure(go.Bar(
        x=dados["RISCO_SOCIAL_FINAL"],
        y=dados["nome_municipio"],
        orientation="h",
        marker=dict(
            color=cores,
            line=dict(color="rgba(255,255,255,0.1)", width=0.5),
        ),
        text=dados["RISCO_SOCIAL_FINAL"].round(1),
        textposition="outside",
        textfont=dict(color="#e2e8f0"),
        hovertemplate="<b>%{y}</b><br>Risco Social: %{x:.2f}<extra></extra>",
    ))

    _aplicar_layout(
        fig,
        titulo=titulo,
        xaxis_title="Índice de Risco Social",
        yaxis_title="",
        height=max(350, top_n * 28),
        showlegend=False,
    )
    fig.update_layout(yaxis=dict(autorange="reversed", gridcolor="rgba(148,163,184,0.08)"))
    return fig


# ─── 7. Comparação entre Municípios ─────────────────────────────────────────

def grafico_comparacao_municipios(df: pd.DataFrame, municipios: list[str]) -> go.Figure:
    """
    Radar/Spider chart para comparar perfil de saneamento entre municípios.
    """
    metricas = {
        "Água (%)": "indice_atendimento_total_agua",
        "Esgoto (%)": "indice_atendimento_esgoto_agua",
        "Tratamento (%)": "indice_tratamento_esgoto",
        "Sem Perdas (%)": None,  # 100 - perda
        "Eficiência (%)": "eficiencia_arrecadacao",
        "Sem Risco": None,  # 100 - RISCO_SOCIAL_FINAL
    }

    if "nome_municipio" not in df.columns:
        df = df.copy()
        df["nome_municipio"] = df.get("id_municipio", "?")

    dados = df.copy()
    if "ano" in dados.columns:
        idx = dados.groupby("nome_municipio")["ano"].idxmax()
        dados = dados.loc[idx]
    dados = dados[dados["nome_municipio"].isin(municipios)].set_index("nome_municipio")

    fig = go.Figure()
    categories = ["Cobertura Água", "Cobertura Esgoto", "Tratamento Esgoto",
                  "Eficiência Rede", "Efic. Arrecadação", "Baixo Risco"]
    col_map = [
        "indice_atendimento_total_agua",
        "indice_atendimento_esgoto_agua",
        "indice_tratamento_esgoto",
        "indice_perda_distribuicao_agua",
        "eficiencia_arrecadacao",
        "RISCO_SOCIAL_FINAL",
    ]

    cores_radar = ["#60a5fa", "#34d399", "#f472b6", "#fb923c", "#a78bfa"]

    for i, mun in enumerate(municipios):
        if mun not in dados.index:
            continue
        row = dados.loc[mun]
        vals = []
        for col in col_map:
            v = float(row.get(col, 0) or 0)
            if col == "indice_perda_distribuicao_agua":
                v = max(0, 100 - v)  # inverter: menor perda = melhor
            elif col == "RISCO_SOCIAL_FINAL":
                v = max(0, 100 - v)  # inverter: menor risco = melhor
            vals.append(min(100, max(0, v)))

        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]],
            theta=categories + [categories[0]],
            fill="toself",
            fillcolor=_hex_to_rgba(cores_radar[i % len(cores_radar)], 0.15),
            line=dict(color=cores_radar[i % len(cores_radar)], width=2),
            name=mun,
            hovertemplate="<b>%{theta}</b><br>Valor: %{r:.1f}<extra></extra>",
        ))

    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    fig.update_layout(
        **_get_layout_base(),
        **_get_grid_style(),
        polar=dict(
            bgcolor=tokens['surface'],
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor="rgba(148,163,184,0.08)" if is_dark else "rgba(148,163,184,0.15)",
                tickfont=dict(color=tokens['text'], size=10),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.08)" if is_dark else "rgba(148,163,184,0.15)",
                linecolor=tokens['surface_border'],
            ),
        ),
        title=dict(
            text="Perfil Comparativo de Saneamento",
            font=dict(size=17, color=tokens['title'], family="Inter, sans-serif"),
            x=0.02,
        ),
    )
    return fig


# ─── 8. Indicador Investimento vs Risco ──────────────────────────────────────

def grafico_investimento_vs_risco(df: pd.DataFrame) -> go.Figure:
    """
    Bubble chart: investimento per capita vs risco social, tamanho = população.
    Permite visualizar se municípios que investem mais têm menor risco.
    """
    cols_req = ["investimento_total_consolidado", "RISCO_SOCIAL_FINAL", "populacao_ref", "nome_municipio"]
    dados = df.dropna(subset=["investimento_total_consolidado", "RISCO_SOCIAL_FINAL"]).copy()

    if "ano" in dados.columns:
        idx = dados.groupby("nome_municipio" if "nome_municipio" in dados.columns else "id_municipio")["ano"].idxmax()
        dados = dados.loc[idx]

    if "populacao_ref" in dados.columns:
        dados = dados[dados["populacao_ref"] > 0]
        dados["invest_per_capita"] = dados["investimento_total_consolidado"] / dados["populacao_ref"]
    else:
        dados["invest_per_capita"] = dados["investimento_total_consolidado"]

    if "nome_municipio" not in dados.columns:
        dados["nome_municipio"] = dados["id_municipio"]

    dados = dados.dropna(subset=["invest_per_capita"])

    cores = dados["zona_vulnerabilidade"].map(ZONA_CORES).fillna("#94a3b8") if "zona_vulnerabilidade" in dados.columns else "#60a5fa"
    tamanho = (dados["populacao_ref"] / dados["populacao_ref"].max() * 35 + 6).clip(6, 40) if "populacao_ref" in dados.columns else 12

    fig = go.Figure(go.Scatter(
        x=dados["invest_per_capita"],
        y=dados["RISCO_SOCIAL_FINAL"],
        mode="markers",
        marker=dict(color=cores, size=tamanho, opacity=0.75,
                    line=dict(color="rgba(255,255,255,0.2)", width=1)),
        text=dados["nome_municipio"],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Investimento per capita: R$ %{x:,.0f}<br>"
            "Risco Social: %{y:.2f}<extra></extra>"
        ),
    ))

    _aplicar_layout(
        fig,
        titulo="Investimento per Capita × Risco Social",
        xaxis_title="Investimento per Capita (R$)",
        yaxis_title="Índice de Risco Social",
        showlegend=False,
    )
    return fig


# ─── 9. Top 10 Saneamento ────────────────────────────────────────────────────

def grafico_top10_saneamento(df: pd.DataFrame) -> go.Figure:
    """
    Gráfico de barras duplo: top 10 melhores e piores municípios por saneamento.
    """
    col = "indice_atendimento_total_agua"
    if "nome_municipio" not in df.columns:
        df = df.copy()
        df["nome_municipio"] = df.get("id_municipio", "?")

    dados = df.dropna(subset=[col]).copy()
    if "ano" in dados.columns:
        idx = dados.groupby("nome_municipio")["ano"].idxmax()
        dados = dados.loc[idx]

    melhores = dados.nlargest(10, col)[["nome_municipio", col]].assign(tipo="🟢 Melhores")
    piores = dados.nsmallest(10, col)[["nome_municipio", col]].assign(tipo="🔴 Piores")
    combinado = pd.concat([melhores, piores])

    fig = px.bar(
        combinado,
        x="nome_municipio",
        y=col,
        color="tipo",
        barmode="group",
        color_discrete_map={"🟢 Melhores": "#22c55e", "🔴 Piores": "#ef4444"},
        labels={col: "Atendimento de Água (%)", "nome_municipio": "Município", "tipo": "Grupo"},
        text=combinado[col].round(1),
    )
    fig.update_traces(textposition="outside", textfont_color="#e2e8f0")
    _aplicar_layout(
        fig,
        titulo="Top 10: Melhores e Piores em Cobertura de Água",
        xaxis_title="Município",
        yaxis_title="Índice de Atendimento de Água (%)",
        xaxis_tickangle=-40,
    )
    return fig


# ─── 10. Perfil do Município ──────────────────────────────────────────────────

def grafico_perfil_municipio(df: pd.DataFrame, municipio: str) -> go.Figure:
    """
    Gauge + barras mostrando todos os indicadores de um município específico.
    """
    if "nome_municipio" not in df.columns:
        df = df.copy()
        df["nome_municipio"] = df.get("id_municipio", "?")

    dados = df[df["nome_municipio"] == municipio].copy()
    if "ano" in dados.columns and not dados.empty:
        dados = dados.sort_values("ano").tail(1)

    tokens = _get_tokens()
    is_dark = get_theme_mode() == "Escuro"

    if dados.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="Município não encontrado.",
            showarrow=False,
            font=dict(color=tokens['text'], size=14)
        )
        return _aplicar_layout(fig)

    row = dados.iloc[0]
    risco = float(row.get("RISCO_SOCIAL_FINAL", 50) or 50)

    # Determinar cor do gauge (consistente com SEQUENCIAL_RISK)
    if risco < 25:
        gauge_color = "#22C55E"  # Verde
    elif risco < 50:
        gauge_color = "#EAB308"  # Amarelo
    elif risco < 75:
        gauge_color = "#F97316"  # Laranja
    else:
        gauge_color = "#EF4444"  # Vermelho

    # Steps com opacidade ajustada ao tema
    step_opacity = 0.12 if is_dark else 0.15

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=risco,
        title={
            "text": f"Risco Social — {municipio}",
            "font": {"color": tokens['title'], "size": 17, "family": "Inter, sans-serif"}
        },
        number={"font": {"color": tokens['title'], "size": 40, "family": "Inter, sans-serif"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickcolor": tokens['text'],
                "tickfont": {"color": tokens['text'], "size": 11}
            },
            "bar": {"color": gauge_color, "thickness": 0.25},
            "bgcolor": tokens['surface'],
            "bordercolor": tokens['surface_border'],
            "borderwidth": 2,
            "steps": [
                {"range": [0, 25], "color": f"rgba(34,197,94,{step_opacity})"},
                {"range": [25, 50], "color": f"rgba(234,179,8,{step_opacity})"},
                {"range": [50, 75], "color": f"rgba(249,115,22,{step_opacity})"},
                {"range": [75, 100], "color": f"rgba(239,68,68,{step_opacity})"},
            ],
            "threshold": {
                "line": {"color": tokens['text'], "width": 3},
                "value": risco,
                "thickness": 0.8
            },
        },
    ))

    fig.update_layout(
        **_get_layout_base(),
        **_get_grid_style(),
        height=280,
    )
    return fig
