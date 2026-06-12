"""
pages/2_🗺️_Mapa_Interativo.py
Mapa do Espírito Santo com zonas de vulnerabilidade.
Usa Folium com círculos coloridos por zona de risco.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from utils.data_loader import carregar_dados_zonas, ZONA_CORES_FOLIUM, COORDS_ES
from utils.theme import apply_theme
from utils.components import cabecalho_pagina, secao_com_tooltip, caixa_explicacao, separador, badge_zona

st.set_page_config(
    page_title="Mapa Interativo | Saneamento ES",
    page_icon="🗺️",
    layout="wide",
)

apply_theme(show_toggle=True)

# ─── Dados ────────────────────────────────────────────────────────────────────
df = carregar_dados_zonas()

cabecalho_pagina(
    titulo="Mapa de Vulnerabilidade — Espírito Santo",
    descricao=(
        "Visualize a distribuição geográfica do risco social nos municípios do ES. "
        "Cada círculo representa um município: clique nele para ver os detalhes. "
        "A cor indica a zona de risco: verde é seguro, vermelho é crítico."
    ),
    icone="🗺️",
)

# ─── Filtros ──────────────────────────────────────────────────────────────────
col_f1, col_f2, col_f3 = st.columns([2, 3, 3])

anos = sorted(df["ano"].dropna().unique().astype(int), reverse=True)
with col_f1:
    ano_sel = st.selectbox("📅 Ano", anos, index=0)

zonas_disponiveis = sorted(df["zona_vulnerabilidade"].dropna().unique())
with col_f2:
    zonas_sel = st.multiselect(
        "🎨 Filtrar por Zona",
        zonas_disponiveis,
        default=zonas_disponiveis,
    )

df_mapa = df[(df["ano"] == ano_sel) & (df["zona_vulnerabilidade"].isin(zonas_sel))].copy()

# ─── Legenda de Zonas ─────────────────────────────────────────────────────────
separador()
st.markdown("#### Legenda de Zonas")
cols_legenda = st.columns(4)
zonas_info = [
    ("Zona Verde - Baixo Risco", "Municípios com bom saneamento e baixas taxas de internação."),
    ("Zona Amarela - Risco Moderado", "Saneamento parcial; atenção necessária mas sem emergência."),
    ("Zona Laranja - Risco Elevado", "Déficits significativos de saneamento e saúde."),
    ("Zona Vermelha - Risco Crítico", "Situação de emergência: falta grave de saneamento e alta morbidade."),
]
for col, (zona, desc) in zip(cols_legenda, zonas_info):
    with col:
        badge_zona(zona)
        st.caption(desc)

separador()

# ─── Construção do Mapa Folium ────────────────────────────────────────────────
secao_com_tooltip(
    "Mapa Interativo dos Municípios",
    tooltip=(
        "O que mostra: Localização geográfica de cada município com sua zona de risco. "
        "Como interagir: Clique em um círculo para ver detalhes do município. "
        "Como interpretar: Círculos maiores = maior risco social. Cores = zona de vulnerabilidade."
    ),
)
caixa_explicacao(
    "Os círculos são posicionados nas coordenadas aproximadas dos municípios. "
    "O tamanho do círculo é proporcional ao Índice de Risco Social.",
    tipo="info",
)

# Centro do ES
mapa = folium.Map(
    location=[-19.8, -40.3],
    zoom_start=7,
    tiles=None,
)

# Tile dark
folium.TileLayer(
    tiles="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    attr="&copy; OpenStreetMap &copy; CARTO",
    name="Dark Map",
    max_zoom=19,
).add_to(mapa)

# Adicionar municípios ao mapa
for _, row in df_mapa.iterrows():
    mid = str(row.get("id_municipio", ""))[:7]
    coords = COORDS_ES.get(mid)
    if coords is None:
        continue

    zona = str(row.get("zona_vulnerabilidade", ""))
    cor_folium = ZONA_CORES_FOLIUM.get(zona, "gray")
    risco = float(row.get("RISCO_SOCIAL_FINAL", 0) or 0)
    nome = str(row.get("nome_municipio", mid))

    # Tamanho proporcional ao risco
    raio = max(5, min(20, risco / 5))

    # Construir popup
    agua = row.get("indice_atendimento_total_agua")
    esgoto = row.get("indice_atendimento_esgoto_agua")
    morbidade = row.get("Taxa_Morbidade_100k_Hab")

    popup_html = f"""
    <div style="font-family:sans-serif; min-width:200px;">
        <h4 style="margin:0 0 6px 0; color:#1e293b;">{nome}</h4>
        <hr style="margin:4px 0; border-color:#e2e8f0;">
        <b>Zona:</b> {zona}<br>
        <b>Risco Social:</b> {risco:.2f}<br>
        <b>Cobertura Água:</b> {f"{agua:.1f}%" if agua is not None and str(agua) != "nan" else "N/D"}<br>
        <b>Cobertura Esgoto:</b> {f"{esgoto:.1f}%" if esgoto is not None and str(esgoto) != "nan" else "N/D"}<br>
        <b>Morbidade:</b> {f"{morbidade:.1f} por 100k" if morbidade is not None and str(morbidade) != "nan" else "N/D"}
    </div>
    """

    folium.CircleMarker(
        location=coords,
        radius=raio,
        color="white",
        weight=0.5,
        fill=True,
        fill_color=cor_folium,
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=250),
        tooltip=f"{nome} — Risco: {risco:.1f}",
    ).add_to(mapa)

# Renderizar o mapa
st_folium(mapa, width="100%", height=560)

separador()

# ─── Tabela com detalhes do ano selecionado ───────────────────────────────────
secao_com_tooltip(
    "Tabela de Municípios",
    tooltip=(
        "O que mostra: Todos os municípios do ano selecionado com seus indicadores. "
        "Como usar: Ordene clicando no cabeçalho da coluna. Filtre usando os controles acima."
    ),
)

cols_tabela = [
    "nome_municipio", "zona_vulnerabilidade", "RISCO_SOCIAL_FINAL",
    "indice_atendimento_total_agua", "indice_atendimento_esgoto_agua",
    "indice_tratamento_esgoto", "Taxa_Morbidade_100k_Hab"
]
cols_existentes = [c for c in cols_tabela if c in df_mapa.columns]

if cols_existentes:
    df_tabela = df_mapa[cols_existentes].sort_values("RISCO_SOCIAL_FINAL", ascending=False).copy()
    df_tabela.columns = [
        "Município", "Zona de Risco", "Risco Social",
        "Água (%)", "Esgoto (%)", "Tratamento (%)", "Morbidade"
    ][:len(cols_existentes)]

    for col in ["Risco Social", "Água (%)", "Esgoto (%)", "Tratamento (%)"]:
        if col in df_tabela.columns:
            df_tabela[col] = df_tabela[col].round(1)

    st.dataframe(df_tabela, use_container_width=True, hide_index=True)
