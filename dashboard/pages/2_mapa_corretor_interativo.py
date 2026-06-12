"""
🔧 Corretor Interativo de Coordenadas
Página para ajustar manualmente as coordenadas dos municípios do ES
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium

from utils.data_loader import carregar_dados_zonas, COORDS_ES
from utils.theme import apply_theme
from utils.components import cabecalho_pagina, caixa_explicacao, separador

st.set_page_config(
    page_title="🔧 Corretor de Coordenadas | Saneamento ES",
    page_icon="🔧",
    layout="wide",
)

apply_theme(show_toggle=True)

cabecalho_pagina(
    titulo="🔧 Corretor Interativo de Coordenadas",
    descricao="Use esta ferramenta para verificar e corrigir as coordenadas dos municípios no mapa.",
    icone="🔧",
)

caixa_explicacao(
    "📍 Como usar: 1) Selecione um município abaixo, 2) Veja onde ele está no mapa, "
    "3) Se estiver errado, clique no mapa no local correto, 4) Copie o código gerado.",
    tipo="info"
)

# Carregar dados
df = carregar_dados_zonas()
ano_recente = df["ano"].max()
df_ano = df[df["ano"] == ano_recente].copy()

# Lista de municípios
df_ano["id_mun_7"] = df_ano["id_municipio"].astype(str).str[:7]
municipios_disponiveis = df_ano.sort_values("nome_municipio")[["id_mun_7", "nome_municipio"]].drop_duplicates()

separador()

# Seleção de município
col1, col2 = st.columns([2, 3])

with col1:
    st.markdown("### 🎯 Selecionar Município")

    municipio_opcoes = {
        f"{row['nome_municipio']} ({row['id_mun_7']})": row['id_mun_7']
        for _, row in municipios_disponiveis.iterrows()
    }

    municipio_selecionado = st.selectbox(
        "Escolha um município para verificar/corrigir:",
        options=list(municipio_opcoes.keys())
    )

    codigo_sel = municipio_opcoes[municipio_selecionado]
    nome_sel = municipio_selecionado.split(" (")[0]

    # Coordenadas atuais
    coords_atual = COORDS_ES.get(codigo_sel)

    if coords_atual:
        st.success(f"✅ Coordenada cadastrada: {coords_atual}")
        lat_atual, lon_atual = coords_atual
    else:
        st.error("❌ Município sem coordenada cadastrada!")
        lat_atual, lon_atual = -20.0, -40.5  # Centro do ES

with col2:
    st.markdown("### 📊 Informações do Município")

    municipio_data = df_ano[df_ano["id_mun_7"] == codigo_sel].iloc[0]

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        zona = municipio_data.get("zona_vulnerabilidade", "N/D")
        st.metric("Zona", zona.replace("Zona ", "").replace(" - ", "\n"))
    with col_b:
        risco = municipio_data.get("RISCO_SOCIAL_FINAL", 0)
        st.metric("Risco Social", f"{risco:.1f}")
    with col_c:
        agua = municipio_data.get("indice_atendimento_total_agua", 0)
        st.metric("Cobertura Água", f"{agua:.1f}%")

separador()

# Mapa interativo
st.markdown("### 🗺️ Mapa para Verificação")

caixa_explicacao(
    "🖱️ Clique no mapa no local CORRETO onde o município deveria estar. "
    "As coordenadas do clique aparecerão abaixo para você copiar.",
    tipo="aviso"
)

# Criar mapa centrado no município atual ou no ES
mapa = folium.Map(
    location=[lat_atual, lon_atual],
    zoom_start=11 if coords_atual else 7,
    tiles=None,
)

# Adicionar tile
folium.TileLayer(
    tiles="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    attr="&copy; OpenStreetMap contributors",
    name="OpenStreetMap",
    max_zoom=19,
).add_to(mapa)

# Marcar posição ATUAL do município (se existir)
if coords_atual:
    folium.Marker(
        location=[lat_atual, lon_atual],
        popup=f"<b>POSIÇÃO ATUAL</b><br>{nome_sel}<br>Lat: {lat_atual:.4f}<br>Lon: {lon_atual:.4f}",
        tooltip=f"Posição ATUAL: {nome_sel}",
        icon=folium.Icon(color="red", icon="info-sign"),
    ).add_to(mapa)

# Adicionar plugin de clique para capturar coordenadas
folium.LatLngPopup().add_to(mapa)

# Renderizar mapa
mapa_output = st_folium(mapa, width="100%", height=500)

separador()

# Capturar coordenadas do clique
st.markdown("### 📋 Coordenadas Clicadas")

if mapa_output and mapa_output.get("last_clicked"):
    lat_clicado = mapa_output["last_clicked"]["lat"]
    lon_clicado = mapa_output["last_clicked"]["lng"]

    st.success(f"🎯 Você clicou em: **Lat: {lat_clicado:.6f}, Lon: {lon_clicado:.6f}**")

    # Gerar código para copiar
    codigo_python = f'    "{codigo_sel}": ({lat_clicado:.6f}, {lon_clicado:.6f}),  # {nome_sel}'

    st.markdown("#### 📝 Código para Copiar:")
    st.code(codigo_python, language="python")

    st.markdown("**Como aplicar:**")
    st.markdown("""
    1. Copie o código acima
    2. Abra o arquivo: `dashboard/utils/data_loader.py`
    3. Encontre a linha do município (busque por `{codigo_sel}`)
    4. Substitua a linha antiga pela nova
    5. Salve o arquivo
    6. Reinicie o dashboard
    """.format(codigo_sel=codigo_sel))

    # Comparação
    if coords_atual:
        st.markdown("#### 📊 Comparação:")
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            st.markdown("**❌ ANTES (Errado)**")
            st.code(f'"{codigo_sel}": ({lat_atual:.6f}, {lon_atual:.6f})', language="python")
        with col_comp2:
            st.markdown("**✅ DEPOIS (Correto)**")
            st.code(f'"{codigo_sel}": ({lat_clicado:.6f}, {lon_clicado:.6f})', language="python")
else:
    st.info("👆 Clique no mapa no local CORRETO onde o município deveria estar.")

separador()

# Lista de municípios prioritários para correção
st.markdown("### 🎯 Municípios Prioritários para Verificar")

municipios_prioridade = [
    "Vila Velha",
    "Viana",
    "Serra",
    "Cariacica",
    "Vitória",
    "Guarapari",
    "Vila Valério",
    "São Roque do Canaã",
    "Venda Nova do Imigrante",
]

st.markdown("Verifique especialmente estes municípios:")
cols_prior = st.columns(3)
for idx, mun in enumerate(municipios_prioridade):
    with cols_prior[idx % 3]:
        codigo_mun = df_ano[df_ano["nome_municipio"] == mun]["id_mun_7"].values
        if len(codigo_mun) > 0:
            st.markdown(f"- **{mun}** (`{codigo_mun[0]}`)")

separador()

# Instruções finais
st.markdown("### 📚 Instruções Completas")

with st.expander("🔍 Como encontrar coordenadas corretas"):
    st.markdown("""
    **Opção 1: Google Maps**
    1. Abra: https://www.google.com/maps
    2. Busque pelo município (ex: "Vila Velha, ES")
    3. Clique com botão direito no centro da cidade
    4. Selecione "Ver coordenadas" (ou copie da URL)

    **Opção 2: IBGE**
    1. Acesse: https://cidades.ibge.gov.br/brasil/es
    2. Selecione o município
    3. As coordenadas aparecem na ficha

    **Opção 3: OpenStreetMap**
    1. Abra: https://www.openstreetmap.org
    2. Busque o município
    3. Clique no local e veja as coordenadas na URL
    """)

with st.expander("✏️ Como aplicar múltiplas correções"):
    st.markdown("""
    1. Crie um arquivo temporário `coordenadas_corrigidas.txt`
    2. Para cada município errado:
       - Selecione o município aqui
       - Clique no local correto no mapa
       - Copie o código gerado
       - Cole no arquivo .txt
    3. Quando tiver todas as correções:
       - Abra `dashboard/utils/data_loader.py`
       - Encontre o dicionário `COORDS_ES`
       - Substitua as linhas erradas pelas corretas
       - Salve o arquivo
    4. Limpe o cache:
       ```bash
       cd dashboard
       find . -name "*.pyc" -delete
       find . -type d -name "__pycache__" -exec rm -rf {} +
       ```
    5. Reinicie o dashboard:
       ```bash
       streamlit run app.py
       ```
    """)

separador()

st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#64748b; font-size:0.9rem;">
    💡 <b>Dica:</b> Verifique o mapa após cada correção para garantir que está correto!
</div>
""", unsafe_allow_html=True)
