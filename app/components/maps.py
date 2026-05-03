import streamlit as st
import folium
from folium.plugins import HeatMap
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def render_dynamic_ranking(df_filtered):
    """
    Renderiza o Ranking Dinâmico das top cidades mais críticas.
    """
    st.subheader("📌 Ranking de Criticidade (Top Municípios)")
    
    if df_filtered.empty or 'RISCO_SOCIAL_FINAL' not in df_filtered.columns:
        st.warning("Sem dados para exibir o ranking.")
        return
        
    df_rank = df_filtered.sort_values(by='RISCO_SOCIAL_FINAL', ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Paleta de cores com base no risco
    sns.barplot(
        data=df_rank,
        x='RISCO_SOCIAL_FINAL',
        y='id_municipio',
        orient='h',
        palette="Reds_r",
        ax=ax,
        order=df_rank['id_municipio']
    )
    
    ax.set_xlabel("Risco Social Final (0 a 100)", fontsize=11)
    ax.set_ylabel("ID Município", fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)


def render_map_placeholder(df_filtered):
    """
    Renderiza o mapa geoespacial ou um aviso caso não haja coordenadas.
    """
    st.subheader("🗺️ Mapa Geoespacial (Zonas Críticas)")
    st.markdown("Aqui será exibido o heatmap de vulnerabilidade. Como o dataset atual baseia-se apenas no código IBGE, precisaríamos enriquecer com latitudes/longitudes ou um arquivo `.geojson` do Espírito Santo para visualização coroplética.")
    
    st.info("💡 Sugestão: Baixar a malha municipal do ES via IBGE ou geocodificar as cidades para habilitar a camada Folium aqui.")
    
    # Se tivéssemos lat/lon, seria algo assim:
    # m = folium.Map(location=[-19.18, -40.30], zoom_start=7, tiles='CartoDB dark_matter')
    # heat_data = [[row['lat'], row['lon'], row['RISCO_SOCIAL_FINAL']] for index, row in df_filtered.iterrows()]
    # HeatMap(heat_data).add_to(m)
    # st_folium(m, width=700, height=500)
