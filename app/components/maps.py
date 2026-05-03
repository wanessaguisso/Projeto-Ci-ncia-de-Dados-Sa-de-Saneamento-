import streamlit as st
import plotly.express as px
from components.charts import COLORS

def render_map(df_filtered):
    st.subheader("🗺️ Mapa Geográfico (Distribuição)")
    
    if 'lat' in df_filtered.columns and 'lon' in df_filtered.columns:
        # Mapa com intensidade (Scatter Mapbox)
        fig = px.scatter_mapbox(df_filtered, lat="lat", lon="lon", hover_name="nome_municipio" if 'nome_municipio' in df_filtered.columns else "id_municipio",
                                hover_data=["Taxa_Morbidade_100k_Hab", "vazio_sanitario"],
                                color="RISCO_SOCIAL_FINAL" if 'RISCO_SOCIAL_FINAL' in df_filtered.columns else "Taxa_Morbidade_100k_Hab",
                                size="Taxa_Morbidade_100k_Hab",
                                color_continuous_scale="Reds", zoom=6, height=500,
                                title="Mapa de Intensidade de Doenças e Risco")
        
        # Plotly mapbox configuration (using open street map to avoid token requirement)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("O dataset atual não contém as colunas `lat` e `lon` necessárias para renderizar o mapa interativo.")
