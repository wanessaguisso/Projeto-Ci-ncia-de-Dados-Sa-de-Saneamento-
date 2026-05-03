import streamlit as st
import warnings
from data_loader import load_data
from components.sidebar import render_sidebar
from components.charts import render_kpis, render_scatter_regression, render_correlation_matrix
from components.maps import render_dynamic_ranking, render_map_placeholder
from components.simulator import render_simulator

# Ignorar FutureWarnings do Pandas/Seaborn
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(
    page_title="Saneamento & Saúde ES",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estética WOW / CSS Customizado
st.markdown("""
<style>
    /* Estilos principais */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Títulos com gradiente */
    h1, h2, h3 {
        background: -webkit-linear-gradient(45deg, #48cae4, #0077b6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Inter', sans-serif;
    }
    
    /* Cartões de KPI */
    div[data-testid="metric-container"] {
        background-color: #1a1c23;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        border: 1px solid #2d303a;
        transition: transform 0.2s ease-in-out;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-5px);
        border-color: #48cae4;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("💧 Inteligência em Saneamento e Saúde (ES)")
    st.markdown("Plataforma analítica para correlação entre infraestrutura de água/esgoto e incidência de doenças de veiculação hídrica.")
    
    # 1. Carregar Dados
    with st.spinner('Carregando e processando base de dados...'):
        df = load_data()
        
    if df.empty:
        st.stop()
        
    # Calcular e adicionar anomalias
    from src.model_utils import detectar_outliers_saneamento_saude
    df = detectar_outliers_saneamento_saude(df)
        
    # 2. Renderizar Sidebar e obter Filtros
    ano_sel, zona_sel, mun_sel = render_sidebar(df)
    
    # 3. Aplicar Filtros
    df_filtered = df.copy()
    if ano_sel:
        df_filtered = df_filtered[df_filtered['ano'] == ano_sel]
    if zona_sel and zona_sel != "Todas":
        df_filtered = df_filtered[df_filtered['zona_vulnerabilidade'] == zona_sel]
    if mun_sel and mun_sel != "Todos":
        df_filtered = df_filtered[df_filtered['id_municipio'].astype(str) == mun_sel]
        
    # 4. Renderizar KPIs (Topo)
    st.markdown("### Visão Geral do Cenário Filtrado")
    render_kpis(df_filtered)
    st.markdown("---")
    
    # 5. Layout em Colunas (Análises)
    col1, col2 = st.columns([3, 2])
    
    with col1:
        render_scatter_regression(df_filtered)
        render_simulator(df_filtered)
        
    with col2:
        render_map_placeholder(df_filtered)
        st.markdown("<br>", unsafe_allow_html=True)
        render_dynamic_ranking(df_filtered)
        st.markdown("<br>", unsafe_allow_html=True)
        render_correlation_matrix(df_filtered)

if __name__ == "__main__":
    main()
