import streamlit as st
import warnings
from data_loader import load_data
from components.sidebar import render_sidebar
from components.kpis import render_kpis, render_insights
from components.charts import (
    render_correlation_heatmap, render_scatter_regression, render_pairplot,
    render_time_series, render_comparisons, render_advanced_relations, render_distributions
)
from components.maps import render_map
from components.models_panel import (
    render_linear_regression, render_random_forest, render_arima, render_kmeans
)

# Ignorar FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

st.set_page_config(
    page_title="Saneamento & Saúde Analytics",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estética Light e Elegante
st.markdown("""
<style>
    .stApp {
        background-color: #F8F9FB;
        color: #2E2E2E;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #6C63FF;
        font-family: 'Inter', sans-serif;
    }
    div[data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        border-left: 5px solid #00BFA6;
    }
    div.stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    div.stTabs [data-baseweb="tab"] {
        background-color: #FFFFFF;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        box-shadow: 0 -2px 5px rgba(0,0,0,0.02);
    }
    div.stTabs [aria-selected="true"] {
        border-bottom: 3px solid #6C63FF !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def main():
    st.title("💧 Saneamento & Saúde Analytics")
    st.markdown("Plataforma interativa para correlação preditiva entre infraestrutura sanitária e saúde pública.")
    
    # 1. Carregar Dados
    with st.spinner('Carregando base de dados...'):
        df = load_data()
        
    if df.empty:
        st.stop()
        
    # 2. Sidebar e Filtros
    ano_sel, zona_sel, mun_sel = render_sidebar(df)
    
    df_filtered = df.copy()
    if ano_sel:
        df_filtered = df_filtered[df_filtered['ano'] == ano_sel]
    if zona_sel and zona_sel != "Todas":
        df_filtered = df_filtered[df_filtered['zona_vulnerabilidade'] == zona_sel]
    if mun_sel and mun_sel != "Todos":
        df_filtered = df_filtered[df_filtered['id_municipio'].astype(str) == mun_sel]
        
    # 3. Topo: KPIs e Insights
    render_kpis(df_filtered)
    render_insights(df_filtered)
    st.markdown("---")
    
    # 4. Meio: Visualizações Avançadas em Abas
    st.header("📊 Painel Analítico Interativo")
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Correlações", "Série Temporal", "Comparações", "Geográfico", "Avançado", "Distribuição"
    ])
    
    with tab1:
        colA, colB = st.columns([1, 1])
        with colA:
            render_correlation_heatmap(df_filtered)
        with colB:
            render_pairplot(df_filtered)
        render_scatter_regression(df_filtered)
        
    with tab2:
        render_time_series(df_filtered)
        
    with tab3:
        render_comparisons(df_filtered)
        
    with tab4:
        render_map(df_filtered)
        
    with tab5:
        render_advanced_relations(df_filtered)
        
    with tab6:
        render_distributions(df_filtered)
        
    st.markdown("---")
    
    # 5. Inferior: Modelos Preditivos
    st.header("🤖 Modelos Preditivos e Machine Learning")
    mtab1, mtab2, mtab3, mtab4 = st.tabs([
        "Regressão Linear", "Random Forest", "Série Temporal (ARIMA)", "Clusterização (K-Means)"
    ])
    
    with mtab1:
        render_linear_regression(df_filtered)
    with mtab2:
        render_random_forest(df_filtered)
    with mtab3:
        # ARIMA funciona melhor com série histórica completa
        render_arima(df)
    with mtab4:
        render_kmeans(df_filtered)

if __name__ == "__main__":
    main()
