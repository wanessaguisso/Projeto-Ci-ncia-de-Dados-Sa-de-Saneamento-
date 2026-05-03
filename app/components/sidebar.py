import streamlit as st

def render_sidebar(df):
    """
    Renderiza a barra lateral com filtros dinâmicos e retorna os filtros selecionados.
    """
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3268/3268886.png", width=80)
    st.sidebar.title("Filtros de Análise")
    
    if df.empty:
        return None, None, None
    
    # Filtro de Ano
    anos_disponiveis = sorted(df['ano'].dropna().unique().astype(int).tolist())
    ano_selecionado = st.sidebar.selectbox("Selecione o Ano", anos_disponiveis, index=len(anos_disponiveis)-1)
    
    # Filtro de Zona de Vulnerabilidade (Risco)
    if 'zona_vulnerabilidade' in df.columns:
        zonas = ["Todas"] + sorted(df['zona_vulnerabilidade'].dropna().unique().tolist())
        zona_selecionada = st.sidebar.selectbox("Zona de Risco", zonas)
    else:
        zona_selecionada = "Todas"
        
    # Filtro de Município
    municipios = ["Todos"] + sorted(df['id_municipio'].dropna().unique().astype(str).tolist()) # Em um projeto real, fariamos um join com o nome do município
    municipio_selecionado = st.sidebar.selectbox("Município (ID IBGE)", municipios)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sobre o Projeto:**\nEste dashboard mapeia a correlação entre o déficit de infraestrutura de saneamento básico e as taxas de internação por doenças de veiculação hídrica.")
    
    return ano_selecionado, zona_selecionada, municipio_selecionado
