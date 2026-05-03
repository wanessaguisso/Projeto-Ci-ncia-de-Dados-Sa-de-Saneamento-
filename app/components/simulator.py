import streamlit as st
from src.model_utils import regressao_linear_saneamento_saude, simular_cenario_melhoria_saneamento

def render_simulator(df_filtered):
    """
    Renderiza o Simulador de Cenários Interativo.
    """
    st.subheader("🔄 Simulador de Melhoria Sanitária")
    st.markdown("Brinque com o slider abaixo para simular o impacto de investimentos em saneamento na saúde pública da região selecionada.")
    
    if df_filtered.empty or len(df_filtered) < 3:
        st.warning("Dados insuficientes para treinar o simulador (mínimo de 3 registros necessários).")
        return
        
    # Treinar modelo de regressão com os dados atuais filtrados (ou base toda)
    modelo, r2, coef, intercept = regressao_linear_saneamento_saude(df_filtered)
    
    if modelo is None:
        st.error("Não foi possível ajustar o modelo com os dados atuais.")
        return
        
    st.info(f"💡 **Diagnóstico do Modelo:** Para cada 1% de aumento no déficit sanitário, estima-se um acréscimo de **{coef:.2f}** na taxa de morbidade por 100k habitantes (R² = {r2:.2f}).")
    
    melhoria_percentual = st.slider(
        "Simular Melhoria no Saneamento (%)", 
        min_value=0.0, 
        max_value=100.0, 
        value=20.0, 
        step=5.0,
        help="Redução percentual no Vazio Sanitário."
    )
    
    if melhoria_percentual > 0:
        df_simulado = simular_cenario_melhoria_saneamento(df_filtered, modelo, melhoria_percentual)
        
        reducao_media_morb = df_simulado['Reducao_Morbidade_Abs'].mean()
        
        st.success(f"🌟 **Impacto Estimado:** Uma melhoria de {melhoria_percentual}% no saneamento resultaria em uma redução média de **{reducao_media_morb:.1f}** internações a cada 100 mil habitantes nas áreas selecionadas.")
        
        # Opcional: mostrar um gauge ou barra visual
        st.progress(int(melhoria_percentual))
