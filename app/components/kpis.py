import streamlit as st
import pandas as pd

def render_kpis(df_filtered):
    st.markdown("### 📊 Visão Geral do Cenário")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    # Cálculos
    vazio_sanitario = df_filtered['vazio_sanitario'].mean()
    taxa_doencas = df_filtered['Taxa_Morbidade_100k_Hab'].mean()
    
    if not df_filtered.empty and 'RISCO_SOCIAL_FINAL' in df_filtered.columns:
        idx_critico = df_filtered['RISCO_SOCIAL_FINAL'].idxmax()
        regiao_critica = df_filtered.loc[idx_critico, 'nome_municipio'] if 'nome_municipio' in df_filtered.columns else str(df_filtered.loc[idx_critico, 'id_municipio'])
    else:
        regiao_critica = "N/D"

    # Tendência (simulação simples baseada no último ano)
    if 'ano' in df_filtered.columns and df_filtered['ano'].nunique() > 1:
        ano_max = df_filtered['ano'].max()
        ano_ant = ano_max - 1
        taxa_atual = df_filtered[df_filtered['ano'] == ano_max]['Taxa_Morbidade_100k_Hab'].mean()
        taxa_ant = df_filtered[df_filtered['ano'] == ano_ant]['Taxa_Morbidade_100k_Hab'].mean()
        tendencia = ((taxa_atual - taxa_ant) / taxa_ant) * 100 if taxa_ant > 0 else 0
        tend_text = f"{tendencia:+.1f}% vs ano ant."
    else:
        tend_text = "Estável"
        tendencia = 0
        
    previsao_futura = taxa_doencas * (1 + (tendencia/100))
        
    with col1:
        st.metric("Vazio Sanitário", f"{vazio_sanitario:.1f}%", "- Risco Crítico" if vazio_sanitario > 40 else "+ Adequado", delta_color="inverse")
    with col2:
        st.metric("Taxa Doenças (100k)", f"{taxa_doencas:.1f}", tend_text, delta_color="inverse")
    with col3:
        st.metric("Região Mais Crítica", regiao_critica)
    with col4:
        st.metric("Tendência", "Crescente" if tendencia > 0 else "Decrescente", f"{tendencia:+.1f}%", delta_color="inverse")
    with col5:
        st.metric("Previsão (Próx. Ano)", f"{previsao_futura:.1f}")

def render_insights(df_filtered):
    st.markdown("#### 🤖 Insights Automáticos")
    
    # Correlação
    if 'vazio_sanitario' in df_filtered.columns and 'Taxa_Morbidade_100k_Hab' in df_filtered.columns:
        corr = df_filtered[['vazio_sanitario', 'Taxa_Morbidade_100k_Hab']].corr().iloc[0,1]
        explica = (corr ** 2) * 100
        st.info(f"💡 **Impacto Direto**: A falta de infraestrutura de saneamento básico explica aproximadamente **{explica:.1f}%** da variação nas taxas de doenças de veiculação hídrica nas regiões filtradas.")
        
    # Análise regional
    if 'RISCO_SOCIAL_FINAL' in df_filtered.columns:
        alta_vuln = len(df_filtered[df_filtered['RISCO_SOCIAL_FINAL'] > 60])
        total = len(df_filtered)
        if total > 0:
            perc_vuln = (alta_vuln / total) * 100
            st.warning(f"⚠️ **Alerta de Vulnerabilidade**: **{perc_vuln:.1f}%** dos registros filtrados encontram-se em zonas de Risco Social Final elevado (score > 60).")
