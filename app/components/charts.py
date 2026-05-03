import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def render_kpis(df):
    """
    Renderiza os KPIs principais no topo da página.
    """
    col1, col2, col3, col4 = st.columns(4)
    
    media_morbidade = df['Taxa_Morbidade_100k_Hab'].mean()
    media_vazio = df['vazio_sanitario'].mean()
    
    # Identificar a cidade mais crítica no contexto filtrado
    if not df.empty and 'RISCO_SOCIAL_FINAL' in df.columns:
        cidade_critica_row = df.loc[df['RISCO_SOCIAL_FINAL'].idxmax()]
        cidade_critica_id = str(cidade_critica_row['id_municipio'])
        risco_critico = cidade_critica_row['RISCO_SOCIAL_FINAL']
    else:
        cidade_critica_id = "N/A"
        risco_critico = 0
        
    media_investimento = df['investimento_total_consolidado'].mean() / 1e6 if 'investimento_total_consolidado' in df.columns else 0
    
    with col1:
        st.metric(label="Média de Morbidade (100k hab)", value=f"{media_morbidade:.1f}", delta="- Meta" if media_morbidade < 50 else "+ Alerta", delta_color="inverse")
    with col2:
        st.metric(label="Média de Vazio Sanitário (%)", value=f"{media_vazio:.1f}%", delta="Crítico" if media_vazio > 30 else "Controlado", delta_color="inverse")
    with col3:
        st.metric(label="Município Crítico (ID)", value=cidade_critica_id, delta=f"Risco: {risco_critico:.1f}", delta_color="off")
    with col4:
        st.metric(label="Média de Investimentos", value=f"R$ {media_investimento:.1f} M")


def render_scatter_regression(df_filtered):
    """
    Renderiza scatter plot interativo de Saneamento vs Doenças.
    """
    st.subheader("📉 Relação Saneamento vs Doenças")
    st.markdown("O gráfico abaixo mostra como a taxa de morbidade se comporta à medida que o déficit de saneamento (vazio sanitário) aumenta.")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    sns.regplot(
        data=df_filtered, 
        x='vazio_sanitario', 
        y='Taxa_Morbidade_100k_Hab',
        scatter_kws={'alpha': 0.6, 's': 80, 'color': '#2a9d8f'},
        line_kws={'color': '#e76f51', 'linewidth': 3},
        ax=ax
    )
    
    ax.set_xlabel("Vazio Sanitário (%)", fontsize=12)
    ax.set_ylabel("Taxa de Morbidade (por 100k hab)", fontsize=12)
    
    # Destacar outliers, se existirem na coluna
    if 'outlier' in df_filtered.columns:
        outliers = df_filtered[df_filtered['outlier'] == True]
        if not outliers.empty:
            ax.scatter(outliers['vazio_sanitario'], outliers['Taxa_Morbidade_100k_Hab'], 
                       color='red', s=100, edgecolors='black', label='Anomalias Detectadas')
            ax.legend()
            
    # Estilizar para visual mais premium
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3)
    
    st.pyplot(fig)


def render_correlation_matrix(df_filtered):
    """
    Renderiza Heatmap de Correlação.
    """
    st.subheader("📊 Matriz de Correlação")
    
    cols_to_corr = ['vazio_sanitario', 'def_agua', 'def_esgoto', 'Taxa_Morbidade_100k_Hab', 'investimento_total_consolidado', 'RISCO_SOCIAL_FINAL']
    cols_exist = [c for c in cols_to_corr if c in df_filtered.columns]
    
    if len(cols_exist) > 1:
        corr = df_filtered[cols_exist].corr(method='spearman')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Máscara para o triângulo superior
        mask = np.triu(np.ones_like(corr, dtype=bool))
        
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
        
        st.pyplot(fig)
    else:
        st.info("Colunas insuficientes para calcular correlação.")
