import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# Paleta de cores oficial
COLORS = {
    'primary': '#6C63FF',
    'secondary': '#FF6584',
    'accent': '#00BFA6',
    'text': '#2E2E2E',
    'bg': '#F8F9FB'
}

def render_correlation_heatmap(df_filtered):
    st.subheader("📊 Heatmap de Correlação")
    
    cols_to_corr = ['vazio_sanitario', 'Taxa_Morbidade_100k_Hab', 'investimento_total_consolidado', 
                    'RISCO_SOCIAL_FINAL', 'indice_atendimento_total_agua', 'indice_tratamento_esgoto']
    cols_exist = [c for c in cols_to_corr if c in df_filtered.columns]
    
    if len(cols_exist) > 1:
        corr = df_filtered[cols_exist].corr()
        fig = px.imshow(corr, text_auto=".2f", aspect="auto",
                        color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlação Linear de Pearson")
        st.plotly_chart(fig, use_container_width=True)

def render_scatter_regression(df_filtered):
    st.subheader("📈 Scatter com Regressão (Água/Esgoto vs Doenças)")
    if 'vazio_sanitario' in df_filtered.columns and 'Taxa_Morbidade_100k_Hab' in df_filtered.columns:
        fig = px.scatter(df_filtered, x='vazio_sanitario', y='Taxa_Morbidade_100k_Hab',
                         color='zona_vulnerabilidade' if 'zona_vulnerabilidade' in df_filtered.columns else None,
                         trendline="ols", hover_name='nome_municipio' if 'nome_municipio' in df_filtered.columns else 'id_municipio',
                         color_discrete_sequence=[COLORS['primary'], COLORS['secondary'], COLORS['accent']])
        st.plotly_chart(fig, use_container_width=True)

def render_pairplot(df_filtered):
    st.subheader("🔄 Pairplot (Multivariável)")
    cols = ['vazio_sanitario', 'Taxa_Morbidade_100k_Hab', 'RISCO_SOCIAL_FINAL']
    cols_exist = [c for c in cols if c in df_filtered.columns]
    if len(cols_exist) >= 2:
        fig = px.scatter_matrix(df_filtered, dimensions=cols_exist,
                                color='zona_vulnerabilidade' if 'zona_vulnerabilidade' in df_filtered.columns else None)
        st.plotly_chart(fig, use_container_width=True)

def render_time_series(df_filtered):
    st.subheader("📆 Série Temporal")
    if 'ano' in df_filtered.columns:
        df_agg = df_filtered.groupby('ano')[['Taxa_Morbidade_100k_Hab', 'vazio_sanitario']].mean().reset_index()
        
        # Gráfico Linha Dupla
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_agg['ano'], y=df_agg['Taxa_Morbidade_100k_Hab'], mode='lines+markers', name='Morbidade', line=dict(color=COLORS['secondary'], width=3)))
        fig1.add_trace(go.Scatter(x=df_agg['ano'], y=df_agg['vazio_sanitario'], mode='lines+markers', name='Vazio Sanitário', yaxis='y2', line=dict(color=COLORS['primary'], width=3)))
        
        fig1.update_layout(
            title="Evolução: Saneamento vs Doenças",
            yaxis=dict(title='Morbidade (100k)'),
            yaxis2=dict(title='Vazio Sanitário (%)', overlaying='y', side='right')
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Área empilhada
        st.markdown("##### Composição ao longo do tempo")
        fig2 = px.area(df_filtered, x='ano', y='Taxa_Morbidade_100k_Hab', color='zona_vulnerabilidade' if 'zona_vulnerabilidade' in df_filtered.columns else None)
        st.plotly_chart(fig2, use_container_width=True)

def render_comparisons(df_filtered):
    st.subheader("⚖️ Comparações e Rankings")
    
    col1, col2 = st.columns(2)
    with col1:
        # Barras Horizontais
        if 'RISCO_SOCIAL_FINAL' in df_filtered.columns:
            df_rank = df_filtered.groupby(df_filtered['nome_municipio'] if 'nome_municipio' in df_filtered.columns else 'id_municipio')['RISCO_SOCIAL_FINAL'].mean().reset_index()
            df_rank = df_rank.sort_values(by='RISCO_SOCIAL_FINAL', ascending=False).head(10)
            fig1 = px.bar(df_rank, x='RISCO_SOCIAL_FINAL', y=df_rank.columns[0], orientation='h', title="Top 10 Municípios mais Críticos", color_discrete_sequence=[COLORS['secondary']])
            fig1.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig1, use_container_width=True)
            
    with col2:
        # Barras agrupadas
        if 'zona_vulnerabilidade' in df_filtered.columns:
            df_grp = df_filtered.groupby('zona_vulnerabilidade')[['vazio_sanitario', 'Taxa_Morbidade_100k_Hab']].mean().reset_index()
            fig2 = go.Figure(data=[
                go.Bar(name='Vazio Sanitário', x=df_grp['zona_vulnerabilidade'], y=df_grp['vazio_sanitario'], marker_color=COLORS['primary']),
                go.Bar(name='Morbidade', x=df_grp['zona_vulnerabilidade'], y=df_grp['Taxa_Morbidade_100k_Hab'], marker_color=COLORS['secondary'])
            ])
            fig2.update_layout(barmode='group', title="Médias por Região de Risco")
            st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if 'zona_vulnerabilidade' in df_filtered.columns:
            fig3 = px.box(df_filtered, x='zona_vulnerabilidade', y='Taxa_Morbidade_100k_Hab', color='zona_vulnerabilidade', title="Boxplot (Outliers e Distribuição)")
            st.plotly_chart(fig3, use_container_width=True)
    with col4:
        if 'zona_vulnerabilidade' in df_filtered.columns:
            fig4 = px.violin(df_filtered, x='zona_vulnerabilidade', y='Taxa_Morbidade_100k_Hab', color='zona_vulnerabilidade', box=True, title="Violin Plot (Densidade Detalhada)")
            st.plotly_chart(fig4, use_container_width=True)

def render_advanced_relations(df_filtered):
    st.subheader("🔥 Relações Avançadas")
    
    col1, col2 = st.columns(2)
    with col1:
        # Scatter 3D
        if 'vazio_sanitario' in df_filtered.columns and 'investimento_total_consolidado' in df_filtered.columns:
            fig1 = px.scatter_3d(df_filtered, x='vazio_sanitario', y='investimento_total_consolidado', z='Taxa_Morbidade_100k_Hab',
                                 color='zona_vulnerabilidade' if 'zona_vulnerabilidade' in df_filtered.columns else None,
                                 title="Scatter 3D")
            st.plotly_chart(fig1, use_container_width=True)
    with col2:
        # Bubble Chart
        if 'vazio_sanitario' in df_filtered.columns:
            # Tamanho = População ou Risco
            size_col = 'populacao_ref' if 'populacao_ref' in df_filtered.columns else 'RISCO_SOCIAL_FINAL'
            if size_col in df_filtered.columns:
                df_clean = df_filtered.dropna(subset=[size_col])
                fig2 = px.scatter(df_clean, x='vazio_sanitario', y='Taxa_Morbidade_100k_Hab',
                                  size=size_col, color='zona_vulnerabilidade' if 'zona_vulnerabilidade' in df_clean.columns else None,
                                  hover_name='nome_municipio' if 'nome_municipio' in df_clean.columns else 'id_municipio',
                                  size_max=60, title="Bubble Chart (Tamanho = Impacto/População)")
                st.plotly_chart(fig2, use_container_width=True)
                
    # Hexbin Plot
    st.markdown("##### Hexbin Plot (Densidade de Pontos)")
    if 'vazio_sanitario' in df_filtered.columns:
        fig_hex = plt.figure(figsize=(10, 5))
        plt.hexbin(df_filtered['vazio_sanitario'], df_filtered['Taxa_Morbidade_100k_Hab'], gridsize=20, cmap='Purples')
        plt.colorbar(label='Contagem')
        plt.xlabel('Vazio Sanitário')
        plt.ylabel('Taxa Morbidade')
        # plt.gca().set_facecolor('#F8F9FB') # Aplicado no layout main.py
        st.pyplot(fig_hex)

def render_distributions(df_filtered):
    st.subheader("📉 Distribuições")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1 = px.histogram(df_filtered, x='Taxa_Morbidade_100k_Hab', title="Histograma (Morbidade)", color_discrete_sequence=[COLORS['primary']])
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        # KDE
        fig_kde = plt.figure(figsize=(6, 4))
        sns.kdeplot(data=df_filtered, x='Taxa_Morbidade_100k_Hab', fill=True, color=COLORS['secondary'])
        plt.title('KDE (Curva de Densidade)')
        st.pyplot(fig_kde)
    with col3:
        # ECDF
        fig3 = px.ecdf(df_filtered, x='Taxa_Morbidade_100k_Hab', title="ECDF (Dist. Acumulada)", color_discrete_sequence=[COLORS['accent']])
        st.plotly_chart(fig3, use_container_width=True)
