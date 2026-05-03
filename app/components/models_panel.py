import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
from components.charts import COLORS

def render_linear_regression(df_filtered):
    st.subheader("📈 Regressão Linear Simples")
    if 'vazio_sanitario' in df_filtered.columns and 'Taxa_Morbidade_100k_Hab' in df_filtered.columns:
        df_clean = df_filtered.dropna(subset=['vazio_sanitario', 'Taxa_Morbidade_100k_Hab'])
        X = df_clean[['vazio_sanitario']]
        y = df_clean['Taxa_Morbidade_100k_Hab']
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        coef = model.coef_[0]
        intercept = model.intercept_
        
        st.write(f"**Equação:** Morbidade = {coef:.2f} * Vazio Sanitário + {intercept:.2f}")
        st.write(f"**R² (Qualidade do ajuste):** {r2:.4f}")
        
        fig = px.scatter(df_clean, x='vazio_sanitario', y='Taxa_Morbidade_100k_Hab', opacity=0.6, title="Morbidade vs Vazio Sanitário (Regressão)")
        fig.add_trace(go.Scatter(x=df_clean['vazio_sanitario'], y=y_pred, mode='lines', name='Regressão', line=dict(color=COLORS['secondary'], width=3)))
        st.plotly_chart(fig, use_container_width=True)

def render_random_forest(df_filtered):
    st.subheader("🌲 Random Forest Regressor")
    
    features = ['vazio_sanitario', 'investimento_total_consolidado', 'indice_atendimento_total_agua', 'indice_tratamento_esgoto']
    features_exist = [f for f in features if f in df_filtered.columns]
    
    if len(features_exist) > 1 and 'Taxa_Morbidade_100k_Hab' in df_filtered.columns:
        df_clean = df_filtered.dropna(subset=features_exist + ['Taxa_Morbidade_100k_Hab'])
        X = df_clean[features_exist]
        y = df_clean['Taxa_Morbidade_100k_Hab']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        r2 = r2_score(y_test, y_pred)
        st.write(f"**R² (Base de Teste):** {r2:.4f}")
        
        col1, col2 = st.columns(2)
        with col1:
            # Feature Importance
            importances = rf.feature_importances_
            df_imp = pd.DataFrame({'Feature': features_exist, 'Importância': importances}).sort_values(by='Importância', ascending=True)
            fig1 = px.bar(df_imp, x='Importância', y='Feature', orientation='h', title="Feature Importance", color_discrete_sequence=[COLORS['primary']])
            st.plotly_chart(fig1, use_container_width=True)
            
        with col2:
            # Real vs Previsto
            fig2 = px.scatter(x=y_test, y=y_pred, labels={'x': 'Real', 'y': 'Previsto'}, title="Real vs Previsto", color_discrete_sequence=[COLORS['accent']])
            fig2.add_shape(type="line", line=dict(dash='dash'), x0=y.min(), y0=y.min(), x1=y.max(), y1=y.max())
            st.plotly_chart(fig2, use_container_width=True)
            
        # Resíduos
        residuos = y_test - y_pred
        fig3 = px.histogram(residuos, title="Distribuição dos Resíduos (Erros)", color_discrete_sequence=[COLORS['secondary']])
        st.plotly_chart(fig3, use_container_width=True)

def render_arima(df_filtered):
    st.subheader("🔮 Previsão Temporal (ARIMA)")
    
    if 'ano' in df_filtered.columns and 'Taxa_Morbidade_100k_Hab' in df_filtered.columns:
        ts = df_filtered.groupby('ano')['Taxa_Morbidade_100k_Hab'].mean().sort_index()
        if len(ts) >= 3:
            st.write("Série agregada estadual/regional:")
            
            # Simple ARIMA model (1, 1, 0)
            try:
                model = sm.tsa.ARIMA(ts, order=(1, 1, 0))
                fitted = model.fit()
                
                # Previsão 3 anos
                forecast = fitted.forecast(steps=3)
                anos_futuros = [ts.index[-1] + i for i in range(1, 4)]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ts.index, y=ts.values, mode='lines+markers', name='Histórico', line=dict(color=COLORS['primary'])))
                fig.add_trace(go.Scatter(x=anos_futuros, y=forecast.values, mode='lines+markers', name='Previsão', line=dict(color=COLORS['secondary'], dash='dash')))
                fig.update_layout(title="Previsão de Morbidade (3 Anos)", xaxis_title="Ano", yaxis_title="Morbidade (100k)")
                
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Não foi possível ajustar o modelo ARIMA com os dados disponíveis: {e}")
        else:
            st.warning("É necessário pelo menos 3 anos de dados para gerar a previsão simplificada.")

def render_kmeans(df_filtered):
    st.subheader("🧠 Clusterização (K-Means)")
    
    features = ['vazio_sanitario', 'Taxa_Morbidade_100k_Hab']
    if all(f in df_filtered.columns for f in features):
        df_clean = df_filtered.dropna(subset=features)
        
        n_clusters = st.slider("Número de Clusters (k)", min_value=2, max_value=7, value=4)
        
        km = KMeans(n_clusters=n_clusters, random_state=42)
        df_clean['Cluster_ML'] = km.fit_predict(df_clean[features]).astype(str)
        
        fig = px.scatter(df_clean, x='vazio_sanitario', y='Taxa_Morbidade_100k_Hab', color='Cluster_ML', 
                         title=f"K-Means ({n_clusters} Clusters)", hover_name='nome_municipio' if 'nome_municipio' in df_clean.columns else 'id_municipio',
                         color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar centróides
        centros = pd.DataFrame(km.cluster_centers_, columns=features)
        st.write("**Centróides identificados:**")
        st.dataframe(centros.round(2))
