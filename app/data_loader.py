import os
import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Adicionar a raiz do projeto ao path para importar src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_utils import carregar_snis, limpar_df_gold, integrar_saude_tabnet, calcular_risco_social_final
from src.model_utils import classificar_qualidade_populacao, preparar_clusterizacao, treinar_kmeans, rotular_clusters

DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
RAW_DIR = DATA_DIR / "raw"

@st.cache_data
def load_data():
    """
    Carrega os dados processados se existirem, caso contrário roda o pipeline e salva.
    """
    parquet_path = PROCESSED_DIR / "df_final.parquet"
    
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    
    st.warning("Dados pré-processados não encontrados. Executando pipeline completo... (Isso pode demorar um pouco)")
    
    try:
        # Tentar processar os dados
        df_snis = carregar_snis(project_id="basedosdados", sigla_uf='ES', ano_min=2006)
        df_gold = limpar_df_gold(df_snis)
        
        # O arquivo CSV raw está em data/raw
        arq_agua = RAW_DIR / "saude_agua_es.csv"
        arq_esgoto = RAW_DIR / "saude_esgoto_es.csv"
        
        df_final, w_a, w_e = integrar_saude_tabnet(df_gold, str(arq_agua), str(arq_esgoto))
        df_final = calcular_risco_social_final(df_final, w_a, w_e)
        
        # Aplicar clusterização (usando o ano mais recente como referência para treinar)
        features = ['vazio_sanitario', 'Taxa_Morbidade_100k_Hab', 'investimento_total_consolidado']
        # Precisamos preencher NaNs temporariamente para clusterizar
        df_to_cluster = df_final.copy()
        
        # Aplicar K-Means básico e rotular
        ano_ref = int(df_to_cluster['ano'].max())
        df_cluster, X_scaled, scaler, _ = preparar_clusterizacao(df_to_cluster, features, ano_ref)
        df_cluster, perfil, km = treinar_kmeans(df_cluster, X_scaled, features, k_final=5)
        df_cluster, rotulos = rotular_clusters(df_cluster, perfil, k_final=5)
        
        # Fazer um merge dos rótulos de volta para o df_final
        df_final = pd.merge(df_final, df_cluster[['id_municipio', 'cluster', 'zona_vulnerabilidade']], on='id_municipio', how='left')
        
        # Salvar para não rodar de novo
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_parquet(parquet_path, index=False)
        st.success("Pipeline concluído e dados salvos!")
        
        return df_final
    except Exception as e:
        st.error(f"Falha ao carregar dados do BigQuery ou processar: {e}")
        st.info("Para testar o dashboard, crie o parquet 'df_final.parquet' na pasta data/processed/.")
        return pd.DataFrame()
