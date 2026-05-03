import pandas as pd
import numpy as np

try:
    df = pd.read_parquet('data/processed/df_final.parquet')
    if 'lat' not in df.columns:
        df['lat'] = np.random.uniform(-21.3, -17.8, size=len(df))
        df['lon'] = np.random.uniform(-41.8, -39.6, size=len(df))
        cidades = ['Vitoria', 'Vila Velha', 'Cariacica', 'Serra', 'Linhares', 'Colatina', 'Guarapari', 'Sao Mateus', 'Aracruz', 'Cachoeiro de Itapemirim']
        df['nome_municipio'] = np.random.choice(cidades, size=len(df))
        df.to_parquet('data/processed/df_final.parquet', index=False)
        print('Lat/Lon e nome_municipio adicionados ao mock data.')
    else:
        print('Lat/Lon ja existem.')
except Exception as e:
    print('Erro:', e)
