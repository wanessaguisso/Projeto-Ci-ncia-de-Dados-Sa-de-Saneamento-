"""
Funcoes reutilizaveis para analise estatistica e clusterizacao.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest


def teste_shapiro(
	df: pd.DataFrame,
	colunas: Iterable[str],
	sample_limit: int = 5000,
	random_state: int = 42
) -> pd.DataFrame:
	"""
	Executa o teste de Shapiro-Wilk para cada coluna informada.

	Retorna DataFrame com p-valor e classificacao de distribuicao.
	"""
	resultados = []

	for col in colunas:
		serie = df[col].dropna()
		if serie.empty:
			resultados.append({
				'coluna': col,
				'p_valor': np.nan,
				'distribuicao': 'sem_dados'
			})
			continue

		amostra = serie.sample(min(sample_limit, len(serie)), random_state=random_state)
		_, p_valor = stats.shapiro(amostra)
		distribuicao = 'normal' if p_valor > 0.05 else 'nao-normal'

		resultados.append({
			'coluna': col,
			'p_valor': p_valor,
			'distribuicao': distribuicao
		})

	return pd.DataFrame(resultados)


def correlacao_spearman(df: pd.DataFrame, colunas: Sequence[str]) -> pd.DataFrame:
	"""
	Calcula a matriz de correlacao de Spearman para as colunas informadas.
	"""
	df_corr = df[list(colunas)].dropna()
	corr_matrix, _ = stats.spearmanr(df_corr)
	return pd.DataFrame(corr_matrix, index=df_corr.columns, columns=df_corr.columns)


def plotar_heatmap_spearman(
	corr_df: pd.DataFrame,
	titulo: str = 'Correlacao de Spearman - Saneamento vs Saude (ES)',
	figsize: tuple[int, int] = (10, 8)
):
	"""
	Plota heatmap da matriz de Spearman.
	"""
	fig, ax = plt.subplots(figsize=figsize)
	mask = np.triu(np.ones_like(corr_df, dtype=bool))

	sns.heatmap(
		corr_df,
		mask=mask,
		annot=True,
		fmt='.2f',
		cmap='RdYlGn_r',
		center=0,
		vmin=-1,
		vmax=1,
		linewidths=0.5,
		ax=ax
	)
	ax.set_title(titulo, fontsize=13, pad=15)
	plt.tight_layout()

	return fig, ax


def teste_kruskal_wallis_por_tercis(
	df: pd.DataFrame,
	coluna_saneamento: str,
	coluna_saude: str,
	q: int = 3,
	labels: Sequence[str] | None = None
):
	"""
	Aplica Kruskal-Wallis usando tercis (ou quantis) do saneamento.
	"""
	if labels is None:
		labels = ['Baixo', 'Medio', 'Alto']

	df_hip = df[[coluna_saneamento, coluna_saude]].dropna().copy()
	df_hip['grupo_saneamento'] = pd.qcut(
		df_hip[coluna_saneamento],
		q=q,
		labels=labels
	)

	grupos = [
		df_hip[df_hip['grupo_saneamento'] == label][coluna_saude]
		for label in labels
	]

	stat, p_valor = stats.kruskal(*grupos)
	return stat, p_valor, df_hip


def plotar_boxplot_kruskal(
	df_hip: pd.DataFrame,
	coluna_saude: str,
	coluna_grupo: str = 'grupo_saneamento',
	p_valor: float | None = None,
	titulo_prefixo: str = 'Morbidade por Grupo de Saneamento',
	figsize: tuple[int, int] = (8, 5)
):
	"""
	Plota boxplot para o teste de Kruskal-Wallis.
	"""
	fig, ax = plt.subplots(figsize=figsize)
	df_hip.boxplot(column=coluna_saude, by=coluna_grupo, ax=ax)

	if p_valor is not None:
		ax.set_title(f'{titulo_prefixo} (p={p_valor:.4f})')
	else:
		ax.set_title(titulo_prefixo)

	ax.set_xlabel('Nivel de Atendimento de Agua')
	ax.set_ylabel('Taxa de Morbidade (por 100k hab)')
	plt.suptitle('')
	plt.tight_layout()

	return fig, ax


def preparar_clusterizacao(
	df_final: pd.DataFrame,
	features_cluster: Sequence[str],
	ano_ref: int | None = None
):
	"""
	Prepara dados e normaliza features para clusterizacao.
	"""
	if ano_ref is None:
		ano_ref = int(df_final['ano'].max())

	df_cluster = (
		df_final[df_final['ano'] == ano_ref][['id_municipio'] + list(features_cluster)]
		.dropna()
		.copy()
	)

	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(df_cluster[list(features_cluster)])

	return df_cluster, X_scaled, scaler, ano_ref


def avaliar_k_elbow_silhouette(
	X_scaled: np.ndarray,
	k_range: Iterable[int] = range(2, 10),
	random_state: int = 42,
	n_init: int = 10,
	figsize: tuple[int, int] = (13, 4)
):
	"""
	Calcula inercias e silhuetas para o metodo do cotovelo.
	"""
	inercias = []
	silhouettes = []

	for k in k_range:
		km = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
		km.fit(X_scaled)
		inercias.append(km.inertia_)
		silhouettes.append(silhouette_score(X_scaled, km.labels_))

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

	ax1.plot(list(k_range), inercias, marker='o', color='steelblue')
	ax1.set_title('Metodo do Cotovelo (Inercia)')
	ax1.set_xlabel('Numero de Clusters (k)')
	ax1.set_ylabel('Inercia')

	ax2.plot(list(k_range), silhouettes, marker='s', color='seagreen')
	ax2.set_title('Coeficiente de Silhueta')
	ax2.set_xlabel('Numero de Clusters (k)')
	ax2.set_ylabel('Silhueta (maior = melhor)')

	plt.tight_layout()

	return inercias, silhouettes, fig, (ax1, ax2)


def treinar_kmeans(
	df_cluster: pd.DataFrame,
	X_scaled: np.ndarray,
	features_cluster: Sequence[str],
	k_final: int,
	random_state: int = 42,
	n_init: int = 10
):
	"""
	Treina KMeans e calcula o perfil medio por cluster.
	"""
	km_final = KMeans(n_clusters=k_final, random_state=random_state, n_init=n_init)
	df_out = df_cluster.copy()
	df_out['cluster'] = km_final.fit_predict(X_scaled)

	perfil = df_out.groupby('cluster')[list(features_cluster)].mean().round(2)
	return df_out, perfil, km_final


def rotular_clusters(
	df_cluster: pd.DataFrame,
	perfil: pd.DataFrame,
	k_final: int,
	labels: Sequence[str] | None = None
):
	"""
	Rotula clusters ordenados por risco social.
	"""
	if labels is None:
		niveis = [
			'Baixo Risco',
			'Risco Moderado',
			'Risco Elevado',
			'Risco Critico',
			'Risco Muito Critico'
		]
		prefixos = [
			'Zona Verde',
			'Zona Amarela',
			'Zona Laranja',
			'Zona Vermelha',
			'Zona Preta'
		]
		if k_final > len(niveis):
			raise ValueError('labels deve ser informado quando k_final > 5')
		labels = [f"{prefixos[i]} - {niveis[i]}" for i in range(k_final)]

	ordem_risco = perfil['RISCO_SOCIAL_FINAL'].sort_values().index.tolist()
	rotulos = {ordem_risco[i]: labels[i] for i in range(len(ordem_risco))}

	df_out = df_cluster.copy()
	df_out['zona_vulnerabilidade'] = df_out['cluster'].map(rotulos)

	return df_out, rotulos


def plotar_zonas_vulnerabilidade(
	df_cluster: pd.DataFrame,
	ano_ref: int,
	cores: dict[str, str] | None = None,
	figsize: tuple[int, int] = (10, 6)
):
	"""
	Plota scatter de vazio sanitario vs morbidade por zona de vulnerabilidade.
	"""
	if cores is None:
		cores = {
			'Zona Verde - Baixo Risco': 'green',
			'Zona Amarela - Risco Moderado': 'gold',
			'Zona Laranja - Risco Elevado': 'darkorange',
			'Zona Vermelha - Risco Critico': 'crimson'
		}

	fig, ax = plt.subplots(figsize=figsize)

	for zona, grupo in df_cluster.groupby('zona_vulnerabilidade'):
		ax.scatter(
			grupo['vazio_sanitario'],
			grupo['Taxa_Morbidade_100k_Hab'],
			label=zona,
			color=cores.get(zona, 'gray'),
			alpha=0.75,
			edgecolors='white',
			s=80
		)

	ax.set_xlabel('Vazio Sanitario (%)')
	ax.set_ylabel('Taxa de Morbidade (por 100k hab)')
	ax.set_title(f'Zonas de Vulnerabilidade - ES {ano_ref}')
	ax.legend(loc='upper left')
	plt.tight_layout()

	return fig, ax


def regressao_linear_saneamento_saude(
	df: pd.DataFrame, 
	coluna_x: str = 'vazio_sanitario', 
	coluna_y: str = 'Taxa_Morbidade_100k_Hab'
):
	"""
	Treina um modelo de regressão linear simples para estimar morbidade com base no déficit sanitário.
	Retorna o modelo, o R², o coeficiente (impacto) e o intercepto.
	"""
	df_reg = df[[coluna_x, coluna_y]].dropna()
	if df_reg.empty:
		return None, np.nan, np.nan, np.nan

	X = df_reg[[coluna_x]].values
	y = df_reg[coluna_y].values

	modelo = LinearRegression()
	modelo.fit(X, y)

	r2 = modelo.score(X, y)
	coef = modelo.coef_[0]
	intercept = modelo.intercept_

	return modelo, r2, coef, intercept


def detectar_outliers_saneamento_saude(
	df: pd.DataFrame, 
	coluna_x: str = 'vazio_sanitario', 
	coluna_y: str = 'Taxa_Morbidade_100k_Hab',
	contamination: float = 0.05
):
	"""
	Usa Isolation Forest para identificar municípios anômalos (outliers).
	"""
	df_out = df.copy()
	# Preencher naN com média para não dropar linhas na identificação de outliers
	X = df_out[[coluna_x, coluna_y]].fillna(df_out[[coluna_x, coluna_y]].mean())

	iso = IsolationForest(contamination=contamination, random_state=42)
	df_out['outlier'] = iso.fit_predict(X)
	df_out['outlier'] = df_out['outlier'].map({1: False, -1: True})

	return df_out


def simular_cenario_melhoria_saneamento(
	df: pd.DataFrame,
	modelo: LinearRegression,
	melhoria_percentual: float = 20.0,
	coluna_x: str = 'vazio_sanitario'
):
	"""
	Simula um cenário onde o vazio sanitário diminui em X%.
	Estima a nova taxa de morbidade com base no modelo de regressão.
	"""
	df_sim = df.copy()
	
	fator = 1.0 - (melhoria_percentual / 100.0)
	df_sim['vazio_sanitario_simulado'] = df_sim[coluna_x] * fator

	# Previsão da morbidade
	X_sim = df_sim[['vazio_sanitario_simulado']].fillna(0).values
	y_sim_pred = modelo.predict(X_sim)
	
	# Evitar morbidade negativa
	y_sim_pred = np.clip(y_sim_pred, a_min=0, a_max=None)
	
	df_sim['Morbidade_Simulada_100k'] = y_sim_pred
	df_sim['Reducao_Morbidade_Abs'] = df_sim['Taxa_Morbidade_100k_Hab'] - df_sim['Morbidade_Simulada_100k']
	
	return df_sim
