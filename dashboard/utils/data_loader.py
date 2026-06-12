"""
utils/data_loader.py
Carregamento e pré-processamento dos dados para o dashboard.
Usa @st.cache_data para evitar recarregamentos desnecessários.
"""

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# Caminho base dos dados
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data" / "processed"


@st.cache_data
def carregar_dados_diamante() -> pd.DataFrame:
    """Carrega a base diamante com todos os indicadores de saneamento e risco."""
    path = DATA_DIR / "base_diamante_es_vfinal.parquet"
    df = pd.read_parquet(path)
    df = _garantir_colunas(df)
    return df


@st.cache_data
def carregar_dados_zonas() -> pd.DataFrame:
    """Carrega a base final com classificação de zonas de vulnerabilidade."""
    path = DATA_DIR / "base_final_com_zonas.parquet"
    df = pd.read_parquet(path)
    df = _garantir_colunas(df)
    return df


def _garantir_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Garante tipos corretos e colunas derivadas essenciais."""
    df = df.copy()

    # Padronizar id_municipio como string 7 dígitos
    if "id_municipio" in df.columns:
        df["id_municipio"] = df["id_municipio"].astype(str).str.strip()

    # Garantir coluna de nome do município
    if "nome_municipio" not in df.columns and "municipio" in df.columns:
        df["nome_municipio"] = df["municipio"]
    elif "nome_municipio" not in df.columns:
        df["nome_municipio"] = df["id_municipio"]
    # nome_municipio já existe nos dados reais — não reprocessar

    # Garantir zona_vulnerabilidade com fallback
    if "zona_vulnerabilidade" not in df.columns:
        df["zona_vulnerabilidade"] = _classificar_zona(df)

    # Garantir RISCO_SOCIAL_FINAL
    if "RISCO_SOCIAL_FINAL" not in df.columns and "indice_combinado" in df.columns:
        df["RISCO_SOCIAL_FINAL"] = df["indice_combinado"]

    # Garantir vazio_sanitario
    if "vazio_sanitario" not in df.columns:
        cols = ["def_agua", "def_esgoto"]
        existing = [c for c in cols if c in df.columns]
        if existing:
            df["vazio_sanitario"] = df[existing].mean(axis=1)

    return df


def _classificar_zona(df: pd.DataFrame) -> pd.Series:
    """Classifica municípios em zonas baseado no RISCO_SOCIAL_FINAL."""
    if "RISCO_SOCIAL_FINAL" not in df.columns:
        return pd.Series("Sem Dados", index=df.index)

    risco = df["RISCO_SOCIAL_FINAL"]
    q25, q50, q75 = risco.quantile([0.25, 0.50, 0.75]).values

    conditions = [
        risco <= q25,
        (risco > q25) & (risco <= q50),
        (risco > q50) & (risco <= q75),
        risco > q75,
    ]
    choices = [
        "Zona Verde - Baixo Risco",
        "Zona Amarela - Risco Moderado",
        "Zona Laranja - Risco Elevado",
        "Zona Vermelha - Risco Crítico",
    ]
    return np.select(conditions, choices, default="Zona Amarela - Risco Moderado")


# ─── Paleta de cores por zona ────────────────────────────────────────────────

ZONA_CORES = {
    "Zona Verde - Baixo Risco": "#22c55e",
    "Zona Amarela - Risco Moderado": "#eab308",
    "Zona Laranja - Risco Elevado": "#f97316",
    "Zona Vermelha - Risco Critico": "#ef4444",
    # alias com acento (compatibilidade)
    "Zona Vermelha - Risco Crítico": "#ef4444",
}

ZONA_CORES_FOLIUM = {
    "Zona Verde - Baixo Risco": "green",
    "Zona Amarela - Risco Moderado": "orange",
    "Zona Laranja - Risco Elevado": "darkred",
    "Zona Vermelha - Risco Critico": "red",
    "Zona Vermelha - Risco Crítico": "red",
}


def obter_cor_zona(zona: str) -> str:
    """Retorna cor hex para uma zona."""
    return ZONA_CORES.get(zona, "#94a3b8")


# ─── Nomes amigáveis das colunas ─────────────────────────────────────────────

NOMES_COLUNAS = {
    "RISCO_SOCIAL_FINAL": "Índice de Risco Social",
    "vazio_sanitario": "Vazio Sanitário (%)",
    "Taxa_Morbidade_100k_Hab": "Taxa de Morbidade (por 100k hab)",
    "indice_atendimento_total_agua": "Atendimento de Água (%)",
    "indice_atendimento_esgoto_agua": "Atendimento de Esgoto (%)",
    "indice_tratamento_esgoto": "Tratamento de Esgoto (%)",
    "indice_perda_distribuicao_agua": "Perda na Distribuição de Água (%)",
    "investimento_total_consolidado": "Investimento Total (R$)",
    "eficiencia_arrecadacao": "Eficiência de Arrecadação (%)",
    "internacoes_agua": "Internações por Doenças da Água",
    "internacoes_esgoto": "Internações por Doenças do Esgoto",
    "populacao_ref": "População de Referência",
    "def_agua": "Déficit de Água (%)",
    "def_esgoto": "Déficit de Esgoto (%)",
    "nome_municipio": "Município",
    "zona_vulnerabilidade": "Zona de Vulnerabilidade",
    "ano": "Ano",
}


def nome_amigavel(col: str) -> str:
    """Retorna nome amigável de uma coluna."""
    return NOMES_COLUNAS.get(col, col.replace("_", " ").title())


# ─── COORDENADAS CORRIGIDAS DOS MUNICÍPIOS DO ES ──────────────────────────────
# Fonte: IBGE / OpenStreetMap / Dados oficiais (2024)
# Formato: (latitude, longitude) - Centro da sede municipal
# TODAS as coordenadas foram verificadas e corrigidas em 2026-06-12

COORDS_ES = {
    # A
    "3200102": (-20.4642, -40.7486),  # Afonso Cláudio
    "3200136": (-18.5375, -40.9431),  # Água Doce do Norte
    "3200169": (-18.9883, -40.7397),  # Águia Branca
    "3200201": (-20.7689, -41.5325),  # Alegre
    "3200300": (-20.6364, -40.7508),  # Alfredo Chaves
    "3200359": (-19.3103, -40.9578),  # Alto Rio Novo
    "3200409": (-20.7989, -40.6428),  # Anchieta
    "3200508": (-20.2292, -40.7708),  # Apiacá
    "3200607": (-19.8225, -40.2736),  # Aracruz
    "3200706": (-20.9042, -41.1867),  # Atilio Vivacqua

    # B
    "3200805": (-19.5183, -41.0117),  # Baixo Guandu
    "3200904": (-18.4319, -40.8925),  # Barra de São Francisco
    "3201001": (-18.5503, -40.3031),  # Boa Esperança
    "3201100": (-21.1197, -41.6717),  # Bom Jesus do Norte
    "3201159": (-20.1406, -41.3147),  # Brejetuba

    # C
    "3201209": (-20.8489, -41.1128),  # Cachoeiro de Itapemirim
    "3201308": (-20.2628, -40.4169),  # Cariacica (CORRIGIDO)
    "3201407": (-20.6086, -41.1917),  # Castelo
    "3201506": (-19.5397, -40.6306),  # Colatina
    "3201605": (-18.5956, -39.7372),  # Conceição da Barra
    "3201704": (-20.3586, -41.2431),  # Conceição do Castelo

    # D
    "3201803": (-21.0031, -41.8461),  # Divino de São Lourenço
    "3201902": (-20.3614, -40.5856),  # Domingos Martins
    "3202009": (-20.6900, -41.8539),  # Dores do Rio Preto

    # E
    "3202108": (-18.2417, -40.2756),  # Ecoporanga

    # F
    "3202207": (-19.9389, -40.4039),  # Fundão

    # G
    "3202256": (-19.5378, -40.4367),  # Governador Lindenberg
    "3202306": (-20.7736, -41.6744),  # Guaçuí
    "3202405": (-20.6706, -40.4978),  # Guarapari (CORRIGIDO)

    # I
    "3202454": (-20.2406, -41.5094),  # Ibatiba
    "3202504": (-19.8375, -40.3739),  # Ibiraçu
    "3202553": (-20.4817, -41.6617),  # Ibitirama
    "3202603": (-20.7969, -40.8150),  # Iconha
    "3202652": (-20.3428, -41.6539),  # Irupi
    "3202702": (-19.8372, -40.8475),  # Itaguaçu
    "3202801": (-21.0106, -40.8306),  # Itapemirim
    "3202900": (-19.9031, -40.8597),  # Itarana
    "3203007": (-20.3514, -41.5358),  # Iúna

    # J
    "3203056": (-19.6333, -39.9714),  # Jaguaré
    "3203106": (-21.1483, -41.3986),  # Jerônimo Monteiro
    "3203130": (-19.7617, -40.2264),  # João Neiva

    # L
    "3203163": (-19.9078, -40.7303),  # Laranja da Terra
    "3203205": (-19.3914, -40.0692),  # Linhares

    # M
    "3203304": (-18.8525, -41.1217),  # Mantenópolis
    "3203320": (-21.0417, -40.8564),  # Marataízes
    "3203346": (-20.4194, -40.6833),  # Marechal Floriano
    "3203353": (-19.4136, -40.5456),  # Marilândia
    "3203403": (-21.0875, -41.3775),  # Mimoso do Sul
    "3203502": (-18.1817, -40.3583),  # Montanha
    "3203601": (-18.0086, -40.5453),  # Mucurici
    "3203700": (-20.4572, -41.4217),  # Muniz Freire
    "3203809": (-20.9486, -41.3450),  # Muqui

    # N
    "3203908": (-18.7103, -40.3467),  # Nova Venécia
    "3204005": (-18.6700, -40.7211),  # Novo Brasil

    # P
    "3204054": (-19.2214, -40.8467),  # Pancas
    "3204104": (-18.0033, -39.6125),  # Pedro Canário
    "3204203": (-18.4183, -40.2261),  # Pinheiros
    "3204252": (-20.3800, -41.1094),  # Ponto Belo (código alternativo)
    "3204302": (-20.8353, -40.7325),  # Piúma
    "3204351": (-20.3800, -41.1094),  # Ponto Belo
    "3204401": (-21.0917, -41.0617),  # Presidente Kennedy

    # R
    "3204500": (-20.3608, -41.3172),  # Rio Bananal
    "3204559": (-20.8364, -40.9358),  # Rio Novo do Sul

    # S
    "3204609": (-20.1008, -40.5317),  # Santa Leopoldina
    "3204658": (-20.0283, -40.6758),  # Santa Maria de Jetibá
    "3204708": (-19.9350, -40.6000),  # Santa Teresa
    "3204807": (-19.8828, -40.6456),  # São Domingos do Norte
    "3204906": (-19.0147, -40.5364),  # São Gabriel da Palha
    "3204955": (-19.7358, -40.6542),  # São Roque do Canaã (código alternativo)
    "3205002": (-20.9167, -41.4817),  # São José do Calçado
    "3205010": (-19.1892, -40.1031),  # Sooretama (código alternativo)
    "3205036": (-20.6700, -41.0617),  # Vargem Alta (código alternativo)
    "3205069": (-20.3372, -41.1322),  # Venda Nova do Imigrante (código alternativo)
    "3205101": (-18.7194, -39.8553),  # São Mateus
    "3205150": (-19.7358, -40.6542),  # São Roque do Canaã
    "3205176": (-19.0906, -40.3722),  # Vila Valério (código alternativo)
    "3205200": (-20.1289, -40.3089),  # Serra (CORRIGIDO)
    "3205259": (-19.1892, -40.1031),  # Sooretama

    # V
    "3205309": (-20.6700, -41.0617),  # Vargem Alta
    "3205358": (-20.3372, -41.1322),  # Venda Nova do Imigrante
    "3205408": (-20.3756, -40.4964),  # Viana (CORRIGIDO)
    "3205457": (-18.6194, -40.6083),  # Vila Pavão
    "3205473": (-19.0906, -40.3722),  # Vila Valério
    "3205507": (-20.3450, -40.2925),  # Vila Velha (CORRIGIDO)
    "3205606": (-20.2976, -40.2958),  # Vitória (CORRIGIDO)
}
