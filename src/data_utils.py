import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_gbq

# ---------------------------------------------------------------------------
# 1. Carregamento
# ---------------------------------------------------------------------------

def carregar_snis(project_id: str, sigla_uf: str = 'ES', ano_min: int = 2006) -> pd.DataFrame:
    """
    Carrega dados do SNIS via BigQuery e retorna DataFrame bruto.
    """
    
    sql = f"""
    SELECT
      ano,
      id_municipio,
      sigla_uf,
      quantidade_economia_residencial_ativa_agua,
      quantidade_economia_residencial_ativa_esgoto,
      quantidade_ligacao_total_agua,
      quantidade_ligacao_total_esgoto,
      populacao_urbana,
      populacao_atendida_agua,
      indice_atendimento_total_agua,
      indice_atendimento_esgoto_agua,
      indice_atendimento_urbano_agua,
      indice_tratamento_esgoto,
      indice_perda_distribuicao_agua,
      indice_consumo_agua_per_capita,
      volume_esgoto_coletado,
      volume_esgoto_tratado,
      extensao_rede_agua,
      extensao_rede_esgoto,
      populacao_atentida_esgoto AS populacao_urbana_atendida_esgoto,
      quantidade_ligacao_ativa_esgoto,
      investimento_total_municipio,
      investimento_total_estado,
      investimento_total_prestador,
      despesa_exploracao,
      arrecadacao_total,
      receita_operacional
    FROM `basedosdados.br_mdr_snis.municipio_agua_esgoto`
    WHERE sigla_uf = '{sigla_uf}' AND ano >= {ano_min}
    """

    try:
        df = pandas_gbq.read_gbq(sql, project_id=project_id)
        print(f"✅ Dados do {sigla_uf} carregados: {df.shape}")
        return df
    except Exception as e:
        raise RuntimeError(f"Erro ao acessar a tabela de saneamento: {e}")


# ---------------------------------------------------------------------------
# 2. Limpeza de população
# ---------------------------------------------------------------------------

def preparar_populacao_referencia(df_base: pd.DataFrame) -> pd.DataFrame:
    """
    Imputa e interpola populacao_ref com fallback para populacao_atendida_agua.

    Reconstrói a coluna a partir das colunas-base para evitar estado
    inconsistente quando as células forem reexecutadas fora de ordem.
    """
    df_base = df_base.copy().sort_values(['id_municipio', 'ano'])

    if 'populacao_ref_bruta' in df_base.columns:
        pop_base = df_base['populacao_ref_bruta']
    else:
        pop_base = df_base['populacao_urbana']

    df_base['populacao_urbana_limpa'] = pop_base
    df_base['populacao_urbana_era_nula'] = df_base['populacao_urbana_limpa'].isna()

    df_base['populacao_ref'] = df_base['populacao_urbana_limpa']
    mask_fallback = df_base['populacao_ref'].isna()
    df_base.loc[mask_fallback, 'populacao_ref'] = df_base.loc[mask_fallback, 'populacao_atendida_agua']

    df_base['populacao_usou_fallback_agua'] = (
        df_base['populacao_urbana_era_nula'] & df_base['populacao_atendida_agua'].notna()
    )

    df_base['fonte_populacao'] = np.where(
        df_base['populacao_urbana_limpa'].notna(),
        'urbana',
        np.where(
            df_base['populacao_atendida_agua'].notna(),
            'agua',
            'missing'
        )
    )

    # Nula = ainda nula após todos os fallbacks (para interpolação)
    df_base['populacao_ref_era_nula'] = df_base['populacao_ref'].isna()
    df_base['populacao_ref'] = pd.to_numeric(df_base['populacao_ref'], errors='coerce')
    df_base['populacao_ref'] = df_base.groupby('id_municipio')['populacao_ref'].transform(
        lambda x: x.astype('float64').interpolate(method='linear', limit=2, limit_area='inside')
    )

    return df_base


# ---------------------------------------------------------------------------
# 3. Flags de evidência de esgoto
# ---------------------------------------------------------------------------

def calcular_flags_evidencia(df: pd.DataFrame):
    """
    Retorna (tem_rede, tem_trat_real) como Series booleanas.

    Recalcula a partir das colunas-base do df recebido.
    Evita estado global inconsistente quando células são reexecutadas fora de ordem.
    """
    evidencias_coleta = [c for c in [
        'extensao_rede_esgoto',
        'populacao_urbana_atendida_esgoto',
        'quantidade_ligacao_ativa_esgoto'
    ] if c in df.columns]

    evidencias_tratamento = [c for c in ['volume_esgoto_tratado'] if c in df.columns]

    tem_rede = (
        df[evidencias_coleta].fillna(0).gt(0).any(axis=1)
        if evidencias_coleta else pd.Series(False, index=df.index)
    )
    tem_trat_real = (
        df[evidencias_tratamento].fillna(0).gt(0).any(axis=1)
        if evidencias_tratamento else pd.Series(False, index=df.index)
    )

    return tem_rede, tem_trat_real


# ---------------------------------------------------------------------------
# 3b. Limpeza principal (df_gold)
# ---------------------------------------------------------------------------

def limpar_df_gold(df_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e enriquece df_silver, gerando df_gold pronto para integração.

    Inclui tratamento de esgoto, interpolação conservadora, flags de qualidade,
    cálculo de gaps e preparação da população de referência.
    """
    df_gold = df_silver.sort_values(['id_municipio', 'ano']).copy()

    # Snapshot pré-imputação para auditoria/validação posterior
    df_gold['populacao_ref_bruta'] = df_gold['populacao_urbana']
    mask_pop_urbana_zero = df_gold['populacao_ref_bruta'].eq(0)
    df_gold.loc[mask_pop_urbana_zero, 'populacao_ref_bruta'] = np.nan

    df_gold['populacao_atendida_agua'] = df_gold['populacao_atendida_agua'].replace(0, np.nan)

    # Lógica rigorosa de evidência (coleta vs tratamento)
    tem_rede, tem_trat_real = calcular_flags_evidencia(df_gold)

    mask_coleta_nulo = df_gold['indice_atendimento_esgoto_agua'].isna()
    df_gold.loc[mask_coleta_nulo & ~tem_rede, 'indice_atendimento_esgoto_agua'] = 0.0

    mask_trat_nulo = df_gold['indice_tratamento_esgoto'].isna()
    df_gold.loc[mask_trat_nulo & ~tem_trat_real, 'indice_tratamento_esgoto'] = 0.0

    mask_vol_nulo = df_gold['volume_esgoto_tratado'].isna()
    df_gold.loc[mask_vol_nulo & ~tem_trat_real, 'volume_esgoto_tratado'] = 0.0

    # Interpolação conservadora
    cols_interp = [
        'indice_atendimento_esgoto_agua',
        'indice_tratamento_esgoto',
        'volume_esgoto_tratado',
        'volume_esgoto_coletado'
    ]

    for col in cols_interp:
        df_gold[col] = df_gold.groupby('id_municipio')[col].transform(
            lambda x: x.interpolate(method='linear', limit=2, limit_area='inside')
        )

    # Preencher zero apenas quando a ausência representa inexistência estrutural comprovada
    df_gold.loc[
        df_gold['indice_atendimento_esgoto_agua'].isna() & ~tem_rede,
        'indice_atendimento_esgoto_agua'
    ] = 0.0
    df_gold.loc[
        df_gold['indice_tratamento_esgoto'].isna() & ~tem_trat_real,
        'indice_tratamento_esgoto'
    ] = 0.0
    df_gold.loc[
        df_gold['volume_esgoto_tratado'].isna() & ~tem_trat_real,
        'volume_esgoto_tratado'
    ] = 0.0
    df_gold.loc[
        df_gold['volume_esgoto_coletado'].isna() & ~tem_rede,
        'volume_esgoto_coletado'
    ] = 0.0

    # Flags de completude para evitar tratar dado ausente como zero real
    df_gold['flag_insumos_esgoto_incompletos'] = df_gold[cols_interp].isna().any(axis=1)

    # Cálculo dos gaps (com proteção contra NaN)
    df_gold['Volume_Esgoto_Nao_Tratado_m3'] = np.where(
        df_gold['volume_esgoto_coletado'].notna() & df_gold['volume_esgoto_tratado'].notna(),
        (df_gold['volume_esgoto_coletado'] - df_gold['volume_esgoto_tratado']).clip(lower=0).round(2),
        np.nan
    )

    indice_trat_limite = df_gold['indice_tratamento_esgoto'].clip(upper=100)

    df_gold['Atendimento_Com_Tratamento_Efetivo_Percentual'] = np.where(
        df_gold['indice_atendimento_esgoto_agua'].notna() & indice_trat_limite.notna(),
        (df_gold['indice_atendimento_esgoto_agua'] * (indice_trat_limite / 100)).round(2),
        np.nan
    )

    df_gold['Deficit_Cobertura_Tratamento_Percentual'] = np.where(
        df_gold['indice_atendimento_esgoto_agua'].notna()
        & df_gold['Atendimento_Com_Tratamento_Efetivo_Percentual'].notna(),
        (df_gold['indice_atendimento_esgoto_agua']
         - df_gold['Atendimento_Com_Tratamento_Efetivo_Percentual']).clip(lower=0).round(2),
        np.nan
    )

    # Investimentos (não assumir zero quando todos os componentes estiverem ausentes)
    cols_invest = [
        'investimento_total_municipio',
        'investimento_total_estado',
        'investimento_total_prestador'
    ]
    df_gold['investimento_total_consolidado'] = df_gold[cols_invest].sum(axis=1, min_count=1)
    df_gold['flag_investimento_parcial_ou_ausente'] = df_gold[cols_invest].isna().any(axis=1)

    # População de referência
    df_gold = preparar_populacao_referencia(df_gold)

    # Flags de qualidade dos dados de esgoto
    tem_rede_flags, _ = calcular_flags_evidencia(df_gold)
    df_gold['qualidade_dados_esgoto'] = 'dados_preenchidos'

    df_gold.loc[
        tem_rede_flags & (df_gold['indice_tratamento_esgoto'] == 0),
        'qualidade_dados_esgoto'
    ] = 'possivel_omissao_informativa'

    mask_sem_rede_atendimento_zero = (
        (df_gold['extensao_rede_esgoto'] == 0)
        & (df_gold['indice_atendimento_esgoto_agua'] == 0)
    )

    df_gold.loc[
        mask_sem_rede_atendimento_zero & (~df_gold['flag_insumos_esgoto_incompletos']),
        'qualidade_dados_esgoto'
    ] = 'provavelmente_inexistente'

    df_gold.loc[
        mask_sem_rede_atendimento_zero & df_gold['flag_insumos_esgoto_incompletos'],
        'qualidade_dados_esgoto'
    ] = 'ausencia_de_dados'

    # Limpeza de resíduos de arredondamento no déficit
    df_gold.loc[
        df_gold['Deficit_Cobertura_Tratamento_Percentual'] < 1e-6,
        'Deficit_Cobertura_Tratamento_Percentual'
    ] = 0

    # Limpeza final de infraestrutura (mantém NaN e cria colunas _display)
    colunas_contagem = [
        'extensao_rede_esgoto',
        'populacao_urbana_atendida_esgoto',
        'quantidade_ligacao_ativa_esgoto',
        'quantidade_economia_residencial_ativa_esgoto',
        'quantidade_ligacao_total_esgoto'
    ]

    for col in colunas_contagem:
        if col in df_gold.columns:
            df_gold[f'{col}_display'] = df_gold[col].fillna(0)

    return df_gold


# ---------------------------------------------------------------------------
# 4. Integração com DATASUS/TabNet
# ---------------------------------------------------------------------------

def processar_saude(arquivo: str, nome_metrica: str) -> pd.DataFrame:
    """
    Carrega e limpa CSV do TabNet (DATASUS), retorna long format.

    Detecta dinamicamente a faixa útil de dados e evita dependência de
    skipfooter fixo, que não funciona com engine='c'.
    """
    df_raw = pd.read_csv(
        arquivo,
        sep=';',
        encoding='iso-8859-1',
        header=None,
        dtype=str,
        names=list(range(40)),
        engine='python'
    )

    col0 = df_raw.iloc[:, 0].fillna('').astype(str).str.strip()
    mask_id = col0.str.match(r'^\d{6,}')

    if not mask_id.any():
        raise ValueError(f'Não foi possível identificar linhas de municípios em {arquivo}.')

    primeiro_dado = int(mask_id.idxmax())
    inicio = max(primeiro_dado - 1, 0)  # linha de cabeçalho logo antes do primeiro município

    apos_primeiro = col0.iloc[primeiro_dado + 1:]
    mask_fim = ~apos_primeiro.str.match(r'^\d{6,}')

    # idxmax() retornaria o PRIMEIRO True, mas nonzero() torna a intenção explícita:
    # queremos parar na primeira linha que NÃO seja um município válido
    # (ex: rodapé "Total", linha vazia, nota de rodapé).
    # Se todas as linhas forem municípios válidos, usamos len(df_raw) como sentinela.
    indices_fim = mask_fim.to_numpy().nonzero()[0]
    fim = int(apos_primeiro.index[indices_fim[0]]) if len(indices_fim) > 0 else len(df_raw)

    nrows = max(fim - inicio - 1, 1)

    df_s = pd.read_csv(
        arquivo,
        sep=';',
        encoding='iso-8859-1',
        skiprows=inicio,
        nrows=nrows
    )

    df_s = df_s.drop(columns=['Total', 'total'], errors='ignore')
    col_mun = df_s.columns[0]
    df_s = df_s[df_s[col_mun].astype(str).str.extract(r'^(\d{6})')[0].notna()].copy()

    df_long = df_s.melt(
        id_vars=[col_mun],
        var_name='ano_bruto',
        value_name=nome_metrica
    )

    df_long['id_municipio_6'] = df_long[col_mun].astype(str).str.extract(r'^(\d{6})')[0].str.strip()
    df_long['ano'] = pd.to_numeric(
        df_long['ano_bruto'].astype(str).str.extract(r'(\d{4})')[0], errors='coerce'
    )

    # Conversão robusta de contagem (formato brasileiro com ponto como milhar)
    df_long[nome_metrica] = (
        df_long[nome_metrica]
        .astype(str)
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.strip()
        .replace({'-': None, '': None, 'nan': None})
    )
    df_long[nome_metrica] = pd.to_numeric(df_long[nome_metrica], errors='coerce')

    df_limpo = df_long.dropna(subset=['ano', 'id_municipio_6']).copy()
    df_limpo['ano'] = df_limpo['ano'].astype(int)
    df_limpo[nome_metrica] = df_limpo[nome_metrica].fillna(0)

    return df_limpo[['id_municipio_6', 'ano', nome_metrica]]


def integrar_saude_tabnet(
    df_gold: pd.DataFrame,
    arquivo_agua: str,
    arquivo_esgoto: str,
    versao_pesos: str = '2025-04',
    w_agua_calculado: float = 0.52,
    w_esgoto_calculado: float = 0.48,
    recalcular_pesos: bool = False,
    strict: bool = False
) -> tuple[pd.DataFrame, float, float]:
    """
    Integra dados de saúde (TabNet) ao df_gold e calcula morbidade.

    Retorna (df_final, W_AGUA, W_ESGOTO).
    """
    try:
        linhas_antes = df_gold.shape[0]
        df_base = df_gold.copy()
        df_base['id_municipio_6'] = df_base['id_municipio'].astype(str).str.slice(0, 6).str.strip()

        # Validação de unicidade no SNIS (lado esquerdo do merge)
        dupes_snis = df_base.duplicated(subset=['id_municipio_6', 'ano'], keep=False)
        if dupes_snis.any():
            exemplos = (
                df_base.loc[dupes_snis, ['id_municipio', 'id_municipio_6', 'ano']]
                .drop_duplicates()
                .sort_values(['id_municipio_6', 'ano'])
                .head(20)
            )
            raise ValueError(
                "Colisão/duplicidade no SNIS para a chave (id_municipio_6, ano). "
                f"Exemplos:\n{exemplos}"
            )

        df_s_agua = processar_saude(arquivo_agua, 'internacoes_agua')
        df_s_esgoto = processar_saude(arquivo_esgoto, 'internacoes_esgoto')

        # Validação de unicidade na saúde (água)
        dupes_saude_agua = df_s_agua.duplicated(subset=['id_municipio_6', 'ano'], keep=False)
        if dupes_saude_agua.any():
            exemplos = (
                df_s_agua.loc[dupes_saude_agua, ['id_municipio_6', 'ano']]
                .drop_duplicates()
                .sort_values(['id_municipio_6', 'ano'])
                .head(20)
            )
            raise ValueError(
                "Colisão/duplicidade na base de saúde (água) para a chave (id_municipio_6, ano). "
                f"Exemplos:\n{exemplos}"
            )

        # Validação de unicidade na saúde (esgoto)
        dupes_saude_esgoto = df_s_esgoto.duplicated(subset=['id_municipio_6', 'ano'], keep=False)
        if dupes_saude_esgoto.any():
            exemplos = (
                df_s_esgoto.loc[dupes_saude_esgoto, ['id_municipio_6', 'ano']]
                .drop_duplicates()
                .sort_values(['id_municipio_6', 'ano'])
                .head(20)
            )
            raise ValueError(
                "Colisão/duplicidade na base de saúde (esgoto) para a chave (id_municipio_6, ano). "
                f"Exemplos:\n{exemplos}"
            )

        anos_agua = set(df_s_agua['ano'])
        anos_esgoto = set(df_s_esgoto['ano'])
        anos_comuns = anos_agua.intersection(anos_esgoto)

        print("\n📊 DIAGNÓSTICO DE COBERTURA")
        print(f"Água: {min(anos_agua)}–{max(anos_agua)} ({len(anos_agua)} anos)")
        print(f"Esgoto: {min(anos_esgoto)}–{max(anos_esgoto)} ({len(anos_esgoto)} anos)")

        if anos_comuns:
            print(f"Interseção: {min(anos_comuns)}–{max(anos_comuns)} ({len(anos_comuns)} anos)")
        else:
            print("Interseção: nenhuma. Mantidos pesos fixos versionados.")

        w_agua = float(w_agua_calculado)
        w_esgoto = float(w_esgoto_calculado)

        if recalcular_pesos:
            if anos_comuns:
                df_agua_common = df_s_agua[df_s_agua['ano'].isin(anos_comuns)]
                df_esgoto_common = df_s_esgoto[df_s_esgoto['ano'].isin(anos_comuns)]

                total_a = float(df_agua_common['internacoes_agua'].sum())
                total_e = float(df_esgoto_common['internacoes_esgoto'].sum())
                total_g = total_a + total_e

                if total_g > 0:
                    w_agua = total_a / total_g
                    w_esgoto = total_e / total_g
                    print(
                        f"🔄 Pesos recalculados com base na interseção ({min(anos_comuns)}–{max(anos_comuns)})."
                    )
                else:
                    print("⚠️ Soma de internações na interseção igual a zero. Mantidos pesos fixos.")
            else:
                print("⚠️ Sem interseção entre arquivos. Mantidos pesos fixos.")

        print(f"\n⚖️ Pesos utilizados (versão {versao_pesos}):")
        print(f"W_AGUA: {w_agua:.3f}")
        print(f"W_ESGOTO: {w_esgoto:.3f}")

        df_final = pd.merge(
            df_base, df_s_agua, on=['id_municipio_6', 'ano'], how='left', validate='many_to_one'
        )
        df_final = pd.merge(
            df_final, df_s_esgoto, on=['id_municipio_6', 'ano'], how='left', validate='many_to_one'
        )

        assert df_final.shape[0] == linhas_antes, (
            f"⚠️ Erro no Merge! Linhas antes: {linhas_antes}, linhas atuais: {df_final.shape[0]}"
        )

        df_final['tem_dado_saude_agua'] = df_final['internacoes_agua'].notna()
        df_final['tem_dado_saude_esgoto'] = df_final['internacoes_esgoto'].notna()
        df_final['tem_dado_saude'] = df_final['tem_dado_saude_agua'] | df_final['tem_dado_saude_esgoto']

        df_final[['internacoes_agua', 'internacoes_esgoto']] = (
            df_final[['internacoes_agua', 'internacoes_esgoto']]
            .fillna(0)
            .round()
            .astype(int)
        )

        df_final['Taxa_Morbidade_100k_Hab'] = pd.to_numeric(
            np.where(
                (df_final['populacao_ref'] > 0) & df_final['tem_dado_saude'],
                ((df_final['internacoes_agua'] + df_final['internacoes_esgoto']) / df_final['populacao_ref'])
                * 100000,
                np.nan
            ),
            errors='coerce'
        )

        print("\n✅ Integração concluída com pesos robustos!")
        return df_final, w_agua, w_esgoto

    except Exception as e:
        if strict:
            raise

        print(f"❌ Erro crítico na integração de saúde: {e}")
        print("⚠️ Pipeline continuará com df_final = df_gold e pesos default.")

        df_final = df_gold.copy()
        if 'internacoes_agua' not in df_final.columns:
            df_final['internacoes_agua'] = 0
        if 'internacoes_esgoto' not in df_final.columns:
            df_final['internacoes_esgoto'] = 0
        df_final['tem_dado_saude_agua'] = False
        df_final['tem_dado_saude_esgoto'] = False
        df_final['tem_dado_saude'] = False
        if 'Taxa_Morbidade_100k_Hab' not in df_final.columns:
            df_final['Taxa_Morbidade_100k_Hab'] = np.nan

        return df_final, float(w_agua_calculado), float(w_esgoto_calculado)


def calcular_risco_social_final(
    df_final: pd.DataFrame,
    w_agua: float,
    w_esgoto: float,
    peso_saude_com_dado: float = 0.4,
    peso_saude_sem_dado: float = 0.0
) -> pd.DataFrame:
    """
    Calcula índice de risco social final combinando infraestrutura e saúde.

    Retorna df_final com colunas de risco e eficiência financeira adicionadas.
    """
    df_final = df_final.copy()

    # Conversão numérica
    cols_infra = [
        'indice_atendimento_total_agua',
        'indice_atendimento_esgoto_agua',
        'indice_tratamento_esgoto'
    ]
    for col in cols_infra:
        df_final[col] = pd.to_numeric(df_final[col], errors='coerce')

    df_final['Taxa_Morbidade_100k_Hab'] = pd.to_numeric(
        df_final['Taxa_Morbidade_100k_Hab'], errors='coerce'
    )
    df_final['arrecadacao_total'] = pd.to_numeric(df_final['arrecadacao_total'], errors='coerce')
    df_final['receita_operacional'] = pd.to_numeric(df_final['receita_operacional'], errors='coerce')

    # Déficit de água (não tratar NaN como cobertura zero)
    df_final['indice_atendimento_total_agua'] = df_final['indice_atendimento_total_agua'].clip(0, 100)
    df_final['def_agua'] = np.where(
        df_final['indice_atendimento_total_agua'].notna(),
        (100 - df_final['indice_atendimento_total_agua']).clip(0, 100),
        np.nan
    )

    # Déficit de esgoto (não tratar NaN como ausência total)
    df_final['indice_atendimento_esgoto_agua'] = df_final['indice_atendimento_esgoto_agua'].clip(0, 100)
    df_final['indice_tratamento_esgoto'] = df_final['indice_tratamento_esgoto'].clip(0, 100)

    df_final['eficiencia_esgoto_calc'] = np.where(
        df_final['indice_atendimento_esgoto_agua'].notna()
        & df_final['indice_tratamento_esgoto'].notna(),
        df_final['indice_atendimento_esgoto_agua'] * (df_final['indice_tratamento_esgoto'] / 100),
        np.nan
    )

    df_final['def_esgoto'] = np.where(
        df_final['eficiencia_esgoto_calc'].notna(),
        (100 - df_final['eficiencia_esgoto_calc']).clip(0, 100),
        np.nan
    )

    # Vazio sanitário com reponderação pelos componentes disponíveis
    peso_agua_disp = np.where(df_final['def_agua'].notna(), float(w_agua), 0.0)
    peso_esgoto_disp = np.where(df_final['def_esgoto'].notna(), float(w_esgoto), 0.0)
    peso_total_disp = peso_agua_disp + peso_esgoto_disp

    df_final['vazio_sanitario'] = np.where(
        peso_total_disp > 0,
        ((df_final['def_agua'].fillna(0) * peso_agua_disp)
         + (df_final['def_esgoto'].fillna(0) * peso_esgoto_disp)) / peso_total_disp,
        np.nan
    )
    df_final['flag_insumos_risco_incompletos'] = df_final[['def_agua', 'def_esgoto']].isna().any(axis=1)

    # Índice combinado em escala única (0-100)
    df_final['sem_dados_saude'] = df_final['Taxa_Morbidade_100k_Hab'].isna()

    vs = df_final['vazio_sanitario'].fillna(0)
    taxa = df_final['Taxa_Morbidade_100k_Hab']
    max_taxa = taxa.max()

    if pd.notna(max_taxa) and max_taxa > 0:
        taxa_norm = np.where(taxa.notna(), (taxa / max_taxa) * 100, 0.0)
    else:
        taxa_norm = np.zeros(len(df_final), dtype=float)

    peso_saude = np.where(df_final['sem_dados_saude'], peso_saude_sem_dado, peso_saude_com_dado)
    peso_sanitario = 1.0 - peso_saude

    df_final['indice_combinado'] = (vs * peso_sanitario + taxa_norm * peso_saude).round(2)
    df_final['RISCO_SOCIAL_FINAL'] = df_final['indice_combinado'].round(2)

    # Eficiência financeira com rastreabilidade de truncamento
    df_final['eficiencia_arrecadacao_bruta'] = np.where(
        df_final['receita_operacional'] > 0,
        (df_final['arrecadacao_total'] / df_final['receita_operacional']) * 100,
        np.nan
    )
    df_final['flag_eficiencia_arrecadacao_truncada'] = df_final['eficiencia_arrecadacao_bruta'] > 150
    df_final['eficiencia_arrecadacao'] = df_final['eficiencia_arrecadacao_bruta'].clip(upper=150).round(2)

    return df_final


# ---------------------------------------------------------------------------
# 5. Validação de qualidade da população
# ---------------------------------------------------------------------------

def validar_populacao(df: pd.DataFrame):
    """
    Valida qualidade da coluna populacao_ref e retorna relatório + df anotado.
    """
    df = df.sort_values(['id_municipio', 'ano']).copy()
    relatorio = {}

    # 1. Valores inválidos
    invalidos = df[df['populacao_ref'].isna() | (df['populacao_ref'] <= 0)]
    relatorio['valores_invalidos'] = invalidos

    # 2. Crescimento ano a ano (%)
    df['populacao_anterior'] = df.groupby('id_municipio')['populacao_ref'].shift(1)
    df['crescimento_pct'] = (
        (df['populacao_ref'] - df['populacao_anterior']) / df['populacao_anterior']
    ) * 100
    relatorio['crescimento_absurdo'] = df[df['crescimento_pct'].abs() > 20]

    # 3. Outliers (z-score por município via transform vetorizado)
    df['outlier'] = df.groupby('id_municipio', group_keys=False)['populacao_ref'].transform(
        lambda s: ((s - s.mean()) / s.std()).abs().gt(3)
        if pd.notna(s.std()) and s.std() != 0
        else pd.Series(False, index=s.index)
    )
    relatorio['outliers'] = df[df['outlier']]

    # 4. Saltos absolutos grandes
    df['delta_abs'] = (df['populacao_ref'] - df['populacao_anterior']).abs()
    relatorio['saltos_grandes'] = df[df['delta_abs'] > df['populacao_ref'] * 0.15]

    # 5. Lacunas originalmente imputadas
    if 'populacao_ref_era_nula' in df.columns:
        relatorio['valores_imputados'] = df[df['populacao_ref_era_nula']]
    else:
        relatorio['valores_imputados'] = df.iloc[0:0].copy()

    print("\n📊 RELATÓRIO DE VALIDAÇÃO DA POPULAÇÃO\n")
    print(f"❌ Valores inválidos: {len(relatorio['valores_invalidos'])}")
    print(f"📈 Crescimentos suspeitos (>20%): {len(relatorio['crescimento_absurdo'])}")
    print(f"📊 Outliers estatísticos: {len(relatorio['outliers'])}")
    print(f"⚠️ Saltos absolutos grandes: {len(relatorio['saltos_grandes'])}")
    print(f"🩹 Valores imputados/interpolados: {len(relatorio['valores_imputados'])}")

    # Apelidos e coluna adicional para células de classificação posteriores usarem sem recalcular
    df['pop_ant'] = df['populacao_anterior']
    df['pop_prox'] = df.groupby('id_municipio')['populacao_ref'].shift(-1)

    return relatorio, df


def classificar_qualidade_populacao(
    df_validado: pd.DataFrame,
    df_final: pd.DataFrame | None = None
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Classifica qualidade da população e, se fornecido, propaga flags ao df_final.

    Retorna (df_validado_classificado, df_final_atualizado).
    """
    df_class = df_validado.sort_values(['id_municipio', 'ano']).copy()

    if 'crescimento_pct' not in df_class.columns:
        df_class['populacao_anterior'] = df_class.groupby('id_municipio')['populacao_ref'].shift(1)
        df_class['crescimento_pct'] = (
            (df_class['populacao_ref'] - df_class['populacao_anterior'])
            / df_class['populacao_anterior']
        ) * 100

    if 'pop_prox' not in df_class.columns:
        df_class['pop_prox'] = df_class.groupby('id_municipio')['populacao_ref'].shift(-1)

    df_class['erro_pontual'] = (
        (df_class['crescimento_pct'].abs() > 30)
        & (
            (df_class['pop_prox'] - df_class['populacao_ref']).abs()
            > (df_class['populacao_ref'] * 0.2)
        )
    )

    df_class['mudanca_base'] = (
        (df_class['crescimento_pct'] > 30)
        & (df_class['pop_prox'] > df_class['populacao_ref'] * 0.9)
    )

    if 'populacao_ref_era_nula' in df_class.columns:
        df_class['buraco_antes'] = df_class.groupby('id_municipio')['populacao_ref_era_nula'].transform(
            lambda x: x.astype(int).rolling(2, min_periods=1).sum()
        )
    else:
        df_class['buraco_antes'] = 0

    df_class['erro_interpolacao'] = (
        (df_class['crescimento_pct'].abs() > 20)
        & (df_class['buraco_antes'] > 0)
    )

    df_class['municipio_pequeno'] = df_class['populacao_ref'] < 20000

    df_class['variacao_pequeno'] = (
        df_class['municipio_pequeno']
        & (df_class['crescimento_pct'].abs() > 20)
    )

    for col in ['erro_pontual', 'mudanca_base', 'erro_interpolacao', 'variacao_pequeno']:
        df_class[col] = df_class[col].fillna(False)

    condicoes_erro = [
        df_class['erro_pontual'],
        df_class['mudanca_base'],
        df_class['erro_interpolacao'],
        df_class['variacao_pequeno'],
    ]
    escolhas_erro = ['erro_pontual', 'mudanca_base', 'interpolacao_ruim', 'municipio_pequeno']
    df_class['tipo_erro'] = np.select(condicoes_erro, escolhas_erro, default='ok')

    df_final_out = df_final
    if df_final is not None:
        cols_flags_pop = [
            'id_municipio',
            'ano',
            'erro_pontual',
            'mudanca_base',
            'erro_interpolacao',
            'variacao_pequeno',
            'tipo_erro'
        ]
        cols_existentes = [c for c in cols_flags_pop if c in df_class.columns]

        df_final_out = df_final.copy()
        if len(cols_existentes) >= 3:
            flags_para_merge = df_class[cols_existentes].copy()
            cols_regrava = [c for c in cols_existentes if c not in ['id_municipio', 'ano']]

            df_final_out = df_final_out.drop(columns=cols_regrava, errors='ignore')
            df_final_out = df_final_out.merge(flags_para_merge, on=['id_municipio', 'ano'], how='left')

            if 'tipo_erro' in df_final_out.columns:
                df_final_out['tipo_erro'] = df_final_out['tipo_erro'].fillna('ok')

            for col_bool in ['erro_pontual', 'mudanca_base', 'erro_interpolacao', 'variacao_pequeno']:
                if col_bool in df_final_out.columns:
                    df_final_out[col_bool] = df_final_out[col_bool].fillna(False).astype(bool)

    return df_class, df_final_out


# ---------------------------------------------------------------------------
# 6. Diagnóstico por município
# ---------------------------------------------------------------------------

def analisar_municipio(df: pd.DataFrame, municipio_id) -> None:
    """
    Exibe série temporal de populacao_ref com crescimento % para um município.
    """
    df_m = df[df['id_municipio'].astype(str) == str(municipio_id)].sort_values('ano')

    print(df_m[['ano', 'populacao_ref', 'crescimento_pct']])

    plt.plot(df_m['ano'], df_m['populacao_ref'], marker='o')
    plt.title(f"Município {municipio_id}")
    plt.grid()
    plt.show()
