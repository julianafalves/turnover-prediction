"""
Módulo de Carregamento de Dados e Engenharia de Atributos (Feature Engineering)

Este módulo fornece funções para carregar dados de turnover e criar
atributos de séries temporais para modelos de machine learning. Ele implementa o
pipeline de engenharia de atributos da V2, incluindo:
    - Atributos sazonais (mês, trimestre)
    - Atributos de atraso/lag (headcount, admissões, alvo)
    - Atributos de momento (taxas de crescimento)

Funções:
    create_features: Gera atributos de séries temporais a partir de dados brutos
    load_data: Carrega e pré-processa o conjunto de dados de turnover
"""

import pandas as pd
import numpy as np


def create_features(df, config):
    """
    Cria atributos de séries temporais para modelos de predição de turnover.
    
    Esta função implementa o pipeline de engenharia de atributos da V2,
    gerando atributos temporais que capturam sazonalidade, padrões
    históricos e o momento nos dados.
    
    Atributos Criados:
        - Sazonais: mês (1-12), trimestre (1-4)
        - Lags: target_lag_N, headcount_lag_N, admissions_lag_N (onde N é o período de lag)
        - Momento: headcount_growth_1m, admissions_growth_1m (se habilitado)
    
    Args:
        df (pd.DataFrame): Dataframe bruto com dados de turnover
        config (dict): Dicionário de configuração contendo:
            - data.date_col: Nome da coluna de data
            - data.area_col: Nome da coluna de área/agrupamento
            - data.target_col: Nome da coluna da variável alvo
            - features.lags: Lista de períodos de lag a serem criados
            - features.use_momentum: Booleano para habilitar atributos de taxa de crescimento
    
    Returns:
        pd.DataFrame: Dataframe processado com atributos engenheirados.
                      Linhas com valores NaN (da criação de lags) são removidas.
    
    Nota:
        - Os dados são ordenados por área e data antes da criação dos atributos
        - Atributos de lag são computados dentro de cada grupo de área
        - Valores NaN da criação de lags são descartados (primeiras N linhas por área)
    """
    df = df.copy()
    
    # Extrair parâmetros de configuração
    date_col = config['data']['date_col']
    area_col = config['data']['area_col']
    target_col = config['data']['target_col']
    lags = config['features']['lags']

    # Passo 1: Ordenar dados por área e data (crítico para computação de lag)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=[area_col, date_col])

    # Passo 2: Criar atributos sazonais
    df['month'] = df[date_col].dt.month
    df['quarter'] = df[date_col].dt.quarter
    
    # Passo 3: Agrupar por área para operações de série temporal
    grouped = df.groupby(area_col)
    
    # Passo 4: Criar atributos de lag para cada período especificado
    for lag in lags:
        # Lag da variável alvo (taxas de turnover passadas)
        df[f'target_lag_{lag}'] = grouped[target_col].shift(lag)
        
        # Lag de atributos operacionais (se presentes)
        if 'headcount' in df.columns:
            df[f'headcount_lag_{lag}'] = grouped['headcount'].shift(lag)
        if 'admissions' in df.columns:
            df[f'admissions_lag_{lag}'] = grouped['admissions'].shift(lag)
            
    # Passo 5: Criar atributos de momento (taxas de crescimento do mês anterior)
    if config['features'].get('use_momentum', False):
        if 'headcount' in df.columns:
            df['headcount_growth_1m'] = grouped['headcount'].diff(1)
        if 'admissions' in df.columns:
            df['admissions_growth_1m'] = grouped['admissions'].diff(1)

    # Passo 6: Remover valores NaN gerados pelas operações de lag
    # Isso ocorre nas primeiras N linhas da série temporal de cada área
    df = df.dropna().reset_index(drop=True)
    
    return df


def load_data(config):
    """
    Carrega e pré-processa o conjunto de dados de turnover.
    
    Esta função realiza os seguintes passos de pré-processamento:
        1. Lê os dados brutos de um CSV
        2. Remove colunas de vazamento (leakage - atributos que revelam o alvo)
        3. Remove colunas de informações pessoais/identificáveis
        4. Renomeia colunas para nomes padronizados (se o mapeamento for fornecido)
        5. Remove linhas com valores alvo ausentes
        6. Gera atributos de séries temporais
    
    Args:
        config (dict): Dicionário de configuração contendo:
            - data.input_path: Caminho para o arquivo CSV de entrada
            - data.drop_cols: Lista de colunas a remover (leakage + pessoais)
            - data.rename_map: Dicionário mapeando nomes originais para novos nomes de colunas
            - data.target_col: Nome da coluna da variável alvo
            - Todos os outros parâmetros de config necessários para create_features()
    
    Returns:
        pd.DataFrame: Dataframe totalmente processado e com engenharia de atributos
                      pronto para o treinamento do modelo.
    
    Nota:
        - Colunas de vazamento são removidas para evitar contaminação de dados
        - Colunas de dados pessoais são removidas para privacidade e conformidade (LGPD)
        - A renomeação de colunas padroniza os nomes dos atributos em todo o pipeline
    """
    # Passo 1: Carregar dados brutos do CSV
    df = pd.read_csv(config['data']['input_path'])
    
    # Passo 2: Remover colunas de vazamento (leakage) e dados pessoais
    # Vazamento: colunas que contêm informações do mês atual
    #           que revelariam o valor alvo (target)
    # Pessoal: colunas contendo PII ou informações de identificação
    cols_to_drop = [c for c in config['data']['drop_cols'] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    # Passo 3: Renomear colunas para nomes padronizados
    # Isso melhora a legibilidade e a consistência em todo o pipeline
    if 'rename_map' in config['data']:
        df = df.rename(columns=config['data']['rename_map'])
    
    # Passo 4: Remover linhas com valores alvo ausentes
    # Estes não podem ser usados para treinamento ou avaliação
    df = df.dropna(subset=[config['data']['target_col']])
    
    # Passo 5: Gerar atributos de séries temporais
    df_processed = create_features(df, config)
    
    return df_processed