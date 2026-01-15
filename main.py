"""
Pipeline de Predição de Turnover - Ponto de Entrada Principal

Este script é o ponto de entrada principal para o pipeline de predição de turnover.
Ele orquestra todo o fluxo de trabalho, desde o carregamento de dados até a avaliação do modelo,
incluindo:
    - Carregamento de dados e pré-processamento
    - Engenharia de atributos (feature engineering) para séries temporais
    - Treinamento de modelos de machine learning com validação walk-forward
    - Treinamento de modelos de séries temporais tradicionais (ARIMA, ETS, Naive)
    - Comparação de modelos e análise de importância de atributos
    - Geração de resultados e relatórios

Visão Geral do Pipeline:
    1. Carregar e explorar dados
    2. Analisar padrões temporais (teste de estacionariedade)
    3. Descrever engenharia de atributos
    4. Preparar configuração de validação walk-forward
    5. Treinar modelos de ML (XGBoost, RF, GB, Ridge)
    6. Treinar modelos de séries temporais tradicionais
    7. Comparar desempenho dos modelos
    8. Analisar importância dos atributos
    9. Gerar saídas e relatórios

Uso:
    python main.py

Saídas:
    - reports/predictions_v3_validation.csv: Resultados de validação para todos os modelos
    - reports/feature_importance_v3.csv: Ranking de importância dos atributos
    - pipeline.log: Log detalhado de execução
"""

import sys
import yaml
import pandas as pd
import numpy as np
import warnings
from datetime import timedelta

# Importações internas
from src.data_loader import load_data
from src.model_trainer import TurnoverTrainer

# Importações para análise estatística e séries temporais
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb

# Suprimir avisos para uma saída mais limpa
warnings.filterwarnings('ignore')

# ==============================================================================
# FUNÇÕES UTILITÁRIAS
# ==============================================================================

def print_header(text):
    """
    Imprime um cabeçalho formatado com o texto centralizado entre sinais de igual.
    
    Args:
        text (str): O texto do cabeçalho a ser exibido
    
    Exemplo:
        >>> print_header("TÍTULO DA SEÇÃO")
        ================================================================================
        TÍTULO DA SEÇÃO
        ================================================================================
    """
    print("=" * 80)
    print(text)
    print("=" * 80)
    print()


def load_config(path="config/params.yaml"):
    """
    Carrega a configuração do pipeline a partir de um arquivo YAML.
    
    Esta função lê o arquivo de configuração e retorna seu conteúdo
    como um dicionário Python. Se o arquivo não for encontrado, imprime uma
    mensagem de erro e encerra o programa.
    
    Args:
        path (str, opcional): Caminho para o arquivo de configuração.
                              Padrão é "config/params.yaml".
    
    Returns:
        dict: Parâmetros de configuração carregados do arquivo YAML.
    
    Exits:
        Se o arquivo de configuração não for encontrado.
    """
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de configuração '{path}' não encontrado.")
        sys.exit(1)


def import_logging():
    """
    Função auxiliar para importar o módulo de logging.
    
    Isso é usado para definir o nível de log para a instância do trainer
    e suprimir saídas verbosas durante a execução principal do pipeline.
    
    Returns:
        logging: O módulo de logging.
    """
    import logging
    return logging


# ==============================================================================
# PIPELINE PRINCIPAL
# ==============================================================================

def main():
    """
    Função principal de execução para o pipeline de predição de turnover.
    
    Esta função orquestra o fluxo de trabalho completo:
        1. Carregar configuração e dados
        2. Realizar análise temporal (teste ADF)
        3. Descrever engenharia de atributos
        4. Treinar modelos de ML com validação walk-forward
        5. Treinar modelos de séries temporais tradicionais
        6. Comparar desempenho dos modelos
        7. Analisar importância dos atributos
        8. Gerar relatórios e saídas
    
    O pipeline utiliza validação cruzada aninhada para evitar vazamento de dados (leakage)
    e fornece estimativas de desempenho robustas em múltiplas janelas de tempo.
    """
    # Carregar configuração
    config = load_config()
    
    print_header("TURNOVER COM SÉRIES TEMPORAIS E ML (V3 ROBUSTO)")

    # --------------------------------------------------------------------------
    # PASSO 1: Carregamento e Exploração de Dados
    # --------------------------------------------------------------------------
    print("[1/9] Carregando e explorando dados...")
    
    # Carregar dados usando o carregador oficial
    # Isso lida com leitura de CSV, renomeação de colunas e pré-processamento inicial
    df = load_data(config)
    
    # Contar áreas únicas (unidades de negócio) no conjunto de dados
    n_areas = df[config['data']['area_col']].nunique()
    print(f"  Formato: {df.shape}")
    print(f"  Áreas: {n_areas}")
    print()

    # --------------------------------------------------------------------------
    # PASSO 2: Análise Temporal Agregada
    # --------------------------------------------------------------------------
    print("[2/9] Analisando série temporal agregada...")
    
    target_col = config['data']['target_col']
    date_col = config['data']['date_col']
    
    # Criar série temporal agregada (média de turnover de todas as áreas por mês)
    # Isso fornece uma visão de alto nível das tendências gerais de turnover
    ts_agg = df.groupby(date_col)[target_col].mean()
    
    print(f"  Observações temporais: {len(ts_agg)}")
    print(f"  Período: {ts_agg.index.min().date()} até {ts_agg.index.max().date()}")
    
    # Realizar teste Augmented Dickey-Fuller para estacionariedade
    # A estacionariedade é importante para modelos de séries temporais tradicionais
    try:
        adf_result = adfuller(ts_agg.dropna())
        status = "(Estacionária)" if adf_result[1] < 0.05 else "(Não-estacionária)"
        print(f"  p-valor ADF: {adf_result[1]:.4f} {status}")
    except:
        print("  p-valor ADF: N/A")
    print()

    # --------------------------------------------------------------------------
    # PASSO 3: Descrição da Engenharia de Atributos
    # --------------------------------------------------------------------------
    print("[3/9] Engenharia de Atributos (Feature Engineering)...")
    
    # Imprimir descrição de todos os atributos gerados
    print("   Mês (1-12): Captura sazonalidade intra-anual")
    print("   Trimestre (1-4): Captura ciclos trimestrais")
    print("   Dummies de mês (12 atributos): Permite coeficientes diferentes por mês")
    
    # Atributos de Lag - capturam padrões históricos
    for lag in config['features']['lags']:
        print(f"   headcount_lag_{lag}: Tamanho da equipe {lag} meses atrás")
    for lag in config['features']['lags']:
        print(f"   admissions_lag_{lag}: Admissões {lag} meses atrás")
    for lag in config['features']['lags']:
        print(f"   target_lag_{lag}: Turnover {lag} meses atrás (memória temporal)")
    
    # Atributos de Momento - capturam taxa de mudança
    if config['features'].get('use_momentum'):
        print("   crescimento headcount/admissões: Taxa de mudança (momentum)")
    
    # Identificar colunas de atributos (excluir alvo e metadados)
    ignore_cols = [target_col, date_col, config['data']['area_col'], 'area_encoded']
    features = [c for c in df.columns if c not in ignore_cols]
    
    print()
    print(f"  Total de Atributos: {len(features)}")
    print(f"  Formato Final: {df.shape}")
    print()

    # --------------------------------------------------------------------------
    # PASSO 4: Preparação do Treinamento (Configuração Walk-Forward)
    # --------------------------------------------------------------------------
    print("[4/9] Preparando dados para treinamento (Configuração Walk-Forward)...")
    
    # Exibir configuração da validação walk-forward
    n_splits = config['training']['n_splits']
    print(f"  Configuração: {n_splits} dobras temporais (janelas deslizantes)")
    print(f"  Validando robustez ao longo do tempo...")
    print()

    # --------------------------------------------------------------------------
    # PASSO 5: Treinamento de Modelos de ML (Orquestrado pelo TurnoverTrainer)
    # --------------------------------------------------------------------------
    print("[5/9] Treinando modelos de ML...")
    
    # Exibir quais modelos serão treinados
    for model_name in config['models']:
        if config['models'][model_name].get('enabled'):
            print(f"  - {model_name}...")
    
    # Inicializar trainer e executar treinamento
    # O trainer lida com a validação cruzada walk-forward internamente
    trainer = TurnoverTrainer(df, config)
    
    # Suprimir logs internos do trainer para uma saída mais limpa
    # Define o nível de log para WARNING para ocultar mensagens INFO
    trainer.logger.setLevel(import_logging().WARNING)
    
    # Executar treinamento e validação
    results_df = trainer.train_and_validate()
    
    n_models = len(results_df['model'].unique())
    print(f"   {n_models} modelos de ML treinados e validados via Walk-Forward")
    print()

    # --------------------------------------------------------------------------
    # PASSO 6: Modelos de Séries Temporais Tradicionais (Escala Agregada)
    # --------------------------------------------------------------------------
    print("[6/9] Treinando modelos de séries temporais (Escala Agregada)...")
    
    # Preparar série temporal agregada para teste (últimos 20% dos dados)
    split_date = ts_agg.index[int(len(ts_agg) * 0.8)]
    train_ts = ts_agg[ts_agg.index < split_date]
    test_ts = ts_agg[ts_agg.index >= split_date]
    
    ts_metrics = []

    # Modelo ARIMA
    print("  - ARIMA (1,0,1)...")
    try:
        arima = ARIMA(train_ts, order=(1, 0, 1)).fit()
        pred_arima = arima.forecast(steps=len(test_ts))
        r2_arima = r2_score(test_ts, pred_arima)
        mae_arima = mean_absolute_error(test_ts, pred_arima)
        ts_metrics.append({'model': 'ARIMA', 'r2': r2_arima, 'mae': mae_arima})
        print("     ARIMA treinado")
    except:
        print("     ARIMA falhou")

    # Modelo de Suavização Exponencial (ETS)
    print("  - Suavização Exponencial (ETS)...")
    try:
        ets = ExponentialSmoothing(train_ts, trend='add').fit()
        pred_ets = ets.forecast(steps=len(test_ts))
        r2_ets = r2_score(test_ts, pred_ets)
        mae_ets = mean_absolute_error(test_ts, pred_ets)
        ts_metrics.append({'model': 'ExponentSmoothing', 'r2': r2_ets, 'mae': mae_ets})
        print("     Suavização Exponencial treinada")
    except:
        print("     ES falhou")

    # Previsão Naive (Linha de Base)
    print("  - Previsão Naive (Baseline)...")
    # Previsão Naive: prevê o último valor observado para todos os períodos futuros
    pred_naive = np.full(len(test_ts), train_ts.iloc[-1])
    r2_naive = r2_score(test_ts, pred_naive)
    mae_naive = mean_absolute_error(test_ts, pred_naive)
    ts_metrics.append({'model': 'Naive', 'r2': r2_naive, 'mae': mae_naive})
    print("     Naive (baseline) criada")
    print()

    # --------------------------------------------------------------------------
    # PASSO 7: Comparação de Modelos
    # --------------------------------------------------------------------------
    print("[7/9] Comparando modelos...")
    
    # Calcular métricas médias entre todas as dobras para os modelos de ML
    ml_summary = results_df.groupby('model')[['r2', 'mae', 'rmse']].mean().reset_index()
    ml_summary = ml_summary.sort_values('r2', ascending=False)
    
    # Exibir resultados dos modelos de ML
    print("  Modelos ML (Média da Validação Cruzada Temporal):")
    for _, row in ml_summary.iterrows():
        print(f"    {row['model']:20s} | R²: {row['r2']:6.4f} | MAE: {row['mae']:6.2f}%")
        
    # Exibir resultados dos modelos de séries temporais
    print("  Modelos Séries Temporais (Escala agregada - Teste único):")
    for m in ts_metrics:
        print(f"    {m['model']:20s} | R²: {m['r2']:6.4f} | MAE: {m['mae']:6.2f}%")

    # Identificar o melhor modelo de ML
    best_model_name = ml_summary.iloc[0]['model']
    best_model_r2 = ml_summary.iloc[0]['r2']
    best_model_mae = ml_summary.iloc[0]['mae']
    best_model_rmse = ml_summary.iloc[0]['rmse']
    
    print()
    print(f"  Melhor modelo: {best_model_name} (R² Médio = {best_model_r2:.4f})")
    print()

    # --------------------------------------------------------------------------
    # PASSO 8: Análise de Importância de Atributos (Retreinamento Final)
    # --------------------------------------------------------------------------
    print("[8/9] Analisando Engenharia de Atributos...")
    print("  (Retreinando o melhor modelo com todo o histórico recente para extrair importância...)")
    
    # Retreinar o melhor modelo em todo o conjunto de dados para extrair importância dos atributos
    X_all = df[features]
    y_all = df[target_col]
    
    # Obter instância do modelo baseada no nome do vencedor
    model_factory = trainer.get_model_factory(best_model_name)
    
    # Extrair importância dos atributos (o método depende do tipo de modelo)
    if best_model_name == 'ridge':
        # Ridge usa coeficientes (valores absolutos para importância)
        model_factory.fit(X_all, y_all)
        importances = np.abs(model_factory.coef_)
    else:
        # Modelos baseados em árvore usam o atributo feature_importances_
        model_factory.fit(X_all, y_all)
        importances = model_factory.feature_importances_

    # Criar e ordenar dataframe de importância de atributos
    feat_df = pd.DataFrame({'feature': features, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=False)
    
    print()
    print("  Top 10 Atributos Gerais:")
    for i, row in feat_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['feature']:30s} | {row['importance']:.4f}")
    
    # Salvar importância de atributos em CSV
    feat_df.to_csv("reports/feature_importance_v3.csv", index=False)
    print()

    # --------------------------------------------------------------------------
    # PASSO 9: Geração de Saídas
    # --------------------------------------------------------------------------
    print("[9/9] Gerando saídas e documentação...")
    
    # Salvar resultados da validação
    results_df.to_csv("reports/predictions_v3_validation.csv", index=False)
    print("   Tabela de validação salva: reports/predictions_v3_validation.csv")
    print("   Importância de atributos salva: reports/feature_importance_v3.csv")
    print()

    # --------------------------------------------------------------------------
    # RESUMO EXECUTIVO
    # --------------------------------------------------------------------------
    print_header("RESUMO EXECUTIVO V3 (ROBUSTO)")
    
    print(" CONJUNTO DE DADOS:")
    print(f"   Total: {len(df):,} observações")
    print(f"   Áreas: {n_areas}")
    print(f"   Período: {df[date_col].min().date()} até {df[date_col].max().date()}")
    print()
    print(f" MELHOR MODELO: {best_model_name}")
    print(f"   R² Score (Média): {best_model_r2:.4f}")
    print(f"   MAE (Média):      {best_model_mae:.2f}%")
    print(f"   RMSE (Média):     {best_model_rmse:.2f}%")
    print()
    print(" ENGENHARIA DE ATRIBUTOS:")
    print(f"   Total de Atributos: {len(features)}")
    print("   - Operacionais: Headcount, Admissões (atual e lags)")
    print("   - Sazonalidade: Mês, Trimestre")
    print("   - Memória: Lags do Alvo (Target)")
    print("   - Momento: Taxas de Crescimento")
    print()
    print(" MODELOS COMPARADOS:")
    print(f"   ML: {len(ml_summary)}")
    print(f"   Séries Temporais (TS): {len(ts_metrics)}")
    print()
    print_header(" Pipeline V3 concluído com sucesso!")


if __name__ == "__main__":
    main()