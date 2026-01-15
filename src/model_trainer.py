"""
Módulo de Treinamento e Validação de Modelos

Este módulo implementa a classe TurnoverTrainer, que fornece uma 
validação cruzada de séries temporais robusta para modelos de machine learning.
Utiliza validação cruzada aninhada (nested cross-validation) com walk-forward 
para evitar vazamento de dados e garantir estimativas de desempenho realistas.

Recursos Principais:
    - Validação cruzada de séries temporais walk-forward
    - Ajuste de hiperparâmetros aninhado (loop interno)
    - Ordenação temporal estrita para evitar vazamento (leakage)
    - Suporte para múltiplos algoritmos (XGBoost, RF, GB, Ridge)
    - Log detalhado e rastreamento de métricas

Classes:
    TurnoverTrainer: Classe principal para treinamento e validação de modelos
"""

import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
import xgboost as xgb


class TurnoverTrainer:
    """
    Classe de treinamento para modelos de predição de turnover com validação temporal.
    
    Esta classe implementa um pipeline de treinamento robusto que:
        1. Garante a ordenação temporal estrita dos dados
        2. Realiza validação cruzada walk-forward
        3. Otimiza hiperparâmetros dentro de cada janela de treinamento
        4. Evita vazamento de dados através de pré-processamento cuidadoso
        5. Rastreia métricas abrangentes em todas as dobras (folds)
    
    Atributos:
        df (pd.DataFrame): Dataframe de entrada com atributos e alvo
        config (dict): Dicionário de configuração para os parâmetros do pipeline
        logger (logging.Logger): Instância de logger para rastrear o progresso
        target_col (str): Nome da coluna da variável alvo
        date_col (str): Nome da coluna de data
        area_col (str): Nome da coluna de área/agrupamento
        features (list): Lista de nomes das colunas de atributos para treinamento
    """
    
    def __init__(self, df, config):
        """
        Inicializa o trainer com dados e configuração.
        
        Este construtor realiza a seguinte configuração:
            - Armazena os dados de entrada e a configuração
            - Extrai os nomes das colunas do config
            - Garante a ordenação temporal estrita dos dados
            - Identifica as colunas de atributos (exclui alvo e metadados)
        
        Args:
            df (pd.DataFrame): Dataframe de entrada com atributos e alvo
            config (dict): Dicionário de configuração contendo:
                - data.target_col: Nome da coluna alvo
                - data.date_col: Nome da coluna de data
                - data.area_col: Nome da coluna de área/agrupamento
                - Todos os outros parâmetros de configuração para treinamento
        
        Nota:
            - Os dados são ordenados por data para garantir consistência temporal
            - Os atributos são identificados automaticamente (todas as colunas exceto alvo, data, área)
        """
        self.df = df
        self.config = config
        self.logger = logging.getLogger("TurnoverTrainer")
        
        # Extrair nomes das colunas da configuração
        self.target_col = config['data']['target_col']
        self.date_col = config['data']['date_col']
        self.area_col = config['data']['area_col']
        
        # Garantir ordenação temporal para que o TimeSeriesSplit funcione corretamente
        # Isso é crítico para evitar vazamento de dados (look-ahead bias) em séries temporais
        self.df = self.df.sort_values(by=self.date_col).reset_index(drop=True)
        
        # Definir colunas de atributos: todas as colunas exceto alvo e metadados
        ignore_cols = [self.target_col, self.date_col, self.area_col, 'area_encoded']
        self.features = [c for c in df.columns if c not in ignore_cols]
        
        self.logger.info(f"Trainer inicializado. Atributos selecionados: {len(self.features)}")

    def get_model_factory(self, model_name):
        """
        Método de fábrica que retorna uma instância básica do modelo.
        
        Este método cria e retorna uma instância de modelo inicializada
        com base no nome do algoritmo solicitado. Todos os modelos são configurados
        com padrões apropriados para reprodutibilidade e desempenho.
        
        Modelos Suportados:
            - xgboost: Regressor XGBoost com objetivo de erro quadrático
            - random_forest: Regressor Random Forest
            - gradient_boosting: Regressor Gradient Boosting
            - ridge: Regressão Ridge (modelo linear com regularização L2)
        
        Args:
            model_name (str): Nome do modelo a ser instanciado
        
        Returns:
            objeto de modelo compatível com sklearn: Instância do modelo inicializada
        
        Raises:
            ValueError: Se o model_name não for suportado
        
        Nota:
            - Modelos baseados em árvore usam n_jobs=-1 para processamento paralelo
            - Todos os modelos usam random_state=42 para reprodutibilidade
        """
        if model_name == 'xgboost':
            # XGBoost para regressão
            # n_jobs=-1 utiliza todos os núcleos de CPU disponíveis
            return xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
        
        elif model_name == 'random_forest':
            # Random Forest com processamento paralelo
            return RandomForestRegressor(random_state=42, n_jobs=-1)
        
        elif model_name == 'gradient_boosting':
            # Gradient Boosting (sequencial, sem processamento paralelo)
            return GradientBoostingRegressor(random_state=42)
        
        elif model_name == 'ridge':
            # Regressão Ridge (linear com L2)
            return Ridge()
        
        else:
            raise ValueError(f"Modelo '{model_name}' não implementado na fábrica.")

    def train_and_validate(self):
        """
        Executa a validação cruzada aninhada para avaliação robusta do modelo.
        
        Este método implementa uma estratégia de validação cruzada aninhada (nested):
            1. Loop Externo (Walk-Forward): Simula o progresso do tempo em janelas
            2. Loop Interno (Tuning): Otimiza hiperparâmetros dentro da janela de treino
            3. Prevenção de Vazamento: O scaler é ajustado APENAS nos dados de treino
        
        Estratégia de Validação:
            - Usa TimeSeriesSplit para validação walk-forward
            - Cada dobra treina em dados passados e testa em dados futuros
            - O ajuste de hiperparâmetros respeita a ordem temporal
            - O pré-processamento (scaling) é ajustado apenas nos dados de treinamento
        
        Returns:
            pd.DataFrame: Dataframe de resultados com as colunas:
                - fold: Número da dobra (1 a n_splits)
                - model: Nome do modelo
                - train_end_date: Última data no conjunto de treinamento
                - test_period_start: Primeira data no conjunto de teste
                - test_period_end: Última data no conjunto de teste
                - mae: Erro Médio Absoluto
                - rmse: Raiz do Erro Quadrático Médio
                - r2: Coeficiente de Determinação (R-squared)
                - best_params: Melhores hiperparâmetros encontrados
        
        Nota:
            - Resultados de todas as dobras e modelos são combinados
            - Cada dobra representa uma divisão treino/teste realista
            - As métricas são computadas em dados futuros não vistos pelo modelo
        """
        # Preparar atributos e alvo
        X = self.df[self.features]
        y = self.df[self.target_col]
        dates = self.df[self.date_col]
        
        # Configurar a divisão de série temporal
        n_splits = self.config['training']['n_splits']
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        results = []
        
        self.logger.info(f"Iniciando Validação Walk-Forward com {n_splits} dobras...")

        # --- LOOP EXTERNO: Simular o progresso do tempo ---
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            
            # Passo 1: Separação física dos dados (treino vs teste futuro)
            X_train_raw, X_test_raw = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Extrair referências de data para o log
            train_end = dates.iloc[train_idx].max().date()
            test_start = dates.iloc[test_idx].min().date()
            test_end = dates.iloc[test_idx].max().date()
            
            self.logger.info(f"\n=== DOBRA (FOLD) {fold+1}/{n_splits} ===")
            self.logger.info(f"Treino até: {train_end} | Teste (Futuro): {test_start} a {test_end}")
            
            # Passo 2: Prevenção de vazamento através de escala correta
            # fit() acontece APENAS nos dados passados (treino). transform() em ambos.
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)
            
            # Iterar por todos os modelos habilitados na configuração
            for model_name, cfg in self.config['models'].items():
                if not cfg.get('enabled', False):
                    continue
                
                self.logger.info(f"  > Otimizando {model_name}...")
                
                base_model = self.get_model_factory(model_name)
                param_dist = cfg['params']
                n_iter = cfg['n_iter']
                
                # --- LOOP INTERNO: Ajuste de hiperparâmetros ---
                # Usa TimeSeriesSplit internamente (3 dobras) para garantir que o
                # ajuste também respeite a ordem temporal dentro dos dados de treino
                inner_cv = TimeSeriesSplit(n_splits=3)
                
                search = RandomizedSearchCV(
                    estimator=base_model,
                    param_distributions=param_dist,
                    n_iter=n_iter,
                    scoring=self.config['training']['scoring'],
                    cv=inner_cv,
                    n_jobs=-1,
                    verbose=0,
                    random_state=42
                )
                
                # Otimizar usando apenas os dados de treino desta janela
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                
                # Passo 3: Avaliação final na janela de teste (futuro desconhecido)
                preds = best_model.predict(X_test)
                
                # Calcular métricas de avaliação
                mae = mean_absolute_error(y_test, preds)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
                r2 = r2_score(y_test, preds)
                
                self.logger.info(f"    Melhores Params: {search.best_params_}")
                self.logger.info(f"    Resultado: MAE={mae:.4f} | R2={r2:.4f}")
                
                results.append({
                    'fold': fold + 1,
                    'model': model_name,
                    'train_end_date': train_end,
                    'test_period_start': test_start,
                    'test_period_end': test_end,
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'best_params': search.best_params_
                })
        
        return pd.DataFrame(results)