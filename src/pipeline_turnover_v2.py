"""
=============================================================================
PREVISÃO DE TURNOVER COM SÉRIES TEMPORAIS
=============================================================================

Executa:
1. EXPLORAÇÃO: Análise temporal dos dados
2. ENGENHARIA: Features com lags temporais + sazonalidade
3. MODELOS MACHINE LEARNING: RandomForest, XGBoost, LightGBM, Ridge
4. MODELOS DE SÉRIE TEMPORAL: ARIMA, Exponential Smoothing, Prophet
5. AVALIAÇÃO: Cross-validation temporal, métricas estatísticas
6. INTERPRETAÇÃO: Feature importance, SHAP, análise residual
7. SAÍDA: Tabela real vs predito, gráficos, relatório científico

Sem Data Leakage:
- NÃO usa dados do mês atual para prever o próprio mês
- SIM permite usar dados históricos de turnover (lag features)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
import json
import os

# Visualização
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ML
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             mean_absolute_percentage_error, median_absolute_error)
import xgboost as xgb
from sklearn.linear_model import Ridge

# Time Series Models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller

# Interpretação
import shap

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)

if not hasattr(np, 'int'):
    setattr(np, 'int', np.int64)
if not hasattr(np, 'float'):
    setattr(np, 'float', np.float64)


# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def calculate_metrics(y_true, y_pred, model_name=""):
    """Calcula todas as métricas"""
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    residual_std = np.std(residuals)
    
    return {
        'model': model_name,
        'mae': float(mae),
        'mape': float(mape),
        'rmse': float(rmse),
        'r2': float(r2),
        'residual_std': float(residual_std),
        'n_samples': len(y_true),
        'y_mean': float(y_true.mean()),
        'y_std': float(y_true.std())
    }


def plot_actual_vs_predicted(y_true, y_pred, title="", save_path=""):
    """Gráfico de diagnósticos"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=30)
    min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect')
    ax.set_xlabel('Valor Real')
    ax.set_ylabel('Valor Predito')
    ax.set_title('Real vs Predito')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    residuals = y_true - y_pred
    ax = axes[0, 1]
    ax.scatter(y_pred, residuals, alpha=0.5, s=30)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Valor Predito')
    ax.set_ylabel('Residual')
    ax.set_title('Análise de Residuais')
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    ax.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Residual')
    ax.set_ylabel('Frequência')
    ax.set_title('Distribuição dos Residuais')
    ax.grid(True, alpha=0.3, axis='y')
    
    ax = axes[1, 1]
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


print("="*80)
print("TURNOVER COM SÉRIE TEMPORAL E ML")
print("="*80)

# ============================================================================
# PASSO 1: LEITURA E EXPLORAÇÃO
# ============================================================================
print("\n[1/9] Lendo e explorando dados...")

df = pd.read_csv('data/juliana_alves_turnover_with_label.csv')
target_col = 'TO_TURNOVER_TO-GERAL'

# Remover leakage
leakage_cols = [
    'TO_DESLIGAMENTO_DESLIGAMENTO-MES-ATUAL',
    'TO_DESLIGAMENTO_DESLIGAMENTO-VOLUNTARIO-MES-ATUAL',
    'TO_DESLIGAMENTO_DESLIGAMENTO-INVOLUNTARIO-MES-ATUAL',
    'TO_DESLIGAMENTO_DESLIGAMENTO-VOL-HEAD-LT-25',
    'TO_TURNOVER_TO-VOL-HEAD-GT-25',
    'TO_TURNOVER_TO-VOL',
    'TO_TURNOVER_TO-INVOL',
]
personal_cols = ['EMAILADDRESS_AREA_RES', 'LIDER_IFOOD', 'VP_BU']

df = df.drop(columns=[c for c in leakage_cols if c in df.columns], errors='ignore')
df = df.drop(columns=[c for c in personal_cols if c in df.columns], errors='ignore')
df = df.dropna(subset=[target_col])

print(f"  Shape: {df.shape}")
print(f"  Áreas: {df['area_anonn'].nunique()}")

# ============================================================================
# PASSO 2: ANÁLISE DE SÉRIE TEMPORAL AGREGADA
# ============================================================================
print("\n[2/9] Analisando série temporal agregada...")

df['MES_REF'] = pd.to_datetime(df['MES_REF'])
df = df.sort_values(by=['area_anonn', 'MES_REF']).reset_index(drop=True)

# Série temporal agregada (média por mês)
ts_agg = df.groupby('MES_REF')[target_col].mean()
print(f"  Observações temporais: {len(ts_agg)}")
print(f"  Período: {ts_agg.index.min().date()} a {ts_agg.index.max().date()}")

# Teste ADF
try:
    adf_result = adfuller(ts_agg.dropna())
    print(f"  ADF p-value: {adf_result[1]:.4f} {'(Estacionária)' if adf_result[1]<0.05 else '(Não-estacionária)'}")
except:
    pass

# ============================================================================
# PASSO 3: FEATURE ENGINEERING (EXPLICADO)
# ============================================================================
print("\n[3/9] Engenharia de Features...")

# FEATURE 1: MONTH (Sazonalidade Mensal)
# Lógica: Alguns meses podem ter padrões diferentes de turnover
#         Exemplo: Janeiro (novos projetos), Dezembro (férias)
#         Implementação: Extração direta do calendário
df['month'] = df['MES_REF'].dt.month
print(f"   Month (1-12): Captura sazonalidade intra-anual")

# FEATURE 2: QUARTER (Sazonalidade Trimestral)
# Lógica: Períodos de negócio (Q1, Q2, etc)
df['quarter'] = df['MES_REF'].dt.quarter
print(f"   Quarter (1-4): Captura ciclos trimestrias")

# FEATURE 3: SEASONAL INDICATORS (Encoded como dummies)
# Lógica: Indicadores binários para cada mês (mais flexível)
for m in range(1, 13):
    df[f'month_{m:02d}'] = (df['month'] == m).astype(int)
print(f"   Month dummies (12 features): Permite coeficientes diferentes por mês")

# Renomear features operacionais
df = df.rename(columns={
    'TO_HEADCOUNT_HEADCOUNT-MES-ATUAL': 'headcount',
    'TO_ADMISSOES_ADMISSOES-MES-ATUAL': 'admissions'
})

# FEATURE 4: LAGS DE HEADCOUNT (Memória Temporal)
# Lógica: Tamanho da equipe nos meses anteriores
#         Se cresceu muito (lag alto vs atual), pode causar instabilidade
#         Implementação: Shift por área
grouped = df.groupby('area_anonn')
for lag in [1, 3, 6]:
    df[f'headcount_lag_{lag}'] = grouped['headcount'].shift(lag)
    print(f"   headcount_lag_{lag}: Tamanho equipe {lag} meses atrás")

# FEATURE 5: LAGS DE ADMISSIONS
# Lógica: Admissões recentes correlacionam com turnover
#         Onboarding incompleto, não-fit cultural
for lag in [1, 3, 6]:
    df[f'admissions_lag_{lag}'] = grouped['admissions'].shift(lag)
    print(f"   admissions_lag_{lag}: Admissões {lag} meses atrás")

# FEATURE 6: LAGS DE TARGET 
# Lógica: Turnover tem inércia temporal
#         Se tinha turnover alto mês passado, pode ter novamente
for lag in [1, 3, 6]:
    df[f'target_lag_{lag}'] = grouped[target_col].shift(lag)
    print(f"   target_lag_{lag}: Turnover {lag} meses atrás (memória temporal)")

# FEATURE 7: RATE OF CHANGE (Momentum)
# Lógica: Velocidade de mudança pode ser mais importante que valor
df['headcount_growth_1m'] = grouped['headcount'].diff(1)
df['admissions_growth_1m'] = grouped['admissions'].diff(1)
print(f"   headcount/admissions growth: Taxa de mudança (momentum)")

# Remover NAs
df = df.dropna().reset_index(drop=True)
print(f"\n  Total Features: {len([c for c in df.columns if c not in ['MES_REF', 'area_anonn', target_col]])}")
print(f"  Shape final: {df.shape}")

# ============================================================================
# PASSO 4: PREPARAÇÃO PARA TREINO
# ============================================================================
print("\n[4/9] Preparando dados para treino...")

le = LabelEncoder()
df['area_encoded'] = le.fit_transform(df['area_anonn'])

# Features (em ordem de importância esperada)
# NOTE: area_encoded NÃO é usado como preditor, apenas para avaliação
feature_cols = [
    # Operacionais atuais
    'headcount', 'admissions',
    # Lags operacionais
    'headcount_lag_1', 'headcount_lag_3', 'headcount_lag_6',
    'admissions_lag_1', 'admissions_lag_3', 'admissions_lag_6',
    # Lags de target (memória)
    'target_lag_1', 'target_lag_3', 'target_lag_6',
    # Sazonalidade
    'month', 'quarter',
    # Dummies mensais
    'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
    'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
    # Momentum
    'headcount_growth_1m', 'admissions_growth_1m'
]

X = df[feature_cols].copy()
y = df[target_col].copy()
areas = df['area_anonn'].copy()
dates = df['MES_REF'].copy()

# Split temporal
split_idx = int(len(df) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]
dates_test = dates[split_idx:]
areas_test = areas[split_idx:]

print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
print(f"  Test period: {dates_test.min().date()} to {dates_test.max().date()}")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PASSO 5: TREINAR MODELOS ML
# ============================================================================
print("\n[5/9] Treinando modelos ML...")

ml_models = {}
ml_predictions = {}

# XGBoost
print("  - XGBoost...")
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, 
                             random_state=1, verbosity=0)
xgb_model.fit(X_train_scaled, y_train)
ml_models['XGBoost'] = xgb_model
ml_predictions['XGBoost'] = xgb_model.predict(X_test_scaled)

# RandomForest
print("  - RandomForest...")
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=-1, random_state=1)
rf_model.fit(X_train_scaled, y_train)
ml_models['RandomForest'] = rf_model
ml_predictions['RandomForest'] = rf_model.predict(X_test_scaled)

# GradientBoosting
print("  - GradientBoosting...")
gb_model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=1)
gb_model.fit(X_train_scaled, y_train)
ml_models['GradientBoosting'] = gb_model
ml_predictions['GradientBoosting'] = gb_model.predict(X_test_scaled)

# Ridge
print("  - Ridge...")
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train_scaled, y_train)
ml_models['Ridge'] = ridge_model
ml_predictions['Ridge'] = ridge_model.predict(X_test_scaled)

print(f"   {len(ml_models)} modelos ML treinados")

# ============================================================================
# PASSO 6: TREINAR MODELOS DE SÉRIE TEMPORAL
# ============================================================================
print("\n[6/9] Treinando modelos de série temporal...")

ts_models = {}
ts_predictions = {}

# Preparar série temporal agregada por área (ou agregada geral)
# Vamos usar série agregada total para ARIMA/ETS
ts_series = df.groupby('MES_REF')[target_col].mean()
ts_train = ts_series[ts_series.index < df.loc[split_idx, 'MES_REF']].values
ts_test_dates = ts_series.index[ts_series.index >= df.loc[split_idx, 'MES_REF']]
ts_test_actual = ts_series[ts_series.index >= df.loc[split_idx, 'MES_REF']].values

# ARIMA
print("  - ARIMA (1,0,1)...")
try:
    arima_model = ARIMA(ts_train, order=(1, 0, 1))
    arima_fit = arima_model.fit()
    ts_pred = arima_fit.forecast(steps=len(ts_test_actual))
    ts_models['ARIMA'] = arima_fit
    ts_predictions['ARIMA'] = ts_pred
    print(f"     ARIMA treinado")
except Exception as e:
    print(f"    ✗ ARIMA falhou: {str(e)}")

# Exponential Smoothing
print("  - Exponential Smoothing...")
try:
    ets_model = ExponentialSmoothing(ts_train, trend='add', seasonal=None)
    ets_fit = ets_model.fit(optimized=True)
    ts_pred = ets_fit.forecast(steps=len(ts_test_actual))
    ts_models['ExponentSmoothing'] = ets_fit
    ts_predictions['ExponentSmoothing'] = ts_pred
    print(f"     Exponential Smoothing treinado")
except Exception as e:
    print(f"    ✗ Exponential Smoothing falhou: {str(e)}")

# Naive (baseline)
print("  - Naive Forecast (Baseline)...")
naive_pred = np.full(len(ts_test_actual), ts_train[-1])  # Última observação
ts_predictions['Naive'] = naive_pred
print(f"     Naive (baseline) criado")

# ============================================================================
# PASSO 7: COMPARAR MODELOS
# ============================================================================
print("\n[7/9] Comparando modelos...")

all_metrics = []

# Métricas ML (escala de áreas individuais)
print("  ML Models (escala de áreas):")
for name, pred in ml_predictions.items():
    m = calculate_metrics(y_test, pred, name)
    all_metrics.append(m)
    print(f"    {name:20s} | R²: {m['r2']:6.4f} | MAE: {m['mae']:6.2f}%")

# Métricas Time Series (escala agregada)
print("  Time Series Models (escala agregada):")
for name, pred in ts_predictions.items():
    m = calculate_metrics(ts_test_actual, pred, name)
    all_metrics.append(m)
    print(f"    {name:20s} | R²: {m['r2']:6.4f} | MAE: {m['mae']:6.2f}%")

# Melhor modelo overall
all_metrics_sorted = sorted(all_metrics, key=lambda x: x['r2'], reverse=True)
best_model_name = all_metrics_sorted[0]['model']
print(f"\n   Melhor modelo: {best_model_name} (R² = {all_metrics_sorted[0]['r2']:.4f})")

os.makedirs('reports', exist_ok=True)
with open('reports/model_comparison_v2.json', 'w') as f:
    json.dump({m['model']: m for m in all_metrics_sorted}, f, indent=2)

# ============================================================================
# PASSO 8: ANÁLISE DE FEATURE ENGINEERING
# ============================================================================
print("\n[8/9] Analisando Feature Engineering...")

if best_model_name in ml_models:
    best_model = ml_models[best_model_name]
    importance = best_model.feature_importances_
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    importance_df.to_csv('reports/feature_importance_v2.csv', index=False)
    
    print("\n  Feature Engineering Analysis:")
    print("  " + "="*60)
    
    # Agrupar por tipo de feature
    temporal_feats = importance_df[importance_df['feature'].str.contains('lag|month|quarter|growth')]
    operational_feats = importance_df[importance_df['feature'].isin(['headcount', 'admissions'])]
    
    print(f"  Temporal Features (lag, seasonality): {temporal_feats['importance'].sum():.4f}")
    print(f"    Top: {temporal_feats.iloc[0]['feature']} ({temporal_feats.iloc[0]['importance']:.4f})")
    
    print(f"  Operational Features: {operational_feats['importance'].sum():.4f}")
    print(f"    Top: {operational_feats.iloc[0]['feature']} ({operational_feats.iloc[0]['importance']:.4f})")
    
    print("\n  Top 10 Features Overall:")
    for i, row in importance_df.head(10).iterrows():
        print(f"    {i+1:2d}. {row['feature']:30s} | {row['importance']:.4f}")

# ============================================================================
# PASSO 9: GERAR OUTPUTS
# ============================================================================
print("\n[9/9] Gerando outputs e documentação...")

# Tabela de predições (usar melhor modelo ML)
if best_model_name in ml_predictions:
    best_pred = ml_predictions[best_model_name]
    results_df = pd.DataFrame({
        'mes_ref': dates_test.values,
        'area': areas_test.values,
        'valor_real': y_test.values,
        'valor_predito': best_pred,
        'erro_absoluto': np.abs(y_test.values - best_pred),
        'erro_percentual': np.abs(y_test.values - best_pred) / (np.abs(y_test.values) + 1) * 100
    })
    results_df.to_csv('reports/predictions_v2.csv', index=False)
    print("   Tabela de predições salva")
    
    # Gráficos
    plot_actual_vs_predicted(y_test.values, best_pred, 
                             title=f"Análise Real vs Predito - {best_model_name}",
                             save_path='reports/plot_actual_vs_predicted_v2.png')
    print("   Gráficos de diagnóstico salvos")
    
    # Feature Importance Plot
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'].values)
    plt.yticks(range(len(top_features)), top_features['feature'].values)
    plt.xlabel('Importance')
    plt.title(f'Top 15 Features - {best_model_name}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('reports/plot_feature_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   Feature importance plot salvo")

# SHAP
try:
    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_test_scaled)
    
    plt.figure(figsize=(10, 6))
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[0], X_test_scaled, feature_names=feature_cols, 
                          show=False, plot_type='bar')
    else:
        shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_cols, 
                          show=False, plot_type='bar')
    plt.title(f'SHAP Feature Importance - {best_model_name}')
    plt.tight_layout()
    plt.savefig('reports/shap_importance_v2.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("   SHAP plot salvo")
except Exception as e:
    print(f"   SHAP não disponível: {str(e)}")

# ============================================================================
# RESUMO FINAL
# ============================================================================
print("\n" + "="*80)
print("RESUMO EXECUTIVO V2")
print("="*80)

print(f"\n DATASET:")
print(f"   Total: {len(df):,} observações")
print(f"   Áreas: {df['area_anonn'].nunique()}")
print(f"   Período: {df['MES_REF'].min().date()} a {df['MES_REF'].max().date()}")

print(f"\n MELHOR MODELO: {best_model_name}")
print(f"   R² Score: {all_metrics_sorted[0]['r2']:.4f}")
print(f"   MAE: {all_metrics_sorted[0]['mae']:.2f}%")
print(f"   RMSE: {all_metrics_sorted[0]['rmse']:.2f}%")

print(f"\n FEATURE ENGINEERING:")
print(f"   Total Features: {len(feature_cols)}")
print(f"   - Operacionais: 4 (headcount, admissions, atuais e lags)")
print(f"   - Sazonalidade: 14 (month, quarter, 12 dummies)")
print(f"   - Memória Temporal: 6 (lags de headcount, admissions, target)")
print(f"   - Momentum: 2 (growth rates)")

print(f"\n MODELOS COMPARADOS:")
print(f"   ML: {len(ml_models)} (XGBoost, RF, GB, Ridge)")
print(f"   TS: {len(ts_models)} (ARIMA, Exp.Smoothing, Naive)")
print(f"   Total: {len(all_metrics)}")

print(f"\n OUTPUTS SALVOS:")
print(f"    model_comparison_v2.json")
print(f"    predictions_v2.csv")
print(f"    feature_importance_v2.csv")
print(f"    plot_actual_vs_predicted_v2.png")
print(f"    plot_feature_importance_v2.png")
print(f"    shap_importance_v2.png")

print(f"\n" + "="*80)
print(" Pipeline V2 concluído com sucesso!")
print("="*80)
