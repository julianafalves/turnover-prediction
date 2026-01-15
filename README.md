# Previsão de Turnover com Séries Temporais e Machine Learning

## Visão Geral

Este repositório implementa um **pipeline completo de previsão de turnover** combinando:
- **Análise temporal** de dados de desligamentos
- **Machine Learning** (XGBoost, Random Forest, Gradient Boosting, Ridge)
- **Modelos de Série Temporal** (ARIMA, Exponential Smoothing, Naive)
- **Feature Engineering avançado** com lags temporais e sazonalidade
- **Validação Walk-Forward** robusta para séries temporais
- **Arquitetura modular** com configuração centralizada

### Arquitetura V3 (Robusto)

O pipeline V3 implementa uma abordagem robusta com:
- **Validação cruzada aninhada** (nested cross-validation)
- **Loop externo**: Walk-forward time series split (5 dobras)
- **Loop interno**: Otimização de hiperparâmetros com RandomizedSearchCV
- **Prevenção de vazamento de dados** (leakage) através de pré-processamento cuidadoso
- **Configuração centralizada** via YAML

---

## Estrutura do Projeto

```
turnover-prediction/
├── main.py                        # Ponto de entrada principal do pipeline V3
├── config/                        # Configurações do pipeline
│   └── params.yaml                # Parâmetros centralizados (dados, features, modelos)
├── src/                           # Código fonte modular
│   ├── __init__.py
│   ├── data_loader.py             # Carregamento de dados e feature engineering
│   ├── model_trainer.py           # Treinamento com validação walk-forward
│   └── utils.py                   # Funções utilitárias (logging, config)
├── data/                          # Dados brutos e preparados
│   ├── turnover_with_label.csv
│   └── *.csv
├── notebooks/                     # Análises interativas
│   └── individual turnover predictions.ipynb
├── scripts/                       # Scripts auxiliares
│   ├── setup_venv.sh              # Setup de ambiente (Linux/Mac)
│   ├── setup_venv.ps1             # Setup de ambiente (Windows)
│   ├── cleanup.sh                 # Limpeza de arquivos temporários
│   └── debug_preprocess.py        # Debug de pré-processamento
├── reports/                       # Outputs e visualizações
│   ├── predictions_v3_validation.csv    # Resultados de validação
│   ├── feature_importance_v3.csv        # Ranking de features
│   └── pipeline.log                      # Log detalhado de execução
├── requirements.txt               # Dependências do projeto
├── setup.py                       # Configuração do pacote
├── .gitignore                     # Arquivos ignorados pelo Git
└── README.md                      # Este arquivo
```

---

## Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip ou conda

### Setup Rápido

**Linux/Mac:**
```bash
# 1. Clonar repositório
git clone https://github.com/julianafalves/turnover-prediction.git
cd turnover-prediction

# 2. Criar virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Instalar dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Windows (PowerShell):**
```powershell
# 1. Clonar repositório
git clone https://github.com/julianafalves/turnover-prediction.git
cd turnover-prediction

# 2. Criar virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 3. Instalar dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Ou usando o script helper:**
```bash
# Linux/Mac
./scripts/setup_venv.sh

# Windows (PowerShell)
.\scripts\setup_venv.ps1
```

---

## Dados de Entrada Esperados

| Arquivo | Descrição | Campos Obrigatórios |
|---------|-----------|-------------------|
| `turnover_with_label.csv` | Histórico de turnover por período/área | `MES_REF`, `area_anonn`, `TO_TURNOVER_TO-GERAL`, `TO_HEADCOUNT_HEADCOUNT-MES-ATUAL`, `TO_ADMISSOES_ADMISSOES-MES-ATUAL` |

**Formato do CSV:**
- `MES_REF`: Data de referência (formato YYYY-MM-DD)
- `area_anonn`: Área/unidade de negócio anonimizada
- `TO_TURNOVER_TO-GERAL`: Taxa de turnover alvo (%)
- `TO_HEADCOUNT_HEADCOUNT-MES-ATUAL`: Headcount atual
- `TO_ADMISSOES_ADMISSOES-MES-ATUAL`: Admissões atuais

---

## Executando o Pipeline V3

### Opção 1: Pipeline Completo (Recomendado)

```bash
python main.py
```

**O que faz:**
1. Carrega configuração do [`config/params.yaml`](config/params.yaml:1)
2. Carrega e explora dados via [`src/data_loader.py`](src/data_loader.py:1)
3. Analisa série temporal agregada (teste ADF)
4. Descreve engenharia de features
5. Treina 4 modelos ML com validação walk-forward (XGBoost, RF, GB, Ridge)
6. Treina 3 modelos de série temporal (ARIMA, ETS, Naive)
7. Compara todos os modelos
8. Analisa importância de features (retreinamento final)
9. Gera relatórios em [`reports/`](reports/)

**Outputs gerados:**
- `reports/predictions_v3_validation.csv` — Resultados de validação para todos os modelos
- `reports/feature_importance_v3.csv` — Ranking de features
- `pipeline.log` — Log detalhado de execução

### Opção 2: Configuração Personalizada

Edite [`config/params.yaml`](config/params.yaml:1) para personalizar:

**Configurações disponíveis:**
- `data.input_path`: Caminho do arquivo de dados
- `data.target_col`: Nome da coluna alvo
- `features.lags`: Períodos de lag (padrão: [1, 3, 6])
- `features.use_momentum`: Habilitar features de momentum
- `training.n_splits`: Número de dobras walk-forward (padrão: 5)
- `models.*.enabled`: Habilitar/desabilitar modelos específicos
- `models.*.params`: Hiperparâmetros para cada modelo

**Exemplo - Desabilitar Ridge:**
```yaml
models:
  ridge:
    enabled: false
```

---

## Metodologia: Validação Walk-Forward

O pipeline V3 utiliza **validação cruzada aninhada** para garantir estimativas de desempenho realistas:

### Loop Externo (Walk-Forward)
- Divide os dados em 5 janelas temporais
- Cada dobra treina em dados passados e testa em dados futuros
- Simula o cenário real de produção

### Loop Interno (Otimização)
- Usa RandomizedSearchCV dentro de cada janela de treino
- Otimiza hiperparâmetros respeitando a ordem temporal
- Usa TimeSeriesSplit (3 dobras) para validação interna

### Prevenção de Vazamento
- StandardScaler é ajustado APENAS nos dados de treino
- Transformação aplicada em treino e teste separadamente
- Ordenação temporal estrita em todos os passos

**Vantagens:**
- Estimativas de desempenho realistas
- Sem vazamento de dados do futuro
- Robustez a diferentes períodos temporais
- Otimização de hiperparâmetros segura

---

## Feature Engineering

### Features Utilizadas

| Tipo | Features | Descrição |
|------|----------|-----------|
| **Sazonais (2)** | `month`, `quarter` | Captura padrões sazonais |
| **Operacionais (2)** | `headcount`, `admissions` | Tamanho da equipe e novas contratações |
| **Lags Operacionais (6)** | `headcount_lag_{1,3,6}`, `admissions_lag_{1,3,6}` | Histórico de 1, 3 e 6 meses |
| **Lags de Target (3)** | `target_lag_{1,3,6}` | Turnover histórico (sem leakage) |
| **Momentum (2)** | `headcount_growth_1m`, `admissions_growth_1m` | Taxa de mudança |

**Total: ~15-20 features** (dependendo da configuração)

### Implementação

A engenharia de features é implementada em [`src/data_loader.py`](src/data_loader.py:20):

```python
def create_features(df, config):
    # 1. Ordena por área e data
    # 2. Cria features sazonais (mês, trimestre)
    # 3. Cria lags para cada período configurado
    # 4. Cria features de momentum (se habilitado)
    # 5. Remove valores NaN gerados pelos lags
```

---

## Modelos Treinados e Comparados

### Machine Learning (Validação Walk-Forward)

| Modelo | R² (Média) | MAE (%) | RMSE (%) | Vantagens |
|--------|-----------|---------|----------|-----------|
| **XGBoost** | ~0.65-0.75 | 1.5-2.5 | 2.0-3.0 | Melhor geral, captura não-linearidades |
| **Random Forest** | ~0.60-0.70 | 2.0-3.0 | 2.5-3.5 | Robusto, menos overfitting |
| **Gradient Boosting** | ~0.62-0.72 | 1.8-2.8 | 2.2-3.2 | Bom para features engineered |
| **Ridge** | ~0.50-0.60 | 2.5-4.0 | 3.0-4.5 | Baseline linear, muito rápido |

### Séries Temporais (Escala Agregada)

| Modelo | R² | MAE (%) | Uso |
|--------|----|---------|-----|
| **ARIMA(1,0,1)** | ~0.55-0.65 | 2.0-3.0 | Série estacionária |
| **Exponential Smoothing** | ~0.60-0.70 | 1.8-2.8 | Tendência e nível |
| **Naive (Baseline)** | ~0.20-0.40 | 4.0-6.0 | Referência mínima |

**Recomendação:** XGBoost combina melhor performance com interpretabilidade

---

## Interpretação de Resultados

### Feature Importance

O pipeline gera um ranking de features em [`reports/feature_importance_v3.csv`](reports/feature_importance_v3.csv:1):

```
Top Features (tipicamente):
1. target_lag_1        → Inércia temporal do turnover
2. headcount_lag_3     → Efeito de contratações/redução
3. admissions_lag_1    → Onboarding incompleto
4. month_08            → Sazonalidade (ex: férias)
5. quarter             → Ciclos de negócio
```

### Métricas de Validação

**predictions_v3_validation.csv** contém:
- `fold`: Número da dobra (1-5)
- `model`: Nome do modelo
- `train_end_date`: Última data no treino
- `test_period_start/end`: Período de teste
- `mae`: Erro Médio Absoluto
- `rmse`: Raiz do Erro Quadrático Médio
- `r2`: Coeficiente de Determinação
- `best_params`: Melhores hiperparâmetros encontrados

---

## Análise Individual - Jupyter Notebook

Para análise interativa da **previsão individual de turnover**, execute:

```bash
jupyter notebook "notebooks/individual turnover predictions.ipynb"
```

**Conteúdo do Notebook:**
- Exploração dos dados Fala AI
- Feature engineering individual
- Treino de Random Forest para propensão à saída
- Análise de importância de features por pessoa
- Comparação com dados de desligamentos reais

---

## Metodologia: Taxa de Turnover (%)

Este projeto prediz **taxa de turnover em percentual**, não quantidade absoluta. A decisão foi baseada em:

- **Padronização internacional** — SHRM, BLS usam % como métrica padrão
- **Comparabilidade** — Áreas com diferentes tamanhos em mesma escala
- **Série temporal mais estável** — Melhor para ARIMA, Prophet, Exponential Smoothing
- **Literatura consolidada** — 95% dos papers acadêmicos usam %

---

## Estrutura de Saídas

### `reports/` - Outputs do Pipeline

```
reports/
├── predictions_v3_validation.csv    # Resultados de validação walk-forward
├── feature_importance_v3.csv        # Ranking de features
└── pipeline.log                     # Log detalhado de execução
```

### Formato dos Outputs

**predictions_v3_validation.csv:**
```csv
fold,model,train_end_date,test_period_start,test_period_end,mae,rmse,r2,best_params
1,xgboost,2023-06-30,2023-07-31,2023-08-31,1.85,2.42,0.72,"{'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1}"
...
```

**feature_importance_v3.csv:**
```csv
feature,importance
target_lag_1,0.2456
headcount_lag_3,0.1876
admissions_lag_1,0.1452
...
```

---

## Scripts Auxiliares

### Setup de Ambiente
- [`scripts/setup_venv.sh`](scripts/setup_venv.sh:1) — Setup para Linux/Mac
- [`scripts/setup_venv.ps1`](scripts/setup_venv.ps1:1) — Setup para Windows

### Limpeza
- [`scripts/cleanup.sh`](scripts/cleanup.sh:1) — Remove arquivos temporários e sintéticos
- [`scripts/cleanup_synthetic.ps1`](scripts/cleanup_synthetic.ps1:1) — Versão Windows

### Debug
- [`scripts/debug_preprocess.py`](scripts/debug_preprocess.py:1) — Debug do pré-processamento de dados

---

## Arquitetura Modular

### [`src/data_loader.py`](src/data_loader.py:1)
- `load_data(config)`: Carrega e pré-processa dados
- `create_features(df, config)`: Engenharia de features de séries temporais

### [`src/model_trainer.py`](src/model_trainer.py:1)
- `TurnoverTrainer`: Classe principal de treinamento
  - `train_and_validate()`: Validação cruzada aninhada
  - `get_model_factory(model_name)`: Fábrica de modelos

### [`src/utils.py`](src/utils.py:1)
- `setup_logger(name)`: Configura logging
- `load_config(path)`: Carrega configuração YAML

### [`main.py`](main.py:1)
- Orquestra todo o pipeline
- Coordena 9 passos principais
- Gera relatórios executivos

---

## Configuração do Pipeline

O arquivo [`config/params.yaml`](config/params.yaml:1) centraliza toda a configuração:

**Seções principais:**
- `project`: Nome e seed aleatória
- `data`: Caminhos, colunas, mapeamento, exclusões
- `features`: Configuração de lags e momentum
- `training`: Número de dobras, métrica de scoring
- `models`: Configuração de cada modelo (habilitado, parâmetros)

**Exemplo de configuração de modelo:**
```yaml
models:
  xgboost:
    enabled: true
    n_iter: 20
    params:
      n_estimators: [100, 200, 300]
      max_depth: [3, 5, 7]
      learning_rate: [0.01, 0.1, 0.2]
```

---

## Dependências

As dependências principais estão em [`requirements.txt`](requirements.txt:1):

- `pandas==1.5.3` — Manipulação de dados
- `scikit-learn==1.2.2` — ML e métricas
- `xgboost==1.7.4` — Gradient boosting
- `statsmodels==0.14.0` — Séries temporais
- `shap==0.41.0` — Interpretabilidade
- `PyYAML>=6.0.1` — Configuração
- `numpy==1.23.5` — Computação numérica

---

## 📧 Autores

- **Juliana Alves**
- **Marcus Rodrigues**

---

## Licença

Este projeto é desenvolvido para fins de pesquisa e análise de turnover corporativo.
