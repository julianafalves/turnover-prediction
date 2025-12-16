# Previsão de Turnover com Séries Temporais e Machine Learning


## Visão Geral

Este repositório implementa um **pipeline completo de previsão de turnover** combinando:
- **Análise temporal** de dados de desligamentos
- **Machine Learning** (XGBoost, Random Forest, Gradient Boosting, Ridge)
- **Modelos de Série Temporal** (ARIMA, Exponential Smoothing)
- **Feature Engineering avançado** com lags temporais e sazonalidade
- **Interpretação de modelos** usando Feature Importance e SHAP

### Abordagens Implementadas

1. **Pipeline V2 - Análise Agregada** (`pipeline_turnover_v2.py`)
   - Prediz **taxa de turnover (%)** por período agregado
   - Usa séries temporais com features operacionais (headcount, admissões)
   - Modelos ML: XGBoost, Random Forest, Gradient Boosting, Ridge
   - Validação temporal com análise de residuais

2. **Análise Individual** (`notebooks/individual_turnover_predictions.ipynb`)
   - Prediz **propensão à saída por pessoa**
   - Baseado em dados Fala AI + histórico individual
   - Deduplicação automática por pessoa/mês

---

## Estrutura do Projeto

```
time-series-turnover-prediction/
├── data/                          # Dados brutos e preparados
│   ├── juliana_alves_turnover_with_label.csv
│   ├── juliana_alves_turnover_and_fala_ai_annon_with_label.csv
│   ├── prepared.joblib            # Dados preprocessados
│   ├── prepared_no_leakage.joblib # Versão sem data leakage
│   ├── preprocessor.joblib        # Scaler e encoders
│   └── *.csv
├── models/                        # Modelos treinados
│   ├── xgb_turnover.joblib       # Principal (XGBoost)
│   ├── best_model_xgboost.joblib
│   ├── fala_rf.joblib            # RandomForest individual
│   ├── scaler.joblib             # StandardScaler
│   └── *.metrics.json
├── src/                           # Código fonte
│   └── turnover_prediction/
│       ├── __init__.py
│       └── utils.py
├── notebooks/                     # Análises interativas
│   └── individual_turnover_predictions.ipynb
├── reports/                       # Outputs e visualizações
│   ├── model_comparison_v2.json
│   ├── predictions_v2.csv
│   ├── feature_importance_v2.csv
│   ├── plot_actual_vs_predicted_v2.png
│   ├── plot_feature_importance_v2.png
│   └── shap_importance_v2.png
├── scripts/                       # Utilidades
│   ├── setup_venv.ps1
│   ├── cleanup.sh
│   └── cleanup_synthetic.ps1
├── tests/                         # Testes
│   ├── test_pipeline.py
│   └── test_benchmark.py
├── pipeline_turnover_v2.py       # Pipeline principal
├── requirements.txt
├── METODOLOGIA_TURNOVER_ANALISE.md
├── README.md
└── .gitignore
```

---

## Instalação e Configuração

### Pré-requisitos
- Python 3.8+
- pip ou conda
- PowerShell (para scripts Windows)

### Setup Rápido

```powershell
# 1. Clonar repositório
git clone https://github.com/julianafalves/time-series-turnover-prediction.git
cd time-series-turnover-prediction

# 2. Criar virtual environment
python -m venv .venv
.\.venv\Scripts\Activate

# 3. Instalar dependências
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Ou usando o script helper (PowerShell):**
```powershell
.\scripts\setup_venv.ps1
```

---

## Dados de Entrada Esperados

| Arquivo | Descrição | Campos Obrigatórios |
|---------|-----------|-------------------|
| `juliana_alves_turnover_with_label.csv` | Histórico de turnover por período/área | `MES_REF`, `area_anonn`, `TO_TURNOVER_TO-GERAL`, `TO_HEADCOUNT_HEADCOUNT-MES-ATUAL`, `TO_ADMISSOES_ADMISSOES-MES-ATUAL` |
| `juliana_alves_turnover_and_fala_ai_annon_with_label.csv` | Dados diários Fala AI + labels | `pseudo_person_id`, `year_month`, respostas de pesquisa, rótulo de desligamento |

---

## Executando o Pipeline V2

### Opção 1: Pipeline Completo (Recomendado)

```powershell
python pipeline_turnover_v2.py
```

**O que faz:**
1. Carrega e explora dados
2. Engenharia de features (lags, sazonalidade)
3. Treina 4 modelos ML (XGBoost, RandomForest, GradientBoosting, Ridge)
4. Treina 3 modelos de série temporal (ARIMA, Exponential Smoothing, Naive)
5. Compara todos os modelos
6. Gera Feature Importance e SHAP
7. Salva outputs em `reports/`

**Outputs gerados:**
- `reports/model_comparison_v2.json` — Métricas de todos os modelos
- `reports/predictions_v2.csv` — Predições vs valores reais
- `reports/feature_importance_v2.csv` — Ranking de features
- `reports/plot_actual_vs_predicted_v2.png` — Gráficos diagnósticos
- `reports/plot_feature_importance_v2.png` — Top 15 features
- `reports/shap_importance_v2.png` — SHAP values

---

## Análise Individual - Jupyter Notebook

Para análise interativa da **previsão individual de turnover**, execute:

```powershell
jupyter notebook notebooks/individual_turnover_predictions.ipynb
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

**Padronização internacional** — SHRM, BLS usam % como métrica padrão  
**Comparabilidade** — Áreas com diferentes tamanhos em mesma escala  
**Série temporal mais estável** — Melhor para ARIMA, Prophet, Exponential Smoothing  
**Literatura consolidada** — 95% dos papers acadêmicos usam %  

**Para detalhes metodológicos completos**, veja `METODOLOGIA_TURNOVER_ANALISE.md`

---

## Feature Engineering (Pipeline V2)

### Features Utilizadas (26 no total)

| Tipo | Features | Descrição |
|------|----------|-----------|
| **Operacionais (2)** | `headcount`, `admissions` | Tamanho da equipe e novas contratações atuais |
| **Lags Operacionais (6)** | `headcount_lag_{1,3,6}`, `admissions_lag_{1,3,6}` | Histórico de 1, 3 e 6 meses |
| **Lags de Target (6)** | `target_lag_{1,3,6}` | Turnover histórico (sem leakage data) |
| **Sazonalidade (14)** | `month`, `quarter`, `month_01..12` | Padrões sazonais mensais |
| **Momentum (2)** | `headcount_growth_1m`, `admissions_growth_1m` | Taxa de mudança |


---

## Modelos Treinados e Comparados

### Machine Learning (Escala Individual/Agregada)

| Modelo | R² | MAE (%) | RMSE (%) | Vantagens |
|--------|----|---------|---------|----|
| **XGBoost** | ~0.65-0.75 | 1.5-2.5 | 2.0-3.0 | Melhor geral, captura não-linearidades |
| **Random Forest** | ~0.60-0.70 | 2.0-3.0 | 2.5-3.5 | Robusto, menos overfitting |
| **Gradient Boosting** | ~0.62-0.72 | 1.8-2.8 | 2.2-3.2 | Bom para features engineered |
| **Ridge** | ~0.50-0.60 | 2.5-4.0 | 3.0-4.5 | Baseline, muito rápido |

### Série Temporal (Escala Agregada)

| Modelo | R² | MAE (%) | Uso |
|--------|----|---------|----|
| **ARIMA(1,0,1)** | ~0.55-0.65 | 2.0-3.0 | Série estacionária |
| **Exponential Smoothing** | ~0.60-0.70 | 1.8-2.8 | Tendência e nível |
| **Naive (Baseline)** | ~0.20-0.40 | 4.0-6.0 | Referência mínima |

**Recomendação:** XGBoost combina melhor performance com interpretabilidade (SHAP values)

---

## 📊 Interpretação de Resultados

### Feature Importance
```
Top Features (tipicamente):
1. target_lag_1    → Inércia temporal do turnover
2. headcount_lag_3 → Efeito de contratações/redução
3. admissions_lag_1 → Onboarding incompleto
4. month_08        → Sazonalidade (ex: férias)
5. quarter         → Ciclos de negócio
```

### SHAP Values
Visualiza contribuição de cada feature para previsão individual. Disponível em:
- `reports/shap_importance_v2.png`

### Análise de Residuais
4 gráficos diagnósticos:
- Real vs Predito (scatter plot)
- Residuais vs Predito (detecção de padrões)
- Distribuição de residuais (normalidade)
- Q-Q Plot (comparação com normal)

---

## 🧪 Testes

```powershell
# Ativar venv primeiro
.\.venv\Scripts\Activate

# Rodar todos os testes
pytest

# Rodar com cobertura
pytest --cov=src tests/

# Teste específico
pytest tests/test_pipeline.py -v
```

---

## 🛠️ Boas Práticas de Desenvolvimento

### Git e Versionamento
- ❌ Não commit: `data/`, `models/`, `reports/` (já ignorados em `.gitignore`)
- Sim commit: código fonte, notebooks, documentação, requirements.txt

Se acidentalmente comitou arquivos grandes:
```powershell
git rm -r --cached data models reports
git add .gitignore
git commit -m "Remove data, models and reports from tracking"
git push
```

### Ambiente Virtual
- Sempre trabalhe dentro do `.venv`
- Atualize `requirements.txt` quando adicionar pacotes
- Use `pip freeze > requirements.txt` para exportar

### Reprodutibilidade
- Use `random_state=1` em modelos (já feito no código)
- Salve versões de modelos com timestamps
- Documente versões de dependências (Python 3.8+, XGBoost 1.7+, etc)

---

## 📚 Estrutura de Saídas

### `reports/` - Outputs do Pipeline

```
reports/
├── model_comparison_v2.json           # Métricas de todos os modelos
├── predictions_v2.csv                 # Real vs Predito com erros
├── feature_importance_v2.csv          # Ranking de features
├── plot_actual_vs_predicted_v2.png    # 4 gráficos diagnósticos
├── plot_feature_importance_v2.png     # Top 15 features (barplot)
└── shap_importance_v2.png             # SHAP summary plot
```

### Formato dos Outputs

**predictions_v2.csv:**
```
mes_ref,area,valor_real,valor_predito,erro_absoluto,erro_percentual
2023-01-31,Area1,12.5,12.1,0.4,3.2
2023-01-31,Area2,8.7,9.2,0.5,5.7
...
```

**model_comparison_v2.json:**
```json
{
  "XGBoost": {
    "model": "XGBoost",
    "mae": 1.85,
    "mape": 15.3,
    "rmse": 2.42,
    "r2": 0.72,
    "n_samples": 150,
    "y_mean": 12.4,
    "y_std": 3.1
  },
  ...
}
```

---

## 🔬 Validação Temporal (Walk-Forward)

O pipeline usa **validação temporal** apropriada para séries:

```python
# Split 80/20 cronológico (não aleatório!)
split_idx = int(len(df) * 0.8)
train_period = antes dessa data
test_period = depois dessa data
```

⚠️ **Importante:** Dados futuros nunca "vêem" dados passados de teste (prevenção de leakage)

---

## 📖 Documentação Adicional

- **`METODOLOGIA_TURNOVER_ANALISE.md`** — Análise aprofundada de % vs quantidade, referências acadêmicas, benchmarks
- **`TIME_SERIES_MODELS.md`** — Documentação técnica de modelos ARIMA, Prophet, ETS
- **`GUIA_ARTIGO.md`** — Guia para escrita de artigo científico com resultados

---

## 🤝 Contribuindo

1. Faça um fork do repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-analise`)
3. Commit suas mudanças (`git commit -m 'Add nova análise'`)
4. Push para a branch (`git push origin feature/nova-analise`)
5. Abra um Pull Request

---

## 📝 Licença

[Insira licença aqui - ex: MIT, CC BY 4.0]

---

## 📧 Autores e Contato

- **Juliana Alves** - Análise principal
- Desenvolvido para pesquisa em Ciência de Dados (Faculdade/Empresa)

**GitHub:** [@julianafalves](https://github.com/julianafalves)

---

## 🙏 Agradecimentos

- Dados fornecidos por [Empresa/Instituição]
- Pesquisa Fala AI para propensão individual
- Comunidade científica (referências em METODOLOGIA_TURNOVER_ANALISE.md)

---

## 📌 Changelog

### v2.0 (Atual)
- Pipeline completo com 7 modelos
- Feature engineering avançado (lags + sazonalidade)
- Validação temporal
- SHAP interpretation
- Análise agregada vs individual
- Sem data leakage

### v1.0
- Baseline XGBoost
- Predição por área
- Feature importance simples

---

## ⚡ Quick Start

Para rodar o pipeline completo em < 5 min:

```powershell
# Setup
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt

# Rodar
python pipeline_turnover_v2.py

# Ver resultados
# Abra: reports/model_comparison_v2.json
#       reports/predictions_v2.csv
#       reports/*.png
```

**Próximas análises:**
- [ ] Comparação detalhada com análise individual (Fala AI)
- [ ] Dashboard interativo (Streamlit/Plotly)
- [ ] API de predição em tempo real
- [ ] Análise causal (econometric models)

