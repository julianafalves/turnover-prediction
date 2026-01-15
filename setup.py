"""
Configuração de setup para o Pipeline de Predição de Turnover.

Este módulo define os metadados do pacote e os requisitos de instalação
para o projeto de predição de turnover, que implementa modelos de machine learning
para prever taxas de rotatividade de funcionários usando análise de séries temporais.

Estrutura do Pacote:
    - src/: Módulos principais para carregamento de dados, treinamento de modelo e utilitários
    - config/: Arquivos de configuração YAML para parâmetros do pipeline
    - main.py: Ponto de entrada para o pipeline de predição

Versão: 2.0
"""

from setuptools import setup, find_packages

setup(
    # Metadados do pacote
    name="turnover-prediction",
    version="2.0",
    
    # Descobre automaticamente todos os pacotes no projeto
    # Isso inclui o pacote 'src' e seus submódulos
    packages=find_packages(),
    
    # Metadados adicionais (podem ser expandidos conforme necessário)
    description="Pipeline de Machine Learning para Predição de Turnover de Funcionários",
    author="Equipe de Predição de Turnover",
    python_requires=">=3.8",
)