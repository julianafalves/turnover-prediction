"""
Módulo de Funções Utilitárias

Este módulo fornece funções utilitárias comuns usadas em todo o
pipeline de predição de turnover, incluindo a configuração de logging
e o carregamento de arquivos de configuração YAML.

Funções:
    setup_logger: Configura o logging tanto para o console quanto para arquivo
    load_config: Carrega arquivos de configuração YAML
"""

import logging
import yaml
import sys


def setup_logger(name="TurnoverPipeline"):
    """
    Configura e retorna um logger com manipuladores (handlers) de console e arquivo.
    
    Esta função define um sistema de log abrangente que envia mensagens de log
    tanto para o console (stdout) quanto para um arquivo de log. Isso permite
    o monitoramento em tempo real durante a execução e o registro persistente
    para fins de depuração e auditoria.
    
    Configuração do Logging:
        - Nível: INFO (captura mensagens informativas, avisos e erros)
        - Formato: timestamp - nome_do_logger - nível - mensagem
        - Saída de Console: stdout (visível no terminal)
        - Saída de Arquivo: pipeline.log (arquivo de log persistente)
    
    Args:
        name (str, opcional): Nome da instância do logger. O padrão é 
                              "TurnoverPipeline". O uso de nomes específicos
                              permite a filtragem granular dos logs.
    
    Returns:
        logging.Logger: Instância do logger configurada e pronta para uso.
    
    Exemplo:
        >>> logger = setup_logger("MeuModulo")
        >>> logger.info("Processamento iniciado")
        >>> logger.error("Ocorreu um erro")
    
    Nota:
        - Se chamada várias vezes com o mesmo nome, os manipuladores podem ser
          duplicados. Verifique se os manipuladores já existem antes de adicionar.
        - O arquivo de log é criado no diretório de trabalho atual.
        - O arquivo de log existente é sobrescrito a cada execução.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Define o formato da mensagem de log
    # Formato: data/hora - nome_do_logger - nível - mensagem
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Manipulador de console: envia a saída para o terminal/stdout
    # Útil para monitoramento em tempo real durante a execução
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # Manipulador de arquivo: envia a saída para um arquivo de log persistente
    # Útil para depuração, auditoria e análise pós-execução
    fh = logging.FileHandler('pipeline.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    return logger


def load_config(path="config/params.yaml"):
    """
    Carrega os parâmetros de configuração a partir de um arquivo YAML.
    
    Esta função lê um arquivo de configuração YAML e retorna seu 
    conteúdo como um dicionário Python. O arquivo de configuração contém
    todos os parâmetros do pipeline, incluindo caminhos de dados, configurações
    de atributos, configurações de modelos e parâmetros de treinamento.
    
    Estrutura Esperada do YAML:
        project:
            name: str
            random_seed: int
        data:
            input_path: str
            target_col: str
            date_col: str
            area_col: str
            rename_map: dict
            drop_cols: list
        features:
            lags: list
            use_momentum: bool
        training:
            n_splits: int
            scoring: str
        models:
            nome_do_modelo:
                enabled: bool
                n_iter: int
                params: dict
    
    Args:
        path (str, opcional): Caminho para o arquivo de configuração YAML.
                              O padrão é "config/params.yaml".
    
    Returns:
        dict: Parâmetros de configuração carregados do arquivo YAML.
    
    Raises:
        FileNotFoundError: Se o arquivo de configuração especificado não existir
        yaml.YAMLError: Se o arquivo YAML estiver malformado ou contiver sintaxe inválida
    
    Exemplo:
        >>> config = load_config("config/params.yaml")
        >>> print(config['project']['name'])
        'Turnover Prediction Pipeline V2'
    
    Nota:
        - Utiliza yaml.safe_load() por segurança (evita execução de código)
        - O caminho é relativo ao diretório de trabalho atual
        - A configuração deve ser validada após o carregamento
    """
    with open(path, 'r') as f:
        return yaml.safe_load(f)