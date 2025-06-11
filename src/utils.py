import pandas as pd
import numpy as np

from scipy.stats import chi2_contingency


def gerar_metadados(df, ids, targets, orderby = 'PC_NULOS'):
    """
    Esta função retorna uma tabela com informações descritivas sobre um DataFrame.

    Parâmetros:
    - df: DataFrame que você quer descrever.
    - ids: Lista de colunas que são identificadores.
    - targets: Lista de colunas que são variáveis alvo.

    Retorna:
    Um DataFrame com informações sobre o df original.
    """
   
    summary = pd.DataFrame({
        'USO_FEATURE': ['ID' if col in ids else 'Target' if col in targets else 'Explicativa' for col in df.columns],
        'QT_NULOS': df.isnull().sum(),
        'PC_NULOS': round((df.isnull().sum() / len(df))* 100,2),
        'CARDINALIDADE': df.nunique(),
        'TIPO_FEATURE': df.dtypes
    })

    summary_sorted = summary.sort_values(by=orderby, ascending=False)
    summary_sorted = summary_sorted.reset_index()
    # Renomeando a coluna 'index' para 'FEATURES'
    summary_sorted = summary_sorted.rename(columns={'index': 'FEATURE'})
    return summary_sorted