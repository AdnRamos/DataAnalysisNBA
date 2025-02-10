import pandas as pd
import numpy as np
from scipy.stats import gumbel_r
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, classification_report
from pygam import PoissonGAM, LinearGAM
import pygam

import matplotlib.pyplot as plt
import seaborn as sns
from parte_2_run import load_all_players_data, process_player_log
from scipy.sparse import csr_matrix
# Corrigir depreciação do np.int
np.int = int

# ============================================================================
# 1. Carregar e Processar os Dados
# ============================================================================
seasons = ['2023-24', '2024-25']
df_raw = load_all_players_data(seasons)
df_processed = process_player_log(df_raw)

# ============================================================================
# 2. Método de Gumbel para Modelar Eventos Extremos
# ============================================================================
def ajustar_gumbel(data, coluna):
    loc, scale = gumbel_r.fit(data[coluna])
    print(f"Parâmetros para {coluna} - Loc: {loc:.2f}, Scale: {scale:.2f}")
    return loc, scale

def calcular_probabilidades_gumbel(data, coluna, valor):
    loc, scale = ajustar_gumbel(data, coluna)
    prob_acima = 1 - gumbel_r.cdf(valor, loc=loc, scale=scale)
    prob_ate = gumbel_r.cdf(valor, loc=loc, scale=scale)
    print(f"Probabilidade de {coluna} > {valor}: {prob_acima:.4f}")
    print(f"Probabilidade de {coluna} <= {valor}: {prob_ate:.4f}")
    return prob_acima, prob_ate

def plotar_gumbel(data, coluna):
    loc, scale = ajustar_gumbel(data, coluna)
    valores = np.linspace(data[coluna].min(), data[coluna].max(), 100)
    pdf = gumbel_r.pdf(valores, loc=loc, scale=scale)

    plt.figure(figsize=(8, 5))
    plt.hist(data[coluna], bins=20, density=True, alpha=0.6, color='skyblue', label='Dados Reais')
    plt.plot(valores, pdf, 'r-', label='Distribuição de Gumbel')
    plt.title(f'Distribuição de Gumbel para {coluna}')
    plt.xlabel(coluna)
    plt.ylabel('Densidade de Probabilidade')
    plt.legend()
    plt.show()

# ============================================================================
# 3. Regressão Linear para Previsão de Desempenho
# ============================================================================
def regressao_linear(data, dependente, independentes):
    X = data[independentes]
    y = data[dependente]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel('Valores Reais')
    plt.ylabel('Valores Preditos')
    plt.title(f'Regressão Linear: {dependente}')
    plt.show()

    return modelo

# ============================================================================
# 4. Regressão Logística para Classificação de Desempenho
# ============================================================================
def regressao_logistica(data, dependente, independentes, threshold):
    data.loc[:, f'{dependente}_high'] = (data[dependente] > threshold).astype(int)
    
    X = data[independentes]
    y = data[f'{dependente}_high']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão: {dependente}')
    plt.xlabel('Predito')
    plt.ylabel('Real')
    plt.show()

    print(classification_report(y_test, y_pred))
    return modelo

# ============================================================================
# 5. GAMLSS para Previsão de Jogos Futuros
# ============================================================================
def gam_previsao(data, dependente, independentes):
    X = data[independentes].to_numpy()  # Converter para matriz densa
    y = data[dependente].to_numpy()  # Garantir que y está no formato correto

    modelo_poisson = PoissonGAM().fit(X, y)
    modelo_linear = LinearGAM().fit(X, y)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(y, modelo_poisson.predict(X), 'o', label='PoissonGAM')
    plt.title(f'PoissonGAM: {dependente}')
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(y, modelo_linear.predict(X), 'o', color='green', label='LinearGAM')
    plt.title(f'LinearGAM: {dependente}')
    plt.xlabel('Real')
    plt.ylabel('Predito')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

    return modelo_poisson, modelo_linear

def patched_initial_estimate(self, Y, modelmat):
    if isinstance(modelmat, csr_matrix):
        modelmat = modelmat.toarray()  # Converter para matriz densa
    return np.linalg.solve(np.dot(modelmat.T, modelmat), np.dot(modelmat.T, Y))

# ============================================================================
# 6. Execução das Funções para um Jogador Específico
# ============================================================================
if __name__ == '__main__':
    jogador = 'Ja Morant'
    df_jogador = df_processed[df_processed['Player_Name'] == jogador].copy()
    
    

    pygam.pygam.PoissonGAM._initial_estimate = patched_initial_estimate
    # Gumbel
    plotar_gumbel(df_jogador, 'PTS')
    calcular_probabilidades_gumbel(df_jogador, 'PTS', 30)

    # Regressão Linear
    regressao_linear(df_jogador, 'PTS', ['Tempo em Quadra', 'Tentativas de Cestas de 3'])

    # Regressão Logística
    regressao_logistica(df_jogador, 'PTS', ['Tempo em Quadra', 'Tentativas de Cestas de 3'], threshold=25)

    # GAMLSS
    gam_previsao(df_jogador, 'PTS', ['Tempo em Quadra', 'Tentativas de Cestas de 3'])
