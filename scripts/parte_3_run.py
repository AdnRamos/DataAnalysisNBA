import numpy as np
np.int = int  # Monkey patch para np.int, se necessário

import scipy.sparse
if not hasattr(scipy.sparse.csr_matrix, 'A'):
    @property
    def A(self):
        return self.toarray()
    scipy.sparse.csr_matrix.A = A

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import gumbel_r
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
from pygam import PoissonGAM, LinearGAM

#####################################
# 1. MODELO DE GUMBEL – EVENTOS EXTREMOS
#####################################
def fit_gumbel_model(data):
    print("Ajustando modelo Gumbel...", flush=True)
    loc, scale = gumbel_r.fit(data)
    print(f"Modelo ajustado: loc={loc}, scale={scale}", flush=True)
    return loc, scale

def gumbel_probability(x, loc, scale, tail='above'):
    cdf_val = gumbel_r.cdf(x, loc=loc, scale=scale)
    if tail=='above':
        return 1 - cdf_val
    elif tail=='below':
        return cdf_val
    else:
        raise ValueError("O parâmetro 'tail' deve ser 'above' ou 'below'.")

def plot_gumbel_fit(data, loc, scale, metric_name="Pontos"):
    x = np.linspace(min(data), max(data), 100)
    pdf = gumbel_r.pdf(x, loc=loc, scale=scale)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=data, histnorm='probability density', name='Dados'))
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='Gumbel PDF'))
    fig.update_layout(title=f"Fit Gumbel para {metric_name}",
                      xaxis_title=metric_name,
                      yaxis_title="Densidade")
    return fig

def calculate_gumbel_benchmark_probabilities(data):
    mean_val = np.mean(data)
    median_val = np.median(data)
    mode_series = pd.Series(data).mode()
    mode_val = mode_series.iloc[0] if not mode_series.empty else None
    max_val = np.max(data)
    min_val = np.min(data)
    loc, scale = fit_gumbel_model(data)
    probabilities = {
        'above_mean': gumbel_probability(mean_val, loc, scale, tail='above'),
        'above_median': gumbel_probability(median_val, loc, scale, tail='above'),
        'above_mode': gumbel_probability(mode_val, loc, scale, tail='above') if mode_val is not None else None,
        'above_max': gumbel_probability(max_val, loc, scale, tail='above'),
        'above_min': gumbel_probability(min_val, loc, scale, tail='above'),
        'below_mean': gumbel_probability(mean_val, loc, scale, tail='below'),
        'below_median': gumbel_probability(median_val, loc, scale, tail='below'),
        'below_mode': gumbel_probability(mode_val, loc, scale, tail='below') if mode_val is not None else None,
        'below_max': gumbel_probability(max_val, loc, scale, tail='below'),
        'below_min': gumbel_probability(min_val, loc, scale, tail='below')
    }
    return {'benchmarks': {'mean': mean_val, 'median': median_val, 'mode': mode_val, 'max': max_val, 'min': min_val},
            'probabilities': probabilities}

#####################################
# 2. REGRESSÃO LINEAR – PREDIÇÃO CONTÍNUA
#####################################
def linear_regression_model(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, X_test, y_test, predictions

def plot_linear_regression_results(y_test, predictions, metric_name="Pontos"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test, mode='markers', name='Real'))
    fig.add_trace(go.Scatter(x=list(range(len(predictions))), y=predictions, mode='markers', name='Previsto'))
    fig.update_layout(title=f"Regressão Linear: Real vs. Previsto para {metric_name}",
                      xaxis_title="Amostras",
                      yaxis_title=metric_name)
    return fig

def plot_regression_coefficients(model, feature_names):
    coef = model.coef_
    fig = px.bar(x=feature_names, y=coef,
                 labels={'x': 'Variável', 'y': 'Coeficiente'},
                 title="Coeficientes da Regressão Linear")
    return fig

def calculate_linear_regression_benchmark_probabilities(y_true, predictions):
    mean_val = np.mean(y_true)
    median_val = np.median(y_true)
    mode_series = pd.Series(y_true).mode()
    mode_val = mode_series.iloc[0] if not mode_series.empty else None
    max_val = np.max(y_true)
    min_val = np.min(y_true)
    benchmarks = {'mean': mean_val, 'median': median_val, 'mode': mode_val, 'max': max_val, 'min': min_val}
    probabilities = {}
    for key, value in benchmarks.items():
        probabilities['above_' + key] = np.mean(predictions > value)
        probabilities['below_' + key] = np.mean(predictions < value)
    return {'benchmarks': benchmarks, 'probabilities': probabilities}

#####################################
# 3. REGRESSÃO LOGÍSTICA – CLASSIFICAÇÃO BINÁRIA
#####################################
def logistic_regression_model(df, features, target):
    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)
    return model, X_test, y_test, proba, preds

def plot_confusion_matrix(y_test, preds):
    cm = confusion_matrix(y_test, preds)
    fig = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                    labels=dict(x="Previsto", y="Real", color="Contagem"),
                    x=["0", "1"], y=["0", "1"],
                    title="Matriz de Confusão")
    return fig

def plot_roc_curve(y_test, proba):
    fpr, tpr, _ = roc_curve(y_test, proba)
    auc = roc_auc_score(y_test, proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.2f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Linha de Base', line=dict(dash='dash')))
    fig.update_layout(title="Curva ROC",
                      xaxis_title="Taxa de Falso Positivo",
                      yaxis_title="Taxa de Verdadeiro Positivo")
    return fig

def plot_logistic_coefficients(model, feature_names):
    coef = model.coef_[0]
    fig = px.bar(x=feature_names, y=coef,
                 labels={'x': 'Variável', 'y': 'Coeficiente'},
                 title="Coeficientes da Regressão Logística")
    return fig

def calculate_logistic_benchmark_probabilities(proba):
    pred_series = pd.Series(proba)
    mean_val = np.mean(proba)
    median_val = np.median(proba)
    mode_series = pred_series.mode()
    mode_val = mode_series.iloc[0] if not mode_series.empty else None
    max_val = np.max(proba)
    min_val = np.min(proba)
    benchmarks = {'mean': mean_val, 'median': median_val, 'mode': mode_val, 'max': max_val, 'min': min_val}
    probabilities = {}
    for key, value in benchmarks.items():
        probabilities['above_' + key] = np.mean(proba > value)
        probabilities['below_' + key] = np.mean(proba < value)
    return {'benchmarks': benchmarks, 'probabilities': probabilities}

#####################################
# 4. GAM / GAMLSS – MODELO GAMA (USANDO PYGAM)
#####################################
def gamlss_model(df, features, target, model_type="poisson"):
    print("Iniciando pré-processamento para GAM...", flush=True)
    X = df[features].values
    y = df[target].values
    print("Dividindo dados em treino e teste...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"X_train.shape = {X_train.shape}, X_test.shape = {X_test.shape}", flush=True)
    
    if model_type == "poisson":
        print("Treinando PoissonGAM...", flush=True)
        model = PoissonGAM().fit(X_train, y_train)
    elif model_type == "linear":
        print("Treinando LinearGAM...", flush=True)
        model = LinearGAM().fit(X_train, y_train)
    else:
        raise ValueError("model_type deve ser 'poisson' ou 'linear'")
    
    print("Treinamento do GAM concluído. Gerando predições...", flush=True)
    predictions = model.predict(X_test)
    print("Predições geradas.", flush=True)
    return model, X_test, y_test, predictions

def plot_gam_coefficients(model, feature_names):
    edof = model.statistics_['edof_per_coef']
    aggregated_edof = []
    aggregated_feature_names = []
    # O primeiro termo é o intercepto; pulamos
    start = model.terms[0].n_coefs
    for i in range(1, len(model.terms)):
        term = model.terms[i]
        n_coefs = term.n_coefs
        term_edof = np.sum(edof[start:start+n_coefs])
        aggregated_edof.append(term_edof)
        if i - 1 < len(feature_names):
            aggregated_feature_names.append(feature_names[i - 1])
        else:
            aggregated_feature_names.append(f"Term {i}")
        start += n_coefs
    fig = px.bar(
        x=aggregated_feature_names,
        y=aggregated_edof,
        labels={'x': 'Variável', 'y': 'EDOF'},
        title="Coeficientes do GAM (EDOF) - agregados por variável"
    )
    return fig

# Para GAM, reutilizamos a mesma função de benchmark dos valores contínuos.
calculate_gam_benchmark_probabilities = calculate_linear_regression_benchmark_probabilities

#####################################
# EXEMPLO DE USO – INTEGRANDO OS MODELOS
#####################################
if __name__ == "__main__":
    np.random.seed(42)
    df_exemplo = pd.DataFrame({
        "PTS": np.random.poisson(lam=25, size=200),
        "REB": np.random.poisson(lam=7, size=200),
        "AST": np.random.poisson(lam=5, size=200),
        "Tempo_Quadra": np.random.uniform(20, 40, size=200),
        "Arremessos_Tentados": np.random.uniform(5, 20, size=200),
        "Turnovers": np.random.poisson(lam=2, size=200)
    })

    # 1. Modelo Gumbel
    pts_data = df_exemplo["PTS"]
    loc, scale = fit_gumbel_model(pts_data)
    x_value = 30
    prob_acima = gumbel_probability(x_value, loc, scale, tail='above')
    prob_abaixo = gumbel_probability(x_value, loc, scale, tail='below')
    print(f"Probabilidade de marcar acima de {x_value} pontos: {prob_acima:.2f}")
    print(f"Probabilidade de marcar abaixo de {x_value} pontos: {prob_abaixo:.2f}")
    results_gumbel = calculate_gumbel_benchmark_probabilities(pts_data)
    print("Benchmarks e probabilidades (Gumbel):", results_gumbel)
    fig_gumbel = plot_gumbel_fit(pts_data, loc, scale, metric_name="Pontos")
    fig_gumbel.show()

    # 2. Regressão Linear
    features_lr = ["Tempo_Quadra", "Arremessos_Tentados", "Turnovers"]
    target_lr = "PTS"
    lr_model, X_test_lr, y_test_lr, preds_lr = linear_regression_model(df_exemplo, features_lr, target_lr)
    fig_lr = plot_linear_regression_results(y_test_lr, preds_lr, metric_name="Pontos")
    fig_lr.show()
    fig_lr_coef = plot_regression_coefficients(lr_model, features_lr)
    fig_lr_coef.show()
    results_linear = calculate_linear_regression_benchmark_probabilities(y_test_lr, preds_lr)
    print("Benchmarks e probabilidades (Regressão Linear):", results_linear)

    # 3. Regressão Logística
    median_pts = df_exemplo["PTS"].median()
    df_exemplo["PTS_bin"] = (df_exemplo["PTS"] > median_pts).astype(int)
    features_log = ["Tempo_Quadra", "Arremessos_Tentados", "Turnovers"]
    target_log = "PTS_bin"
    log_model, X_test_log, y_test_log, proba_log, preds_log = logistic_regression_model(df_exemplo, features_log, target_log)
    fig_cm = plot_confusion_matrix(y_test_log, preds_log)
    fig_cm.show()
    fig_roc = plot_roc_curve(y_test_log, proba_log)
    fig_roc.show()
    fig_log_coef = plot_logistic_coefficients(log_model, features_log)
    fig_log_coef.show()
    results_log = calculate_logistic_benchmark_probabilities(proba_log)
    print("Benchmarks e probabilidades (Logística):", results_log)

    # 4. GAM / GAMLSS
    features_gam = ["Tempo_Quadra", "Arremessos_Tentados", "Turnovers"]
    target_gam = "PTS"
    gam_model, X_test_gam, y_test_gam, preds_gam = gamlss_model(df_exemplo, features_gam, target_gam, model_type="poisson")
    fig_gam = go.Figure()
    fig_gam.add_trace(go.Scatter(x=list(range(len(y_test_gam))), y=y_test_gam, mode='markers', name='Real'))
    fig_gam.add_trace(go.Scatter(x=list(range(len(preds_gam))), y=preds_gam, mode='markers', name='Previsto'))
    fig_gam.update_layout(title="GAMLSS (PoissonGAM): Real vs. Previsto para Pontos",
                          xaxis_title="Amostras", yaxis_title="Pontos")
    fig_gam.show()
    fig_gam_coef = plot_gam_coefficients(gam_model, features_gam)
    fig_gam_coef.show()
    results_gam = calculate_linear_regression_benchmark_probabilities(y_test_gam, preds_gam)
    print("Benchmarks e probabilidades (GAM):", results_gam)
