import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import parte_1_run as parte1
import parte_2_run as parte2
import parte_3_run as parte3   # Certifique-se de que o módulo parte_3_run.py NÃO executa o bloco __main__
import data_colection as data_collect
import data_processed as data_process
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go

print("Coletando e processando dados...")
data_collect.main()
data_process.main()

app = dash.Dash(__name__, suppress_callback_exceptions=True)

# Carrega dados da Parte 1
df_team = parte1.carregar_dados_team(['2023-24', '2024-25'])
wins_losses_df = parte1.calcular_wins_losses(df_team)
df_classificacao, df_times = parte1.carregar_dados_conferencia()

# Carrega dados da Parte 2
df_players = parte2.load_all_players_data(['2023-24', '2024-25'])
df_processed_players = parte2.process_player_log(df_players)
df_metrics = parte2.aggregate_player_metrics(df_processed_players)

melted_df_metrics = pd.melt(df_metrics, id_vars=['Player_Name'], 
                            value_vars=['Média_PTS', 'Média_REB', 'Média_AST'], 
                            var_name='Métrica', value_name='Valor').dropna()

def calcular_porcentagens(df, coluna, referencia):
    total_jogos = len(df)
    abaixo_referencia = len(df[df[coluna] < referencia])
    return (abaixo_referencia / total_jogos) * 100

app.layout = html.Div([
    dcc.Tabs(id="tabs", value='parte1', children=[
        dcc.Tab(label='Análise do Time (Parte 1)', value='parte1'),
        dcc.Tab(label='Análise dos Jogadores (Parte 2)', value='parte2'),
        dcc.Tab(label='Modelos Estatísticos (Parte 3)', value='parte3')
    ]),
    html.Div(id='tabs-content'),
    html.Button('Salvar Dados em CSV', id='save-button', n_clicks=0),
    html.Div(id='save-status')
])

@app.callback(
    Output('tabs-content', 'children'),
    [Input('tabs', 'value')]
)
def update_tab(tab):
    if tab == 'parte1':
        return html.Div([
            html.H2("Análise do Time"),
            dcc.Tabs(id="tabs-parte1", value='tab1', children=[
                dcc.Tab(label='Vitórias e Derrotas', value='tab1'),
                dcc.Tab(label='Distribuição de Resultados', value='tab2'),
                dcc.Tab(label='Média de Pontos', value='tab3'),
                dcc.Tab(label='Vitórias/Derrotas por Local', value='tab4'),
                dcc.Tab(label='Histograma de Resultados', value='tab5'),
                dcc.Tab(label='Sequência de Vitórias e Derrotas', value='tab6'),
                dcc.Tab(label='Performance Defensiva', value='tab7'),
                dcc.Tab(label='Classificação dos Times', value='tab8'),
            ]),
            html.Div(id='tabs-content-parte1')
        ])
    elif tab == 'parte2':
        return html.Div([
            html.H2("Análise dos Jogadores"),
            dcc.Tabs(id="tabs-parte2", value='tab1', children=[
                dcc.Tab(label='Métricas dos Jogadores', value='tab1'),
                dcc.Tab(label='Gráficos de Desempenho', value='tab2'),
                dcc.Tab(label='Comparativo Carreira vs Temporada', value='tab3'),
                dcc.Tab(label='Jogos Específicos', value='tab4')
            ]),
            html.Div(id='tabs-content-parte2')
        ])
    elif tab == 'parte3':
        return html.Div([
            html.H2("Modelos Estatísticos e Preditivos (Parte 3)"),
            dcc.Tabs(id="tabs-parte3", value='gumbel', children=[
                dcc.Tab(label="Gumbel", value="gumbel"),
                dcc.Tab(label="Regressão Linear", value="linear"),
                dcc.Tab(label="Regressão Logística", value="logistica"),
                dcc.Tab(label="GAMLSS (GAM)", value="gam")
            ]),
            html.Div(id="tabs-content-parte3")
        ])

@app.callback(
    Output('tabs-content-parte1', 'children'),
    [Input('tabs-parte1', 'value')]
)
def update_parte1(tab):
    if tab == 'tab1':
        return dcc.Graph(figure=parte1.grafico_barras_empilhado(wins_losses_df))
    elif tab == 'tab2':
        return dcc.Graph(figure=parte1.grafico_pizza(wins_losses_df))
    elif tab == 'tab3':
        return dcc.Graph(figure=parte1.grafico_radar(df_team))
    elif tab == 'tab4':
        return dcc.Graph(figure=parte1.grafico_barras_agrupado(wins_losses_df))
    elif tab == 'tab5':
        return dcc.Graph(figure=parte1.grafico_histograma(df_team))
    elif tab == 'tab6':
        return dcc.Graph(figure=parte1.grafico_linha_sequencia(df_team))
    elif tab == 'tab7':
        return dcc.Graph(figure=parte1.grafico_custom_defensivo(df_team))
    elif tab == 'tab8':
        return dcc.Graph(figure=parte1.grafico_classificacao_conferencia(df_classificacao))

@app.callback(
    Output('tabs-content-parte2', 'children'),
    [Input('tabs-parte2', 'value')]
)
def update_parte2(tab):
    if tab == 'tab1':
        porcentagens = []
        for player in df_processed_players['Player_Name'].unique():
            df_player = df_processed_players[df_processed_players['Player_Name'] == player]
            media_pts = df_player['PTS'].mean()
            mediana_pts = df_player['PTS'].median()
            moda_pts = df_player['PTS'].mode().iloc[0] if not df_player['PTS'].mode().empty else np.nan
            porcentagem_media = calcular_porcentagens(df_player, 'PTS', media_pts)
            porcentagem_mediana = calcular_porcentagens(df_player, 'PTS', mediana_pts)
            porcentagem_moda = calcular_porcentagens(df_player, 'PTS', moda_pts)
            porcentagens.append({
                'Player_Name': player,
                'Porcentagem_Abaixo_Media_PTS': porcentagem_media,
                'Porcentagem_Abaixo_Mediana_PTS': porcentagem_mediana,
                'Porcentagem_Abaixo_Moda_PTS': porcentagem_moda
            })
        porcentagem_df = pd.DataFrame(porcentagens)
        return html.Div([
            html.H3("Métricas dos Jogadores"),
            dcc.Graph(figure=px.bar(melted_df_metrics, x='Player_Name', y='Valor', color='Métrica',
                                    title='Métricas de Desempenho por Jogador', barmode='group')),
            html.H3("Porcentagem de Jogos Abaixo da Média, Mediana e Moda de Pontos"),
            dcc.Graph(figure=px.bar(porcentagem_df, x='Player_Name', 
                                    y=['Porcentagem_Abaixo_Media_PTS', 'Porcentagem_Abaixo_Mediana_PTS', 'Porcentagem_Abaixo_Moda_PTS'],
                                    title='Porcentagem de Jogos com Desempenho Abaixo da Média, Mediana e Moda',
                                    barmode='group'))
        ])
    elif tab == 'tab2':
        return html.Div([
            html.H3("Gráficos de Desempenho"),
            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': player, 'value': player} for player in df_processed_players['Player_Name'].unique()],
                value=df_processed_players['Player_Name'].unique()[0]
            ),
            dcc.Tabs(id="tabs-graficos", value='pts', children=[
                dcc.Tab(label='Distribuição de Pontos', value='pts'),
                dcc.Tab(label='Distribuição de Rebotes', value='reb'),
                dcc.Tab(label='Distribuição de Assistências', value='ast')
            ]),
            dcc.RadioItems(
                id='graph-type',
                options=[
                    {'label': 'Histograma', 'value': 'hist'},
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Time Line', 'value': 'line'}
                ],
                value='hist',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            html.Div(id='graficos-content')
        ])
    elif tab == 'tab3':
        comparativo_df = df_metrics.copy()
        comparativo_df['Carreira_PTS'] = comparativo_df['Média_PTS'] * comparativo_df['Total_Jogos']
        return html.Div([
            html.H3("Comparativo Carreira vs Temporada Atual"),
            dcc.Graph(figure=px.bar(comparativo_df, x='Player_Name', y=['Carreira_PTS', 'Média_PTS'],
                                    title='Comparativo de Pontos: Carreira vs Temporada Atual', barmode='group'))
        ])
    elif tab == 'tab4':
        return html.Div([
            html.H3("Filtragem de Jogos Específicos"),
            dcc.Dropdown(
                id='team-dropdown',
                options=[{'label': team, 'value': team} for team in df_processed_players['Adversário'].unique()],
                value=df_processed_players['Adversário'].unique()[0]
            ),
            dcc.RadioItems(
                id='location-filter',
                options=[
                    {'label': 'Todos', 'value': 'Todos'},
                    {'label': 'Casa', 'value': 'Casa'},
                    {'label': 'Fora', 'value': 'Fora'}
                ],
                value='Todos',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            html.Div(id='filtered-games-content')
        ])

@app.callback(
    Output('filtered-games-content', 'children'),
    [Input('team-dropdown', 'value'), Input('location-filter', 'value')]
)
def update_filtered_games(selected_team, location_filter):
    filtered_df = df_processed_players[df_processed_players['Adversário'] == selected_team]
    if location_filter != 'Todos':
        filtered_df = filtered_df[filtered_df['Casa/Fora'] == location_filter]
    jogos_casa = len(filtered_df[filtered_df['Casa/Fora'] == 'Casa'])
    jogos_fora = len(filtered_df[filtered_df['Casa/Fora'] == 'Fora'])
    return html.Div([
        html.H4(f"Jogos contra {selected_team}"),
        html.P(f"Total de Jogos em Casa: {jogos_casa}"),
        html.P(f"Total de Jogos Fora: {jogos_fora}"),
        dcc.Graph(figure=px.bar(filtered_df, x='Data do Jogo', y='PTS', color='Player_Name', title='Pontos por Jogo')),
        dcc.Graph(figure=px.bar(filtered_df, x='Data do Jogo', y='REB', color='Player_Name', title='Rebotes por Jogo')),
        dcc.Graph(figure=px.bar(filtered_df, x='Data do Jogo', y='AST', color='Player_Name', title='Assistências por Jogo'))
    ])

@app.callback(
    Output('graficos-content', 'children'),
    [Input('tabs-graficos', 'value'), Input('player-dropdown', 'value'), Input('graph-type', 'value')]
)
def update_graficos(metric, player, graph_type):
    df_player = df_processed_players[df_processed_players['Player_Name'] == player]
    if graph_type == 'hist':
        if metric == 'pts':
            fig = parte2.generate_histogram(df_player, 'PTS', f'Distribuição de Pontos - {player}', 'output/charts/pts_histogram.html')
        elif metric == 'reb':
            fig = parte2.generate_histogram(df_player, 'REB', f'Distribuição de Rebotes - {player}', 'output/charts/reb_histogram.html')
        elif metric == 'ast':
            fig = parte2.generate_histogram(df_player, 'AST', f'Distribuição de Assistências - {player}', 'output/charts/ast_histogram.html')
    elif graph_type == 'box':
        if metric == 'pts':
            fig = parte2.generate_boxplot(df_player, 'PTS', f'Box Plot de Pontos - {player}', 'output/charts/pts_boxplot.html')
        elif metric == 'reb':
            fig = parte2.generate_boxplot(df_player, 'REB', f'Box Plot de Rebotes - {player}', 'output/charts/reb_boxplot.html')
        elif metric == 'ast':
            fig = parte2.generate_boxplot(df_player, 'AST', f'Box Plot de Assistências - {player}', 'output/charts/ast_boxplot.html')
    elif graph_type == 'line':
        if metric == 'pts':
            fig = parte2.generate_time_series(df_player, player, 'PTS', 'output/charts/pts_time_series.html')
        elif metric == 'reb':
            fig = parte2.generate_time_series(df_player, player, 'REB', 'output/charts/reb_time_series.html')
        elif metric == 'ast':
            fig = parte2.generate_time_series(df_player, player, 'AST', 'output/charts/ast_time_series.html')
    return dcc.Graph(figure=fig)

@app.callback(
    Output('save-status', 'children'),
    [Input('save-button', 'n_clicks')]
)
def save_data(n_clicks):
    if n_clicks > 0:
        save_all_data_to_csv(df_processed_players, df_metrics, df_team)
        return html.P('Dados salvos com sucesso!', style={'color': 'green'})
    return ''

def save_all_data_to_csv(df_players, df_metrics, df_team):
    df_players.to_csv('output/players_data.csv', index=False)
    df_metrics.to_csv('output/players_metrics.csv', index=False)
    df_team.to_csv('output/team_data.csv', index=False)
    print("Todos os dados foram salvos em arquivos CSV.")

@app.callback(
    Output('tabs-content-parte3', 'children'),
    Input('tabs-parte3', 'value')
)
def update_parte3(sub_tab):
    # Dataset de exemplo – substitua por seus dados reais se necessário.
    np.random.seed(42)
    df_exemplo = pd.DataFrame({
        "PTS": np.random.poisson(lam=25, size=200),
        "REB": np.random.poisson(lam=7, size=200),
        "AST": np.random.poisson(lam=5, size=200),
        "Tempo_Quadra": np.random.uniform(20, 40, size=200),
        "Arremessos_Tentados": np.random.uniform(5, 20, size=200),
        "Turnovers": np.random.poisson(lam=2, size=200)
    })
    features = ["Tempo_Quadra", "Arremessos_Tentados", "Turnovers"]

    if sub_tab == 'gumbel':
        pts_data = df_exemplo["PTS"]
        loc, scale = parte3.fit_gumbel_model(pts_data)
        x_value = 30  # valor de referência
        prob_acima = parte3.gumbel_probability(x_value, loc, scale, tail='above')
        prob_abaixo = parte3.gumbel_probability(x_value, loc, scale, tail='below')
        fig_gumbel = parte3.plot_gumbel_fit(pts_data, loc, scale, metric_name="Pontos")
        results_gumbel = parte3.calculate_gumbel_benchmark_probabilities(pts_data)
        benchmarks = results_gumbel['benchmarks']
        probs = results_gumbel['probabilities']
        prob_text = []
        for key, value in benchmarks.items():
            prob_text.append(html.P(f"{key.capitalize()}: {value:.2f}"))
        for key, value in probs.items():
            prob_text.append(html.P(f"Probabilidade {key.replace('_', ' ')}: {value*100:.2f}%"))
        return html.Div([
            html.H3("Modelo Gumbel"),
            html.Div(prob_text),
            dcc.Graph(figure=fig_gumbel)
        ])
    elif sub_tab == 'linear':
        lr_model, X_test_lr, y_test_lr, preds_lr = parte3.linear_regression_model(df_exemplo, features, "PTS")
        fig_lr = parte3.plot_linear_regression_results(y_test_lr, preds_lr, metric_name="Pontos")
        fig_lr_coef = parte3.plot_regression_coefficients(lr_model, features)
        results_linear = parte3.calculate_linear_regression_benchmark_probabilities(y_test_lr, preds_lr)
        benchmarks_lin = results_linear['benchmarks']
        probs_lin = results_linear['probabilities']
        prob_text_lin = []
        for key, value in benchmarks_lin.items():
            prob_text_lin.append(html.P(f"{key.capitalize()}: {value:.2f}"))
        for key, value in probs_lin.items():
            prob_text_lin.append(html.P(f"Probabilidade {key.replace('_', ' ')}: {value*100:.2f}%"))
        return html.Div([
            html.H3("Regressão Linear"),
            html.Div(prob_text_lin),
            dcc.Graph(figure=fig_lr),
            dcc.Graph(figure=fig_lr_coef)
        ])
    elif sub_tab == 'logistica':
        median_pts = df_exemplo["PTS"].median()
        df_exemplo["PTS_bin"] = (df_exemplo["PTS"] > median_pts).astype(int)
        log_model, X_test_log, y_test_log, proba_log, preds_log = parte3.logistic_regression_model(df_exemplo, features, "PTS_bin")
        fig_cm = parte3.plot_confusion_matrix(y_test_log, preds_log)
        fig_roc = parte3.plot_roc_curve(y_test_log, proba_log)
        fig_log_coef = parte3.plot_logistic_coefficients(log_model, features)
        # Calcula benchmarks e probabilidades para o modelo logístico (usando as probabilidades preditas)
        pred_series = pd.Series(proba_log)
        mean_log = np.mean(proba_log)
        median_log = np.median(proba_log)
        mode_series = pred_series.mode()
        mode_log = mode_series.iloc[0] if not mode_series.empty else None
        max_log = np.max(proba_log)
        min_log = np.min(proba_log)
        benchmarks_log = {'mean': mean_log, 'median': median_log, 'mode': mode_log, 'max': max_log, 'min': min_log}
        probabilities_log = {
            'above_mean': np.mean(proba_log > mean_log),
            'below_mean': np.mean(proba_log < mean_log),
            'above_median': np.mean(proba_log > median_log),
            'below_median': np.mean(proba_log < median_log),
            'above_mode': np.mean(proba_log > mode_log) if mode_log is not None else None,
            'below_mode': np.mean(proba_log < mode_log) if mode_log is not None else None,
            'above_max': 0.0,
            'below_min': 1.0
        }
        results_log = {'benchmarks': benchmarks_log, 'probabilities': probabilities_log}
        prob_text_log = []
        for key, value in benchmarks_log.items():
            prob_text_log.append(html.P(f"{key.capitalize()}: {value:.2f}"))
        for key, value in probabilities_log.items():
            prob_text_log.append(html.P(f"Probabilidade {key.replace('_', ' ')}: {value*100:.2f}%"))
        return html.Div([
            html.H3("Regressão Logística"),
            html.Div(prob_text_log),
            dcc.Graph(figure=fig_cm),
            dcc.Graph(figure=fig_roc),
            dcc.Graph(figure=fig_log_coef)
        ])
    elif sub_tab == 'gam':
        try:
            print("Iniciando treinamento do modelo GAM (PoissonGAM)...", flush=True)
            gam_model, X_test_gam, y_test_gam, preds_gam = parte3.gamlss_model(df_exemplo, features, "PTS", model_type="poisson")
            print("Treinamento do GAM concluído.", flush=True)
            
            print("Criando gráfico de Real vs. Previsto para o GAM...", flush=True)
            fig_gam = go.Figure()
            fig_gam.add_trace(go.Scatter(x=list(range(len(y_test_gam))), y=y_test_gam, mode='markers', name='Real'))
            fig_gam.add_trace(go.Scatter(x=list(range(len(preds_gam))), y=preds_gam, mode='markers', name='Previsto'))
            fig_gam.update_layout(title="GAMLSS (PoissonGAM): Real vs. Previsto para Pontos",
                                  xaxis_title="Amostras", yaxis_title="Pontos")
            print("Gráfico de Real vs. Previsto criado.", flush=True)
            
            print("Criando gráfico de coeficientes (EDOF) do GAM...", flush=True)
            fig_gam_coef = parte3.plot_gam_coefficients(gam_model, features)
            print("Gráfico de coeficientes criado.", flush=True)
            
            results_gam = parte3.calculate_linear_regression_benchmark_probabilities(y_test_gam, preds_gam)
            benchmarks_gam = results_gam['benchmarks']
            probs_gam = results_gam['probabilities']
            prob_text_gam = []
            for key, value in benchmarks_gam.items():
                prob_text_gam.append(html.P(f"{key.capitalize()}: {value:.2f}"))
            for key, value in probs_gam.items():
                prob_text_gam.append(html.P(f"Probabilidade {key.replace('_', ' ')}: {value*100:.2f}%"))
            
            print("Retornando conteúdo para a aba GAM.", flush=True)
            return dcc.Loading(
                id="loading-gam",
                type="default",
                children=html.Div([
                    html.H3("GAMLSS (GAM)"),
                    html.Div(prob_text_gam),
                    dcc.Graph(figure=fig_gam),
                    dcc.Graph(figure=fig_gam_coef)
                ])
            )
        except Exception as e:
            print("Erro no processamento do GAM:", e, flush=True)
            return html.Div("Erro no processamento do GAM: " + str(e))

if __name__ == '__main__':
    app.run_server(debug=True)
