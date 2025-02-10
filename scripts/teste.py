import dash
from dash import html, dcc
from dash.dependencies import Input, Output
import parte_1_run as parte1
import parte_2_run as parte2
import data_colection as data_collect
import data_processed as data_process
import pandas as pd
import plotly.express as px
import numpy as np

# Inicializa o app Dash
print("Coletando e processando dados...")
# data_collect.main()
# data_process.main()

app = dash.Dash(__name__)

# Carrega dados da Parte 1
df_team = parte1.carregar_dados_team(['2023-24', '2024-25'])
wins_losses_df = parte1.calcular_wins_losses(df_team)
df_classificacao, df_times = parte1.carregar_dados_conferencia()

# Carrega dados da Parte 2
df_players = parte2.load_all_players_data(['2023-24', '2024-25'])
df_processed_players = parte2.process_player_log(df_players)
df_metrics = parte2.aggregate_player_metrics(df_processed_players)

# Transforma o DataFrame para formato long para o gráfico e remove valores nulos
melted_df_metrics = pd.melt(df_metrics, id_vars=['Player_Name'], 
                            value_vars=['Média_PTS', 'Média_REB', 'Média_AST'], 
                            var_name='Métrica', value_name='Valor').dropna()

# Função para calcular a porcentagem de desempenhos abaixo da média/mediana/moda
def calcular_porcentagens(df, coluna, referencia):
    total_jogos = len(df)
    abaixo_referencia = len(df[df[coluna] < referencia])
    return (abaixo_referencia / total_jogos) * 100

# Layout com abas para Parte 1 e Parte 2
app.layout = html.Div([
    dcc.Tabs(id="tabs", value='parte1', children=[
        dcc.Tab(label='Análise do Time (Parte 1)', value='parte1'),
        dcc.Tab(label='Análise dos Jogadores (Parte 2)', value='parte2'),
    ]),
    html.Div(id='tabs-content')
])

# Callback para atualizar o conteúdo conforme a aba
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

# Callback para Parte 1
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

# Callback para Parte 2
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
            porcentagem_pts = calcular_porcentagens(df_player, 'PTS', media_pts)
            porcentagens.append({'Player_Name': player, 'Porcentagem_Abaixo_Media_PTS': porcentagem_pts})

        porcentagem_df = pd.DataFrame(porcentagens)

        return html.Div([
            html.H3("Métricas dos Jogadores"),
            dcc.Graph(figure=px.bar(melted_df_metrics, x='Player_Name', y='Valor', color='Métrica',
                                    title='Métricas de Desempenho por Jogador', barmode='group')),
            html.H3("Porcentagem de Jogos Abaixo da Média de Pontos"),
            dcc.Graph(figure=px.bar(porcentagem_df, x='Player_Name', y='Porcentagem_Abaixo_Media_PTS',
                                    title='Porcentagem de Jogos com Pontos Abaixo da Média'))
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
                    {'label': 'Box Plot', 'value': 'box'}
                ],
                value='hist',
                labelStyle={'display': 'inline-block', 'margin-right': '10px'}
            ),
            html.Div(id='graficos-content')
        ])
    elif tab == 'tab3':
        comparativo_df = df_metrics.copy()
        comparativo_df['Carreira_PTS'] = comparativo_df['Média_PTS'] * comparativo_df['Total_Jogos']  # Simula dados de carreira
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
            html.Div(id='filtered-games-content')
        ])

# Callback para gráficos detalhados dos jogadores
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
    return dcc.Graph(figure=fig)

# Callback para filtragem de jogos específicos
@app.callback(
    Output('filtered-games-content', 'children'),
    [Input('team-dropdown', 'value')]
)
def update_filtered_games(selected_team):
    filtered_df = df_processed_players[df_processed_players['Adversário'] == selected_team]
    return html.Div([
        html.H4(f"Jogos contra {selected_team}"),
        dcc.Graph(figure=px.bar(filtered_df, x='Data do Jogo', y='PTS', color='Player_Name', title='Pontos por Jogo')),
        dcc.Graph(figure=px.bar(filtered_df, x='Data do Jogo', y='REB', color='Player_Name', title='Rebotes por Jogo')),
        dcc.Graph(figure=px.bar(filtered_df, x='Data do Jogo', y='AST', color='Player_Name', title='Assistências por Jogo'))
    ])

# Roda o app
if __name__ == '__main__':
    app.run_server(debug=True)
