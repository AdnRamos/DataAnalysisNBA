import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

# ---------------------------
# Funções de apoio
# ---------------------------
def carregar_dados_team(seasons, base_dir="data/processed"):
    """
    Carrega os dados dos jogos do time para as temporadas indicadas e retorna um DataFrame concatenado.
    
    Parâmetros:
      seasons (list): Lista de temporadas, por exemplo, ['2023-24', '2024-25'].
      base_dir (str): Diretório base onde os dados processados estão salvos.
      
    Retorna:
      DataFrame com os dados dos jogos concatenados de todas as temporadas.
    """
    dfs = []
    for season in seasons:
        season_safe = season.replace('/', '-')
        file_path = os.path.join(base_dir, season, "team", f"games_{season_safe}_clean.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            # Acrescenta uma coluna com a temporada, se ainda não existir
            if "SEASON" not in df.columns:
                df["SEASON"] = season
            dfs.append(df)
        else:
            print(f"Aviso: Arquivo {file_path} não encontrado.")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def definir_local(df):
    """
    Adiciona uma coluna 'Local' para indicar se o jogo foi em Casa ou Fora com base na coluna MATCHUP.
    Pressupõe que:
      - Se 'MATCHUP' contém "vs.", é jogo em casa (Casa).
      - Se contém "@", é jogo fora de casa (Fora).
    """
    df = df.copy()
    df['Local'] = np.where(df['MATCHUP'].str.contains("vs."), "Casa", "Fora")
    return df

# ---------------------------
# RF3 – Totais de Vitórias/Derrotas
# ---------------------------
def calcular_wins_losses(df):
    """
    Calcula o total de vitórias e derrotas, bem como separados por jogos em Casa e Fora.
    
    Retorna um DataFrame com as seguintes colunas:
      - Total de Vitórias, Vitórias em Casa, Vitórias Fora de Casa,
      - Total de Derrotas, Derrotas em Casa, Derrotas Fora de Casa.
    """
    df = definir_local(df)
    total_wins = (df['WL'] == 'W').sum()
    total_losses = (df['WL'] == 'L').sum()
    
    wins_casa = ((df['WL'] == 'W') & (df['Local'] == "Casa")).sum()
    wins_fora = ((df['WL'] == 'W') & (df['Local'] == "Fora")).sum()
    losses_casa = ((df['WL'] == 'L') & (df['Local'] == "Casa")).sum()
    losses_fora = ((df['WL'] == 'L') & (df['Local'] == "Fora")).sum()
    
    dados = {
        "Total de Vitórias": [total_wins],
        "Vitórias em Casa": [wins_casa],
        "Vitórias Fora de Casa": [wins_fora],
        "Total de Derrotas": [total_losses],
        "Derrotas em Casa": [losses_casa],
        "Derrotas Fora de Casa": [losses_fora]
    }
    df_result = pd.DataFrame(dados)
    # Salva em CSV conforme RF9
    df_result.to_csv("output/resultado_wins_losses.csv", index=False)
    return df_result

# ---------------------------
# RF4 – Totais Gerais de Dados do Time
# ---------------------------
def calcular_totais_gerais(df):
    """
    Calcula médias dos principais indicadores por jogo:
      - Pontos, Assistências, Rebotes, Cestas de 3 convertidas.
    Também inclui as derrotas em Casa e Fora (do RF3).
    """
    df = definir_local(df)
    media_pts = df['PTS'].mean() if 'PTS' in df.columns else np.nan
    media_ast = df['AST'].mean() if 'AST' in df.columns else np.nan
    media_reb = df['REB'].mean() if 'REB' in df.columns else np.nan
    media_fg3m = df['FG3M'].mean() if 'FG3M' in df.columns else np.nan

    # Para derrotas em casa e fora, reutilizamos os cálculos do RF3:
    wins_losses = calcular_wins_losses(df)
    derrotas_casa = wins_losses["Derrotas em Casa"].iloc[0]
    derrotas_fora = wins_losses["Derrotas Fora de Casa"].iloc[0]

    dados = {
        "Média de Pontos por Jogo": [media_pts],
        "Média de Assistências por Jogo": [media_ast],
        "Média de Rebotes por Jogo": [media_reb],
        "Média de Cestas de 3 Convertidas por Jogo": [media_fg3m],
        "Derrotas em Casa": [derrotas_casa],
        "Derrotas Fora de Casa": [derrotas_fora]
    }
    df_totais = pd.DataFrame(dados)
    df_totais.to_csv("output/totais_gerais_time.csv", index=False)
    return df_totais

# ---------------------------
# RF5 – Divisão dos Dados (Ex.: Rebotes, Pontos, 2PT, 3PT, Lance Livre)
# ---------------------------
def calcular_divisao_dados(df):
    """
    Calcula a divisão de alguns dados:
      - Total de Rebotes, Rebotes Ofensivos (OREB) e Defensivos (DREB).
      - Total de Pontos, onde os pontos de 2PT são calculados como (FGM - FG3M)
        e os de 3PT já estão na coluna FG3M.
      - Total de Lance Livre (FTM).
    """
    # Para que os cálculos funcionem, convertemos para numérico (já feito no tratamento, mas garantimos)
    df['FGM'] = pd.to_numeric(df['FGM'], errors='coerce')
    df['FG3M'] = pd.to_numeric(df['FG3M'], errors='coerce')
    df['FTM'] = pd.to_numeric(df['FTM'], errors='coerce')
    df['OREB'] = pd.to_numeric(df['OREB'], errors='coerce')
    df['DREB'] = pd.to_numeric(df['DREB'], errors='coerce')
    df['REB'] = pd.to_numeric(df['REB'], errors='coerce')
    df['PTS'] = pd.to_numeric(df['PTS'], errors='coerce')

    total_reb = df['REB'].sum()
    total_oreb = df['OREB'].sum()
    total_dreb = df['DREB'].sum()
    total_pts = df['PTS'].sum()
    total_fg3 = df['FG3M'].sum()
    total_fg2 = (df['FGM'] - df['FG3M']).sum()  # Pressupõe que FGM inclui todos os cestas de 2PT e 3PT
    total_ft = df['FTM'].sum()

    dados = {
        "Total de Rebotes": [total_reb],
        "Total de Rebotes Ofensivos": [total_oreb],
        "Total de Rebotes Defensivos": [total_dreb],
        "Total de Pontos": [total_pts],
        "Total de Cestas de 2 Pontos": [total_fg2],
        "Total de Cestas de 3 Pontos": [total_fg3],
        "Total de Lance Livre Convertidos": [total_ft]
    }
    df_divisao = pd.DataFrame(dados)
    df_divisao.to_csv("output/divisao_dados_time.csv", index=False)
    return df_divisao

# ---------------------------
# RF6 – Dados de Performance Defensiva
# ---------------------------
def calcular_performance_defensiva(df):
    """
    Calcula médias dos indicadores defensivos por jogo:
      - Roubos de bola (STL), Rebotes Defensivos (DREB),
      - Tocos por jogo (BLK), Erros por jogo (TOV) e Faltas por jogo (PF).
    """
    df['STL'] = pd.to_numeric(df['STL'], errors='coerce')
    df['DREB'] = pd.to_numeric(df['DREB'], errors='coerce')
    df['BLK'] = pd.to_numeric(df['BLK'], errors='coerce')
    df['TOV'] = pd.to_numeric(df['TOV'], errors='coerce')
    df['PF'] = pd.to_numeric(df['PF'], errors='coerce')
    
    media_stl = df['STL'].mean()
    media_dreb = df['DREB'].mean()
    media_blk = df['BLK'].mean()
    media_tov = df['TOV'].mean()
    media_pf = df['PF'].mean()
    
    dados = {
        "Média de Roubos de Bola por Jogo": [media_stl],
        "Média de Rebotes Defensivos por Jogo": [media_dreb],
        "Média de Tocos por Jogo": [media_blk],
        "Média de Erros por Jogo": [media_tov],
        "Média de Faltas por Jogo": [media_pf]
    }
    df_defensiva = pd.DataFrame(dados)
    df_defensiva.to_csv("output/performance_defensiva_time.csv", index=False)
    return df_defensiva

# ---------------------------
# RF7 – Tabela de Jogos do Time
# ---------------------------
def gerar_tabela_jogos(df):
    """
    Cria uma tabela com:
      - Data do jogo, Adversário, Resultado (Vitória ou Derrota), Local (Casa ou Fora) e Placar.
    Para 'Adversário', extrai da coluna MATCHUP o nome do time adversário.
    Para 'Placar', utiliza a coluna PTS (como os pontos do time); se houver coluna com o placar do adversário, 
    pode-se incluir essa informação.
    """
    df = definir_local(df)
    # Extrair o adversário a partir da coluna MATCHUP
    def extrair_adversario(matchup):
        if "vs." in matchup:
            return matchup.split("vs.")[1].strip()
        elif "@" in matchup:
            return matchup.split("@")[1].strip()
        else:
            return np.nan
    
    df["Adversario"] = df["MATCHUP"].apply(extrair_adversario)
    # Aqui, usaremos o placar do time (PTS); se houver placar do adversário, poderá ser concatenado.
    df["Placar"] = df["PTS"].apply(lambda pts: f"{pts} - ?")  # "?" indica que a informação do adversário não está disponível
    
    tabela_jogos = df[["GAME_DATE", "Adversario", "WL", "Local", "Placar"]].copy()
    tabela_jogos.to_csv("output/tabela_jogos_time.csv", index=False)
    return tabela_jogos

# ---------------------------
# RF8 – Geração de Gráficos (Exemplos)
# ---------------------------
def grafico_barras_empilhado(wins_losses_df):
    """
    Gráfico de Barras Empilhado para Vitórias (verde) e Derrotas (vermelho).
    """
    categorias = ["Vitórias", "Derrotas"]
    valores = [wins_losses_df["Total de Vitórias"].iloc[0], wins_losses_df["Total de Derrotas"].iloc[0]]
    
    fig = go.Figure(data=[
        go.Bar(name='Vitórias', x=["Total"], y=[valores[0]], marker_color='green'),
        go.Bar(name='Derrotas', x=["Total"], y=[valores[1]], marker_color='red')
    ])
    fig.update_layout(barmode='stack', title="Vitórias vs. Derrotas (Empilhado)")
    fig.write_html("output/grafico_barras_empilhado.html")
    return fig

def grafico_barras_agrupado(wins_losses_df):
    """
    Gráfico de Barras Agrupado para Vitórias em Casa (verde), Vitórias Fora (azul),
    Derrotas em Casa (vermelho) e Derrotas Fora (marrom).
    """
    categorias = ["Vitórias Casa", "Vitórias Fora", "Derrotas Casa", "Derrotas Fora"]
    valores = [
        wins_losses_df["Vitórias em Casa"].iloc[0],
        wins_losses_df["Vitórias Fora de Casa"].iloc[0],
        wins_losses_df["Derrotas em Casa"].iloc[0],
        wins_losses_df["Derrotas Fora de Casa"].iloc[0]
    ]
    cores = ['green', 'blue', 'red', 'brown']
    fig = go.Figure(data=[
        go.Bar(name=cat, x=["Total"], y=[val], marker_color=cor)
        for cat, val, cor in zip(categorias, valores, cores)
    ])
    fig.update_layout(barmode='group', title="Vitórias e Derrotas - Casa vs. Fora")
    fig.write_html("output/grafico_barras_agrupado.html")
    return fig

def grafico_histograma(df):
    """
    Gráfico Histograma da frequência de vitórias e derrotas.
    """
    fig = px.histogram(df, x="WL", color="WL", 
                       color_discrete_map={'W':'green', 'L':'red'},
                       title="Frequência de Vitórias e Derrotas")
    fig.write_html("output/grafico_histograma.html")
    return fig

def grafico_pizza(wins_losses_df):
    """
    Gráfico de Setor (Pizza) para o percentual de Vitórias em Casa, Vitórias Fora, Derrotas em Casa e Derrotas Fora.
    """
    labels = ["Vitórias Casa", "Vitórias Fora", "Derrotas Casa", "Derrotas Fora"]
    values = [
        wins_losses_df["Vitórias em Casa"].iloc[0],
        wins_losses_df["Vitórias Fora de Casa"].iloc[0],
        wins_losses_df["Derrotas em Casa"].iloc[0],
        wins_losses_df["Derrotas Fora de Casa"].iloc[0]
    ]
    fig = px.pie(names=labels, values=values, title="Distribuição de Resultados por Local")
    fig.write_html("output/grafico_pizza.html")
    return fig

def grafico_radar(df):
    """
    Gráfico de Radar exibindo a média de pontos marcados e (se disponível) sofridos nos jogos em Casa e Fora.
    Como não dispomos dos pontos sofridos, usaremos apenas os pontos marcados, agrupados por Local.
    """
    df = definir_local(df)
    media_pts = df.groupby("Local")["PTS"].mean().reset_index()
    # Ajusta para o gráfico radar (precisamos de categorias iguais)
    categorias = media_pts["Local"].tolist()
    valores = media_pts["PTS"].tolist()
    # Criando gráfico radar com Plotly Graph Objects
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=categorias,
        fill='toself',
        name='Pontos Médios'
    ))
    fig.update_layout(
      polar=dict(
        radialaxis=dict(visible=True)
      ),
      showlegend=False,
      title="Média de Pontos por Jogo - Casa vs. Fora"
    )
    fig.write_html("output/grafico_radar.html")
    return fig

def grafico_linha_sequencia(df):
    """
    Gráfico de Linhas exibindo a sequência de vitórias e derrotas ao longo da temporada.
    Para isso, usamos a data do jogo e mapeamos 'W' para 1 e 'L' para 0.
    """
    df = df.copy()
    df["Resultado_Num"] = df["WL"].map({'W': 1, 'L': 0})
    df = df.sort_values("GAME_DATE")
    fig = px.line(df, x="GAME_DATE", y="Resultado_Num", markers=True,
                  title="Sequência de Vitórias (1) e Derrotas (0) ao Longo da Temporada")
    fig.write_html("output/grafico_linha_sequencia.html")
    return fig

def grafico_custom_defensivo(df):
    """
    Gráfico de Dispersão para os dados de performance defensiva (RF6)
    utilizando, por exemplo, a média de roubos de bola e tocos (BLK).
    """
    df["STL"] = pd.to_numeric(df["STL"], errors='coerce')
    df["BLK"] = pd.to_numeric(df["BLK"], errors='coerce')
    media_stl = df["STL"].mean()
    media_blk = df["BLK"].mean()
    dados = pd.DataFrame({
        "Indicador": ["Roubos de Bola", "Tocos"],
        "Média": [media_stl, media_blk]
    })
    fig = px.bar(dados, x="Indicador", y="Média", color="Indicador", 
                 title="Média de Roubos de Bola e Tocos por Jogo", color_discrete_sequence=["purple", "orange"])
    fig.write_html("output/grafico_custom_defensivo.html")
    return fig

def carregar_dados_conferencia(base_dir="data/raw"):
    """
    Carrega os dados de classificação e times por conferência e retorna DataFrames.
    
    Parâmetros:
      base_dir (str): Diretório base onde os dados brutos estão salvos.
      
    Retorna:
      Tuple de DataFrames: (df_classificacao, df_times)
    """
    classificacao_path = os.path.join(base_dir, "classificacao_por_conferencia.csv")
    times_path = os.path.join(base_dir, "times_por_conferencia.csv")

    df_classificacao = pd.read_csv(classificacao_path)
    df_times = pd.read_csv(times_path)
    
    return df_classificacao, df_times

def grafico_classificacao_conferencia(df_classificacao):
    """
    Gera um gráfico de barras com a classificação dos times por conferência.
    
    Parâmetros:
      df_classificacao (DataFrame): Dados de classificação dos times.
    
    Retorna:
      fig: Objeto de figura do Plotly.
    """
    fig = px.bar(df_classificacao, 
                 x='TeamName', 
                 y='WinPCT', 
                 color='Conferencia', 
                 title='Classificação dos Times por Conferência',
                 labels={'WinPCT': 'Percentual de Vitórias', 'TeamName': 'Times'})
    return fig
# ---------------------------
# Função Principal para Gerar Relatórios e Gráficos (RF3 a RF10)
# ---------------------------

# Lista das temporadas
seasons = ['2023-24', '2024-25']
# Carregar os dados dos jogos do time
df_team = carregar_dados_team(seasons)

if df_team.empty:
    print("Nenhum dado de jogos do time foi carregado. Verifique os arquivos processados.")


# Converter GAME_DATE para datetime (caso não esteja)
df_team['GAME_DATE'] = pd.to_datetime(df_team['GAME_DATE'], errors='coerce')

# --- RF3: Totais de Vitórias e Derrotas ---
wins_losses_df = calcular_wins_losses(df_team)
print("RF3 - Totais de Vitórias e Derrotas:")
print(wins_losses_df)

# --- RF4: Totais Gerais de Dados do Time ---
totais_df = calcular_totais_gerais(df_team)
print("RF4 - Totais Gerais:")
print(totais_df)

# --- RF5: Divisão dos Dados ---
divisao_df = calcular_divisao_dados(df_team)
print("RF5 - Divisão dos Dados:")
print(divisao_df)

# --- RF6: Performance Defensiva ---
defensiva_df = calcular_performance_defensiva(df_team)
print("RF6 - Performance Defensiva:")
print(defensiva_df)

# --- RF7: Tabela de Jogos ---
tabela_jogos = gerar_tabela_jogos(df_team)
print("RF7 - Tabela de Jogos:")
print(tabela_jogos.head())

# --- RF8: Gráficos ---
# grafico_barras_empilhado(wins_losses_df)
# grafico_barras_agrupado(wins_losses_df)
# grafico_histograma(df_team)
# grafico_pizza(wins_losses_df)
# grafico_radar(df_team)
# grafico_linha_sequencia(df_team)
# grafico_custom_defensivo(df_team)
# app = dash.Dash(__name__)
# app.title = "Dashboard NBA"

# # Layout do app com abas
# app.layout = html.Div([
#     html.H1("Análise de Desempenho do Time - NBA", style={'textAlign': 'center'}),
    
#     dcc.Tabs(id="tabs", value='tab1', children=[
#         dcc.Tab(label='Vitórias e Derrotas', value='tab1'),
#         dcc.Tab(label='Distribuição de Resultados', value='tab2'),
#         dcc.Tab(label='Média de Pontos', value='tab3'),
#         dcc.Tab(label='Vitórias/Derrotas por Local', value='tab4'),
#         dcc.Tab(label='Histograma de Resultados', value='tab5'),
#         dcc.Tab(label='Sequência de Vitórias e Derrotas', value='tab6'),
#         dcc.Tab(label='Performance Defensiva', value='tab7')
#     ]),
    
#     html.Div(id='tabs-content')
# ])

# @app.callback(
#     Output('tabs-content', 'children'),
#     Input('tabs', 'value')
# )
# def render_content(tab):
#     if tab == 'tab1':
#         return dcc.Graph(figure=grafico_barras_empilhado(wins_losses_df))
#     elif tab == 'tab2':
#         return dcc.Graph(figure=grafico_pizza(wins_losses_df))
#     elif tab == 'tab3':
#         return dcc.Graph(figure=grafico_radar(df_team))
#     elif tab == 'tab4':
#         return dcc.Graph(figure=grafico_barras_agrupado(wins_losses_df))
#     elif tab == 'tab5':
#         return dcc.Graph(figure=grafico_histograma(df_team))
#     elif tab == 'tab6':
#         return dcc.Graph(figure=grafico_linha_sequencia(df_team))
#     elif tab == 'tab7':
#         return dcc.Graph(figure=grafico_custom_defensivo(df_team))

# Os arquivos CSV já foram salvos em cada função (RF9)
# Os gráficos foram salvos em HTML e abertos no browser (RF10)

if __name__ == '__main__':
    # Cria o diretório de output se não existir
    os.makedirs("output", exist_ok=True)
    
    # app.run_server(debug=True)
