import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
from plotly.subplots import make_subplots

# ---------------------------
# Funções de apoio
# ---------------------------
def carregar_dados_team(seasons, base_dir="scripts/data/processed"):
    """
    Carrega os dados dos jogos do time para as temporadas indicadas e retorna um DataFrame concatenado.
    
    Parâmetros:
      - seasons (list): Lista de temporadas, por exemplo, ['2023-24', '2024-25'].
      - base_dir (str): Diretório base onde os dados processados estão salvos.
      
    Retorna:
      - DataFrame com os dados dos jogos concatenados de todas as temporadas.
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


import numpy as np

def definir_local(df):
    """
    Verifica a coluna de identificação do jogo e define a coluna 'Local' com:
      - "Casa" se o valor contiver "vs." (jogo em casa)
      - "Fora" caso contrário.
    
    A função tenta encontrar a coluna 'MATCHUP' (ou 'matchup') no DataFrame.
    Se nenhuma delas for encontrada, gera um erro com mensagem informativa.
    """
    # Verifica se a coluna 'MATCHUP' ou 'matchup' existe
    if "MATCHUP" in df.columns:
        col = "MATCHUP"
    elif "matchup" in df.columns:
        col = "matchup"
    else:
        raise KeyError("A coluna 'MATCHUP' (ou 'matchup') não foi encontrada no DataFrame. Verifique os nomes das colunas.")
    
    # Cria a coluna 'Local' utilizando np.where
    df['Local'] = np.where(df[col].str.contains("vs."), "Casa", "Fora")
    return df


# ---------------------------
# RF3 – Totais de Vitórias/Derrotas
# ---------------------------
def calcular_wins_losses(df):
    """
    Calcula o total de vitórias e derrotas do time, separando os jogos em casa e fora.
    Se a coluna 'SEASON' existir, agrupa os dados por temporada; caso contrário, calcula um total global.
    
    Retorna um DataFrame com as colunas:
      - Season (renomeado de SEASON, se existir)
      - Total de Vitórias
      - Vitórias em Casa
      - Vitórias Fora de Casa
      - Total de Derrotas
      - Derrotas em Casa
      - Derrotas Fora de Casa
    """
    df = definir_local(df)

    if 'SEASON' in df.columns:
        group_obj = df.groupby('SEASON')
        resultados = []
        for season, group in group_obj:
            total_wins = (group['WL'] == 'W').sum()
            total_losses = (group['WL'] == 'L').sum()
            wins_home = ((group['WL'] == 'W') & (group['Local'] == "Casa")).sum()
            wins_away = ((group['WL'] == 'W') & (group['Local'] == "Fora")).sum()
            losses_home = ((group['WL'] == 'L') & (group['Local'] == "Casa")).sum()
            losses_away = ((group['WL'] == 'L') & (group['Local'] == "Fora")).sum()

            resultados.append({
                "SEASON": season,
                "Total de Vitórias": total_wins,
                "Vitórias em Casa": wins_home,
                "Vitórias Fora de Casa": wins_away,
                "Total de Derrotas": total_losses,
                "Derrotas em Casa": losses_home,
                "Derrotas Fora de Casa": losses_away
            })
        df_result = pd.DataFrame(resultados)
        # Renomeia a coluna 'SEASON' para 'Season'
        df_result.rename(columns={"SEASON": "Season"}, inplace=True)
    else:
        total_wins = (df['WL'] == 'W').sum()
        total_losses = (df['WL'] == 'L').sum()
        wins_home = ((df['WL'] == 'W') & (df['Local'] == "Casa")).sum()
        wins_away = ((df['WL'] == 'W') & (df['Local'] == "Fora")).sum()
        losses_home = ((df['WL'] == 'L') & (df['Local'] == "Casa")).sum()
        losses_away = ((df['WL'] == 'L') & (df['Local'] == "Fora")).sum()

        data = {
            "Season": ["Total"],
            "Total de Vitórias": [total_wins],
            "Vitórias em Casa": [wins_home],
            "Vitórias Fora de Casa": [wins_away],
            "Total de Derrotas": [total_losses],
            "Derrotas em Casa": [losses_home],
            "Derrotas Fora de Casa": [losses_away]
        }
        df_result = pd.DataFrame(data)

    # Salva o resultado em CSV (RF9)
    os.makedirs("scripts/output", exist_ok=True)
    output_path = os.path.join("scripts/output", "resultado_wins_losses.csv")
    df_result.to_csv(output_path, index=False)
    print(f"Vitórias/Derrotas salvos em: {output_path}")

    return df_result

def grafico_barras_empilhado_express(wins_losses_df):
    """
    Gera um gráfico de barras empilhadas para "Total de Vitórias" (verde) e "Total de Derrotas" (vermelho)
    separadas por temporada, utilizando Plotly Express.
    """
    # Converte o DataFrame para o formato "longo"
    df_melt = wins_losses_df.melt(
        id_vars="Season",
        value_vars=["Total de Vitórias", "Total de Derrotas"],
        var_name="Resultado",
        value_name="Quantidade"
    )

    # Define um mapeamento de cores para cada resultado
    color_discrete_map = {
        "Total de Vitórias": "green",
        "Total de Derrotas": "red"
    }

    # Cria o gráfico de barras empilhadas
    fig = px.bar(
        df_melt,
        x="Season",
        y="Quantidade",
        color="Resultado",
        title="Vitórias vs. Derrotas (Empilhado) por Temporada",
        color_discrete_map=color_discrete_map,
        text="Quantidade"
    )

    # Define o modo de barras empilhadas e os títulos dos eixos
    fig.update_layout(
        barmode="stack",
        xaxis_title="Temporada",
        yaxis_title="Quantidade de Jogos"
    )

    # Opcional: salva o gráfico em HTML
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_barras_empilhado_express.html")

    return fig
# ---------------------------
# RF4 – Totais Gerais de Dados do Time
# ---------------------------

def calcular_totais_gerais_por_temporada(df, team_name="MEM"):
    """
    Calcula os totais dos principais indicadores do time para cada temporada, conforme NF4.
    
    Para cada temporada, calcula:
      - Média de Pontos por Jogo (PTS)
      - Média de Assistências por Jogo (AST)
      - Média de Rebotes por Jogo (REB)
      - Média de Cestas de 3 Convertidas por Jogo (FG3M)
      - Derrotas em Casa: total de jogos perdidos em casa
      - Derrotas Fora de Casa: total de jogos perdidos fora de casa
    
    Além disso, adiciona uma coluna com o nome do time e renomeia (abrevia) os títulos das colunas.
    
    Requisitos:
      - O DataFrame 'df' deve conter a coluna "SEASON" indicando a temporada.
      - Também deve conter a coluna "MATCHUP" para que a função definir_local() funcione,
        e "WL" (com "W" e "L") para identificar vitórias e derrotas.
    
    Retorna:
      Um DataFrame com uma linha por temporada e as seguintes colunas:
        - Season
        - Time       (Nome do time)   <-- segunda coluna
        - PTS        (Média de Pontos por Jogo)
        - AST        (Média de Assistências por Jogo)
        - REB        (Média de Rebotes por Jogo)
        - 3PM        (Média de Cestas de 3 Convertidas por Jogo)
        - DCasa      (Derrotas em Casa)
        - DFora      (Derrotas Fora de Casa)
    
    O resultado também é salvo em "scripts/output/totais_gerais_time_por_temporada.csv".
    """
    # Certifique-se de que a coluna "Local" esteja definida (a função definir_local deve estar implementada)
    df = definir_local(df)
    
    # Calcula as médias por temporada para os indicadores "por jogo"
    agg_metrics = df.groupby("SEASON").agg({
        "PTS": "mean",
        "AST": "mean",
        "REB": "mean",
        "FG3M": "mean"
    }).rename(columns={
        "PTS": "M Pontos Jogo",
        "AST": "M Assistências Jogo",
        "REB": "M Rebotes Jogo",
        "FG3M": "M Cestas de 3 Convertidas por Jogo"
    })
    
    # Função auxiliar para calcular as derrotas para cada temporada
    def losses_per_season(group):
        # Usa a função calcular_wins_losses para o subset de cada temporada (deve estar implementada)
        wl = calcular_wins_losses(group)
        return pd.Series({
            "Derrotas em Casa": wl["Derrotas em Casa"].iloc[0],
            "Derrotas Fora de Casa": wl["Derrotas Fora de Casa"].iloc[0]
        })
    
    loss_metrics = df.groupby("SEASON").apply(losses_per_season)
    
    # Junta os resultados (médias e totais de derrotas)
    df_totais = pd.merge(agg_metrics, loss_metrics, left_index=True, right_index=True).reset_index()
    
    # Arredonda as colunas numéricas de médias para 2 casas decimais
    for col in ["M Pontos Jogo", "M Assistências Jogo", "M Rebotes Jogo", "M Cestas de 3 Convertidas por Jogo"]:
        if col in df_totais.columns:
            df_totais[col] = df_totais[col].round(2)
    
    # Adiciona a coluna com o nome do time
    df_totais["Time"] = team_name
    
    # Renomeia (abrevia) as colunas
    rename_cols = {
        "SEASON": "Season",
        "M Pontos Jogo": "PTS",
        "M Assistências Jogo": "AST",
        "M Rebotes Jogo": "REB",
        "M Cestas de 3 Convertidas por Jogo": "3PM",
        "Derrotas em Casa": "DCasa",
        "Derrotas Fora de Casa": "DFora"
    }
    df_totais.rename(columns=rename_cols, inplace=True)
    
    # Reordena as colunas para que a segunda coluna seja "Time"
    desired_order = ["Season", "Time", "PTS", "AST", "REB", "3PM", "DCasa", "DFora"]
    df_totais = df_totais[desired_order]
    
    # Salva o resultado em CSV
    os.makedirs("scripts/output", exist_ok=True)
    output_path = os.path.join("scripts/output", "totais_gerais_time_por_temporada.csv")
    df_totais.to_csv(output_path, index=False)
    print(f"Totais gerais por temporada salvos em: {output_path}")
    
    return df_totais


def plot_totais_gerais_por_temporada(df_totais, team_name="MEM"):
    """
    Plota um gráfico de barras agrupado para os principais indicadores do time por temporada,
    exibindo indicadores ofensivos e defensivos em subplots separados.
    
    Indicadores:
      - Ofensivos: PTS (Média de Pontos por Jogo), AST (Média de Assistências por Jogo),
                   REB (Média de Rebotes por Jogo) e 3PM (Média de Cestas de 3 Convertidas por Jogo)
      - Defensivos: DCasa (Derrotas em Casa) e DFora (Derrotas Fora de Casa)
    
    Parâmetros:
      df_totais (DataFrame): DataFrame com os totais gerais por temporada, contendo as colunas:
                             "Season", "Time", "PTS", "AST", "REB", "3PM", "DCasa" e "DFora".
      team_name (str): Nome do time, que será exibido no título do gráfico.
      
    Retorna:
      Objeto figura do Plotly.
    """
    # Se a coluna "Time" não existir, adiciona-a com o valor de team_name.
    if "Time" not in df_totais.columns:
        df_totais["Time"] = team_name

    # Cria subplots: 2 linhas x 1 coluna
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=["Indicadores Ofensivos", "Indicadores Defensivos"],
        vertical_spacing=0.15
    )

    # Grupo de indicadores ofensivos
    offensive_indicators = ["PTS", "AST", "REB", "3PM"]
    for indicador in offensive_indicators:
        fig.add_trace(
            go.Bar(
                x=df_totais["Season"],
                y=df_totais[indicador],
                name=indicador
            ),
            row=1, col=1
        )

    # Grupo de indicadores defensivos
    defensive_indicators = ["DCasa", "DFora"]
    for indicador in defensive_indicators:
        fig.add_trace(
            go.Bar(
                x=df_totais["Season"],
                y=df_totais[indicador],
                name=indicador
            ),
            row=2, col=1
        )

    # Atualiza o layout e os eixos
    fig.update_layout(
        title_text=f"Indicadores Gerais do {team_name} por Temporada",
        barmode="group",
        height=700,
        legend_title_text="Indicadores"
    )
    fig.update_xaxes(title_text="Temporada", row=1, col=1)
    fig.update_xaxes(title_text="Temporada", row=2, col=1)
    fig.update_yaxes(title_text="Média por Jogo", row=1, col=1)
    fig.update_yaxes(title_text="Total", row=2, col=1)

    return fig
# ---------------------------
# RF5 – Divisão dos Dados (Ex.: Rebotes, Pontos, 2PT, 3PT, Lance Livre)
# ---------------------------
def calcular_divisao_dados(df):
    """
    Calcula a divisão dos dados por temporada. Se o DataFrame 'df' contiver a coluna "SEASON",
    os dados serão agrupados por temporada; caso contrário, os dados serão somados globalmente.
    
    São calculados os seguintes totais:
      - Total de Rebotes
      - Total de Rebotes Ofensivos (OREB)
      - Total de Rebotes Defensivos (DREB)
      - Total de Pontos (PTS)
      - Total de Cestas de 2 Pontos (FGM - FG3M)
      - Total de Cestas de 3 Pontos (FG3M)
      - Total de Lance Livre Convertidos (FTM)
    
    Retorna:
      Um DataFrame com uma linha por temporada (ou uma única linha, se não houver coluna "SEASON"),
      com as colunas:
        - Season
        - Total de Rebotes
        - Total de Rebotes Ofensivos
        - Total de Rebotes Defensivos
        - Total de Pontos
        - Total de Cestas de 2 Pontos
        - Total de Cestas de 3 Pontos
        - Total de Lance Livre Convertidos
    """
    # Converte as colunas necessárias para numérico
    for col in ['FGM', 'FG3M', 'FTM', 'OREB', 'DREB', 'REB', 'PTS']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'SEASON' in df.columns:
        # Agrupa por temporada e soma os valores de cada coluna
        grouped = df.groupby('SEASON').agg({
            'REB': 'sum',
            'OREB': 'sum',
            'DREB': 'sum',
            'PTS': 'sum',
            'FGM': 'sum',
            'FG3M': 'sum',
            'FTM': 'sum'
        }).reset_index()
        
        # Calcula os totais desejados
        grouped['Total de Rebotes'] = grouped['REB']
        grouped['Total de Rebotes Ofensivos'] = grouped['OREB']
        grouped['Total de Rebotes Defensivos'] = grouped['DREB']
        grouped['Total de Pontos'] = grouped['PTS']
        grouped['Total de Cestas de 2 Pontos'] = grouped['FGM'] - grouped['FG3M']
        grouped['Total de Cestas de 3 Pontos'] = grouped['FG3M']
        grouped['Total de Lance Livre Convertidos'] = grouped['FTM']
        
        df_divisao = grouped[['SEASON',
                              'Total de Rebotes',
                              'Total de Rebotes Ofensivos',
                              'Total de Rebotes Defensivos',
                              'Total de Pontos',
                              'Total de Cestas de 2 Pontos',
                              'Total de Cestas de 3 Pontos',
                              'Total de Lance Livre Convertidos']]
        # Renomeia a coluna de temporada
        df_divisao.rename(columns={'SEASON': 'Season'}, inplace=True)
    else:
        # Se não houver coluna "SEASON", calcula os totais globais
        total_reb  = df['REB'].sum()
        total_oreb = df['OREB'].sum()
        total_dreb = df['DREB'].sum()
        total_pts  = df['PTS'].sum()
        total_fg3  = df['FG3M'].sum()
        total_fg2  = (df['FGM'] - df['FG3M']).sum()
        total_ft   = df['FTM'].sum()
        dados = {
            "Season": ["Total"],
            "Total de Rebotes": [total_reb],
            "Total de Rebotes Ofensivos": [total_oreb],
            "Total de Rebotes Defensivos": [total_dreb],
            "Total de Pontos": [total_pts],
            "Total de Cestas de 2 Pontos": [total_fg2],
            "Total de Cestas de 3 Pontos": [total_fg3],
            "Total de Lance Livre Convertidos": [total_ft]
        }
        df_divisao = pd.DataFrame(dados)
    
    # Salva o resultado em CSV
    os.makedirs("scripts/output", exist_ok=True)
    output_path = os.path.join("scripts/output", "divisao_dados_time.csv")
    df_divisao.to_csv(output_path, index=False)
    print(f"Divisão dos dados salvos em: {output_path}")
    
    return df_divisao

def grafico_divisao_dados(df):
    """
    Gera um gráfico de barras agrupado por temporada para exibir a divisão dos dados do time.
    
    Retorna:
      Objeto figura do Plotly.
    """
    # Calcula os dados divididos (agrupados por temporada)
    df_div = calcular_divisao_dados(df)
    
    # Converte o DataFrame para formato long para facilitar a plotagem
    df_long = df_div.melt(id_vars="Season", var_name="Indicador", value_name="Total")
    
    # Cria o gráfico de barras agrupado
    fig = px.bar(
        df_long,
        x="Season",
        y="Total",
        color="Indicador",
        barmode="group",
        title="Divisão dos Dados Totais do Time por Temporada",
        text="Total"
    )
    
    fig.update_traces(texttemplate='%{text:.0f}', textposition='outside')
    fig.update_layout(
        xaxis_title="Temporada",
        yaxis_title="Total"
    )
    
    return fig
# ---------------------------
# RF6 – Dados de Performance Defensiva
# ---------------------------
def calcular_performance_defensiva(df):
    """
    Calcula os totais dos indicadores defensivos por temporada:
      - Total de Roubos de Bola (STL)
      - Total de Rebotes Defensivos (DREB)
      - Total de Tocos (BLK)
      - Total de Erros (TOV)
      - Total de Faltas (PF)
    
    Se o DataFrame 'df' contiver a coluna "SEASON", os dados serão agrupados por temporada;
    caso contrário, os totais serão calculados globalmente.
    
    Retorna:
      Um DataFrame com uma linha por temporada (ou uma única linha se não houver coluna "SEASON"),
      contendo as seguintes colunas:
        - Season
        - Total de Roubos de Bola
        - Total de Rebotes Defensivos
        - Total de Tocos
        - Total de Erros
        - Total de Faltas
    """
    # Converte as colunas necessárias para tipo numérico
    for col in ['STL', 'DREB', 'BLK', 'TOV', 'PF']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'SEASON' in df.columns:
        # Agrupa por temporada e soma os valores de cada indicador
        grouped = df.groupby('SEASON').agg({
            'STL': 'sum',
            'DREB': 'sum',
            'BLK': 'sum',
            'TOV': 'sum',
            'PF': 'sum'
        }).reset_index()
        
        # Cria uma cópia para evitar o SettingWithCopyWarning
        df_defensiva = grouped[['SEASON', 'STL', 'DREB', 'BLK', 'TOV', 'PF']].copy()
        
        # Renomeia as colunas conforme a Tabela 5
        df_defensiva.rename(columns={
            'SEASON': 'Season',
            'STL': 'Total de Roubos de Bola',
            'DREB': 'Total de Rebotes Defensivos',
            'BLK': 'Total de Tocos',
            'TOV': 'Total de Erros',
            'PF': 'Total de Faltas'
        }, inplace=True)
    else:
        # Caso não haja a coluna "SEASON", calcula os totais globalmente
        total_stl  = df['STL'].sum()
        total_dreb = df['DREB'].sum()
        total_blk  = df['BLK'].sum()
        total_tov  = df['TOV'].sum()
        total_pf   = df['PF'].sum()
        df_defensiva = pd.DataFrame({
            'Season': ['Total'],
            'Total de Roubos de Bola': [total_stl],
            'Total de Rebotes Defensivos': [total_dreb],
            'Total de Tocos': [total_blk],
            'Total de Erros': [total_tov],
            'Total de Faltas': [total_pf]
        })
    
    # Salva o resultado em CSV
    os.makedirs("scripts/output", exist_ok=True)
    output_path = os.path.join("scripts/output", "performance_defensiva_time.csv")
    df_defensiva.to_csv(output_path, index=False)
    print(f"Performance defensiva salvos em: {output_path}")
    
    return df_defensiva

# ---------------------------
# RF7 – Tabela de Jogos do Time
# ---------------------------
def gerar_tabela_jogos(df):
    """
    Cria uma tabela com os dados dos jogos do time, filtrando para incluir apenas as temporadas desejadas 
    (por exemplo, "23-24" e "2024-25"). A tabela apresenta:
      - Data do Jogo
      - Adversário
      - Resultado (Vitória ou Derrota)
      - Casa ou Fora
      - Placar
    
    Para extrair o adversário, a função utiliza a coluna "MATCHUP". 
    O placar é gerado a partir da coluna "PTS" (pontos do time), com um placeholder para os pontos do adversário.
    
    As informações são extraídas do DataFrame original, que deve conter as colunas utilizadas (por exemplo, 
    "GAME_DATE", "MATCHUP", "WL", "PTS", etc.), normalmente geradas a partir dos arquivos CSV processados.
    """
    # Define as temporadas desejadas (ajuste conforme necessário)
    temporadas_desejadas = ["23-24", "2024-25"]
    
    # Filtra o DataFrame para incluir apenas as temporadas desejadas, se a coluna "SEASON" existir
    if "SEASON" in df.columns:
        df = df[df["SEASON"].isin(temporadas_desejadas)].copy()
    else:
        df = df.copy()
    
    # Chama a função que define a coluna "Local" (supondo que ela utilize a coluna "MATCHUP")
    df = definir_local(df)
    
    # Função para extrair o nome do adversário da coluna "MATCHUP"
    def extrair_adversario(matchup):
        if "vs." in matchup:
            return matchup.split("vs.")[1].strip()
        elif "@" in matchup:
            return matchup.split("@")[1].strip()
        else:
            return np.nan

    df["Adversario"] = df["MATCHUP"].apply(extrair_adversario)
    
    # Cria a coluna "Placar" usando a coluna "PTS" (com um placeholder para os pontos do adversário)
    df["Placar"] = df["PTS"].apply(lambda pts: f"{pts} - ?")
    
    # Converte a coluna de data para datetime e formata como string (se existir)
    if "GAME_DATE" in df.columns:
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
        df["Data do Jogo"] = df["GAME_DATE"].dt.strftime("%d/%m/%Y")
    else:
        df["Data do Jogo"] = np.nan
    
    # Renomeia a coluna de resultado: "WL" para "Resultado"
    df.rename(columns={"WL": "Resultado"}, inplace=True)
    
    # Renomeia a coluna "Local" para "Casa ou Fora" (se necessário)
    df.rename(columns={"Local": "Casa ou Fora"}, inplace=True)
    
    # Seleciona as colunas desejadas na ordem correta
    tabela_jogos = df[["Data do Jogo", "Adversario", "Resultado", "Casa ou Fora", "Placar"]].copy()
    
    # Salva o resultado em CSV
    os.makedirs("scripts/output", exist_ok=True)
    output_path = os.path.join("scripts/output", "tabela_jogos_time.csv")
    tabela_jogos.to_csv(output_path, index=False)
    print(f"Tabela de jogos salva em: {output_path}")
    
    return tabela_jogos

# ---------------------------
# RF8 – Geração de Gráficos (Exemplos)
# ---------------------------

def grafico_barras_agrupado(wins_losses_df):
    """
    Gera um gráfico de barras agrupado para:
      - Vitórias em Casa (verde)
      - Vitórias Fora de Casa (azul)
      - Derrotas em Casa (vermelho)
      - Derrotas Fora de Casa (marrom)
    
    Para cada temporada, é gerado um subplot e os gráficos são dispostos lado a lado.
    """
    # Definição das categorias e cores
    categorias = ["Vitórias Casa", "Vitórias Fora", "Derrotas Casa", "Derrotas Fora"]
    cores = ['green', 'blue', 'red', 'brown']
    
    # Número de temporadas (linhas do DataFrame)
    n_seasons = wins_losses_df.shape[0]
    seasons = wins_losses_df["Season"].tolist()
    
    # Cria subplots com 1 linha e n_seasons colunas
    fig = make_subplots(
        rows=1,
        cols=n_seasons,
        subplot_titles=seasons,
        horizontal_spacing=0.1  # ajuste o espaçamento horizontal conforme necessário
    )
    
    # Itera sobre cada temporada para adicionar os traços em sua coluna correspondente
    for i, (_, row) in enumerate(wins_losses_df.iterrows()):
        # Obtém os valores para as 4 categorias para essa temporada
        valores = [
            row["Vitórias em Casa"],
            row["Vitórias Fora de Casa"],
            row["Derrotas em Casa"],
            row["Derrotas Fora de Casa"]
        ]
        # Para cada categoria, adiciona um traço do tipo Bar no subplot correspondente
        for cat, val, cor in zip(categorias, valores, cores):
            fig.add_trace(
                go.Bar(
                    name=cat,
                    x=[cat],
                    y=[val],
                    marker_color=cor
                ),
                row=1,
                col=i+1
            )
    
    # Atualiza o layout para barras agrupadas e adiciona título geral
    fig.update_layout(
        barmode="group",
        title="Vitórias e Derrotas Agrupadas por Local - Por Temporada (Lado a Lado)",
        showlegend=True
    )
    
    # Salva o gráfico em HTML (opcional)
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_barras_agrupado_horizontal.html")
    
    return fig

def grafico_histograma(df):
    """
    Gráfico Histograma da frequência de Vitórias e Derrotas por temporada.
    Para que o gráfico funcione, o DataFrame deve conter:
      - Uma coluna "WL" com os resultados de cada jogo (ex.: "W" ou "L")
      - Uma coluna "Season" (ou "SEASON") indicando a temporada de cada jogo
    """
    # Se a coluna estiver em maiúsculas, renomeia para "Season"
    if "Season" not in df.columns and "SEASON" in df.columns:
        df = df.rename(columns={"SEASON": "Season"})
    
    # Cria o histograma com facetas para cada temporada
    fig = px.histogram(
        df,
        x="WL",
        color="WL",
        facet_col="Season",
        color_discrete_map={'W': 'green', 'L': 'red'},
        title="Frequência de Vitórias e Derrotas por Temporada"
    )
    
    # Ajuste opcional: remover títulos das facetas se necessário
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    
    # Salva o gráfico em HTML (opcional)
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_histograma.html")
    
    return fig

def grafico_pizza(wins_losses_df):
    """
    Gráfico de Pizza para a distribuição de resultados por local para cada temporada.
    
    Para cada temporada (linha do DataFrame), é exibido um gráfico de pizza com:
      - Vitórias Casa
      - Vitórias Fora
      - Derrotas Casa
      - Derrotas Fora
       
    Cada pizza é posicionada em um subplot (lado a lado) com o título da temporada.
    """
    # Número de temporadas (número de linhas do DataFrame)
    n_seasons = wins_losses_df.shape[0]
    
    # Cria subplots: 1 linha e n_seasons colunas; cada célula do tipo 'domain' (necessário para Pie)
    fig = make_subplots(
        rows=1,
        cols=n_seasons,
        specs=[[{'type': 'domain'}]*n_seasons],
        subplot_titles=wins_losses_df['Season'].tolist()
    )
    
    # Itera sobre cada temporada para criar um gráfico de pizza com os dados correspondentes
    for i, (_, row) in enumerate(wins_losses_df.iterrows()):
        labels = ["Vitórias Casa", "Vitórias Fora", "Derrotas Casa", "Derrotas Fora"]
        values = [
            row["Vitórias em Casa"],
            row["Vitórias Fora de Casa"],
            row["Derrotas em Casa"],
            row["Derrotas Fora de Casa"]
        ]
        fig.add_trace(
            go.Pie(labels=labels, values=values, name=row["Season"]),
            row=1, col=i+1
        )
    
    # Atualiza o layout geral do gráfico
    fig.update_layout(
        title_text="Distribuição de Resultados por Local - Por Temporada"
    )
    
    # Salva o gráfico em HTML (opcional)
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_pizza_por_temporada.html")
    
    return fig

def grafico_radar(df):
    """
    Gráfico de Radar exibindo a média de pontos marcados e (opcionalmente) sofridos 
    nos jogos em casa e fora, para cada temporada.
    
    Se a coluna 'PTS_SOFRIDOS' não existir no DataFrame, o gráfico terá somente 
    a parte de Pontos Marcados.
    
    Requisitos:
      - O DataFrame deve conter as colunas: 
            "Season", "Local", "PTS"
        e opcionalmente: 
            "PTS_SOFRIDOS".
      - "Local" deve conter valores "Casa" ou "Fora".
      - Se houver várias temporadas, serão gerados subplots (um radar por temporada).
    """

    # Garante que existe a coluna "Season". Se não existir, cria com valor padrão.
    if "Season" not in df.columns:
        df["Season"] = "Total"

    # Verifica se a coluna "Local" existe
    if "Local" not in df.columns:
        raise KeyError("A coluna 'Local' não foi encontrada (esperado 'Casa' ou 'Fora').")

    # Verifica se a coluna "PTS" existe
    if "PTS" not in df.columns:
        raise KeyError("A coluna 'PTS' não foi encontrada no DataFrame.")

    # Verifica se a coluna PTS_SOFRIDOS existe
    tem_pts_sofridos = ("PTS_SOFRIDOS" in df.columns)

    # Monta o dicionário de agregação para o groupby
    agg_dict = {"PTS": "mean"}
    if tem_pts_sofridos:
        agg_dict["PTS_SOFRIDOS"] = "mean"

    # Agrupa por Season e Local para calcular as médias
    df_media = df.groupby(["Season", "Local"], as_index=False).agg(agg_dict)

    # Lista de temporadas
    seasons = df_media["Season"].unique()
    n_seasons = len(seasons)

    # Cria subplots do tipo polar: 1 linha e n_seasons colunas
    fig = make_subplots(
        rows=1,
        cols=n_seasons,
        subplot_titles=seasons,
        specs=[[{"type": "polar"}]*n_seasons]
    )

    for i, season in enumerate(seasons):
        df_season = df_media[df_media["Season"] == season]

        # Casa
        df_casa = df_season[df_season["Local"] == "Casa"]
        if not df_casa.empty:
            pts_casa = df_casa["PTS"].iloc[0]
        else:
            pts_casa = 0

        # Fora
        df_fora = df_season[df_season["Local"] == "Fora"]
        if not df_fora.empty:
            pts_fora = df_fora["PTS"].iloc[0]
        else:
            pts_fora = 0

        # Adiciona o traço para Pontos Marcados
        fig.add_trace(
            go.Scatterpolar(
                r=[pts_casa, pts_fora],
                theta=["Casa", "Fora"],
                fill='toself',
                name='Pontos Marcados',
                marker_color='blue',
            ),
            row=1, col=i+1
        )

        # Se tiver a coluna de pontos sofridos, faz o traço
        if tem_pts_sofridos:
            if not df_casa.empty:
                pts_sofridos_casa = df_casa["PTS_SOFRIDOS"].iloc[0]
            else:
                pts_sofridos_casa = 0

            if not df_fora.empty:
                pts_sofridos_fora = df_fora["PTS_SOFRIDOS"].iloc[0]
            else:
                pts_sofridos_fora = 0

            fig.add_trace(
                go.Scatterpolar(
                    r=[pts_sofridos_casa, pts_sofridos_fora],
                    theta=["Casa", "Fora"],
                    fill='toself',
                    name='Pontos Sofridos',
                    marker_color='red'
                ),
                row=1, col=i+1
            )

        # Ajusta se quiser controlar o range radial (opcional)
        max_valor = max(
            pts_casa, 
            pts_fora,
            (pts_sofridos_casa if tem_pts_sofridos else 0),
            (pts_sofridos_fora if tem_pts_sofridos else 0),
            10  # coloca um mínimo para evitar zero ou muito pequeno
        )
        fig.update_polars(
            radialaxis=dict(range=[0, max_valor * 1.2]),
            row=1, col=i+1
        )

    # Atualiza o layout geral do gráfico
    fig.update_layout(
        height=500,
        title="Média de Pontos Marcados e Sofridos (Casa vs. Fora)",
        showlegend=True
    )

    # (Opcional) Salvar em HTML
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_radar.html")

    return fig

def grafico_linha_sequencia(df):
    """
    Gráfico de Linha para a sequência de Vitórias (1) e Derrotas (0) ao longo da temporada,
    separado por temporada (se houver mais de uma).
    
    Requisitos:
      - O DataFrame deve conter:
          * "GAME_DATE": data do jogo
          * "WL": resultado do jogo ('W' para vitória, 'L' para derrota)
          * "Season": temporada (caso queira agrupar por temporada)
    """

    # Faz uma cópia para não alterar o DataFrame original
    df = df.copy()

    # Mapeia 'W' -> 1, 'L' -> 0
    df["Resultado_Num"] = df["WL"].map({'W': 1, 'L': 0})

    # Verifica se "GAME_DATE" está em formato datetime; se não estiver, converte
    if not np.issubdtype(df["GAME_DATE"].dtype, np.datetime64):
        # Tenta converter com dayfirst=True (caso seja dd/mm/yyyy)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], dayfirst=True, errors='coerce')

    # Se "Season" não existir, cria com valor "Total"
    if "Season" not in df.columns:
        df["Season"] = "Total"

    # Ordena os dados pela data do jogo
    df = df.sort_values("GAME_DATE")

    # Verifica quantas temporadas distintas existem
    n_temporadas = df["Season"].nunique()

    # Plota
    if n_temporadas == 1:
        # Apenas uma temporada
        fig = px.line(
            df, 
            x="GAME_DATE", 
            y="Resultado_Num", 
            markers=True,
            title="Sequência de Vitórias (1) e Derrotas (0) ao Longo da Temporada"
        )
    else:
        # Várias temporadas => facet_col
        fig = px.line(
            df,
            x="GAME_DATE",
            y="Resultado_Num",
            markers=True,
            facet_col="Season",
            title="Sequência de Vitórias (1) e Derrotas (0) ao Longo das Temporadas"
        )

    # Salva o gráfico em HTML (opcional)
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_linha_sequencia.html")

    return fig

def grafico_custom_ofensivo_defensivo(df):
    """
    Gráfico de Dispersão exibindo equipes e a média de pontos marcados (PTS) e
    sofridos (PTS_SOFRIDOS) durante a temporada.

    Requisitos mínimos:
      - O DataFrame deve conter colunas para:
          * "TEAM" (ou "Team") ou algo que identifique as equipes
          * "PTS"  (pontos marcados)
          * "PTS_SOFRIDOS" (pontos sofridos)
    """

    # Verifica se existem as colunas necessárias
    col_equipe = "TEAM"  # ou ajuste para o nome correto, caso seja "Team" ou "Equipe"
    if col_equipe not in df.columns:
        raise KeyError(f"A coluna '{col_equipe}' não foi encontrada no DataFrame.")

    if "PTS" not in df.columns or "PTS_SOFRIDOS" not in df.columns:
        raise KeyError("É preciso ter as colunas 'PTS' e 'PTS_SOFRIDOS' no DataFrame.")

    # Calcula a média de PTS e PTS_SOFRIDOS por equipe
    df_medias = df.groupby(col_equipe, as_index=False).agg({
        "PTS": "mean",
        "PTS_SOFRIDOS": "mean"
    })

    # Cria o gráfico de dispersão
    fig = px.scatter(
        df_medias,
        x="PTS",
        y="PTS_SOFRIDOS",
        hover_name=col_equipe,     # ao passar o mouse, mostra o nome da equipe
        text=col_equipe,          # mostra o texto da equipe diretamente no ponto
        color=col_equipe,         # cada equipe com uma cor (caso queira tudo de uma cor, pode remover)
        title="Média de Pontos Marcados vs Pontos Sofridos por Equipe"
    )

    # Ajusta a posição do texto (para não ficar em cima do ponto)
    fig.update_traces(textposition='top center')

    # (Opcional) Ajusta o layout (tamanho, legendas, etc.)
    fig.update_layout(
        xaxis_title="Média de Pontos Marcados (PTS)",
        yaxis_title="Média de Pontos Sofridos (PTS_SOFRIDOS)",
        showlegend=False,  # oculta a legenda de cores se achar muito poluída
        height=600
    )

    # Salva em HTML (opcional)
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_custom_ofensivo_defensivo.html")

    return fig

def grafico_custom_defensivo(df):
    """
    Gráfico de Barras para exibir os totais de indicadores defensivos e de erros
    por temporada.
    
    Espera que o DataFrame contenha as colunas:
      - "Season"
      - "Total de Roubos de Bola"
      - "Total de Rebotes Defensivos"
      - "Total de Tocos"
      - "Total de Erros"
      - "Total de Faltas"
      
    O gráfico resultante é um gráfico de barras agrupadas, onde cada temporada
    possui as barras dos indicadores.
    """
    # Se a coluna Season não existir, cria com valor padrão "Total"
    if "Season" not in df.columns:
        df["Season"] = "Total"
    
    # Lista dos indicadores que queremos exibir
    indicadores = [
        "Total de Roubos de Bola",
        "Total de Rebotes Defensivos",
        "Total de Tocos",
        "Total de Erros",
        "Total de Faltas"
    ]
    
    # Verifica se todas as colunas necessárias existem
    for col in indicadores:
        if col not in df.columns:
            raise KeyError(f"A coluna '{col}' não foi encontrada no DataFrame.")
    
    # Transforma o DataFrame de formato wide para long
    df_melted = df.melt(id_vars="Season", 
                        value_vars=indicadores, 
                        var_name="Indicador", 
                        value_name="Total")
    
    # Cria o gráfico de barras agrupadas (barmode="group")
    fig = px.bar(
        df_melted,
        x="Season",
        y="Total",
        color="Indicador",
        barmode="group",
        title="Indicadores Defensivos e Erros por Temporada",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    fig.update_layout(
        xaxis_title="Temporada",
        yaxis_title="Total",
    )
    
    # Salva o gráfico em HTML (opcional)
    os.makedirs("scripts/output", exist_ok=True)
    fig.write_html("scripts/output/grafico_custom_defensivo_por_temporada.html")
    
    return fig
# ---------------------------
# RF1 – Listar Times por Conferência
# ---------------------------
def listar_times_por_conferencia_rf1(output_dir="scripts/data/raw"):
    
    """
    RF1: Carrega a lista de times por conferência a partir do arquivo 'times_por_conferencia.csv'.
    Retorna um DataFrame com as colunas originais (por exemplo, 'Nome', 'ID' e 'Conferencia').
    """
    file_path = os.path.join(output_dir, "times_por_conferencia.csv")
    if os.path.exists(file_path):
        df_times = pd.read_csv(file_path)
        return df_times
    else:
        print(f"Aviso: Arquivo {file_path} não encontrado.")
        return pd.DataFrame()
def generate_html_table_from_multiindex(df):
    """
    Recebe um DataFrame com MultiIndex nas colunas (exemplo: [("Oeste", "ID"), ("Oeste", "Nome"), ("Leste", "ID"), ("Leste", "Nome")])
    e retorna um componente HTML que exibe uma tabela com um cabeçalho de duas linhas:
      - A primeira linha possui: "Oeste" (colspan=2) e "Leste" (colspan=2)
      - A segunda linha possui: "ID", "Nome" para Oeste e "ID", "Nome" para Leste.
      
    Se houver valores ausentes (NaN), eles serão exibidos como strings vazias.
    """
    # Cria a primeira linha do cabeçalho com colspan
    header_row1 = html.Tr([
        html.Th("Oeste", colSpan=2, style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
        html.Th("Leste", colSpan=2, style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'})
    ])
    
    # Cria a segunda linha do cabeçalho com os subheaders
    header_row2 = html.Tr([
        html.Th("ID", style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
        html.Th("Nome", style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
        html.Th("ID", style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
        html.Th("Nome", style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'})
    ])
    
    # Cria as linhas de dados
    data_rows = []
    for i in range(len(df)):
        # Obtém os valores para cada coluna; se algum valor for NaN, converte para string vazia
        oeste_id = "" if pd.isna(df.loc[i, ("Oeste", "ID")]) else df.loc[i, ("Oeste", "ID")]
        oeste_nome = "" if pd.isna(df.loc[i, ("Oeste", "Nome")]) else df.loc[i, ("Oeste", "Nome")]
        leste_id = "" if pd.isna(df.loc[i, ("Leste", "ID")]) else df.loc[i, ("Leste", "ID")]
        leste_nome = "" if pd.isna(df.loc[i, ("Leste", "Nome")]) else df.loc[i, ("Leste", "Nome")]
        
        row = html.Tr([
            html.Td(oeste_id, style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
            html.Td(oeste_nome, style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
            html.Td(leste_id, style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'}),
            html.Td(leste_nome, style={'textAlign': 'center', 'border': '1px solid black', 'padding': '5px'})
        ])
        data_rows.append(row)
    
    table = html.Table(
        [html.Thead([header_row1, header_row2]), html.Tbody(data_rows)],
        style={'width': '100%', 'borderCollapse': 'collapse'}
    )
    return table

def format_times_by_conference_subcolumns(df_times):
    """
    Formata o DataFrame de times para exibir duas colunas de nível superior:
      - "Oeste": times da Conferência Oeste
      - "Leste": times da Conferência Leste
    
    Cada coluna possui as subcolunas "ID" e "Nome".
    Se um dos grupos tiver menos times, as células faltantes serão preenchidas com NaN.
    """
    # Filtra os times por conferência e seleciona as colunas necessárias
    df_oeste = df_times[df_times["Conferencia"] == "Oeste"][["ID", "Nome"]].reset_index(drop=True)
    df_leste = df_times[df_times["Conferencia"] == "Leste"][["ID", "Nome"]].reset_index(drop=True)
    
    # Determina o número máximo de linhas
    max_rows = max(len(df_oeste), len(df_leste))
    
    # Reindexa para igualar o número de linhas (linhas ausentes serão NaN)
    df_oeste = df_oeste.reindex(range(max_rows))
    df_leste = df_leste.reindex(range(max_rows))
    
    # Define os MultiIndex para as colunas de cada grupo
    west_columns = pd.MultiIndex.from_tuples([("Oeste", "ID"), ("Oeste", "Nome")])
    east_columns = pd.MultiIndex.from_tuples([("Leste", "ID"), ("Leste", "Nome")])
    
    df_oeste.columns = west_columns
    df_leste.columns = east_columns
    
    # Concatena os DataFrames ao longo das colunas
    df_formatted = pd.concat([df_oeste, df_leste], axis=1)
    return df_formatted



# ---------------------------
# RF2 – Classificação Atual dos Times por Conferência
# ---------------------------
def obter_classificacao_atual_rf2(output_dir="scripts/data/raw"):
    """
    RF2: Apresenta a classificação atual dos times da NBA agrupados por Conferência.
    Os dados são carregados a partir do arquivo 'classificacao_por_conferencia.csv'.
    """
    file_path = os.path.join(output_dir, "classificacao_por_conferencia.csv")
    if os.path.exists(file_path):
        df_classificacao = pd.read_csv(file_path)
        return df_classificacao
    else:
        print(f"Aviso: Arquivo {file_path} não encontrado.")
        return pd.DataFrame()

def grafico_classificacao_conferencia(df_classificacao):
    """
    Gera um gráfico de barras horizontal para a classificação atual dos times por conferência,
    aplicando as métricas de acordo com a temporada, se o DataFrame contiver a coluna "SEASON".
    
    Espera que o DataFrame 'df_classificacao' contenha pelo menos as colunas:
      - "TeamName": nome do time
      - "WinPCT": percentual de vitórias (em formato decimal, ex. 0.65 para 65%)
      - "Conferencia": com valores, por exemplo, "Leste" e "Oeste"
      - Opcionalmente, "SEASON": para identificar a temporada
      
    Se "SEASON" existir, o gráfico será facetado por temporada.
    
    Retorna:
      Objeto figura do Plotly.
    """

    if 'SEASON' in df_classificacao.columns:
        fig = px.bar(
            df_classificacao,
            x="WinPCT",
            y="TeamName",
            color="Conferencia",
            orientation="h",
            facet_col="SEASON",
            title="Classificação Atual dos Times por Conferência por Temporada (RF2)",
            labels={"TeamName": "Time", "WinPCT": "Percentual de Vitórias"},
            text="WinPCT",
            color_discrete_map={"Leste": "blue", "Oeste": "green"}
        )
    else:
        fig = px.bar(
            df_classificacao,
            x="WinPCT",
            y="TeamName",
            color="Conferencia",
            orientation="h",
            title="Classificação Atual dos Times por Conferência (RF2)",
            labels={"TeamName": "Time", "WinPCT": "Percentual de Vitórias"},
            text="WinPCT",
            color_discrete_map={"Leste": "blue", "Oeste": "green"}
        )

    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_tickformat='.0%',
        margin=dict(l=150, r=50, t=70, b=50),
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    return fig

# ---------------------------
# Execução dos Cálculos e Geração de Arquivos (RF3 a RF10)
# ---------------------------
# Lista das temporadas
seasons = ['2023-24', '2024-25']
# Carregar os dados dos jogos do time
df_team = carregar_dados_team(seasons)

if df_team.empty:
    print("Nenhum dado de jogos do time foi carregado. Verifique os arquivos processados.")
else:
    # Converter GAME_DATE para datetime (caso não esteja)
    if "GAME_DATE" in df_team.columns:
        df_team['GAME_DATE'] = pd.to_datetime(df_team['GAME_DATE'], errors='coerce')
    else:
        print("Aviso: Coluna GAME_DATE não encontrada no DataFrame.")
    
    # RF3 – Totais de Vitórias e Derrotas
    wins_losses_df = calcular_wins_losses(df_team)
    
    # RF4 – Totais Gerais de Dados do Time
    totais_df = calcular_totais_gerais_por_temporada(df_team)
    
    # RF5 – Divisão dos Dados
    divisao_df = calcular_divisao_dados(df_team)
    
    # RF6 – Performance Defensiva
    defensiva_df = calcular_performance_defensiva(df_team)
    
    # RF7 – Tabela de Jogos
    tabela_jogos = gerar_tabela_jogos(df_team)
    
    # RF8 – Geração de Gráficos (os gráficos são salvos em HTML)
    # Exemplos (descomente conforme necessário):
    # grafico_barras_empilhado(wins_losses_df)
    # grafico_barras_agrupado(wins_losses_df)
    # grafico_histograma(df_team)
    # grafico_pizza(wins_losses_df)
    # grafico_radar(df_team)
    # grafico_linha_sequencia(df_team)
    # grafico_custom_defensivo(df_team)
    
    # RF1 e RF2: Carregar e exibir os dados de times e classificação a partir dos arquivos locais
    df_times_rf1 = listar_times_por_conferencia_rf1()
    
    df_classificacao_rf2 = obter_classificacao_atual_rf2()


if __name__ == '__main__':
    os.makedirs("scripts/output", exist_ok=True)
    # app.run_server(debug=True)
