import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# 1. Função para carregar os arquivos de logs dos jogadores de uma temporada
# ============================================================================
def load_players_data_from_folder(season, base_dir="data/processed"):
    folder_path = os.path.join(base_dir, season, "players")
    if not os.path.exists(folder_path):
        print(f"Diretório {folder_path} não encontrado para a temporada {season}.")
        return pd.DataFrame()
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith("_clean.csv")]
    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file, parse_dates=['GAME_DATE'])
        except Exception as e:
            print(f"Erro ao ler {file}: {e}")
            continue
        player_name = os.path.basename(file).split("_games_")[0].replace("_", " ")
        df["Player_Name"] = player_name
        df["Season"] = season
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

# ============================================================================
# 2. Função para carregar dados de todas as temporadas
# ============================================================================
def load_all_players_data(seasons, base_dir="data/processed"):
    df_list = []
    for season in seasons:
        df_season = load_players_data_from_folder(season, base_dir)
        if not df_season.empty:
            df_list.append(df_season)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

# ============================================================================
# 3. Função para processar (padronizar) os logs de jogadores
# ============================================================================
def process_player_log(df):
    df = df.copy()
    if df['GAME_DATE'].dtype == 'O':
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce')
    
    df['Local'] = np.where(df['MATCHUP'].str.contains("vs."), "Casa", "Fora")
    df['Adversário'] = df['MATCHUP'].apply(lambda x: x.split("vs.")[1].strip() if "vs." in x else (x.split("@")[1].strip() if "@" in x else np.nan))
    df['Placar do Jogo'] = df['PTS'].astype(str) + " - ?"

    cols = ["GAME_DATE", "Adversário", "WL", "Local", "PTS", "REB", "AST", "Placar do Jogo", "FG3A", "FG3M", "MIN"]
    for extra in ["Player_Name", "Season"]:
        if extra in df.columns:
            cols.append(extra)
    df_player = df[cols].copy()
    df_player.rename(columns={
        "GAME_DATE": "Data do Jogo",
        "WL": "V/D",
        "MIN": "Tempo em Quadra",
        "FG3A": "Tentativas de Cestas de 3",
        "FG3M": "Cestas de 3 PTS Marcados",
        "Local": "Casa/Fora"
    }, inplace=True)
    return df_player

# ============================================================================
# 4. Função para agregar métricas por jogador e temporada
# ============================================================================
def aggregate_player_metrics(df):
    agg_funcs = {
        "Data do Jogo": "count",
        "PTS": ["mean", "median", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan, "std"],
        "REB": ["mean", "median", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan, "std"],
        "AST": ["mean", "median", lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan, "std"]
    }
    df_agg = df.groupby(["Season", "Player_Name"]).agg(agg_funcs)
    df_agg.columns = ['_'.join(col).strip() for col in df_agg.columns.values]
    df_agg = df_agg.reset_index()
    df_agg.rename(columns={
        "Data do Jogo_count": "Total_Jogos",
        "PTS_mean": "Média_PTS",
        "PTS_median": "Mediana_PTS",
        "PTS_<lambda>": "Moda_PTS",
        "PTS_std": "Desvio_PTS",
        "REB_mean": "Média_REB",
        "REB_median": "Mediana_REB",
        "REB_<lambda>": "Moda_REB",
        "REB_std": "Desvio_REB",
        "AST_mean": "Média_AST",
        "AST_median": "Mediana_AST",
        "AST_<lambda>": "Moda_AST",
        "AST_std": "Desvio_AST"
    }, inplace=True)
    return df_agg

# ============================================================================
# 5. Funções para gerar gráficos interativos com informações aprimoradas
# ============================================================================
def generate_histogram(df, column, title, output_file):
    mean_val = df[column].mean()
    median_val = df[column].median()
    try:
        mode_val = df[column].mode().iloc[0]
    except IndexError:
        mode_val = None

    fig = px.histogram(df, x=column, nbins=20,
                       title=title,
                       labels={column: f"{column} por Jogo", "count": "Frequência"})
    
    fig.add_vline(x=mean_val, line_dash="dash", line_color="green",
                  annotation_text=f"Média: {mean_val:.2f}",
                  annotation_position="top left")
    fig.add_vline(x=median_val, line_dash="dash", line_color="blue",
                  annotation_text=f"Mediana: {median_val:.2f}",
                  annotation_position="top right")
    if mode_val is not None:
        fig.add_vline(x=mode_val, line_dash="dash", line_color="red",
                      annotation_text=f"Moda: {mode_val:.2f}",
                      annotation_position="bottom right")
    
    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Frequência",
        legend_title="Estatísticas",
        title={'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.write_html(output_file)
    return fig

def generate_boxplot(df, column, title, output_file):
    fig = px.box(df, y=column, title=title,
                 labels={column: f"{column}"},
                 points="all")
    fig.update_layout(
        yaxis_title=column,
        title={'x': 0.5, 'xanchor': 'center'},
        margin=dict(l=40, r=40, t=60, b=40)
    )
    fig.write_html(output_file)
    return fig

# ============================================================================
# 6. Função Principal para complementar o que falta
# ============================================================================
def main():
    os.makedirs("output", exist_ok=True)
    os.makedirs(os.path.join("output", "charts"), exist_ok=True)
    
    seasons = ["2023-24", "2024-25"]
    
    df_raw = load_all_players_data(seasons)
    if df_raw.empty:
        print("Nenhum dado encontrado para as temporadas especificadas.")
        return
    
    df_processed = process_player_log(df_raw)
    processed_output = os.path.join("output", "jogadores_logs_processados.csv")
    df_processed.to_csv(processed_output, index=False)
    
    df_metrics = aggregate_player_metrics(df_processed)
    metrics_output = os.path.join("output", "jogadores_metricas_agregadas.csv")
    df_metrics.to_csv(metrics_output, index=False)
    
    unique_players = df_processed["Player_Name"].unique()
    for player in unique_players:
        df_player = df_processed[df_processed["Player_Name"] == player]
        player_folder = os.path.join("output", "charts", player.replace(" ", "_"))
        os.makedirs(player_folder, exist_ok=True)
        for metric in ["PTS", "REB", "AST"]:
            hist_file = os.path.join(player_folder, f"{player.replace(' ', '_')}_{metric}_histogram.html")
            box_file = os.path.join(player_folder, f"{player.replace(' ', '_')}_{metric}_boxplot.html")
            generate_histogram(df_player, metric, f"Distribuição de {metric} para {player}", hist_file).show()
            generate_boxplot(df_player, metric, f"Box Plot de {metric} para {player}", box_file).show()

if __name__ == '__main__':
    main()
