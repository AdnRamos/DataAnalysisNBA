import os
import time
import pandas as pd
from nba_api.stats.endpoints import teamgamelog, commonteamroster, playergamelog

def get_team_game_logs(team_id, season):
    """
    Coleta os dados dos jogos do time para a temporada especificada.
    """
    print(f"Coletando log de jogos para o time {team_id} na temporada {season}...")
    gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=season)
    df_games = gamelog.get_data_frames()[0]
    return df_games

def get_team_roster(team_id, season):
    """
    Coleta o roster (lista de jogadores) do time para a temporada especificada.
    """
    print(f"Coletando roster para o time {team_id} na temporada {season}...")
    roster = commonteamroster.CommonTeamRoster(team_id=team_id, season=season)
    df_roster = roster.get_data_frames()[0]
    return df_roster

def get_player_game_logs(player_id, season):
    """
    Coleta os dados dos jogos de um jogador para a temporada especificada.
    """
    print(f"Coletando logs de jogos para o jogador {player_id} na temporada {season}...")
    pgl = playergamelog.PlayerGameLog(player_id=player_id, season=season)
    df_pgl = pgl.get_data_frames()[0]
    return df_pgl

def get_player_ids(roster_df, players):
    """
    Filtra o DataFrame do roster para retornar os IDs dos jogadores desejados.
    
    Parâmetros:
      roster_df (DataFrame): DataFrame obtido pelo endpoint commonteamroster;
      players (list): Lista com os nomes dos jogadores desejados.
    
    Retorna:
      Dicionário no formato {nome do jogador: player_id}.
    """
    # Verifique qual coluna contém os nomes dos jogadores.
    filtered = roster_df[roster_df['PLAYER'].isin(players)]
    if filtered.empty:
        print("Nenhum jogador encontrado no roster!")
        return {}
    return dict(zip(filtered['PLAYER'], filtered['PLAYER_ID']))

def main():
    # ID do Memphis Grizzlies
    team_id = 1610612763
    # Temporadas a serem analisadas
    seasons = ['2023-24', '2024-25']
    # Lista dos jogadores de interesse
    players = ["Ja Morant", "Desmond Bane", "Jaren Jackson Jr."]
    
    for season in seasons:
        # Cria um nome "seguro" para a pasta, substituindo barras se existirem
        safe_season = season.replace('/', '-')
        # Cria a estrutura de diretórios: data/raw/<temporada>/team e data/<temporada>/players
        season_dir = os.path.join("data/raw", safe_season)
        team_dir = os.path.join(season_dir, "team")
        players_dir = os.path.join(season_dir, "players")
        
        os.makedirs(team_dir, exist_ok=True)
        os.makedirs(players_dir, exist_ok=True)
        
        print(f"\n=== Processando temporada {season} ===")
        
        # Coleta e salva os dados dos jogos do time
        team_games = get_team_game_logs(team_id, season)
        team_games_filepath = os.path.join(team_dir, f"games_{safe_season}.csv")
        team_games.to_csv(team_games_filepath, index=False)
        print(f"Dados dos jogos do time salvos em: {team_games_filepath}")
        
        # Aguarda um pouco para evitar sobrecarregar a API
        time.sleep(1)
        
        # Coleta e salva os dados do roster do time
        roster_df = get_team_roster(team_id, season)
        roster_filepath = os.path.join(team_dir, f"roster_{safe_season}.csv")
        roster_df.to_csv(roster_filepath, index=False)
        print(f"Dados do roster do time salvos em: {roster_filepath}")
        
        # Obtém os IDs dos jogadores de interesse a partir do roster
        player_ids = get_player_ids(roster_df, players)
        
        # Para cada jogador, coleta e salva os logs dos jogos
        for player, player_id in player_ids.items():
            pgl_df = get_player_game_logs(player_id, season)
            safe_player = player.replace(' ', '_')
            player_filepath = os.path.join(players_dir, f"{safe_player}_games_{safe_season}.csv")
            pgl_df.to_csv(player_filepath, index=False)
            print(f"Dados dos jogos de {player} salvos em: {player_filepath}")
            
            # Aguarda um pouco entre as requisições
            time.sleep(1)

if __name__ == "__main__":
    main()
