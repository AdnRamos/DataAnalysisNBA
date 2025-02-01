import os
import pandas as pd
import numpy as np

def tratar_dados(df, tipo='time'):
    """
    Realiza o tratamento dos dados conforme os critérios:
      - Verifica e preenche dados ausentes
      - Remove duplicidades
      - Converte tipos de dados (datas e numéricos)
      - Transforma dados qualitativos para quantitativos (quando aplicável)
      - Exclui colunas irrelevantes (ex.: 'SALARIO')
      - Normaliza colunas numéricas (quando necessário)
      - Trata outliers com base no método IQR

    Parâmetros:
      df (DataFrame): DataFrame a ser tratado.
      tipo (str): Pode ser 'time', 'jogador' ou 'roster' para aplicar
                  transformações específicas a cada padrão.

    Retorna:
      DataFrame tratado.
    """
    df_tratado = df.copy()

    # 1. Remover duplicatas
    df_tratado.drop_duplicates(inplace=True)

    # 2. Converter colunas de datas
    if 'GAME_DATE' in df_tratado.columns:
        df_tratado['GAME_DATE'] = pd.to_datetime(df_tratado['GAME_DATE'], errors='coerce')
    if tipo == 'roster' and "BIRTH_DATE" in df_tratado.columns:
        df_tratado["BIRTH_DATE"] = pd.to_datetime(df_tratado["BIRTH_DATE"], errors="coerce")

    # 3. Definir quais colunas numéricas converter dependendo do tipo do arquivo
    if tipo == 'time':
        # Arquivos de jogos do time
        numeric_cols = [
            "W", "L", "W_PCT", "MIN", "FGM", "FGA", "FG_PCT",
            "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
            "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS"
        ]
    elif tipo == 'jogador':
        # Arquivos de logs dos jogadores
        numeric_cols = [
            "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
            "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST",
            "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS"
        ]
    elif tipo == 'roster':
        # Para o roster, algumas colunas numéricas relevantes
        numeric_cols = ["NUM", "WEIGHT", "AGE", "EXP"]
    else:
        numeric_cols = []

    # 4. Converter colunas numéricas (caso estejam como string)
    for col in numeric_cols:
        if col in df_tratado.columns:
            df_tratado[col] = pd.to_numeric(df_tratado[col], errors='coerce')

    # 5. Transformar dados qualitativos em quantitativos, se aplicável
    # Exemplo: coluna WL (Win/Loss) para os jogos
    if 'WL' in df_tratado.columns:
        df_tratado['WL_BIN'] = df_tratado['WL'].map({'W': 1, 'L': 0})

    # 6. Excluir colunas que não impactam a análise
    # (Aqui é possível adicionar ou remover conforme o caso – por exemplo, 'SALARIO')
    colunas_irrelevantes = ['SALARIO', 'SomeOtherColumn']
    for col in colunas_irrelevantes:
        if col in df_tratado.columns:
            df_tratado.drop(columns=[col], inplace=True)

    # 7. Preencher dados ausentes
    # Para colunas numéricas: usar a mediana
    for col in df_tratado.select_dtypes(include=[np.number]).columns:
        if df_tratado[col].isnull().sum() > 0:
            mediana = df_tratado[col].median()
            df_tratado[col].fillna(mediana, inplace=True)
    # Para colunas categóricas: usar o valor mais frequente
    for col in df_tratado.select_dtypes(include=['object']).columns:
        if df_tratado[col].isnull().sum() > 0:
            df_tratado[col].fillna(df_tratado[col].mode()[0], inplace=True)

    # 8. Identificar e tratar outliers (usando o método IQR) nas colunas numéricas definidas
    for col in numeric_cols:
        if col in df_tratado.columns:
            Q1 = df_tratado[col].quantile(0.25)
            Q3 = df_tratado[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_tratado[col] = np.where(df_tratado[col] < lower_bound, lower_bound, df_tratado[col])
            df_tratado[col] = np.where(df_tratado[col] > upper_bound, upper_bound, df_tratado[col])

    # 9. Normalização (exemplo com a coluna PTS, se aplicável)
    if 'PTS' in numeric_cols and 'PTS' in df_tratado.columns:
        pts_min = df_tratado['PTS'].min()
        pts_max = df_tratado['PTS'].max()
        if pts_max != pts_min:
            df_tratado['PTS_NORM'] = (df_tratado['PTS'] - pts_min) / (pts_max - pts_min)

    # (Opcional) Exibir um resumo dos dados após tratamento
    print("Resumo dos dados após tratamento:")
    print(df_tratado.info())
    print("\nValores ausentes por coluna:")
    print(df_tratado.isnull().sum())

    return df_tratado

def processar_arquivos(input_dir, output_dir, tipo='time'):
    """
    Processa todos os arquivos CSV do diretório de entrada, aplica o tratamento e
    salva os arquivos tratados no diretório de saída.
    
    Parâmetros:
      input_dir (str): Caminho do diretório contendo os arquivos originais.
      output_dir (str): Caminho do diretório onde serão salvos os arquivos tratados.
      tipo (str): 'time', 'jogador' ou 'roster', conforme a estrutura do arquivo.
    """
    if not os.path.exists(input_dir):
        print(f"Atenção: diretório '{input_dir}' não existe. Pulando o processamento desta pasta.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_filepath = os.path.join(input_dir, filename)
            print(f"Lendo arquivo: {input_filepath}")
            df = pd.read_csv(input_filepath)
            print("Visualizando as primeiras linhas antes do tratamento:")
            print(df.head(), "\n")
            
            df_tratado = tratar_dados(df, tipo=tipo)
            
            # Nome de saída: adiciona o sufixo _clean antes da extensão
            output_filename = filename.replace('.csv', '_clean.csv')
            output_filepath = os.path.join(output_dir, output_filename)
            df_tratado.to_csv(output_filepath, index=False)
            print(f"Arquivo tratado salvo em: {output_filepath}\n")

def main():
    # Lista de temporadas a serem processadas
    seasons = ['2023-24', '2024-25']
    
    # Diretório base onde os dados originais estão armazenados
    base_input_dir = "data/raw"
    # Diretório base onde serão salvos os dados tratados
    base_output_dir = os.path.join("data", "processed")
    
    for season in seasons:
        print(f"\n=== Processando temporada {season} ===")
        safe_season = season.replace('/', '-')
        
        # Processar arquivos de jogos do time
        team_input_dir = os.path.join(base_input_dir, season, "team")
        team_output_dir = os.path.join(base_output_dir, season, "team")
        print(">> Processando dados do TIME:")
        processar_arquivos(team_input_dir, team_output_dir, tipo='time')
        
        # Processar arquivos de logs dos jogadores
        players_input_dir = os.path.join(base_input_dir, season, "players")
        players_output_dir = os.path.join(base_output_dir, season, "players")
        print(">> Processando dados dos JOGADORES:")
        processar_arquivos(players_input_dir, players_output_dir, tipo='jogador')
        
        # (Opcional) Se houver arquivos de roster na mesma temporada,
        # por exemplo em data/<season>/roster, processa-os também.
        roster_input_dir = os.path.join(base_input_dir, season, "roster")
        roster_output_dir = os.path.join(base_output_dir, season, "roster")
        print(">> Processando dados do ROSTER:")
        processar_arquivos(roster_input_dir, roster_output_dir, tipo='roster')

if __name__ == '__main__':
    main()
