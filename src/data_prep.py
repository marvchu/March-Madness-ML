import pandas as pd
import numpy as np

def prepare_data():
    df_2 = pd.read_csv(r'C:\Users\Sweat\OneDrive\Desktop\march_madness_ai\data\raw\Barttorvik_Away_Neutral.csv')
    df_3 = pd.read_csv(r'C:\Users\Sweat\OneDrive\Desktop\march_madness_ai\data\raw\mmWinLoss.csv')
    df_4 = pd.read_csv(r'C:\Users\Sweat\OneDrive\Desktop\march_madness_ai\data\raw\teamID.csv')

        #Choose preferred columns
    df_3_col = [
        'Season', 'WTeamID' , 'LTeamID'
    ]

    df_4_col = [
        'TeamID', 'TeamName'
    ]

    df_stats = df_2.drop('ROUND', axis=1)
    df_games = df_3[df_3_col]
    df_ID = df_4[df_4_col]

    #Match ID to team name
    df_merged = df_games.merge(df_ID, left_on='WTeamID', right_on='TeamID', how='left')
    df_merged = df_merged.rename(columns={'TeamName':'WTeamName'})

    df_merged = df_merged.merge(df_ID, left_on='LTeamID', right_on='TeamID', how='left')
    df_merged = df_merged.rename(columns={'TeamName':'LTeamName'})
    df_merged = df_merged.drop(columns=['TeamID_x', 'TeamID_y', 'WTeamID', 'LTeamID'])

    #Name mapping
    stat_name = set(df_games['TEAM'].unique())
    game_name = set(df_merged['WTeamName'].unique()) | set(df_merged['LTeamName'].unique())
    mismatch_name = stat_name - game_name
    name_map = {
        'Abilene Christian' : 'Abilene Chr', 'Alabama St.' : 'Alabama St', 'Albany' : 'SUNY Albany', 'American' : 'American Univ', 'Appalachian St.' : 'Appalachian St',
        'Arizona St.' : 'Arizona St', 'Arkansas Pine Bluff' : 'Ark Pine Bluff', 'Boise St.' : 'Boise St', 'Boston University' : 'Boston Univ', 'Cal St. Bakersfield' : 'CS Bakersfield',
        'Cal St. Fullerton' : 'CS Fullerton', 'Cleveland St.' : 'Cleveland St', 'Coastal Carolina' : 'Coastal Car', 'College of Charleston' : 'Col Charleston', 'Colorado St.' : 'Colorado St',
        'East Tennessee St.' : 'ETSU', 'Eastern Kentucky' : 'E Kentucky', 'Eastern Washington' : 'E Washington', 'Fairleigh Dickinson' : 'F Dickinson', 'Florida Atlantic' : 'FL Atlantic', 
        'Florida Gulf Coast' : 'FGCU', 'Florida St.' : 'Florida St', 'Fresno St.' : 'Fresno St', 'George Washington' : 'G Washington', 'Georgia St.' : 'Georgia St', 'Grambling St.' : 'Grambling',
        'Indiana St.' : 'Indiana St', 'Iowa St.' : 'Iowa St', 'Jacksonville St.' : 'Jacksonville St', 'Kansas St.' : 'Kansas St', 'Kennesaw St.' : 'Kennesaw', 'Kent St.' : 'Kent', 'Little Rock' : 'Ark Little Rock',
        'Long Beach St.' : 'Long Beach St', 'Louisiana Lafayette' : 'Lafayette', 'Loyola Chicago' : 'Loyola-Chicago', 'McNeese St.' : 'McNeese St', 'Michigan St.' : 'Michigan St',
        'Middle Tennessee' : 'MTSU', 'Milwaukee' : 'WI Milwaukee', 'Mississippi St.' : 'Mississippi St', 'Mississippi Valley St.' : 'MS Valley St', 'Montana St.' : 'Montana St', 'Morehead St.' : 'Morehead St',
        'Morgan St.' : 'Morgan St', "Mount St. Mary's" : "Mt St Mary's", 'Murray St.' : 'Murray St', 'Nebraska Omaha' : 'NE Omaha', 'New Mexico St.' : 'New Mexico St', 'Norfolk St.' : 'Norfolk St', 
        'North Carolina Central' : 'NC Central', 'North Carolina A&T' : 'NC A&T', 'North Carolina St.' : 'NC State', 'Northern Colorado' : 'N colorado', 'Northern Kentucky' : 'N Kentucky',
        'Northwestern St.' : 'Northwestern LA', 'Ohio St.' : 'Ohio St', 'Oklahoma St.' : 'Oklahoma St', 'Oregon St.' : 'Oregon St', 'Penn St.' : 'Penn St', 'Prairie View A&M' : 'Prairie View',
        'SIU Edwardsville' : 'S Illinois', 'Saint Francis' : 'St Francis PA', "Saint Joseph's" : "St Joseph's PA", 'Saint Louis' : 'St Louis', "Saint Mary's" : "St Mary's CA", 
        "Saint Peter's" : "St Peter's", 'Sam Houston St.' : 'Sam Houston St', 'San Diego St.' : 'San Diego St', 'South Dakota St.' : 'S Dakota St', 'Southeast Missouri St.' : 'SE Missouri St',
        'Southern' : 'Southern Univ', 'St. Bonaventure' : 'St Bonaventure', "St. John's" : "St John's", 'Stephen F. Austin' : 'SF Austin', 'Texas A&M Corpus Chris' : 'TAM C. Christi',
        'Texas Southern' : 'TX Southern', 'UTSA' : 'UT San Antonio', 'Utah St.' : 'Utah St', 'Washington St.' : 'Washington St', 'Weber St.' : 'Weber St', 'Western Kentucky' : 'WKU',
        'Western Michigan' : 'W Michigan', 'Wichita St.' : 'Wichita St', 'Wright St.' : 'Wright St'
    }
    team_stats['TEAM'] = team_stats['TEAM'].replace(name_map)

    w_stats = team_stats.rename(columns=lambda col: f'W_{col}' if col not in ['YEAR', 'TEAM'] else col)
    l_stats = team_stats.rename(columns=lambda col: f'L_{col}' if col not in ['YEAR', 'TEAM'] else col)

    #Table with individual game and team stats
    df_merged = df_merged[df_merged['Season'] >= 2010]
    df_merged = df_merged.merge(w_stats, left_on=['Season', 'WTeamName'], right_on=['YEAR', 'TEAM'], how='left')
    df_merged = df_merged.merge(l_stats, left_on=['Season', 'LTeamName'], right_on=['YEAR', 'TEAM'], how='left')
    final_df = df_merged.dropna()
    final_df = final_df.drop(columns=['YEAR_x', 'TEAM_x', 'YEAR_y', 'TEAM_y'])

    df = final_df.copy()

    flip = np.random.rand(len(df)) < 0.5


    team1_stats = df.filter(regex='^W_').copy()
    team2_stats = df.filter(regex='^L_').copy()

    team1_stats.columns = team1_stats.columns.str.replace('^W_', 'team1_', regex=True)
    team2_stats.columns = team2_stats.columns.str.replace('^L_', 'team2_', regex=True)

    team1_stats_flipped = df.filter(regex='^L_').copy()
    team2_stats_flipped = df.filter(regex='^W_').copy()

    team1_stats_flipped.columns = team1_stats_flipped.columns.str.replace('^L_', 'team1_', regex=True)
    team2_stats_flipped.columns = team2_stats_flipped.columns.str.replace('^W_', 'team2_', regex=True)

    team1_stats[flip] = team1_stats_flipped[flip]
    team2_stats[flip] = team2_stats_flipped[flip]

    team1_stats.to_csv('team1_stats.csv')
    team2_stats.to_csv('team2_stats.csv')

    X = pd.concat([
        team1_stats, team2_stats
    ], axis=1)

    y = (~flip).astype(int)

    X = X.to_numpy()

    return X.values, y.values