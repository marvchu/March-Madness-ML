import pandas as pd
import numpy as np
import os

class DataPrep:
    def __init__(self, season, base_path):
        self.season = season
        self.base_path = f'{base_path}'
    

    def load_data(self):
        filepath = os.path.join(self.base_path, "processed", f"{self.season}_stats.csv")
        self.df = pd.read_csv(filepath)


    def data_randomization(self):
        df = self.df.copy()

        flip = np.random.rand(len(df)) < 0.5

        team1_cols = df.filter(regex='^team1_').copy()
        team2_cols = df.filter(regex='^team2_').copy()

        team1_flipped = team2_cols.copy()
        team2_flipped = team1_cols.copy()

        team1_cols[flip] = team1_flipped[flip].values
        team2_cols[flip] = team2_flipped[flip].values


        X = pd.concat([
            team1_cols, team2_cols
        ], axis=1)

        X = X.drop(columns=['team1_TEAM NO', 'team1_TEAM ID', 'team2_TEAM NO', 'team2_TEAM ID'])

        y = (~flip).astype(int)
        X = X.to_numpy()

        return X, y