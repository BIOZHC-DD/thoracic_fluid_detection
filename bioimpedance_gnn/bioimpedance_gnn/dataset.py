import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler

class BioimpedanceDataset:
    def __init__(self, csv_path):
        self.df = self.load_and_preprocess_data(csv_path)
        self.graphs = self.create_graphs()

    def load_and_preprocess_data(self, csv_path):
        df = pd.read_csv(csv_path)

        numerical_cols = [
            'Age', 'Height', 'Weight', 'BMI', 'gender', 'chest_size',
            'elec1', 'elec2','elec3','elec4', 'elec5', 'elec6', 'elec7', 'elec8'
        ]

        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        return df

    def create_graph_from_row(self, row):
        # Node features: electrode positions (4 nodes x 8 feature)
        x = torch.tensor([
            [row['elec1'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec2'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec3'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec4'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec5'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec6'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec7'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']],
            [row['elec8'], row['Age'], row['gender'], row['Height'], row['Weight'],
            row['BMI'], row['chest_size']]
        ], dtype=torch.float)

        # Edge indices (4 edges for bidirectional connections)
        edge_index = torch.tensor([
            [0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3],  # Source nodes
            [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3]   # Target nodes
        ], dtype=torch.long)

        # Edge features: impedance values (4 edges with 2 features each)
        edge_attr = torch.tensor([
            [row['bioz_0_0_freq25'], row['bioz_0_0_freq50'], row['bioz_0_0_freq75'], row['bioz_0_0_freq100'], row['bioz_0_0_freq125'], row['bioz_0_0_freq150']],
            [row['bioz_0_2_freq25'], row['bioz_0_2_freq50'], row['bioz_0_2_freq75'], row['bioz_0_2_freq100'], row['bioz_0_2_freq125'], row['bioz_0_2_freq150']],
            [row['bioz_0_3_freq25'], row['bioz_0_3_freq50'], row['bioz_0_3_freq75'], row['bioz_0_3_freq100'], row['bioz_0_3_freq125'], row['bioz_0_3_freq150']],
            [row['bioz_0_5_freq25'], row['bioz_0_5_freq50'], row['bioz_0_5_freq75'], row['bioz_0_5_freq100'], row['bioz_0_5_freq125'], row['bioz_0_5_freq150']],
            [row['bioz_2_0_freq25'], row['bioz_2_0_freq50'], row['bioz_2_0_freq75'], row['bioz_2_0_freq100'], row['bioz_2_0_freq125'], row['bioz_2_0_freq150']],
            [row['bioz_2_2_freq25'], row['bioz_2_2_freq50'], row['bioz_2_2_freq75'], row['bioz_2_2_freq100'], row['bioz_2_2_freq125'], row['bioz_2_2_freq150']],
            [row['bioz_2_3_freq25'], row['bioz_2_3_freq50'], row['bioz_2_3_freq75'], row['bioz_2_3_freq100'], row['bioz_2_3_freq125'], row['bioz_2_3_freq150']],
            [row['bioz_2_5_freq25'], row['bioz_2_5_freq50'], row['bioz_2_5_freq75'], row['bioz_2_5_freq100'], row['bioz_2_5_freq125'], row['bioz_2_5_freq150']],
            [row['bioz_3_0_freq25'], row['bioz_3_0_freq50'], row['bioz_3_0_freq75'], row['bioz_3_0_freq100'], row['bioz_3_0_freq125'], row['bioz_3_0_freq150']],
            [row['bioz_3_2_freq25'], row['bioz_3_2_freq50'], row['bioz_3_2_freq75'], row['bioz_3_2_freq100'], row['bioz_3_2_freq125'], row['bioz_3_2_freq150']],
            [row['bioz_3_3_freq25'], row['bioz_3_3_freq50'], row['bioz_3_3_freq75'], row['bioz_3_3_freq100'], row['bioz_3_3_freq125'], row['bioz_3_3_freq150']],
            [row['bioz_3_5_freq25'], row['bioz_3_5_freq50'], row['bioz_3_5_freq75'], row['bioz_3_5_freq100'], row['bioz_3_5_freq125'], row['bioz_3_5_freq150']],
            [row['bioz_5_0_freq25'], row['bioz_5_0_freq50'], row['bioz_5_0_freq75'], row['bioz_5_0_freq100'], row['bioz_5_0_freq125'], row['bioz_5_0_freq150']],
            [row['bioz_5_2_freq25'], row['bioz_5_2_freq50'], row['bioz_5_2_freq75'], row['bioz_5_2_freq100'], row['bioz_5_2_freq125'], row['bioz_5_2_freq150']],
            [row['bioz_5_3_freq25'], row['bioz_5_3_freq50'], row['bioz_5_3_freq75'], row['bioz_5_3_freq100'], row['bioz_5_3_freq125'], row['bioz_5_3_freq150']],
            [row['bioz_5_5_freq25'], row['bioz_5_5_freq50'], row['bioz_5_5_freq75'], row['bioz_5_5_freq100'], row['bioz_5_5_freq125'], row['bioz_5_5_freq150']]
        ], dtype=torch.float)

        y = torch.tensor([row['nonnormalized_fluid']], dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def create_graphs(self):
        return [self.create_graph_from_row(row) for _, row in self.df.iterrows()]