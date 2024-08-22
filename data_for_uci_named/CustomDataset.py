import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CustomDataset(Dataset):

    def __init__(self, dataset_path):
        data = pd.read_csv(dataset_path)
        self.x = data.drop('stabf', axis=1)
        self.y = data['stabf']

        self.y = self.y.replace('unstable', 0)
        self.y = self.y.replace('stable', 1)
        
        self.x = self.x.values
        self.y = self.y.values 

        print(self.y.shape)
        print(self.y)

data = CustomDataset('./data_for_uci_named/Dataset/Data_for_UCI_named.csv')
