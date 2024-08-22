import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CustomDataset(Dataset):

    def __init__(self, dataset_path):
        data = pd.read_csv(dataset_path)

        self.x = data.drop('stabf', axis=1).values
        self.y = data['stabf'].replace({'unstable': 0, 'stable': 1}).values
       
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self) -> int:
        """
        Returns the size of the dataset.
        """
        return len(self.x)
    
    def __getitem__(self, index: int):
  
        return self.x[index], self.y[index]

data = CustomDataset('./data_for_uci_named/Dataset/Data_for_UCI_named.csv')
print(f"Dataset size: {len(data)}")
