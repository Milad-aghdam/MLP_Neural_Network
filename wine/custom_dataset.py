import torch
import pandas as pd 
from torch.utils.data import Dataset

class CustomDataset(Dataset):

    def __init__(self, path1, path2):
        data1 = pd.read_csv(path1)
        data1['type'] = 'red'

        data2 = pd.read_csv(path2)
        data2['type'] = 'white'

        print("data 1 shape : ", data1.shape)
        print("data 2 shape : ", data2.shape)

    def __len__(self):
        pass

    def __getitem__(self):
        pass

        






path1 = "./wine/Dataset/winequality-red.csv"
path2 = "./wine/Dataset/winequality-white.csv"
data = CustomDataset(path1, path2)

"./data_for_uci_named/Dataset/Data_for_UCI_named.csv"