import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):

    def __init__(self,  dataset_path: str, train=True, test_size=0.2, random_state=42, scaler=None):
        # Check if the dataset file exists
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
        
        data = pd.read_csv(dataset_path)

         # Check if the 'stabf' column exists
        if 'stabf' not in data.columns:
            raise ValueError("Dataset must contain the 'stabf' column.")
        
        # Separate features and labels
        x = data.drop('stabf', axis=1).values
        y = data['stabf'].replace({'unstable': 0, 'stable': 1}).astype(int).values
       
      # Split the data into training and validation sets
        x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size, random_state=random_state)
        
        # Choose the appropriate dataset split (train or validation)
        if train:
            self.x, self.y = x_train, y_train
            if scaler:
                self.x = scaler.fit_transform(self.x)
        else:
            self.x, self.y = x_val, y_val
            if scaler:
                self.x = scaler.transform(self.x)
        
        # Convert to PyTorch tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

