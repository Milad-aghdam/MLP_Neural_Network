import torch
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class CustomDataset(Dataset):

    def __init__(self, path1, path2, scaler=None, test_size=0.2, random_state=42, mode='train'):
        # Load datasets
        data1 = pd.read_csv(path1, delimiter=';')
        data1['type'] = 'red'

        data2 = pd.read_csv(path2, delimiter=';')
        data2['type'] = 'white'

        # Combine datasets
        df = pd.concat([data1, data2])
        df['type'] = df['type'].replace({'red': 0, 'white': 1})

        # Map quality to categories (0: bad, 1: medium, 2: good)
        df['type_quality'] = pd.cut(df['quality'], bins=[0, 3, 6, 9], labels=[0, 1, 2])

        # Features (X) and target (y)
        x = df.drop(columns=['type_quality', 'quality']).values  # Keep 'type' in the features
        y = df['type_quality'].values
        # Split into training and validation sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)

        # Assign data based on mode (train, val, or test)
        if mode == 'train':
            self.x, self.y = x_train, y_train
            if scaler:
                self.x = scaler.fit_transform(self.x)  # Fit and transform on training data
        elif mode == 'val':
            self.x, self.y = x_val, y_val
            if scaler:
                self.x = scaler.transform(self.x)  # Only transform using already fitted scaler
        elif mode == 'test':
            self.x, self.y = x_test, y_test
            if scaler:
                self.x = scaler.transform(self.x)  # Only transform using already fitted scaler
        else:
            raise ValueError("mode must be 'train', 'val', or 'test'")

        # Convert to PyTorch tensors
        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
