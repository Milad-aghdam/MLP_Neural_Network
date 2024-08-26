import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch
from torch.utils.data import Dataset 

class CustomDataset:
    def __init__(self, path):
        data = pd.read_csv(path)
        print("Shape of data: ", data.shape)

        # Check for missing data
        missing_data = data.isnull().mean()  # Calculates the percentage of missing values in each column
        print("Missing data (%): \n",  missing_data * 100)
        # print("Data info: ")
        # print(data.info())

        # Drop the 'id' column
        data = data.drop(columns='id', axis=1)
        print("Dropped 'id' column. New shape: ", data.shape)

        # Drop columns with more than 80% missing data
        null_percent = 0.90
        columns_to_drop = missing_data[missing_data > null_percent].index  # Identifies columns with > 80% missing values
        data.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns with more than {null_percent * 100}% missing values: {columns_to_drop}")
        print("New shape after dropping columns: ", data.shape)

        # split data by data type 
        categorical_columns = [column for column in data.columns if data[column].dtype == 'object']
        # print(categorical_columns)

        mode_of_columns = {column :  data[column].mode()[0] for column in categorical_columns} 
        
        data = data.fillna(value=mode_of_columns)

        numeric_column = [column for column in data.columns if data[column].dtype == 'float64']
        median_of_columns = {column : data[column].median() for column in numeric_column}

        data = data.fillna(value=median_of_columns)

        # print(data.isnull().sum())

        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        data[categorical_columns] = encoder.fit_transform(data[categorical_columns])
        
        # print(data['class'].value_counts())
        # print(data.head(10))
        self.X = data.drop('class').values
        print(self.X.shape)
        self.y = data['class'].values
        print(self.y.shape)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.X[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.long)
        return feature, target



