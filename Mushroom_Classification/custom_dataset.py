import pandas as pd

class CustomDataset:
    def __init__(self, path):
        data = pd.read_csv(path)
        print("Shape of data: ", data.shape)

        # Check for missing data
        missing_data = data.isnull().mean()  # Calculates the percentage of missing values in each column
        print("Missing data (%): \n",  missing_data * 100)
        print("Data info: ")
        print(data.info())

        # Drop the 'id' column
        data = data.drop(columns='id', axis=1)
        print("Dropped 'id' column. New shape: ", data.shape)

        # Drop columns with more than 80% missing data
        null_percent = 0.80
        columns_to_drop = missing_data[missing_data > null_percent].index  # Identifies columns with > 80% missing values
        data.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropped columns with more than {null_percent * 100}% missing values: {columns_to_drop}")
        print("New shape after dropping columns: ", data.shape)

        # split data by data type 
        catogorical_column = [column for column in data.columns if data[column].dtype == 'object']
        print(catogorical_column)

        numeric_column = [column for column in data.columns if data[column].dtype == 'float64']
        print(numeric_column)
data = CustomDataset('./Mushroom_Classification/Dataset/train.csv')

