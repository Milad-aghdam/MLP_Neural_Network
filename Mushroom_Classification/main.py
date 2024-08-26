from custom_dataset import CustomDataset
from model import MLP

import torch.optim
import torch
import torch.nn
from torch.utils.data import DataLoader, random_split

# Set Hyperparameters
input_size = 18
hidden_size = 64
output_size = 2
learning_rate = 0.01
batch_size = 256
num_epochs = 20

# Initialize the Dataset and DataLoader
data = CustomDataset('./Mushroom_Classification/Dataset/train.csv')

train_size = int(0.8 * len(data))  # 80% of data for training
val_size = len(data) - train_size  # Remaining 20% for validation

train_dataset, val_dataset = random_split(data, [train_size, val_size])

rain_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)