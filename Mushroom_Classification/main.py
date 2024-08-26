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

