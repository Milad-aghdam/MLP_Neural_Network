import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, input_size, hiddin_size, output_size):
        super(MLP, self).__init__()
        
        self.f1 = nn.Linear(input_size, hiddin_size)
        self.f2 = nn.Linear(hiddin_size, hiddin_size)
        self.f3 = nn.Linear(hiddin_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  
        return  x
