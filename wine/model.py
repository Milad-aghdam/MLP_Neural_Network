import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Linear):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(MLP, self).__init__()
        self.f1 = nn.Linear(input_size, hidden_size)
        self.f2 = nn.Linear(hidden_size, hidden_size)
        self.f3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_prob)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        
        x = self.dropout(x)
        
        x = self.fc3(x) 
        return x