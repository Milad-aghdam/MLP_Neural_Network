import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(input_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
    
    def forward(self, inputs):
        inputs = F.relu(self.fc1(inputs))  # Apply ReLU activation after the first layer
        inputs = F.relu(self.fc2(inputs))  # Apply ReLU activation after the second layer
        outputs = self.fc3(inputs)         # Output layer (no activation function here)
        return outputs

         



    