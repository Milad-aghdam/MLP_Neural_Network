from custom_dataset import CustomDataset
from model import MLP

import torch.optim as optim
import torch
import torch.nn as nn
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = MLP(input_size, hidden_size, output_size)
loss = nn.CrossEntropyLoss()
optim = optim.Adam(model.parameters(), lr=learning_rate, )

train_accuracies = []
train_loss = []
for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in train_loader:
        y_hat = model(x_batch)
        loss_fn = loss(y_hat, y_batch)
        optim.zero_grad()
        loss_fn.backward()
        optim.step()
        
        running_loss += loss_fn.item() * x_batch.size(0)  # Multiply by batch size
        _, predicted = torch.max(y_hat.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    # Average loss and accuracy for the epoch
    avg_train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct / total

    train_loss.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    print(f'Epoch [{epoch+1}/{num_epochs}], '
        f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%,')