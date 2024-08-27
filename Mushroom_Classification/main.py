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
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, )

train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop with validation
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Training phase
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()  # Clear gradients
        y_hat = model(x_batch)  # Forward pass
        loss = criterion(y_hat, y_batch)  # Compute loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        
        running_loss += loss.item() * x_batch.size(0)
        _, predicted = torch.max(y_hat.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct / total
    
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)
    
    # Validation phase
    model.eval()  # Set the model to evaluation mode
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():  # Disable gradient computation
        for x_batch, y_batch in val_loader:
            y_hat = model(x_batch)
            val_loss = criterion(y_hat, y_batch)
            
            val_running_loss += val_loss.item() * x_batch.size(0)
            _, val_predicted = torch.max(y_hat.data, 1)
            val_total += y_batch.size(0)
            val_correct += (val_predicted == y_batch).sum().item()
    
    avg_val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = 100 * val_correct / val_total
    
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')