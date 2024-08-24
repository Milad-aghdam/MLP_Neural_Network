import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from custom_dataset import CustomDataset  
from model import MLP  


# Hyperparameters
input_size = 12  # Number of features (excluding 'quality')
hidden_size = 64  # Number of neurons in the hidden layer
output_size = 3  # Number of classes (bad, medium, good)
learning_rate = 0.001
batch_size = 64
num_epochs = 100
dropout_prob = 0.6

# Define paths to the dataset
path1 = "./wine/Dataset/winequality-red.csv"
path2 = "./wine/Dataset/winequality-white.csv"

# Initialize the CustomDataset with paths
scaler = StandardScaler()
train_dataset = CustomDataset(path1, path2, train=True, scaler=scaler)
val_dataset = CustomDataset(path1, path2, train=False, scaler=scaler)

# Create DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model
model = MLP(input_size, hidden_size, output_size, dropout_prob)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    train_loss = 0
    correct_train = 0
    total_train = 0

    for x_batch, y_batch in train_loader:
        # Forward pass
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += y_batch.size(0)
        correct_train += (predicted == y_batch).sum().item()

    train_accuracy = 100 * correct_train / total_train
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += y_batch.size(0)
            correct_val += (predicted == y_batch).sum().item()

    val_accuracy = 100 * correct_val / total_val

    # Print epoch statistics
    print(f'Epoch [{epoch+1}/{num_epochs}], '
          f'Train Loss: {train_loss / len(train_loader):.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, '
          f'Val Loss: {val_loss / len(val_loader):.4f}, '
          f'Val Accuracy: {val_accuracy:.2f}%')
