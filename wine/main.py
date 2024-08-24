import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from custom_dataset import CustomDataset  
from model import MLP, SimpleMLP
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler



input_size = 12  # Assuming 12 features in the dataset
hidden_size1 = 128
hidden_size2 = 64
output_size = 3  # 3 classes: bad, medium, good
learning_rate = 0.001  # A reasonable learning rate to start with
batch_size = 32  # Commonly used batch size
num_epochs = 50  # Enough epochs to observe learning
# Define paths to the dataset
path1 = "./wine/Dataset/winequality-red.csv"
path2 = "./wine/Dataset/winequality-white.csv"

# Initialize the CustomDataset with paths
scaler = StandardScaler()

full_dataset = CustomDataset(path1, path2,  scaler)  # Initialize the dataset
dataset_length = len(full_dataset)

train_dataset = CustomDataset(path1, path2, scaler=scaler, mode='train')
print("train_dataset",train_dataset[0][0]) 
val_dataset = CustomDataset(path1, path2, scaler=scaler, mode='val')
test_dataset = CustomDataset(path1, path2, scaler=scaler, mode='test')

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=20, shuffle=False)

# Define class weights for imbalance handling
class_counts = [30, 5190, 1277]
total_samples = sum(class_counts)
class_weights = [total_samples / class_count for class_count in class_counts]
class_weights = torch.tensor(class_weights, dtype=torch.float32)

model = MLP(input_size, hidden_size1, hidden_size2, output_size)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)


# Initialize lists to store loss and accuracy values
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    for x_batch, y_batch in train_loader:
        # Forward pass
        y_hat = model(x_batch)
        loss = criterion(y_hat, y_batch)

        # Backward pass and optimization
        optimizer.step()
        
        running_loss += loss.item() * x_batch.size(0)
        _, predicted = torch.max(y_hat.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()  # Multiply by batch size
     
    
    avg_train_loss = running_loss / len(train_loader.dataset)
    train_accuracy = 100 * correct / total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_accuracy)

# Validation
    model.eval()  # Set model to evaluation mode
    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            y_hat = model(x_batch)
            loss = criterion(y_hat, y_batch)

            # Calculate running loss
            val_running_loss += loss.item() * x_batch.size(0)  # Multiply by batch size

            # Calculate accuracy
            _, predicted = torch.max(y_hat.data, 1)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    avg_val_loss = val_running_loss / len(val_loader.dataset)
    val_accuracy = 100 * val_correct / val_total

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, '
            f'Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%')

plt.figure(figsize=(12, 6))

# Plot training and validation loss
plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st subplot
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot training and validation accuracy
plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd subplot
plt.plot(range(1, num_epochs + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()
    # 6. Evaluation (optional)
model.eval()  # Set model to evaluation mode
correct = 0
total = 0
with torch.no_grad():
    for x_batch, y_batch in test_loader:
        outputs = model(x_batch)
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

test_accuracy = 100 * correct / total
print(f'Accuracy on test data: {test_accuracy:.2f}%')