import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from custom_dataset import CustomDataset
from custom_model import MLP
import matplotlib.pyplot as plt

def main():
    # 1. Hyperparameters
    input_size = 13  # Update this to match the feature size of your dataset
    hidden_size = 50
    output_size = 2   # Example value (e.g., binary classification)
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 10

    # 2. Load Dataset
    dataset_path = './data_for_uci_named/Dataset/Data_for_UCI_named.csv'
    full_dataset = CustomDataset(dataset_path)  # Initialize the dataset

    # Calculate dataset length
    dataset_length = len(full_dataset)

    # Splitting the dataset into training, validation, and testing sets
    train_size = int(0.7 * dataset_length)  # 70% for training
    val_size = int(0.15 * dataset_length)    # 15% for validation
    test_size = dataset_length - train_size - val_size  # Remaining 15% for testing
    
    # Ensure that sizes add up to total length
    assert train_size + val_size + test_size == dataset_length, "Split sizes do not add up to the dataset length"

    # Perform the split
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 3. Initialize Model, Loss Function, and Optimizer
    model = MLP(input_size, hidden_size, output_size)
    criterion = torch.nn.CrossEntropyLoss()  # Example loss function for classification
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists to store loss and accuracy values
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    # 4. Training Loop
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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate running loss
            running_loss += loss.item() * x_batch.size(0)  # Multiply by batch size

            # Calculate accuracy
            _, predicted = torch.max(y_hat.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        # Average loss and accuracy for the epoch
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

    # 5. Plot Learning Curves
    plt.figure(figsize=(12, 10))

    # Plot Loss Curve
    plt.subplot(2, 2, 1)
    plt.plot(range(num_epochs), train_losses, marker='o', label='Training Loss')
    plt.plot(range(num_epochs), val_losses, marker='x', label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()

    # Plot Accuracy Curve
    plt.subplot(2, 2, 2)
    plt.plot(range(num_epochs), train_accuracies, marker='o', label='Training Accuracy')
    plt.plot(range(num_epochs), val_accuracies, marker='x', label='Validation Accuracy', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy Curve')
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

    # 7. Save Model
    torch.save(model.state_dict(), 'model.pth')

if __name__ == '__main__':
    main()
