# MLP Classification Example

This project demonstrates how to build, train, and evaluate a Multi-Layer Perceptron (MLP) model using PyTorch for a classification task. It includes data preprocessing, model definition, training with dropout, and evaluation.

## Project Structure

- `custom_dataset.py`: Defines a custom dataset class for loading and preprocessing data.
- `model.py`: Defines the MLP model architecture.
- `main.py`: Contains the training and evaluation loop.

## Installation

Make sure you have the following packages installed:

- Python 3.x
- PyTorch
- pandas
- scikit-learn

You can install the required packages using pip:

```bash
pip install torch pandas scikit-learn
```

## Usage


1. Prepare Your Dataset:

    Make sure your dataset is formatted correctly and placed in the correct directory.

2. Define the Model:

    In model.py, the MLP model is defined with dropout layers to prevent overfitting. You can adjust the model architecture and hyperparameters as needed.

3. Run Training:

    Execute main.py to start training the model. The script handles loading the dataset, training the model, and evaluating its performance.
```bash
python main.py
```


#### 6. Code Description
Provide a brief description of each script and its main functions. This section helps users understand the purpose of each file in your project.


## Code Description

### `custom_dataset.py`

- **`CustomDataset` class**: Loads and preprocesses the dataset.
  - `__init__(self, dataset_path)`: Initializes the dataset, loads the CSV file, and preprocesses the data.
  - `__len__(self)`: Returns the number of samples in the dataset.
  - `__getitem__(self, index)`: Returns a sample and its corresponding label.

### `model.py`

- **`MLP` class**: Defines a Multi-Layer Perceptron model.
  - `__init__(self, input_size, hidden_size, output_size)`: Initializes the model with layers.
  - `forward(self, x)`: Defines the forward pass with ReLU activations and dropout.

### `main.py`

- **Hyperparameters**: Defines learning rate, batch size, and number of epochs.
- **Data Loading**: Splits the data into training, validation, and test sets, and loads them using `DataLoader`.
- **Training Loop**: Trains the model, evaluates on validation data, and prints metrics.
- **Validation**: Evaluates the model on validation data to monitor performance and prevent overfitting.
