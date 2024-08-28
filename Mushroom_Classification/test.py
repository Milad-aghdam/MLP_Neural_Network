import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import torch
from custom_dataset import CustomDataset
from model import MLP

# Step 1: Load the Test Data
test_data = pd.read_csv('./Mushroom_Classification/Dataset/test.csv')

# Store the 'id' column for later use in the submission file
test_ids = test_data['id']

# Step 2: Preprocess the Test Data
# Drop unnecessary columns
test_data = test_data.drop(columns=['id'], axis=1)

# Handle missing values (same as in the training data)
missing_data = test_data.isnull().mean()  # Calculates the percentage of missing values in each column
print("Missing data (%): \n",  missing_data * 100)
null_percent = 0.90
columns_to_drop = missing_data[missing_data > null_percent].index  # Identifies columns with > 80% missing values
test_data.drop(columns=columns_to_drop, inplace=True)
print(f"Dropped columns with more than {null_percent * 100}% missing values: {columns_to_drop}")
print("New shape after dropping columns: ", test_data.shape)

categorical_columns = [column for column in test_data.columns if test_data[column].dtype == 'object']
mode_of_columns = {column: test_data[column].mode()[0] for column in categorical_columns}
test_data.fillna(value=mode_of_columns, inplace=True)

numeric_columns = [column for column in test_data.columns if test_data[column].dtype == 'float64']
median_of_columns = {column: test_data[column].median() for column in numeric_columns}
test_data.fillna(value=median_of_columns, inplace=True)

# Encode categorical variables (using the encoder fitted on training data)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
test_data[categorical_columns] = encoder.transform(test_data[categorical_columns])

# Convert test data to a tensor
test_tensor = torch.tensor(test_data.values, dtype=torch.float32)

# Step 3: Load the Trained Model
input_size = test_tensor.shape[1]
hidden_size = 64  # Same as used during training
output_size = 2   # Assuming binary classification (editable 'e' or poisonous 'p')

model = MLP(input_size, hidden_size, output_size)
model.load_state_dict(torch.load('best_mlp_model.pth'))
model.eval()  # Set the model to evaluation mode

# Step 4: Make Predictions
with torch.no_grad():
    outputs = model(test_tensor)
    _, predictions = torch.max(outputs, 1)

# Convert numeric predictions back to class labels ('e' or 'p')
# Assuming '0' is 'e' and '1' is 'p', adjust if necessary
label_mapping = {0: 'e', 1: 'p'}
predictions = [label_mapping[p.item()] for p in predictions]

# Step 5: Prepare the Submission File
submission = pd.DataFrame({
    'id': test_ids,
    'class': predictions
})

# Step 6: Save the Submission File
submission.to_csv('submission.csv', index=False)

print("Submission file created: submission.csv")
