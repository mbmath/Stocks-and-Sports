

import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

stock_output = 'AAPL'
stock_input = 'GOOG'

# Define custom dataset
class StockDataset(Dataset):
    def __init__(self, ticker, start_date, end_date):
        # Download historical data for the given ticker and date range
        historical_data = yf.download(ticker, start=start_date, end=end_date)
        self.data = historical_data[['Open', 'Close']]
        self.data['Open'].fillna(method='ffill', inplace=True)

        # Replace remaining NaN values with the mean of non-NaN values
        self.data['Open'].fillna(self.data['Open'].mean(), inplace=True)

        self.labels = torch.tensor(self.data['Open'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.labels[idx]

    def check_for_nans(self):
        # Check for NaN values in labels
        nan_labels = torch.isnan(self.labels).any()

        if nan_labels:
            raise ValueError("Dataset contains NaN labels. Please handle missing data before creating the dataset.")

# Load training dataset
train_dataset_sp500 = StockDataset(ticker= stock_output, start_date='2005-01-01', end_date='2022-12-29')
train_dataset_nasdaq = StockDataset(ticker= stock_input, start_date='2005-01-01', end_date='2022-12-29')

# Concatenate the training datasets
train_dataset = torch.utils.data.ConcatDataset([train_dataset_sp500, train_dataset_nasdaq])

# Load test dataset
test_dataset_sp500 = StockDataset(ticker= stock_output, start_date='2023-01-01', end_date='2023-11-27')

# Check for NaN values
train_dataset_sp500.check_for_nans()
train_dataset_nasdaq.check_for_nans()
test_dataset_sp500.check_for_nans()

# DataLoader for training dataset
train_dataloader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)

# Define neural network model
class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
learning_rate = 0.001
num_epochs = 1

# Initialize the model and optimizer
model = StockPredictor(input_size=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for labels in train_dataloader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(labels.unsqueeze(1))
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward and optimize
        loss.backward()
        optimizer.step()

    # Print training loss
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# DataLoader for test dataset
test_dataloader_sp500 = DataLoader(test_dataset_sp500, batch_size=len(test_dataset_sp500), shuffle=False)

# Evaluation loop
eval_model = StockPredictor(input_size=1)
eval_model.load_state_dict(model.state_dict())  # Copy trained model weights
eval_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for labels in test_dataloader_sp500:
        # Forward pass for the test dataset
        test_outputs = eval_model(labels.unsqueeze(1))
        test_loss = criterion(test_outputs, labels.unsqueeze(1))
        nan_indices = torch.isnan(test_outputs).nonzero()
        if torch.isnan(test_outputs).any():
            raise ValueError("NaN values found in test outputs.")
        else:
            print("No NaN values in test outputs.")

# Print test loss
print(f'Test Loss: {test_loss.item():.4f}')

# Convert tensors to numpy arrays
actual_values_sp500 = test_dataset_sp500.labels.numpy()
predicted_values_sp500 = test_outputs.numpy().flatten()

# Check for missing values in the arrays
if np.isnan(actual_values_sp500).any() or np.isnan(predicted_values_sp500).any():
    raise ValueError("Arrays contain missing values. Please handle missing data before calculating correlations.")


# Plot stock prices for the training dataset
plt.figure(figsize=(12, 6))
plt.plot(train_dataset_sp500.data.index, train_dataset_sp500.labels.numpy(), label=f"{stock_output} Actual", color='blue')
plt.plot(train_dataset_nasdaq.data.index, train_dataset_nasdaq.labels.numpy(), label=f"{stock_input} Actual", color='green')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual Stock Prices - Training Dataset')
plt.legend()
plt.show()

# Calculate changes
actual_changes_sp500 = np.diff(actual_values_sp500)
actual_cumulative_changes_sp500 = np.cumsum(actual_changes_sp500)
predicted_changes_sp500 = np.diff(predicted_values_sp500)
predicted_cumulative_changes_sp500 = np.cumsum(predicted_changes_sp500) + actual_values_sp500[0]

# Trim arrays for length
min_length = min(len(test_dataset_sp500.data.index), len(predicted_cumulative_changes_sp500))
test_index_trimmed = test_dataset_sp500.data.index[:min_length]
predicted_trimmed = predicted_cumulative_changes_sp500[:min_length]


# Plot stock prices for the test dataset
plt.figure(figsize=(12, 6))
plt.plot(test_index_trimmed, actual_values_sp500[:min_length], label=f"{stock_output} Actual", color='blue')
plt.plot(test_index_trimmed, predicted_trimmed, label=f"{stock_output} Predicted", color='orange')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title(f"Predicted Stock Prices - Test Dataset ({stock_output})")
plt.legend()
plt.show()
