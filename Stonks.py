# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 14:24:11 2023

@author: matth
"""
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Define your custom dataset
class StockDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, parse_dates=['Date'])
        self.data = self.data.set_index('Date')
        self.features = torch.tensor(self.data[['WinLoss']].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['Open'].values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Function to fill missing open prices using historical data with forward fill
# def fill_open_prices(dataset, historical_data):
#     for i in range(len(dataset)):
#         if torch.isnan(dataset.labels[i]):
#             date = dataset.data.index[i]
            
#             # Check if the date is not NaT before formatting
#             if pd.notna(date):
#                 date_str = date.strftime('%Y-%m-%d')
#                 if date_str in historical_data.index:
#                     dataset.labels[i] = historical_data.loc[date_str, 'Open']
#                 else:
#                     # Forward fill missing values
#                     dataset.labels[i] = dataset.labels[i - 1]

# Load your dataset
csv_file_path = 'train_sports.csv'
dataset = StockDataset(csv_file_path)

# Download historical data for the entire date range
start_date = dataset.data.index.min().strftime('%Y-%m-%d')
end_date = dataset.data.index.max().strftime('%Y-%m-%d')
historical_data = yf.download('^GSPC', start=start_date, end=end_date)

# Fill missing open prices using historical data with forward fill
# fill_open_prices(dataset, historical_data)
for i in range(len(dataset)):
    if torch.isnan(dataset.labels[i]):
        date = dataset.data.index[i]
        
        # Check if the date is not NaT before formatting
        if pd.notna(date):
            if i == 0:
                dataset.labels[i] = historical_data.loc['2018-01-02', 'Open']
                # print(dataset.labels[i])
            date_str = date.strftime('%Y-%m-%d')
            if date_str in historical_data.index:
                dataset.labels[i] = historical_data.loc[date_str, 'Open']
                # print(dataset.labels[i])
            else:
                # Forward fill missing values
                dataset.labels[i] = dataset.labels[i - 1]
                # print(dataset.labels[i])
                
# Calculate mean and standard deviation of your input features
mean = torch.mean(dataset.features, dim=0)
std = torch.std(dataset.features, dim=0)

# Normalize the input features
normalized_features = (dataset.features - mean) / std

# Replace the original features with the normalized features
dataset.features = normalized_features
# Load your dataset and continue with training and prediction code as before
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Define your neural network model
class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
learning_rate = .0001
num_epochs = 1

# Initialize the model and optimizer
model = StockPredictor(input_size=1)  # Assuming 1 feature (WinLoss)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print training loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Load your test dataset
csv_test_file_path = 'test_sports.csv'
test_dataset = StockDataset(csv_test_file_path)

# Download historical data for the entire test date range
test_start_date = test_dataset.data.index.min().strftime('%Y-%m-%d')
test_end_date = test_dataset.data.index.max().strftime('%Y-%m-%d')
historical_test_data = yf.download('^GSPC', start=test_start_date, end=test_end_date)

# Fill missing open prices for the test dataset using historical data with forward fill
for i in range(len(test_dataset)):
    if torch.isnan(test_dataset.labels[i]):
        date = test_dataset.data.index[i]
        
        # Check if the date is not NaT before formatting
        if pd.notna(date):
            if i == 0:
                test_dataset.labels[i] = historical_test_data.loc['2023-01-03', 'Open']
            date_str = date.strftime('%Y-%m-%d')
            if date_str in historical_test_data.index:
                test_dataset.labels[i] = historical_test_data.loc[date_str, 'Open']
            else:
                # Forward fill missing values
                test_dataset.labels[i] = test_dataset.labels[i - 1]
                
                
labels = test_dataset.labels.numpy()
labels_tensor = torch.from_numpy(labels)
for i in range(1, len(labels)):
    if torch.isnan(labels_tensor[i]):
        j = i - 1
        while torch.isnan(labels_tensor[j]):
            j -= 1
        labels_tensor[i] = labels_tensor[j]

test_dataset.labels = labels_tensor

# Continue with the evaluation
test_dataloader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

# Evaluation loop
eval_model = StockPredictor(input_size=1)
eval_model.load_state_dict(model.state_dict())  # Copy trained model weights
eval_model.eval()  # Set the model to evaluation mode

with torch.no_grad():
    for test_inputs, test_labels in test_dataloader:
        # Forward pass for the test dataset
        test_outputs = eval_model(test_inputs)
        test_loss = criterion(test_outputs, test_labels.unsqueeze(1))
        nan_indices = torch.isnan(test_outputs).nonzero()
        # print("Indices with NaN values in test outputs:", nan_indices)
        # Check for NaN in test outputs
        if torch.isnan(test_outputs).any():
            raise ValueError("NaN values found in test outputs.")
        else:
            print("No NaN values in test outputs.")


# Print test loss
print(f'Test Loss: {test_loss.item():.4f}')

# Visualize the actual and predicted stock prices
plt.figure(figsize=(12, 6))

# Plot actual stock prices
plt.plot(test_dataset.data.index, test_dataset.labels.numpy(), label='Actual Stock Price', color='blue')

# Plot predicted stock prices
plt.plot(test_dataset.data.index, test_outputs.numpy(), label='Predicted Stock Price', color='orange')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()

# Show the plot
plt.show()


