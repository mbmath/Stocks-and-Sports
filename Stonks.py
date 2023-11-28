import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define your custom dataset
class StockDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file, parse_dates=['Date'])
        self.data = self.data.set_index('Date')
        self.features = torch.tensor(self.data[['WinLoss']].values, dtype=torch.float32)
        self.labels = torch.tensor(self.data['Open'].values, dtype=torch.float32)

    def check_for_nans(self):
        # Check for NaN values in features and labels
        nan_features = torch.isnan(self.features).any()
        nan_labels = torch.isnan(self.labels).any()

        if nan_labels:
            raise ValueError("Dataset contains NaN labels. Please handle missing data before creating the dataset.")
        if nan_features:
            raise ValueError("Dataset contains NaN features. Please handle missing data before creating the dataset.")
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Load your dataset
csv_file_path = 'train_sports_sum.csv'
dataset = StockDataset(csv_file_path)

# Download historical data for the entire date range
start_date = dataset.data.index.min().strftime('%Y-%m-%d')
end_date = dataset.data.index.max().strftime('%Y-%m-%d')
historical_data = yf.download('^GSPC', start=start_date, end=end_date)

# Fill missing open prices using historical data with forward fill
# fill_open_prices(dataset, historical_data)

for i in range(len(dataset)):
    if torch.isnan(dataset.features[i]):
        print(i)


dataset.labels[0] = historical_data.loc['2018-01-02', 'Close'] - historical_data.loc['2018-01-02', 'Open']
for i in range(len(dataset)):
    if torch.isnan(dataset.labels[i]):
        date = dataset.data.index[i]
        
       
        date_str = date.strftime('%Y-%m-%d')
        if date_str in historical_data.index:
             dataset.labels[i] = historical_data.loc[date_str, 'Close'] - historical_data.loc[date_str, 'Open']
                # print(dataset.labels[i])
        else:
                # Forward fill missing values
            dataset.labels[i] = dataset.labels[i - 1]
                # print(dataset.labels[i])

for i in range(len(dataset)):
    if torch.isnan(dataset.labels[i]):
        date = dataset.data.index[i]
        
        # Check if the date is not NaT before formatting

        date_str = date.strftime('%Y-%m-%d')
        print(f"NaN label at index {i}, date: {date_str}")
                
dataset.check_for_nans()

dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

# Define your neural network model
class StockPredictor(nn.Module):
    def __init__(self, input_size):
        super(StockPredictor, self).__init__()
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.fc(x)

# Hyperparameters
learning_rate = .001
num_epochs = 1

# Initialize the model and optimizer
model = StockPredictor(input_size=1)  # Assuming 1 feature (WinLoss)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    for inputs, labels in dataloader:
        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))

        # Backward and optimize
        
        loss.backward()
        optimizer.step()

    # Print training loss
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


# Load your test dataset
csv_test_file_path = 'test_sports_sum.csv'
test_dataset = StockDataset(csv_test_file_path)

# Download historical data for the entire test date range
test_start_date = test_dataset.data.index.min().strftime('%Y-%m-%d')
test_end_date = test_dataset.data.index.max().strftime('%Y-%m-%d')
historical_test_data = yf.download('^GSPC', start=test_start_date, end=test_end_date)

# Fill missing open prices for the test dataset using historical data with forward fill
test_dataset.labels[0] = historical_test_data.loc['2023-01-03', 'Open']
test_dataset.labels[1] = historical_test_data.loc['2023-01-03', 'Open']
for i in range(len(test_dataset)):
    if torch.isnan(test_dataset.labels[i]):
        date = test_dataset.data.index[i]
        
        
        date_str = date.strftime('%Y-%m-%d')
        if date_str in historical_test_data.index:
            test_dataset.labels[i] = historical_test_data.loc[date_str, 'Open']
        else:
                # Forward fill missing values
            test_dataset.labels[i] = test_dataset.labels[i - 1]

for i in range(len(test_dataset)):
    if torch.isnan(test_dataset.labels[i]):
        date = test_dataset.data.index[i]
        
        # Check if the date is not NaT before formatting

        date_str = date.strftime('%Y-%m-%d')
        print(f"NaN label at index {i}, date: {date_str}")
                
dataset.check_for_nans()
                


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

# for i in range(len(test_outputs)):
#     print(test_outputs[i])
# Visualize the actual and predicted stock prices
plt.figure(figsize=(12, 6))

# Plot actual stock prices
plt.plot(test_dataset.data.index, test_dataset.labels.numpy(), label='Actual Stock Price', color='blue')

# Output Stock Price instead of Change for Graphing Purposes
test_outputs = test_outputs.numpy()
test_outputs = test_outputs.reshape(1, -1)
start_price_test = test_dataset.labels[0]
test_outputs_final = [start_price_test]
for i in range(len(test_outputs)):
    # for j in test_outputs_final:
    #     temp_sum = sum(test_outputs_final)
    
    # test_outputs_final.append(test_outputs[:, i] + np.sum(test_outputs_final, axis=0))
    current_prediction = test_outputs[[0, i]]
    cumulative_sum = np.sum(test_outputs_final) + current_prediction
    test_outputs_final.append(cumulative_sum)
    
test_outputs_final.pop(0)
test_outputs_final = np.array(test_outputs_final)
test_outputs_final = test_outputs_final.reshape(317, 2)
test_outputs_final[:, 1] = test_outputs_final[:, 0]

# Reshape it back to (317, 1)
test_outputs_final = test_outputs_final[:, 0].reshape(317, 1)
# ************************************************************************************    

# Plot predicted stock prices
plt.plot(test_dataset.data.index, test_outputs_final, label='Predicted Stock Price', color='orange')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices')
plt.legend()

# Show the plot
plt.show()

plt.plot(test_dataset.data.index, test_outputs_final, label='Predicted Stock Price', color='orange')

# Set labels and title
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.title('Predicted Stock Prices')
plt.legend()

# Show the plot
plt.show()

# Show the plot
plt.show()

# Assuming 'WinLoss' and 'Open' are the columns of interest
df = pd.DataFrame()
df['Actual'] = test_dataset.labels.numpy()
df['Predicted'] = test_outputs_final

# Check for missing values in the DataFrame
if df.isnull().values.any():
    raise ValueError("DataFrame contains missing values. Please handle missing data before calculating correlations.")

# Calculate the correlation between 'Actual' and 'Predicted'
correlation = df['Actual'].corr(df['Predicted'])

# Print the correlation
print(f"Correlation between 'Actual' and 'Predicted': {correlation:.4f}")

# Create a scatter plot
plt.scatter(df['Actual'], df['Predicted'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Scatter Plot')
plt.show()


