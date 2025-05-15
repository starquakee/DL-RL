import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import copy
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

data = pd.read_csv("household_power_consumption.txt", sep=";", na_values=["?", "NaN"])
data.dropna(inplace=True)
data["Datetime"] = pd.to_datetime(data["Date"] + " " + data["Time"], format="%d/%m/%Y %H:%M:%S")
data.set_index("Datetime", inplace=True)
data.drop(["Date", "Time"], axis=1, inplace=True)

def remove_outliers(df, column, sigma=3):
    mean_val = df[column].mean()
    std_val = df[column].std()
    return df[(df[column] >= mean_val - sigma * std_val) &
              (df[column] <= mean_val + sigma * std_val)]

data = remove_outliers(data, "Global_active_power")

data["hour"] = data.index.hour
data["weekday"] = data.index.weekday
data["global_active_ma60"] = data["Global_active_power"].rolling(window=60, min_periods=1).mean()

plt.figure(figsize=(12, 4))
plt.plot(data["Global_active_power"].iloc[:1000])
plt.title("Electricity Load Curve")
plt.xlabel("Time")
plt.ylabel("Global Active Power")
plt.savefig("load_curve.png")
plt.show()

plt.figure(figsize=(10, 8))
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

def create_sequences(df, feature_cols, target_col, window_size=60):
    X, y = [], []
    data_arr = df[feature_cols + [target_col]].values
    for i in range(len(data_arr) - window_size):
        X.append(data_arr[i:i+window_size, :-1])
        y.append(data_arr[i+window_size, -1])
    return np.array(X), np.array(y)

feature_cols = ["Global_active_power", "hour", "weekday", "global_active_ma60"]
target_col = "Global_active_power"
window_size = 60

X, y = create_sequences(data, feature_cols, target_col, window_size=window_size)

num_features = len(feature_cols)
scalers = {}
for i, col in enumerate(feature_cols):
    scaler = StandardScaler()
    X_feature = X[:, :, i].reshape(-1, 1)
    scaler.fit(X_feature)
    scalers[col] = scaler
    X[:, :, i] = scaler.transform(X[:, :, i].reshape(-1, 1)).reshape(X[:, :, i].shape)

target_scaler = StandardScaler()
y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

total_samples = len(X)
train_end = int(total_samples * 0.5)
val_end = int(total_samples * 0.75)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

print("Data preprocessing, feature engineering, and sequence generation completed.")

def set_random_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_random_seed(42)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerTimeSeries(nn.Module):
    def __init__(self, input_size, d_model=64, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.input_linear = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.d_model = d_model

    def forward(self, src):
        src = src.transpose(0, 1)
        src = self.input_linear(src) * np.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        out = self.decoder(output[-1, :, :])
        return out.squeeze(-1)

input_size = num_features
d_model = 64
nhead = 4
num_layers = 3
dropout = 0.1

model = TransformerTimeSeries(input_size, d_model, nhead, num_layers, dropout)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 64
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

num_epochs = 50
patience = 5
best_val_loss = float("inf")
best_model_wts = copy.deepcopy(model.state_dict())
patience_counter = 0

train_losses = []
val_losses = []

for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss_epoch = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss_epoch += loss.item() * X_batch.size(0)

    train_loss_epoch /= len(train_loader.dataset)
    train_losses.append(train_loss_epoch)

    model.eval()
    val_loss_epoch = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss_epoch += loss.item() * X_batch.size(0)
    val_loss_epoch /= len(val_loader.dataset)
    val_losses.append(val_loss_epoch)

    print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss_epoch:.4f}, Val Loss: {val_loss_epoch:.4f}")

    if val_loss_epoch < best_val_loss:
        best_val_loss = val_loss_epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

    lr_scheduler.step(val_loss_epoch)

model.load_state_dict(best_model_wts)

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curve")
plt.legend()
plt.savefig("transformer_loss_curve.png")
plt.show()

model.eval()
predictions = []
actuals = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        predictions.extend(outputs.cpu().numpy())
        actuals.extend(y_batch.numpy())

predictions_inv = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
actuals_inv = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()

mse = mean_squared_error(actuals_inv, predictions_inv)
mae = mean_absolute_error(actuals_inv, predictions_inv)
rmse = math.sqrt(mse)

print(f"Test MSE: {mse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(actuals_inv[:200], label="Actual Values")
plt.plot(predictions_inv[:200], label="Predictions")
plt.xlabel("Sample Index")
plt.ylabel("Global Active Power")
plt.title("Predictions vs Actual Comparison")
plt.legend()
plt.savefig("transformer_predictions1.png")
plt.show()
