# import pandas as pd
# import numpy as np
# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['SimHei']  # or ['Microsoft YaHei']
# plt.rcParams['axes.unicode_minus'] = False
# from sklearn.preprocessing import StandardScaler
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import math
# from tqdm import tqdm
# import time
#
# #######################################
# # 数据预处理函数：读取文件、清洗缺失值和异常值、特征工程、归一化
# #######################################
# def data_preprocessing(file_path='household_power_consumption.txt'):
#     # 读取数据（注意：FutureWarning 建议改为先读后 pd.to_datetime，这里仍使用该方式）
#     df = pd.read_csv(file_path, sep=';', parse_dates=[[0, 1]], dayfirst=True)
#     df.columns = ['Datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
#                   'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
#     df.set_index('Datetime', inplace=True)
#
#     # 打印缺失值行（用于检查原始数据问题）
#     missing_rows = df[df.isnull().any(axis=1)]
#     print("Missing rows:")
#     print(missing_rows)
#     print("Raw data preview:")
#     print(df.head())
#
#     # 数据清洗：类型转换和清洗空白字符
#     df = df.infer_objects()
#     numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
#                     'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
#     for col in numeric_cols:
#         df[col] = df[col].astype(str).str.strip()
#         df[col] = pd.to_numeric(df[col], errors='coerce')
#
#     print("\nMissing value counts after conversion:")
#     print(df.isnull().sum())
#
#     # 删除包含缺失值的行
#     df = df.dropna()
#
#     # 异常值处理：IQR方法
#     def remove_outliers(data, col):
#         Q1 = data[col].quantile(0.25)
#         Q3 = data[col].quantile(0.75)
#         IQR = Q3 - Q1
#         lower_bound = Q1 - 1.5 * IQR
#         upper_bound = Q3 + 1.5 * IQR
#         return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
#
#     for col in numeric_cols:
#         df = remove_outliers(df, col)
#
#     # 特征工程：提取时间特征及周期编码
#     df['hour'] = df.index.hour
#     df['dayofweek'] = df.index.dayofweek
#     df['month'] = df.index.month
#
#     df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
#     df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
#     df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
#     df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
#
#     # 滑动窗口统计：过去3个时刻的均值
#     window_size = 3
#     df['global_active_power_ma'] = df['Global_active_power'].rolling(window=window_size).mean()
#     df = df.dropna()
#
#     # 取前10%连续数据（保证时序完整）
#     # df = df.iloc[:int(len(df) * 0.1)]
#
#     # 选择用于建模的特征
#     features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
#                 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour', 'dayofweek', 'month',
#                 'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'global_active_power_ma']
#     df_features = df[features]
#
#     # 数据归一化：使用 StandardScaler
#     scaler = StandardScaler()
#     scaled_features = scaler.fit_transform(df_features)
#     df_scaled = pd.DataFrame(scaled_features, index=df_features.index, columns=features)
#
#     print(f"Preprocessing completed, sample count: {len(df_scaled)}")
#     return df_scaled, features, scaler
#
#
# #######################################
# # 数据集构造
# #######################################
# class TimeSeriesDataset(Dataset):
#     def __init__(self, data, input_seq_len, pred_seq_len):
#         """
#         data: pd.DataFrame (normalized features)
#         input_seq_len: input sequence length
#         pred_seq_len: prediction sequence length (target 'Global_active_power' assumed to be first column)
#         """
#         self.data = data
#         self.input_seq_len = input_seq_len
#         self.pred_seq_len = pred_seq_len
#         self.num_samples = len(data) - input_seq_len - pred_seq_len + 1
#
#     def __len__(self):
#         return self.num_samples
#
#     def __getitem__(self, idx):
#         x = self.data.iloc[idx: idx + self.input_seq_len].values.astype(np.float32)
#         y = self.data.iloc[idx + self.input_seq_len: idx + self.input_seq_len + self.pred_seq_len]['Global_active_power'].values.astype(np.float32)
#         y = y.reshape(-1, 1)
#         return torch.tensor(x), torch.tensor(y)
#
#
# #######################################
# # 模型定义
# #######################################
# # 3.1 MLP Baseline 模型
# class MLPForecast(nn.Module):
#     def __init__(self, input_seq_len, num_features, pred_seq_len):
#         super(MLPForecast, self).__init__()
#         self.input_dim = input_seq_len * num_features
#         self.pred_seq_len = pred_seq_len
#         self.model = nn.Sequential(
#             nn.Linear(self.input_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 64),
#             nn.ReLU(),
#             nn.Linear(64, pred_seq_len)
#         )
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         x = x.view(batch_size, -1)
#         out = self.model(x)
#         out = out.unsqueeze(-1)  # (batch, pred_seq_len, 1)
#         return out
#
#
# # 3.2 Transformer 模型及位置编码
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, dropout=0.1, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         self.dropout = nn.Dropout(dropout)
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0)  # (1, max_len, d_model)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x):
#         x = x + self.pe[:, :x.size(1), :]
#         return self.dropout(x)
#
# class TransformerForecast(nn.Module):
#     def __init__(self, input_seq_len, pred_seq_len, num_features, d_model=64, nhead=4,
#                  num_encoder_layers=2, num_decoder_layers=2, dropout=0.1):
#         super(TransformerForecast, self).__init__()
#         self.input_seq_len = input_seq_len
#         self.pred_seq_len = pred_seq_len
#         self.num_features = num_features
#         self.d_model = d_model
#
#         # Input projection
#         self.input_projection = nn.Linear(num_features, d_model)
#         self.pos_encoder = PositionalEncoding(d_model, dropout)
#
#         # Decoder input projection (target: single variable)
#         self.target_projection = nn.Linear(1, d_model)
#         self.pos_decoder = PositionalEncoding(d_model, dropout)
#
#         self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
#                                           num_encoder_layers=num_encoder_layers,
#                                           num_decoder_layers=num_decoder_layers,
#                                           dropout=dropout)
#         self.output_projection = nn.Linear(d_model, 1)
#
#     def forward(self, src, tgt):
#         src = self.input_projection(src)  # (batch, input_seq_len, d_model)
#         src = self.pos_encoder(src)
#         src = src.transpose(0, 1)  # (input_seq_len, batch, d_model)
#
#         tgt = self.target_projection(tgt)  # (batch, pred_seq_len, d_model)
#         tgt = self.pos_decoder(tgt)
#         tgt = tgt.transpose(0, 1)  # (pred_seq_len, batch, d_model)
#
#         tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
#         out = self.transformer(src, tgt, tgt_mask=tgt_mask)
#         out = out.transpose(0, 1)  # (batch, pred_seq_len, d_model)
#         out = self.output_projection(out)  # (batch, pred_seq_len, 1)
#         return out
#
# # Teacher forcing input for Transformer: right-shift target sequence with zero at beginning
# def generate_tgt_input(y):
#     start_token = torch.zeros((y.size(0), 1, 1), device=y.device)
#     y_input = torch.cat([start_token, y[:, :-1, :]], dim=1)
#     return y_input
#
#
# #######################################
# # 训练与验证函数
# #######################################
# def train_model(model, train_loader, val_loader, num_epochs, lr, model_type="mlp", patience=10):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.to(device)
#     criterion = nn.MSELoss()
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)
#
#     best_val_loss = float('inf')
#     epochs_no_improve = 0
#     train_losses = []
#     val_losses = []
#
#     for epoch in tqdm(range(num_epochs), desc="Training"):
#         model.train()
#         running_loss = 0.0
#         for x, y in train_loader:
#             x = x.to(device)
#             y = y.to(device)
#             optimizer.zero_grad()
#             if model_type == "transformer":
#                 y_input = generate_tgt_input(y)
#                 output = model(x, y_input)
#             else:
#                 output = model(x)
#             loss = criterion(output, y)
#             loss.backward()
#             optimizer.step()
#             running_loss += loss.item() * x.size(0)
#         epoch_train_loss = running_loss / len(train_loader.dataset)
#         train_losses.append(epoch_train_loss)
#
#         model.eval()
#         running_val_loss = 0.0
#         with torch.no_grad():
#             for x_val, y_val in val_loader:
#                 x_val = x_val.to(device)
#                 y_val = y_val.to(device)
#                 if model_type == "transformer":
#                     y_val_input = generate_tgt_input(y_val)
#                     output_val = model(x_val, y_val_input)
#                 else:
#                     output_val = model(x_val)
#                 loss_val = criterion(output_val, y_val)
#                 running_val_loss += loss_val.item() * x_val.size(0)
#         epoch_val_loss = running_val_loss / len(val_loader.dataset)
#         val_losses.append(epoch_val_loss)
#         scheduler.step(epoch_val_loss)
#         print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")
#
#         if epoch_val_loss < best_val_loss:
#             best_val_loss = epoch_val_loss
#             best_model_state = model.state_dict()
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 print("Early stopping triggered.")
#                 break
#
#     model.load_state_dict(best_model_state)
#     return model, train_losses, val_losses
#
#
# #######################################
# # Auto-regressive inference for Transformer
# #######################################
# def transformer_autoregressive_inference(model, x, pred_seq_len):
#     device = x.device
#     batch_size = x.size(0)
#     decoder_input = torch.zeros(batch_size, 1, 1, device=device)
#     predictions = []
#     for t in range(pred_seq_len):
#         output = model(x, decoder_input)
#         next_pred = output[:, -1:, :]
#         predictions.append(next_pred)
#         decoder_input = torch.cat([decoder_input, next_pred], dim=1)
#     predictions = torch.cat(predictions, dim=1)
#     return predictions
#
# #######################################
# # 测试与评估函数
# #######################################
# def evaluate_model(model, test_loader, model_type="mlp", use_autoregressive=False):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model.eval()
#     predictions = []
#     ground_truth = []
#     criterion = nn.MSELoss()
#     total_loss = 0.0
#     with torch.no_grad():
#         for x, y in test_loader:
#             x = x.to(device)
#             y = y.to(device)
#             if model_type == "transformer":
#                 if use_autoregressive:
#                     output = transformer_autoregressive_inference(model, x, y.size(1))
#                 else:
#                     y_input = torch.zeros_like(y)
#                     output = model(x, y_input)
#             else:
#                 output = model(x)
#             loss = criterion(output, y)
#             total_loss += loss.item() * x.size(0)
#             predictions.append(output.cpu().numpy())
#             ground_truth.append(y.cpu().numpy())
#     avg_loss = total_loss / len(test_loader.dataset)
#     predictions = np.concatenate(predictions, axis=0)
#     ground_truth = np.concatenate(ground_truth, axis=0)
#     return avg_loss, predictions, ground_truth
#
# #######################################
# # 辅助函数：将预测（或真实值）从标准化空间逆变换到原始物理值
# #######################################
# def inverse_transform_target(arr, scaler, num_features, target_index=0):
#     # arr shape: (N, 1)
#     N = arr.shape[0]
#     dummy = np.zeros((N, num_features))
#     dummy[:, target_index] = arr[:, 0]
#     inv = scaler.inverse_transform(dummy)
#     return inv[:, target_index].reshape(-1, 1)
#
# #######################################
# # 计算评价指标（均方误差、平均绝对误差、均方根误差）
# #######################################
# def calc_metrics(y_true, y_pred):
#     mse = np.mean((y_true - y_pred) ** 2)
#     mae = np.mean(np.abs(y_true - y_pred))
#     rmse = math.sqrt(mse)
#     return mse, mae, rmse
#
# #######################################
# # 主函数入口
# #######################################
# def main():
#     # Data Preprocessing
#     df_scaled, features, scaler = data_preprocessing(file_path='household_power_consumption.txt')
#
#     # Construct Time-Series Dataset
#     INPUT_SEQ_LEN = 60
#     PRED_SEQ_LEN = 60
#     dataset = TimeSeriesDataset(df_scaled, INPUT_SEQ_LEN, PRED_SEQ_LEN)
#     num_samples = len(dataset)
#     print(f"Number of samples: {num_samples}")
#
#     # Split dataset: 70% train, 15% validation, 15% test
#     train_size = int(0.7 * num_samples)
#     val_size = int(0.15 * num_samples)
#     test_size = num_samples - train_size - val_size
#     train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
#     val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size)))
#     test_dataset = torch.utils.data.Subset(dataset, list(range(train_size + val_size, num_samples)))
#
#     BATCH_SIZE = 64
#     num_workers = 8  # Adjust based on hardware
#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
#     test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
#
#     NUM_FEATURES = len(features)
#     NUM_EPOCHS = 50
#     LEARNING_RATE = 5e-4
#
#     # Train Transformer Model
#     print("\nTraining Transformer Model...")
#     transformer_model = TransformerForecast(INPUT_SEQ_LEN, PRED_SEQ_LEN, NUM_FEATURES, d_model=64, nhead=4,
#                                             num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
#     transformer_model, tf_train_losses, tf_val_losses = train_model(transformer_model, train_loader, val_loader,
#                                                                     num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
#                                                                     model_type="transformer", patience=10)
#
#     # Train MLP Baseline Model
#     print("\nTraining MLP Baseline Model...")
#     mlp_model = MLPForecast(INPUT_SEQ_LEN, NUM_FEATURES, PRED_SEQ_LEN)
#     mlp_model, mlp_train_losses, mlp_val_losses = train_model(mlp_model, train_loader, val_loader,
#                                                               num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
#                                                               model_type="mlp", patience=10)
#
#     # Testing and Evaluation (use auto-regressive for Transformer)
#     mlp_test_loss, mlp_preds, mlp_gt = evaluate_model(mlp_model, test_loader, model_type="mlp")
#     print(f"\nMLP Test MSE Loss (normalized): {mlp_test_loss:.6f}")
#     tf_test_loss, tf_preds, tf_gt = evaluate_model(transformer_model, test_loader, model_type="transformer", use_autoregressive=True)
#     print(f"Transformer Test MSE Loss (normalized): {tf_test_loss:.6f}")
#
#     # Inverse transform predictions and ground truth to physical values (assume target is column 0)
#     # 将所有预测样本展平后逐个逆变换，每个样本是 (pred_seq_len, 1)
#     inv_mlp_preds = []
#     inv_mlp_gt = []
#     inv_tf_preds = []
#     inv_tf_gt = []
#     for i in range(mlp_preds.shape[0]):
#         inv_mlp_preds.append(inverse_transform_target(mlp_preds[i], scaler, NUM_FEATURES, target_index=0))
#         inv_mlp_gt.append(inverse_transform_target(mlp_gt[i], scaler, NUM_FEATURES, target_index=0))
#     for i in range(tf_preds.shape[0]):
#         inv_tf_preds.append(inverse_transform_target(tf_preds[i], scaler, NUM_FEATURES, target_index=0))
#         inv_tf_gt.append(inverse_transform_target(tf_gt[i], scaler, NUM_FEATURES, target_index=0))
#     inv_mlp_preds = np.array(inv_mlp_preds)
#     inv_mlp_gt = np.array(inv_mlp_gt)
#     inv_tf_preds = np.array(inv_tf_preds)
#     inv_tf_gt = np.array(inv_tf_gt)
#
#     # Calculate metrics on original scale
#     mlp_mse, mlp_mae, mlp_rmse = calc_metrics(inv_mlp_gt.flatten(), inv_mlp_preds.flatten())
#     tf_mse, tf_mae, tf_rmse = calc_metrics(inv_tf_gt.flatten(), inv_tf_preds.flatten())
#     print("\nMLP Metrics (original scale):")
#     print(f"MSE: {mlp_mse:.6f}, MAE: {mlp_mae:.6f}, RMSE: {mlp_rmse:.6f}")
#     print("\nTransformer Metrics (original scale):")
#     print(f"MSE: {tf_mse:.6f}, MAE: {tf_mae:.6f}, RMSE: {tf_rmse:.6f}")
#
#     # Plot prediction results (using the first sample from test set) on original scale
#     sample_idx = 0
#     time_steps = np.arange(PRED_SEQ_LEN)
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(time_steps, inv_mlp_gt[sample_idx, :, 0], label='True')
#     plt.plot(time_steps, inv_mlp_preds[sample_idx, :, 0], label='MLP Prediction')
#     plt.xlabel('Prediction Time Steps')
#     plt.ylabel('Global Active Power (Original Scale)')
#     plt.title('MLP Prediction vs True Value (Original Scale)')
#     plt.legend()
#     plt.savefig("mlp_prediction_original.png", dpi=300)
#     # plt.show()
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(time_steps, inv_tf_gt[sample_idx, :, 0], label='True')
#     plt.plot(time_steps, inv_tf_preds[sample_idx, :, 0], label='Transformer Prediction')
#     plt.xlabel('Prediction Time Steps')
#     plt.ylabel('Global Active Power (Original Scale)')
#     plt.title('Transformer Prediction vs True Value (Original Scale)')
#     plt.legend()
#     plt.savefig("transformer_prediction_original.png", dpi=300)
#     # plt.show()
#
#     # Plot training/validation loss curves (normalized loss)
#     plt.figure(figsize=(10, 5))
#     plt.plot(mlp_train_losses, label='MLP Train Loss')
#     plt.plot(mlp_val_losses, label='MLP Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('MLP Model Training/Validation Loss')
#     plt.legend()
#     plt.savefig("mlp_loss.png", dpi=300)
#     # plt.show()
#
#     plt.figure(figsize=(10, 5))
#     plt.plot(tf_train_losses, label='Transformer Train Loss')
#     plt.plot(tf_val_losses, label='Transformer Validation Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.title('Transformer Model Training/Validation Loss')
#     plt.legend()
#     plt.savefig("transformer_loss.png", dpi=300)
#     # plt.show()
#
#
# if __name__ == '__main__':
#     main()
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # or ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
from tqdm import tqdm
import time

#######################################
# Data Preprocessing Function:
# Read file, clean missing/abnormal values, perform feature engineering, and normalize.
#######################################
def data_preprocessing(file_path='household_power_consumption.txt'):
    # Read data. (Note: FutureWarning suggests a different method; here we follow the original way.)
    df = pd.read_csv(file_path, sep=';', parse_dates=[[0, 1]], dayfirst=True)
    df.columns = ['Datetime', 'Global_active_power', 'Global_reactive_power', 'Voltage',
                  'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    df.set_index('Datetime', inplace=True)

    # Print rows with missing values (to check raw data issues)
    missing_rows = df[df.isnull().any(axis=1)]
    print("Missing rows:")
    print(missing_rows)
    print("Raw data preview:")
    print(df.head())

    # Data cleaning: infer types and strip whitespace
    df = df.infer_objects()
    numeric_cols = ['Global_active_power', 'Global_reactive_power', 'Voltage',
                    'Global_intensity', 'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    print("\nMissing value counts after conversion:")
    print(df.isnull().sum())

    # Drop rows containing missing values
    df = df.dropna()

    # Remove outliers using IQR method
    def remove_outliers(data, col):
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
    for col in numeric_cols:
        df = remove_outliers(df, col)

    # Feature Engineering: extract time features and cyclic encoding
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['month'] = df.index.month

    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # Sliding window statistics: compute moving average of Global_active_power over past 3 time steps
    window_size = 3
    df['global_active_power_ma'] = df['Global_active_power'].rolling(window=window_size).mean()
    df = df.dropna()  # drop NaN generated by rolling

    # **** 解决问题1：采用连续数据，这里选择前50%数据，保持时序连续性 ****
    df = df.iloc[:int(len(df) * 0.3)]

    # Select features for modeling
    features = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'hour', 'dayofweek', 'month',
                'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 'global_active_power_ma']
    df_features = df[features]

    # Normalize the features using StandardScaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_features)
    df_scaled = pd.DataFrame(scaled_features, index=df_features.index, columns=features)

    print(f"Preprocessing completed, sample count: {len(df_scaled)}")
    return df_scaled, features, scaler


#######################################
# Construct Time-Series Dataset using sliding window
#######################################
class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_seq_len, pred_seq_len):
        """
        data: pd.DataFrame (normalized features)
        input_seq_len: input sequence length
        pred_seq_len: prediction sequence length (target is "Global_active_power", assumed at column 0)
        """
        self.data = data
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.num_samples = len(data) - input_seq_len - pred_seq_len + 1

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x = self.data.iloc[idx: idx + self.input_seq_len].values.astype(np.float32)
        y = self.data.iloc[idx + self.input_seq_len: idx + self.input_seq_len + self.pred_seq_len]['Global_active_power'].values.astype(np.float32)
        y = y.reshape(-1, 1)
        return torch.tensor(x), torch.tensor(y)


#######################################
# Model Definitions
#######################################
# 1. MLP Baseline Model
class MLPForecast(nn.Module):
    def __init__(self, input_seq_len, num_features, pred_seq_len):
        super(MLPForecast, self).__init__()
        self.input_dim = input_seq_len * num_features
        self.pred_seq_len = pred_seq_len
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, pred_seq_len)
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        out = self.model(x)
        out = out.unsqueeze(-1)  # shape: (batch, pred_seq_len, 1)
        return out


# 2. Transformer Model with Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerForecast(nn.Module):
    def __init__(self, input_seq_len, pred_seq_len, num_features, d_model=64, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2, dropout=0.1):
        super(TransformerForecast, self).__init__()
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.num_features = num_features
        self.d_model = d_model

        self.input_projection = nn.Linear(num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.target_projection = nn.Linear(1, d_model)
        self.pos_decoder = PositionalEncoding(d_model, dropout)

        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dropout=dropout)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, src, tgt):
        src = self.input_projection(src)  # (batch, input_seq_len, d_model)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)  # (input_seq_len, batch, d_model)

        tgt = self.target_projection(tgt)  # (batch, pred_seq_len, d_model)
        tgt = self.pos_decoder(tgt)
        tgt = tgt.transpose(0, 1)  # (pred_seq_len, batch, d_model)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(src.device)
        out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = out.transpose(0, 1)  # (batch, pred_seq_len, d_model)
        out = self.output_projection(out)  # (batch, pred_seq_len, 1)
        return out

# Teacher forcing: generate decoder input by shifting target right (zero start token)
def generate_tgt_input(y):
    start_token = torch.zeros((y.size(0), 1, 1), device=y.device)
    y_input = torch.cat([start_token, y[:, :-1, :]], dim=1)
    return y_input

#######################################
# Training and Validation Function
#######################################
def train_model(model, train_loader, val_loader, num_epochs, lr, model_type="mlp", patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        running_loss = 0.0
        # --- 解决问题2: Scheduled Sampling for Transformer ---
        # Decay teacher forcing ratio from 1.0 to 0.5 linearly over training epochs.
        teacher_forcing_ratio = max(1 - (epoch * 0.5 / num_epochs), 0.5)

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if model_type == "transformer":
                # Use scheduled sampling: with probability teacher_forcing_ratio, use teacher forcing.
                if np.random.rand() < teacher_forcing_ratio:
                    y_input = generate_tgt_input(y)
                    output = model(x, y_input)
                else:
                    # Use auto-regressive inference during training.
                    output = transformer_autoregressive_inference(model, x, y.size(1))
            else:
                output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)
        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        model.eval()
        running_val_loss = 0.0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                if model_type == "transformer":
                    # During validation, use auto-regressive inference for consistency with test.
                    output_val = transformer_autoregressive_inference(model, x_val, y_val.size(1))
                else:
                    output_val = model(x_val)
                loss_val = criterion(output_val, y_val)
                running_val_loss += loss_val.item() * x_val.size(0)
        epoch_val_loss = running_val_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)
        scheduler.step(epoch_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} - Train Loss: {epoch_train_loss:.6f}, Val Loss: {epoch_val_loss:.6f}")

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_model_state)
    return model, train_losses, val_losses

#######################################
# Auto-regressive Inference for Transformer
#######################################
def transformer_autoregressive_inference(model, x, pred_seq_len):
    device = x.device
    batch_size = x.size(0)
    decoder_input = torch.zeros(batch_size, 1, 1, device=device)
    predictions = []
    for t in range(pred_seq_len):
        output = model(x, decoder_input)  # shape: (batch, current_length, 1)
        next_pred = output[:, -1:, :]      # take last time step's prediction
        predictions.append(next_pred)
        decoder_input = torch.cat([decoder_input, next_pred], dim=1)
    predictions = torch.cat(predictions, dim=1)  # (batch, pred_seq_len, 1)
    return predictions

#######################################
# Testing and Evaluation Function
#######################################
def evaluate_model(model, test_loader, model_type="mlp", use_autoregressive=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    predictions = []
    ground_truth = []
    criterion = nn.MSELoss()
    total_loss = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)
            if model_type == "transformer":
                if use_autoregressive:
                    output = transformer_autoregressive_inference(model, x, y.size(1))
                else:
                    y_input = torch.zeros_like(y)
                    output = model(x, y_input)
            else:
                output = model(x)
            loss = criterion(output, y)
            total_loss += loss.item() * x.size(0)
            predictions.append(output.cpu().numpy())
            ground_truth.append(y.cpu().numpy())
    avg_loss = total_loss / len(test_loader.dataset)
    predictions = np.concatenate(predictions, axis=0)
    ground_truth = np.concatenate(ground_truth, axis=0)
    return avg_loss, predictions, ground_truth

#######################################
# Helper function: Inverse transform target from normalized space to original physical values.
# Assumes target variable is at index 0.
#######################################
def inverse_transform_target(arr, scaler, num_features, target_index=0):
    # arr shape: (N, 1)
    N = arr.shape[0]
    dummy = np.zeros((N, num_features))
    dummy[:, target_index] = arr[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, target_index].reshape(-1, 1)

#######################################
# Calculate Metrics: MSE, MAE, RMSE
#######################################
def calc_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = math.sqrt(mse)
    return mse, mae, rmse

#######################################
# Main Function
#######################################
def main():
    # Data Preprocessing
    df_scaled, features, scaler = data_preprocessing(file_path='household_power_consumption.txt')

    # Construct Time-Series Dataset
    INPUT_SEQ_LEN = 60
    PRED_SEQ_LEN = 60
    dataset = TimeSeriesDataset(df_scaled, INPUT_SEQ_LEN, PRED_SEQ_LEN)
    num_samples = len(dataset)
    print(f"Number of samples: {num_samples}")

    # Split dataset: 70% train, 15% validation, 15% test
    train_size = int(0.7 * num_samples)
    val_size = int(0.15 * num_samples)
    test_size = num_samples - train_size - val_size
    train_dataset = torch.utils.data.Subset(dataset, list(range(0, train_size)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_size, train_size + val_size)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(train_size + val_size, num_samples)))

    BATCH_SIZE = 64
    num_workers = 4  # Adjust based on hardware
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    NUM_FEATURES = len(features)
    NUM_EPOCHS = 40
    LEARNING_RATE = 1e-3

    # Train Transformer Model
    print("\nTraining Transformer Model...")
    transformer_model = TransformerForecast(INPUT_SEQ_LEN, PRED_SEQ_LEN, NUM_FEATURES, d_model=64, nhead=4,
                                            num_encoder_layers=2, num_decoder_layers=2, dropout=0.1)
    transformer_model, tf_train_losses, tf_val_losses = train_model(transformer_model, train_loader, val_loader,
                                                                    num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                                                                    model_type="transformer", patience=5)

    # Train MLP Baseline Model
    print("\nTraining MLP Baseline Model...")
    mlp_model = MLPForecast(INPUT_SEQ_LEN, NUM_FEATURES, PRED_SEQ_LEN)
    mlp_model, mlp_train_losses, mlp_val_losses = train_model(mlp_model, train_loader, val_loader,
                                                              num_epochs=NUM_EPOCHS, lr=LEARNING_RATE,
                                                              model_type="mlp", patience=5)

    # Testing and Evaluation (use auto-regressive inference for Transformer)
    mlp_test_loss, mlp_preds, mlp_gt = evaluate_model(mlp_model, test_loader, model_type="mlp")
    print(f"\nMLP Test MSE Loss (normalized): {mlp_test_loss:.6f}")
    tf_test_loss, tf_preds, tf_gt = evaluate_model(transformer_model, test_loader, model_type="transformer",
                                                   use_autoregressive=True)
    print(f"Transformer Test MSE Loss (normalized): {tf_test_loss:.6f}")

    # Inverse transform predictions and ground truth to original physical scale (assume target is column 0)
    inv_mlp_preds = []
    inv_mlp_gt = []
    inv_tf_preds = []
    inv_tf_gt = []
    for i in range(mlp_preds.shape[0]):
        inv_mlp_preds.append(inverse_transform_target(mlp_preds[i], scaler, NUM_FEATURES, target_index=0))
        inv_mlp_gt.append(inverse_transform_target(mlp_gt[i], scaler, NUM_FEATURES, target_index=0))
    for i in range(tf_preds.shape[0]):
        inv_tf_preds.append(inverse_transform_target(tf_preds[i], scaler, NUM_FEATURES, target_index=0))
        inv_tf_gt.append(inverse_transform_target(tf_gt[i], scaler, NUM_FEATURES, target_index=0))
    inv_mlp_preds = np.array(inv_mlp_preds)
    inv_mlp_gt = np.array(inv_mlp_gt)
    inv_tf_preds = np.array(inv_tf_preds)
    inv_tf_gt = np.array(inv_tf_gt)

    # Calculate metrics on original scale
    mlp_mse, mlp_mae, mlp_rmse = calc_metrics(inv_mlp_gt.flatten(), inv_mlp_preds.flatten())
    tf_mse, tf_mae, tf_rmse = calc_metrics(inv_tf_gt.flatten(), inv_tf_preds.flatten())
    print("\nMLP Metrics (original scale):")
    print(f"MSE: {mlp_mse:.6f}, MAE: {mlp_mae:.6f}, RMSE: {mlp_rmse:.6f}")
    print("\nTransformer Metrics (original scale):")
    print(f"MSE: {tf_mse:.6f}, MAE: {tf_mae:.6f}, RMSE: {tf_rmse:.6f}")

    # Plot prediction results (first sample from test set) on original scale
    sample_idx = 0
    time_steps = np.arange(PRED_SEQ_LEN)

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, inv_mlp_gt[sample_idx, :, 0], label='True')
    plt.plot(time_steps, inv_mlp_preds[sample_idx, :, 0], label='MLP Prediction')
    plt.xlabel('Prediction Time Steps')
    plt.ylabel('Global Active Power (Original Scale)')
    plt.title('MLP Prediction vs True Value (Original Scale)')
    plt.legend()
    plt.savefig("mlp_prediction_original.png", dpi=300)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, inv_tf_gt[sample_idx, :, 0], label='True')
    plt.plot(time_steps, inv_tf_preds[sample_idx, :, 0], label='Transformer Prediction')
    plt.xlabel('Prediction Time Steps')
    plt.ylabel('Global Active Power (Original Scale)')
    plt.title('Transformer Prediction vs True Value (Original Scale)')
    plt.legend()
    plt.savefig("transformer_prediction_original.png", dpi=300)
    # plt.show()

    # Plot training/validation loss curves (normalized loss)
    plt.figure(figsize=(10, 5))
    plt.plot(mlp_train_losses, label='MLP Train Loss')
    plt.plot(mlp_val_losses, label='MLP Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLP Model Training/Validation Loss')
    plt.legend()
    plt.savefig("mlp_loss.png", dpi=300)
    # plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(tf_train_losses, label='Transformer Train Loss')
    plt.plot(tf_val_losses, label='Transformer Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Transformer Model Training/Validation Loss')
    plt.legend()
    plt.savefig("transformer_loss.png", dpi=300)
    # plt.show()


if __name__ == '__main__':
    main()
