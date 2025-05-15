# %matplotlib inline

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use('TkAgg')  # 或者使用 'Qt5Agg'（需要安装 PyQt5 或 PySide2）
# import matplotlib.pyplot as plt

import numpy as np


# ============================
# 1. 自定义 Dataset 类与数据生成
# ============================
class CustomDataset(Dataset):
    """
    自定义数据集类，输入为二维特征，输出为一维标签
    """

    def __init__(self, features, outputs):
        self.features = features
        self.outputs = outputs

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.outputs[idx]


def generate_data(num_samples, noise_std=0.1):
    """
    生成数据：
    - features：形状 (num_samples, 2) 的二维输入特征，采样自正态分布
    - outputs：根据函数 y = 3*x1 + 2*x2 + 1 加上噪声生成标签
    """
    # 生成二维输入特征
    X = torch.randn(num_samples, 2)
    # 定义真实权重和偏置
    true_w = torch.tensor([3.0, 2.0]).unsqueeze(1)  # shape: (2, 1)
    true_b = 1.0
    # 根据平面函数生成标签
    y = X.matmul(true_w) + true_b
    # 添加噪声
    noise = noise_std * torch.randn(num_samples, 1)
    y += noise
    return X, y


# ============================
# 2. 模型定义：线性回归模型
# ============================
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        # 使用单层全连接层实现线性回归
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# ============================
# 3. 模型训练与验证函数
# ============================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * features.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)

        # 验证阶段
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                outputs = model(features)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * features.size(0)
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        val_losses.append(epoch_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    return train_losses, val_losses


# ============================
# 4. 数据准备、训练、绘图与结果输出
# ============================
# 设置随机种子，确保实验结果可复现
torch.manual_seed(42)

# 生成数据
num_samples = 1000
noise_std = 0.1
features, outputs = generate_data(num_samples, noise_std)

# 创建自定义 Dataset
dataset = CustomDataset(features, outputs)

# 数据集分割：80% 用于训练，20% 用于验证
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 创建 DataLoader，设置批量大小
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数与优化器
input_dim = 2
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练模型
num_epochs = 100
train_losses, val_losses = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)

# 绘制训练与验证损失曲线（内嵌显示）
epochs = np.arange(1, num_epochs + 1)
plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label='Train Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training and Validation Loss Curve')
plt.legend()
plt.show()

# 输出训练后的模型参数
print("Trained model parameters:")
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data}")

print("\nDiscussion and Conclusion:")
print(
    "从损失曲线来看，训练集和验证集的损失均稳定下降，并在后期趋于平稳，表明模型拟合效果良好，未出现明显的过拟合或欠拟合现象。")
