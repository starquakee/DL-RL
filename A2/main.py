import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

import matplotlib

matplotlib.use('TkAgg')  # 或者换成 'Agg'（非交互模式）

# 1. 数据加载模块：下载并加载 MNIST 数据集，并进行预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)


# 2. 模型实现：Logistic Regression 和 MLP

# Logistic Regression 模型（单层全连接网络）
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


# MLP 模型（包含一个隐藏层）
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 3. 训练模块：训练函数，记录每个 epoch 的平均 Loss
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()  # 设置模型为训练模式
    losses = []
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        print(f'Epoch [{epoch + 1}/ {num_epochs}], Loss: {avg_loss:.4f}')

    return losses


# 2.5 测试模块：测试模型性能
def model_test(model, test_loader):
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 禁止计算梯度，加速测试
        for images, labels in test_loader:
            images, labels = images.view(-1, 28 * 28).to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测类别

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'模型在测试集上的准确率: {accuracy:.2f}%')
    return accuracy


# 训练参数
input_dim = 28 * 28
output_dim = 10
hidden_dim = 128
num_epochs = 10

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# 训练 Logistic Regression 模型
model_lr = LogisticRegression(input_dim, output_dim).to(device)
optimizer_lr = optim.SGD(model_lr.parameters(), lr=0.1)

print("Training Logistic Regression Model:")
losses_lr = train_model(model_lr, train_loader, criterion, optimizer_lr, num_epochs=num_epochs)

# 训练 MLP 模型
model_mlp = MLP(input_dim, hidden_dim, output_dim).to(device)
optimizer_mlp = optim.Adam(model_mlp.parameters(), lr=0.001)

print("\nTraining MLP Model:")
losses_mlp = train_model(model_mlp, train_loader, criterion, optimizer_mlp, num_epochs=num_epochs)

# 测试 Logistic Regression 模型
print("\nTesting Logistic Regression Model:")
accuracy_lr = model_test(model_lr, test_loader)

# 测试 MLP 模型
print("\nTesting MLP Model:")
accuracy_mlp = model_test(model_mlp, test_loader)

# 4. 绘制训练集 Loss 曲线
plt.figure(figsize=(8, 6))
plt.plot(range(1, num_epochs + 1), losses_lr, marker='o', label='Logistic Regression')
plt.plot(range(1, num_epochs + 1), losses_mlp, marker='o', label='MLP')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.legend()
plt.grid(True)

# 解决 Matplotlib 在 PyCharm 中的问题
plt.savefig("loss_curve.png")  # 避免 show() 出错
print("Loss 曲线已保存为 loss_curve.png")
