import torch
from torchvision import datasets, transforms

# 尝试加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=False, transform=transforms.ToTensor())
test_dataset = datasets.MNIST(root='./data', train=False, download=False, transform=transforms.ToTensor())

# 检查数据集大小
print(f"训练集样本数: {len(train_dataset)}")  # 应该是 60000
print(f"测试集样本数: {len(test_dataset)}")  # 应该是 10000

# 显示一张图片
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')  # 或者换成 'Agg'（非交互模式）
image, label = train_dataset[0]  # 取第一张图片
plt.imshow(image.squeeze(), cmap='gray')  # image 是 1x28x28，需要 squeeze() 去掉通道维度
plt.title(f"Label: {label}")
plt.show()
