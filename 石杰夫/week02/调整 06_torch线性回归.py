import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# 生成模拟数据
# 生成sin函数数据 y = sin(x) + noise
# =============================================================================
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
# 形状为 (100, 1)，在 [0, 2π] 范围内均匀分布的100个点

y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)
# 添加高斯噪声，噪声标准差为0.1

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# =============================================================================
# 定义多层神经网络
# 使用多层神经网络（包含非线性激活函数）
# 网络结构：
# 输入层(1) -> 隐藏层1(50) -> ReLU -> 隐藏层2(50) -> ReLU -> 输出层(1)
# =============================================================================
class SinNet(nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.hidden1 = nn.Linear(1, 50)
        self.hidden2 = nn.Linear(50, 50)
        self.output = nn.Linear(50, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = SinNet()

print("模型结构：")
print(model)
print("---" * 10)

# =============================================================================
# 定义损失函数和优化器
# 针对神经网络的所有参数 model.parameters()，使用Adam优化器
# =============================================================================
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# =============================================================================
# 训练模型
# 使用模型前向传播 y_pred = model(X)
# =============================================================================
num_epochs = 2000
losses = []

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# =============================================================================
# 绘制结果
# 绘制sin函数拟合结果和训练损失曲线
# =============================================================================
# 使用训练好的模型进行预测
with torch.no_grad():
    y_predicted = model(X)

# 创建子图，左边显示拟合结果，右边显示损失曲线
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# 子图1：sin函数拟合结果
ax1.scatter(X_numpy, y_numpy, label='Noisy data (sin(x) + noise)', color='blue', alpha=0.6)
ax1.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linewidth=2, linestyle='--')
ax1.plot(X_numpy, y_predicted, label='Neural Network Prediction', color='red', linewidth=2)
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Sin Function Fitting with Multi-layer Network')
ax1.legend()
ax1.grid(True)

# 子图2：训练损失曲线
ax2.plot(range(num_epochs), losses, label='Training Loss', color='purple')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss (MSE)')
ax2.set_title('Training Loss Curve')
ax2.legend()
ax2.grid(True)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()
