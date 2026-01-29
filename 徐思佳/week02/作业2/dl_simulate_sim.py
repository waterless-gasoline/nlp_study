"""
调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""
import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。
mu = 0.1
sigma = 0.05
noise = mu + sigma * np.random.randn(100, 1)
y_numpy = np.sin(X_numpy) + noise

X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("✅ 数据生成完成。")

# 2. 定义神经网络
class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1 = nn.Linear(1, 10)
        self.bn1 = nn.BatchNorm1d(10)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(10, 20)
        self.bn2 = nn.BatchNorm1d(20)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


# 3. 训练模型
model = MyNet()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(1000):
    output = model(X)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch [{epoch + 1}/1000], Loss: {loss.item():.4f}")

# 4. 可视化结果（核心优化：排序数据+散点+平滑曲线）
plt.figure(figsize=(10, 7))

# 4.1 绘制带噪声的真实数据（散点）
plt.scatter(X_numpy, y_numpy, color='lightblue', alpha=0.7, label='Real Data (with Noise)')

# 4.2 生成排序后的输入数据（用于绘制平滑的拟合曲线）
X_sorted = np.linspace(0, 10, 200).reshape(-1, 1)  # 均匀取200个点，覆盖[0,10]
X_sorted_tensor = torch.from_numpy(X_sorted).float()

# 4.3 模型预测（切换评估模式，关闭梯度计算）
model.eval()  # 切换评估模式（固定BN统计量、禁用Dropout等）
with torch.no_grad():
    y_pred_sorted = model(X_sorted_tensor).numpy()

# 4.4 绘制纯净的sin曲线（无噪声，作为参考）
plt.plot(X_sorted, np.sin(X_sorted), color='darkblue', linewidth=2, label='Pure Sin Curve (no Noise)')

# 4.5 绘制模型的预测曲线（平滑）
plt.plot(X_sorted, y_pred_sorted, color='red', linewidth=2, linestyle='--', label='Model Predicted Curve')

# 4.6 设置图表属性
plt.xlabel('X', fontsize=12)
plt.ylabel('y = sin(X) + Noise', fontsize=12)
plt.title('Multi-Layer Network Fitting Sin Function', fontsize=14)
plt.legend(fontsize=10)
plt.grid(alpha=0.3, linestyle='--')
plt.show()
