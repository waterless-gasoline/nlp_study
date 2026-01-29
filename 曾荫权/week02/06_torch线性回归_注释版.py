
import torch
import numpy as np
import matplotlib.pyplot as plt

# 创建100个在[0, 2π]范围内均匀分布的x值
x_values = np.linspace(0, 2 * np.pi, 100)
# 计算对应的sin值
y_values = np.sin(x_values)

# 将数据转换为PyTorch张量
X = torch.tensor(x_values, dtype=torch.float32).reshape(-1, 1)
y = torch.tensor(y_values, dtype=torch.float32).reshape(-1, 1)

print("---" * 10)

class SinApproximator(torch.nn.Module):
    def __init__(self):
        super(SinApproximator, self).__init__()
        self.layer1 = torch.nn.Linear(1, 64)
        self.layer2 = torch.nn.Linear(64, 64)
        self.layer3 = torch.nn.Linear(64, 64)
        self.layer4 = torch.nn.Linear(64, 1)

        self.activation = torch.nn.Tanh()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.layer4(x)
        return x


# 实例化模型
model = SinApproximator()

print("模型结构:")
print(model)
print("---" * 10)

# 使用均方误差(MSE)作为损失函数
loss_fn = torch.nn.MSELoss()

# 使用Adam优化器，适合复杂函数拟合
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# 设置训练轮数
num_epochs = 1000

for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清零梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每500个epoch打印一次训练进度
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 生成更密集的x值用于平滑的可视化
x_dense = np.linspace(0, 2 * np.pi, 1000)
X_dense = torch.tensor(x_dense, dtype=torch.float32).reshape(-1, 1)

# 使用训练好的模型进行预测
with torch.no_grad():
    y_pred_dense = model(X_dense).numpy().flatten()

# 创建图形
plt.figure(figsize=(12, 8))

# sin函数与拟合结果对比
plt.subplot(2, 1, 1)
plt.plot(x_dense, np.sin(x_dense), 'b-', linewidth=2, label='真实 sin(x) 函数')
plt.plot(x_dense, y_pred_dense, 'r--', linewidth=2, label='神经网络拟合结果')
plt.scatter(x_values, y_values, color='green', alpha=0.6, label='训练数据点')
plt.title('正弦函数拟合结果对比', fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.legend()
plt.grid(True)

# 调整子图间距
plt.tight_layout()

# 显示图形
plt.show()

# 一堆报错 报错缺少字形不知道缺了啥 但是能运行就很奇怪
