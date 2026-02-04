import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import numpy as np
import matplotlib.pyplot as plt

# 构建sin函数数据
X_numpy = np.linspace(-3 * np.pi, 3 * np.pi, 200).reshape(-1, 1)
# 对X排序
X_numpy = np.sort(X_numpy, axis=0)
# 在sin值上下添加随机偏移
y_numpy = np.sin(X_numpy) + np.random.uniform(-0.5, 0.5, X_numpy.shape)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("---" * 10)

# 多层网络定义 - 减少网络复杂度
class SinNet(torch.nn.Module):
    def __init__(self, hidden_size=16):  # 减小隐藏层大小
        super(SinNet, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)


# 创建模型、损失函数和优化器
model = SinNet()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.001)

print(f"模型结构: {model}")
print("---" * 10)

# 训练模型
num_epochs = 2000
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 绘制结果
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(12, 6))
plt.scatter(X_numpy, y_numpy, label='True sin(x)', color='blue', alpha=0.5, s=10)
plt.plot(X_numpy, y_predicted, label='Network prediction', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), '--', label='Pure sin(x)', color='green', alpha=0.7)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Neural Network Fitting sin(x)')
plt.legend()
plt.grid(True)
plt.show()