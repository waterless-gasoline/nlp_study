import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的模拟数据
# 生成0到2π之间的均匀分布数据，添加少量噪声增加拟合难度
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # (200, 1)
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 带噪声的sin曲线

# 转换为torch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络模型
class SinFittingNet(nn.Module):
    def __init__(self):
        super(SinFittingNet, self).__init__()
        # 定义多层全连接层：输入层(1) -> 隐藏层1(32) -> 隐藏层2(16) -> 输出层(1)
        self.layers = nn.Sequential(
            nn.Linear(1, 32),   # 第一层：1维输入，32维输出
            nn.ReLU(),          # 激活函数
            nn.Linear(32, 16),  # 第二层：32维输入，16维输出
            nn.ReLU(),          # 激活函数
            nn.Linear(16, 1)    # 输出层：16维输入，1维输出
        )
    
    def forward(self, x):
        # 前向传播
        return self.layers(x)

# 初始化模型、损失函数、优化器
model = SinFittingNet()
criterion = nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器

print("模型初始化完成：")
print(model)
print("---" * 10)

# 3. 训练模型
num_epochs = 1000
loss_history = []  # 记录损失变化

for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)
    loss = criterion(y_pred, y)
    
    # 反向传播与参数更新
    optimizer.zero_grad()  # 梯度清零
    loss.backward()        # 反向传播
    optimizer.step()       # 更新参数
    
    # 记录损失
    loss_history.append(loss.item())
    
    # 每100个epoch打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

# 4. 模型预测
model.eval()  # 切换到评估模式
with torch.no_grad():  # 禁用梯度计算
    y_predicted = model(X).numpy()  # 转换为numpy数组用于可视化

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
plt.figure(figsize=(12, 8))

# 子图1：损失变化曲线
plt.subplot(2, 1, 1)
plt.plot(loss_history, label='Training Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Loss Change During Training')
plt.legend()
plt.grid(True)

# 子图2：sin函数拟合结果
plt.subplot(2, 1, 2)
plt.scatter(X_numpy, y_numpy, label='Raw Data (with noise)', color='blue', alpha=0.6, s=10)
plt.plot(X_numpy, y_predicted, label='Fitted Curve by MLP', color='red', linewidth=2)
plt.plot(X_numpy, np.sin(X_numpy), label='Original sin(x)', color='orange', linestyle='--', linewidth=2)
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('MLP Fitting Result for sin(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()  # 调整子图间距
plt.show()

# 打印模型最终的MSE损失
final_loss = criterion(torch.from_numpy(y_predicted).float(), y).item()
print(f"最终拟合损失 (MSE): {final_loss:.6f}")
