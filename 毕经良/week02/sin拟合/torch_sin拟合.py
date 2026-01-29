import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据：y = sin(x)
# 生成 -2π 到 2π 之间的数据
x_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
print(f"数据生成完成x_numpy:")
# print(x_numpy)
# 加上一些随机噪声
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(200, 1)
print(f"数据生成完成y_numpy:")
# print(y_numpy)
# 转换为 PyTorch Tensor
X = torch.from_numpy(x_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"数据生成完成。样本形状: {X.shape}")
print("---" * 10)

# 2. 定义多层网络模型
# 线性模型无法拟合曲线，需要使用非线性激活函数
# 这里使用 torch.nn.Sequential 快速搭建一个多层感知机 (MLP)
# 结构：输入层(1维) -> 隐藏层(50个神经元) -> 激活函数 -> 输出层(1维)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 50),     # 隐藏层
    torch.nn.Tanh(),            # 激活函数，Tanh 适合拟合平滑曲线（如sin）
    torch.nn.Linear(50, 1)      # 输出层
)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
# 使用 Adam 优化器，通常比 SGD 收敛更快，适合非线性问题
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 2000
print("开始训练...")

for epoch in range(num_epochs):
    # 前向传播：模型自动计算 y_pred
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每200个 epoch 打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化结果
# 不需要计算梯度
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
# 绘制原始带噪声的数据
plt.scatter(x_numpy, y_numpy, label='Noisy Data', color='blue', alpha=0.5, s=20)
# 绘制模型拟合的曲线
plt.plot(x_numpy, y_predicted, label='Fitted Curve (NN)', color='red', linewidth=3)

plt.title('Sin Function Fitting with PyTorch MLP')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()
