import torch
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt

# 1.在 [-5, 5] 区间内生成 200 个点
x_numpy = np.linspace(-5, 5, 200).reshape(-1, 1)
print(x_numpy)
y_numpy = np.sin(x_numpy) + 0.1 * np.random.randn(200, 1)
x = torch.from_numpy(x_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# ---------------------------------------------------------
# 2. 构建多层神经网络 (MLP)
# ---------------------------------------------------------
# 既然要拟合非线性，就不能只用 y = ax+b 了。
# 我们定义一个 3 层的网络：
# 输入层(1) -> 隐藏层(50) -> 激活函数 -> 隐藏层(50) -> 激活函数 -> 输出层(1)
model = torch.nn.Sequential(
    torch.nn.Linear(1, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 50),
    torch.nn.ReLU(),
    torch.nn.Linear(50, 1)
)

print(f"初始参数 a: {model.parameters()}")
print("---" * 10)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss()  # 回归任务

# 使用 Adam 优化器。
# # 对于这种非线性拟合，Adam 通常比 SGD 收敛更快、效果更好。
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)  # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 1000
losses = []  # 用于画 Loss 曲线
for epoch in range(num_epochs):
    # 前向传播：model(X) 会自动流经定义好的所有层
    y_pred = model(x)

    # 计算损失
    loss = loss_fn(y_pred, y)
    losses.append(loss.item())

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
print("---" * 10)

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(x).numpy()

plt.figure(figsize=(12, 5))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_numpy, y_predicted, label=f'Neural Network Fitting', color='red', linewidth=2)
plt.title(f'Fitting Sin(x) with MLP')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

