import numpy as np
import torch
from matplotlib import pyplot as plt

# 1. 生成数据
x_numpy = np.random.uniform(-1, 1, 1000)  # 小范围！
y_numpy = np.sin(x_numpy * np.pi)  # 缩放到 [-π, π] 太大了红线出不来 嘤嘤嘤

# 2. 多项式特征（低次数！）
degree = 3
X = np.column_stack([x_numpy ** i for i in range(degree + 1)])

# 3. 转换为张量
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y_numpy).reshape(-1, 1)

# 4. 初始化权重（小值！）
weights = torch.randn(degree + 1, 1) * 0.01
weights.requires_grad = True

# 5. 训练（小学习率！）
lr = 0.001
epochs = 1000

for epoch in range(epochs):
    y_pred = X_tensor @ weights
    loss = torch.mean((y_pred - y_tensor) ** 2)

    if torch.isnan(loss):
        print(f"NaN at epoch {epoch}")
        break

    # 手动梯度
    grad = 2 / len(X_tensor) * X_tensor.T @ (y_pred - y_tensor)

    # 更新
    with torch.no_grad():
        weights -= lr * grad
        weights.requires_grad = True

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")



# 预测数据
x_numpy_test = np.linspace(-1, 1, 1000)
# x_numpy_test1 = x_numpy_test*
X_test = np.column_stack([x_numpy_test ** i for i in range(degree+ 1 )])
X_test_tensor = torch.FloatTensor(X_test)


# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值


with torch.no_grad():
    y_predicted_test = X_test_tensor @ weights
    y_predicted_numpy = y_predicted_test.numpy()

print(f"y_predicted:{y_predicted_numpy[1]}")

# plt.subplot(1, 3, 1)?
plt.figure(figsize=(15, 5))
plt.scatter(x_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_numpy_test, np.sin(x_numpy_test * np.pi), 'g-', label='sin(x)', color='yellow', linewidth=2)
plt.plot(x_numpy_test, y_predicted_numpy, 'r-', label=f'function:(degree = {degree})次多项式拟合', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
