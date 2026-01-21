import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import matplotlib.pyplot as plt
import torch.nn as nn

# 1. 生成模拟数据 (与之前相同)
# 构造sin函数
# 使用np.linspace() 生成一个等间距的x轴序列
X_numpy = np.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y_numpy = np.sin(X_numpy)

X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络拟合sin曲线
class SinRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SinRegressionModel, self).__init__()
        # 构建多层网络
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        # 输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# 3. 定义损失函数和优化器
model = SinRegressionModel(1, 64, 1)
loss_fn = torch.nn.MSELoss() # 回归任务
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # 优化器，基于 a b 梯度 自动更新

# 4. 训练模型
num_epochs = 5000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * X + b
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")


# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label=f'拟合线', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
