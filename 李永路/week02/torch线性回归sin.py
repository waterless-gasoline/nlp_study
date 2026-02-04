import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据 - sin函数
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 在[-2π, 2π]范围内生成数据
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(1000, 1)  # sin函数加上少量噪声

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"输入数据形状: {X.shape}")
print(f"输出数据形状: {y.shape}")
print("---" * 10)


# 2. 构建神经网络模型
class SinNet(torch.nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        # 多层感知机，包含多个隐藏层
        self.hidden1 = torch.nn.Linear(1, 64)  # 输入层到第一个隐藏层
        self.hidden2 = torch.nn.Linear(64, 128)  # 第一个隐藏层到第二个隐藏层
        self.hidden3 = torch.nn.Linear(128, 64)  # 第二个隐藏层到第三个隐藏层
        self.output = torch.nn.Linear(64, 1)  # 输出层

        # 激活函数
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))  # 第一层+激活
        x = self.relu(self.hidden2(x))  # 第二层+激活
        x = self.relu(self.hidden3(x))  # 第三层+激活
        x = self.output(x)  # 输出层（不使用激活函数，因为sin函数范围是[-1,1]）
        return x


# 实例化模型
model = SinNet()

print("神经网络模型结构:")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = torch.nn.MSELoss()  # 均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam优化器，通常比SGD更适合复杂网络

# 4. 训练模型
num_epochs = 3000
losses = []  # 存储损失值以便绘制

print("开始训练...")
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 记录损失
    losses.append(loss.item())

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 使用训练好的模型进行预测
model.eval()  # 设置为评估模式
with torch.no_grad():  # 不计算梯度
    y_predicted = model(X).numpy()

# 6. 绘制结果
plt.figure(figsize=(12, 8))

# 绘制原始数据和拟合曲线
plt.subplot(2, 1, 1)
plt.scatter(X_numpy, y_numpy, label='原始数据 (含噪声)', color='blue', alpha=0.5, s=10)
plt.plot(X_numpy, np.sin(X_numpy), label='真实sin函数', color='green', linewidth=2, linestyle='--')
plt.plot(X_numpy, y_predicted, label='神经网络拟合', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('神经网络拟合sin函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 绘制损失变化曲线
plt.subplot(2, 1, 2)
plt.plot(losses, label='训练损失', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练过程中损失变化')
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数刻度显示损失变化

plt.tight_layout()
plt.show()

# 7. 额外测试：在训练范围外预测
print("在训练范围外的泛化能力测试...")
test_x = np.linspace(-3 * np.pi, 3 * np.pi, 600).reshape(-1, 1)
test_y_true = np.sin(test_x)
test_x_tensor = torch.from_numpy(test_x).float()

model.eval()
with torch.no_grad():
    test_y_pred = model(test_x_tensor).numpy()

# 绘制泛化能力图
plt.figure(figsize=(12, 5))
plt.plot(test_x, test_y_true, label='真实sin函数', color='green', linewidth=2, linestyle='--')
plt.plot(test_x, test_y_pred, label='神经网络预测', color='red', linewidth=2)
plt.scatter(X_numpy, y_numpy, label='训练数据', color='blue', alpha=0.3, s=10)
plt.xlabel('X')
plt.ylabel('y')
plt.title('神经网络对sin函数的泛化能力（包含训练范围外）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("网络参数统计:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")
