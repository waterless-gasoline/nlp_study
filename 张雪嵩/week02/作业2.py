import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成sin函数的模拟数据
# 生成0到2π之间的均匀分布数据，增加噪声模拟真实场景
X_numpy = np.linspace(0, 2 * np.pi, 200).reshape(-1, 1)  # (200,1)，覆盖sin函数一个完整周期
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(*X_numpy.shape)  # 加小幅高斯噪声
# 转换为torch张量（float类型）
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin函数数据生成完成。")
print("---" * 10)


# 2. 定义多层神经网络模型（替代原线性参数a/b）
# 采用3层全连接网络：输入层(1) → 隐藏层(32) → 隐藏层(16) → 输出层(1)
# 激活函数用ReLU，解决线性模型无法拟合曲线的问题
class SinNet(torch.nn.Module):
    def __init__(self):
        super(SinNet, self).__init__()
        self.fc1 = torch.nn.Linear(1, 32)  # 第一层：1维输入→32维隐藏特征
        self.fc2 = torch.nn.Linear(32, 16)  # 第二层：32维→16维
        self.fc3 = torch.nn.Linear(16, 1)  # 第三层：16维→1维输出（回归任务）
        self.relu = torch.nn.ReLU()  # 非线性激活函数

    def forward(self, x):
        # 前向传播：逐层计算，激活函数穿插其中
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)  # 输出层无激活（回归任务直接输出数值）
        return x


# 初始化模型、损失函数、优化器
model = SinNet()
loss_fn = torch.nn.MSELoss()  # 回归任务仍用均方误差损失
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam优化器（比SGD拟合曲线更高效）

print("模型初始化完成，网络结构：")
print(model)
print("---" * 10)

# 3. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：通过模型计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播与优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 反向传播计算梯度
    optimizer.step()  # 更新网络参数

    # 每50轮打印一次损失，监控训练进度
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
print("---" * 10)

# 4. 模型预测与可视化
# 禁用梯度计算（仅预测，节省资源）
with torch.no_grad():
    y_predicted = model(X).numpy()  # 模型预测值转换为numpy数组，方便绘图

# 绘制结果：原始数据 + 真实sin曲线 + 模型拟合曲线
plt.figure(figsize=(12, 7))
# 原始带噪数据点
plt.scatter(X_numpy, y_numpy, label='Raw Data (with noise)', color='lightblue', alpha=0.8, s=10)
# 真实sin函数曲线
plt.plot(X_numpy, np.sin(X_numpy), label='True sin(x)', color='green', linewidth=2, linestyle='--')
# 模型拟合的曲线
plt.plot(X_numpy, y_predicted, label='Model Fitted Curve', color='red', linewidth=2)

# 图表美化
plt.xlabel('x (0 ~ 2π)', fontsize=12)
plt.ylabel('sin(x)', fontsize=12)
plt.title('Multi-Layer Neural Network Fitting sin(x)', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi], ['0', 'π/2', 'π', '3π/2', '2π'])  # 标注π刻度
plt.show()