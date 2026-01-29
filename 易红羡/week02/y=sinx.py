import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

#设置随机种子
torch.manual_seed(42)
np.random.seed(42)

#生成数据
x = np.linspace(0,2*np.pi,100)
y = np.sin(x)
plt.figure(figsize=(10, 6))
plt.plot(x,y,'b-', linewidth=3, label='真实的 sin(x)')
plt.xlabel('x (0 to 2π)')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

x_tensor = torch.tensor(x, dtype=torch.float32).view(-1,1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1,1)

#线性回归模型
linear_model = nn.Linear(1, 1)    #输入1维输出1维
optimizer = torch.optim.SGD(linear_model.parameters(), lr=0.01)    #优化
criterion = nn.MSELoss()    #均方误差损失

linear_losses = []  # 记录损失

for epoch in range(1000):
    #前向传播
    output = linear_model(x_tensor)
    loss = criterion(output, y_tensor)
    #反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()#更新梯度

#每100轮打印一次损失
    if epoch % 100 == 0:
        print(f'Linear_Epoch[{epoch}/1000],Loss: {loss.item():.4f}')

# 获取预测值
with torch.no_grad():
    linear_prediction = linear_model(x_tensor).detach().numpy()
print(linear_prediction[-1])


#多层网络拟合
mlp_model = nn.Sequential(
    nn.Linear(in_features=1, out_features=64), #输入层到第一个隐藏层
    nn.ReLU(),
    nn.Linear(in_features=64, out_features=32),#第一个隐藏层到第二个隐藏层
    nn.ReLU(),
    nn.Linear(in_features=32, out_features=16),#第二个隐藏层到第三个隐藏层
    nn.ReLU(),
    nn.Linear(in_features=16, out_features=1),#第三个隐藏层到输出层
)

print("模型结构:\n", mlp_model)

optimizer = torch.optim.SGD(mlp_model.parameters(), lr=0.01)
criterion = nn.MSELoss()
mlp_losses = []
for epoch in range(1000):
    #前向传播
    output = mlp_model(x_tensor)
    loss = criterion(output, y_tensor)
    #反向传播与优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'MLP_Epoch[{epoch}/1000],Loss: {loss.item():.4f}')
        mlp_losses.append(loss.item())

# 获取MLP模型预测值
with torch.no_grad():
    mlp_prediction = mlp_model(x_tensor).detach().numpy()
print(mlp_prediction[-1])

#可视化
plt.figure(figsize=(10, 6))
plt.scatter(x, y, label='Raw data', color='blue', alpha=0.6)
plt.plot(x, mlp_prediction, label=f'Model: y = {np.sin(x)}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
