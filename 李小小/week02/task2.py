import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
torch.manual_seed(42)
np.random.seed(42)

# 1. 生成模拟数据：sin函数
# 在[0, 4π]范围内生成随机点
X_numpy = np.random.uniform(0, 4*np.pi, 500).reshape(-1, 1)  # 生成500个随机点
y_numpy = np.sin(X_numpy)  # 计算对应的sin值

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print(f"X shape: {X.shape}, y shape: {y.shape}")
print("---" * 10)

# 2. 定义多层神经网络
class SinNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(SinNet, self).__init__()
        # 第一层：输入到第一个隐藏层
        self.layer1 = nn.Linear(input_size, hidden_size1)
        # 第二层：第一个隐藏层到第二个隐藏层
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        # 第三层：第二个隐藏层到输出层
        self.layer3 = nn.Linear(hidden_size2, output_size)
        # 激活函数
        self.activation = nn.ReLU()  # 使用ReLU激活函数引入非线性
        
    def forward(self, x):
        # 前向传播：输入 -> 隐藏层1 -> 激活 -> 隐藏层2 -> 激活 -> 输出层
        out = self.layer1(x)
        out = self.activation(out)
        out = self.layer2(out)
        out = self.activation(out)
        out = self.layer3(out)
        return out

# 创建网络实例：输入维度为1，两个隐藏层分别有64和32个神经元，输出维度为1
model = SinNet(input_size=1, hidden_size1=64, hidden_size2=32, output_size=1)

print("神经网络结构：")
print(model)
print("---" * 10)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 均方误差损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # 使用Adam优化器，通常比SGD更适合复杂网络

# 4. 训练模型
num_epochs = 2000
loss_history = []  # 记录损失历史

print("开始训练...")
for epoch in range(num_epochs):
    # 前向传播：计算预测值
    y_pred = model(X)
    
    # 计算损失
    loss = loss_fn(y_pred, y)
    
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数
    
    # 记录损失
    loss_history.append(loss.item())
    
    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}')

print("\n训练完成！")
final_loss = loss.item()
print(f"最终损失: {final_loss:.6f}")
print("---" * 10)

# 5. 评估模型
model.eval()  # 设置模型为评估模式
with torch.no_grad():  # 不计算梯度
    # 生成密集的x值用于平滑曲线绘制
    x_test = np.linspace(0, 4*np.pi, 1000).reshape(-1, 1)
    X_test = torch.from_numpy(x_test).float()
    y_pred_test = model(X_test)
    y_pred_test = y_pred_test.numpy()

print("模型评估完成！")
print("---" * 10)

# 6. 绘制结果
plt.figure(figsize=(14, 10))

# 子图1：显示原始数据、拟合曲线和真实sin函数
plt.subplot(2, 1, 1)
plt.scatter(X_numpy[:100], y_numpy[:100], label='训练数据点', color='blue', alpha=0.5, s=20)  # 只显示前100个点避免过密
plt.plot(x_test, np.sin(x_test), label='真实 sin(x)', color='green', linewidth=2)
plt.plot(x_test, y_pred_test, label='神经网络拟合曲线', color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('多层神经网络拟合 sin(x) 函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图2：显示损失函数随训练过程的变化
plt.subplot(2, 1, 2)
plt.plot(loss_history, label='训练损失', color='purple')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'训练损失变化 (最终损失: {final_loss:.6f})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. 计算并显示模型性能指标
with torch.no_grad():
    train_pred = model(X)
    train_loss = loss_fn(train_pred, y)
    print(f"\n训练集上的最终 MSE 损失: {train_loss.item():.6f}")
    
    # 计算平均绝对误差
    mae = torch.mean(torch.abs(train_pred - y))
    print(f"训练集上的平均绝对误差 (MAE): {mae.item():.6f}")