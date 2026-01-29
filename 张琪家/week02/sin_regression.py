import torch
import torch.nn as nn
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 10
# 形状为 (100, 1) 的2维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.01
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
Y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        return out

model = SimpleClassifier(1, 128, 64, 1)
criterion = nn.MSELoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.Adam(model.parameters(), lr=0.02)

# 4. 训练模型
num_epochs = 1000
train_losses = []
total_running_loss = 0.0

# 4. 训练循环
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # 前向传播+损失计算+反向传播+参数更新
    y_pred_train = model(X)
    train_loss = criterion(y_pred_train, Y)
    train_loss.backward()
    optimizer.step()

    # 记录本轮损失
    train_losses.append(train_loss.item())
    total_running_loss += train_loss.item()

    print(f"Epoch [{epoch + 1}/{num_epochs}], Current Loss: {train_loss.item():.6f}")

print("\n训练完成！")
print("---" * 10)

model.eval() # 切换模型为评估模式
with torch.no_grad():
    y_pred = model(X) # 模型预测全量数据
y_pred_numpy = y_pred.numpy()

sorted_indices = np.argsort(X_numpy, axis=0).flatten()
X_sorted = X_numpy[sorted_indices].squeeze()
y_pred_sorted = y_pred_numpy[sorted_indices].squeeze()
y_true_sorted = np.sin(X_numpy[sorted_indices]).squeeze()
y_raw_sorted = y_numpy[sorted_indices].squeeze()

# 3. 绘制拟合曲线
plt.figure(figsize=(10, 6)) # 设置画布大小
# 绘制原始带噪声数据
plt.scatter(X_sorted, y_raw_sorted, label='Raw Data (with noise)', color='blue', alpha=0.6)
# 绘制真实sin(x)曲线
plt.plot(X_sorted, y_true_sorted, label='True sin(x)', color='green', linewidth=2)
# 绘制模型拟合曲线
plt.plot(X_sorted, y_pred_sorted, label='Model Fitting', color='red', linewidth=2, linestyle='--')

# 图表基础设置
plt.xlabel('X') # x轴标签
plt.ylabel('y = sin(X) + noise') # y轴标签
plt.title('sin(X)') # 图表标题
plt.legend() # 显示图例
plt.grid(True, alpha=0.3) # 显示网格（降低透明度，避免遮挡）
plt.show() # 显示图形
