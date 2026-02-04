import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 生成模拟数据
# X_numpy = np.linspace(-np.pi, np.pi, 100).reshape(-1, 1)
X_numpy = np.random.uniform(-np.pi, np.pi, size=(100, 1))
y_numpy = np.sin(X_numpy)                                  # 真实值
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("sin 数据构造完成，样本数:", len(X))
print("---" * 10)

# 非线性模型，需要通过多层网络构建
model = torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 64),
    torch.nn.Tanh(),
    torch.nn.Linear(64, 1)
)

# 3.定义损失函数和优化器
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 4.训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    #前向传播
    y_pred = model(X)
    #计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch +1 ) % 1 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}, loss={loss.item():.4f}")

# 绘制结果
with torch.no_grad():
    y_pred = model(X).numpy()

plt.figure(figsize=(8, 5))
plt.scatter(X_numpy, y_numpy, s=15, color='blue', alpha=0.5, label='Raw data (sin)')
plt.plot(X_numpy, y_pred, color='red', lw=2, label='MLP pred')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()