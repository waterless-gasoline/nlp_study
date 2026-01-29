import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# 生成训练数据

x = torch.linspace(-2 * torch.pi, 2 * torch.pi, 10000)
x = torch.unsqueeze(x, 1)
y = torch.sin(x)

# 定义模型
model = nn.Sequential(
    nn.Linear(1,70),
    nn.Sigmoid(),
    nn.Linear(70,1)
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.01) # Adam优化器比SGD快很多

# 开始训练
epoch_num = 5000
for epoch in range(epoch_num):
    model.train()
    y_pred = model(x)
    loss = criterion(y_pred,y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch%500 == 0:
        print(f"Epoch [{epoch}/{epoch_num}],Loss:{loss:.4f}")
model.eval()
with torch.no_grad():
    y_pred = model(x)

plt.figure()
plt.plot(x,y,linewidth=1.8,color="red",label="Raw Data",alpha = 0.7)
plt.plot(x,y_pred,linewidth=1.8,color = "blue",label = "Predicted Data",linestyle="--",alpha=0.7)
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

