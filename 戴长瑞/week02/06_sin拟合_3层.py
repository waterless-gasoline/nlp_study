import torch
import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("开始调试三层神经网络实现")
print("=" * 60)

# 1. 生成数据
np.random.seed(42)
torch.manual_seed(42)

X_numpy = np.linspace(-np.pi, np.pi, 500).reshape(-1, 1)
y_numpy = np.sin(X_numpy) + 0.05 * np.random.randn(500, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print(f"数据统计:")
print(f"  X范围: [{X.min():.2f}, {X.max():.2f}]")
print(f"  y范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"  样本数: {len(X)}")
print("-" * 40)

# 2. 三层神经网络
input_size = 1
hidden1_size = 16  # 第一隐藏层
hidden2_size = 8  # 第二隐藏层
output_size = 1


# 权重初始化（Xavier初始化）
def xavier_init(size):
    in_dim, out_dim = size
    std = np.sqrt(2.0 / (in_dim + out_dim))
    return torch.randn(size) * std


# 初始化权重
W1 = xavier_init((input_size, hidden1_size))
b1 = torch.zeros(hidden1_size)
W2 = xavier_init((hidden1_size, hidden2_size))
b2 = torch.zeros(hidden2_size)
W3 = xavier_init((hidden2_size, output_size))
b3 = torch.zeros(output_size)

# 设置为需要梯度
W1.requires_grad_(True)
b1.requires_grad_(True)
W2.requires_grad_(True)
b2.requires_grad_(True)
W3.requires_grad_(True)
b3.requires_grad_(True)

print(f"网络参数:")
print(f"  网络结构: {input_size} -> {hidden1_size} -> {hidden2_size} -> {output_size}")
print(
    f"  总参数数: {(input_size * hidden1_size + hidden1_size) + (hidden1_size * hidden2_size + hidden2_size) + (hidden2_size * output_size + output_size)}")
print(f"  W1: {W1.shape}, 初始均方根: {torch.sqrt(torch.mean(W1 ** 2)).item():.4f}")
print(f"  W2: {W2.shape}, 初始均方根: {torch.sqrt(torch.mean(W2 ** 2)).item():.4f}")
print(f"  W3: {W3.shape}, 初始均方根: {torch.sqrt(torch.mean(W3 ** 2)).item():.4f}")
print("-" * 40)


# 3. 使用tanh激活函数（比ReLU更稳定） tanh在所有点上都有非零导数 不容易出现梯度消失问题
def tanh_activation(x):
    return torch.tanh(x)


def forward_three_layer(X, W1, b1, W2, b2, W3, b3):
    # 第一层
    z1 = torch.matmul(X, W1) + b1
    a1 = tanh_activation(z1)

    # 第二层
    z2 = torch.matmul(a1, W2) + b2
    a2 = tanh_activation(z2)

    # 输出层（线性输出，无激活函数）
    z3 = torch.matmul(a2, W3) + b3
    return z3


# 4. 训练参数
learning_rate = 0.01  # 提高学习率
num_epochs = 2000

print(f"训练设置:")
print(f"  学习率: {learning_rate}")
print(f"  训练轮数: {num_epochs}")
print("-" * 40)

# 5. 训练循环
losses = []
grad_norms = []

print("开始训练三层神经网络...")
print("-" * 40)

for epoch in range(num_epochs):
    # 前向传播
    y_pred = forward_three_layer(X, W1, b1, W2, b2, W3, b3)

    # 计算损失
    loss = torch.mean((y_pred - y) ** 2)
    losses.append(loss.item())

    # 清零梯度
    params = [W1, b1, W2, b2, W3, b3]
    for param in params:
        if param.grad is not None:
            param.grad.data.zero_()

    # 反向传播
    loss.backward()

    # 梯度裁剪
    max_grad_norm = 1.0
    total_norm = 0
    for param in params:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    grad_norms.append(total_norm)

    # 如果梯度太大，进行裁剪
    if total_norm > max_grad_norm:
        clip_coef = max_grad_norm / (total_norm + 1e-6)
        for param in params:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)

    # 检查是否有nan
    if torch.isnan(W1.grad).any() or torch.isnan(W2.grad).any() or torch.isnan(W3.grad).any():
        print(f"Epoch {epoch + 1}: 检测到nan梯度，停止训练")
        break

    # 更新参数
    with torch.no_grad():
        for param in params:
            param -= learning_rate * param.grad

    # 打印进度
    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1:4d}/{num_epochs}], Loss: {loss.item():.6f}, Grad Norm: {total_norm:.4f}')

print("-" * 40)
print("三层神经网络训练完成！")

# 6. 绘制训练过程
plt.figure(figsize=(15, 10))

# 损失曲线
plt.subplot(2, 3, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss (Three Layer Network)')
plt.grid(True)
if len(losses) > 0:
    plt.yscale('log')

# 梯度范数
plt.subplot(2, 3, 2)
plt.plot(grad_norms)
plt.xlabel('Epoch')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm During Training')
plt.grid(True)
plt.axhline(y=max_grad_norm, color='r', linestyle='--', label='Max Norm')
plt.legend()

# 预测结果
plt.subplot(2, 3, 3)
X_test = np.linspace(-np.pi, np.pi, 200).reshape(-1, 1)
X_test_tensor = torch.from_numpy(X_test).float()

with torch.no_grad():
    y_test_pred = forward_three_layer(X_test_tensor, W1, b1, W2, b2, W3, b3).numpy()

true_sin = np.sin(X_test)
plt.scatter(X_numpy, y_numpy, alpha=0.3, s=10, color='gray', label='Data')
plt.plot(X_test, true_sin, 'b-', linewidth=2, label='True sin(x)', alpha=0.7)
plt.plot(X_test, y_test_pred, 'r-', linewidth=2, label='Three Layer Network')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Three Layer Network Prediction')
plt.legend()
plt.grid(True)

# 7. 与PyTorch内置三层网络对比
print("\n" + "=" * 60)
print("与PyTorch内置三层网络对比")
print("=" * 60)


# 使用PyTorch的三层网络
class ThreeLayerTorchNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1, 16),
            torch.nn.Tanh(),
            torch.nn.Linear(16, 8),
            torch.nn.Tanh(),
            torch.nn.Linear(8, 1)
        )

    def forward(self, x):
        return self.net(x)


# 训练PyTorch网络
model = ThreeLayerTorchNet()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

torch_losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = criterion(y_pred, y)
    loss.backward()

    # 梯度裁剪
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
    torch_losses.append(loss.item())

    if (epoch + 1) % 200 == 0:
        print(f'Epoch [{epoch + 1:4d}/{num_epochs}], Loss: {loss.item():.6f}')

# 8. 对比两个网络

# 损失对比
plt.subplot(2, 3, 4)
plt.plot(losses[:len(torch_losses)], label='Manual 3-Layer', alpha=0.8)
plt.plot(torch_losses, label='PyTorch 3-Layer', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')

# 预测对比
plt.subplot(2, 3, 5)

model.eval()
with torch.no_grad():
    torch_pred = model(X_test_tensor).numpy()

plt.scatter(X_numpy, y_numpy, alpha=0.2, s=5, color='gray', label='Data')
plt.plot(X_test, true_sin, 'k-', linewidth=2, label='True sin(x)', alpha=0.7)
plt.plot(X_test, y_test_pred, 'r-', linewidth=2, label='Manual 3-Layer', alpha=0.8)
plt.plot(X_test, torch_pred, 'b--', linewidth=2, label='PyTorch 3-Layer', alpha=0.8)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predictions Comparison')
plt.legend()
plt.grid(True)

# 激活值可视化
plt.subplot(2, 3, 6)

# 获取中间层激活值
with torch.no_grad():
    # 手动网络
    z1 = torch.matmul(X_test_tensor, W1) + b1
    a1 = tanh_activation(z1)
    z2 = torch.matmul(a1, W2) + b2
    a2 = tanh_activation(z2)

    # PyTorch网络中间层
    torch_model_layers = list(model.net.children())
    torch_a1 = torch_model_layers[0](X_test_tensor)
    torch_a1 = torch_model_layers[1](torch_a1)
    torch_a2 = torch_model_layers[2](torch_a1)
    torch_a2 = torch_model_layers[3](torch_a2)

# 绘制激活值分布
plt.hist(a1.flatten().numpy(), bins=50, alpha=0.5, label='Manual Layer1', density=True)
plt.hist(a2.flatten().numpy(), bins=50, alpha=0.5, label='Manual Layer2', density=True)
plt.xlabel('Activation Value')
plt.ylabel('Density')
plt.title('Activation Distribution')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. 性能对比
print("\n" + "=" * 60)
print("性能对比")
print("=" * 60)

# 计算最终预测
with torch.no_grad():
    manual_final = forward_three_layer(X, W1, b1, W2, b2, W3, b3).numpy()
    torch_final = model(X).numpy()

# 计算指标
manual_mse = np.mean((manual_final - y_numpy) ** 2)
torch_mse = np.mean((torch_final - y_numpy) ** 2)

print(f"手动三层网络:")
print(f"  最终MSE: {manual_mse:.6f}")
print(f"  初始损失: {losses[0]:.6f}")
print(f"  最终损失: {losses[-1]:.6f}")
print(f"  损失下降: {losses[0] - losses[-1]:.6f}")

print(f"\nPyTorch三层网络:")
print(f"  最终MSE: {torch_mse:.6f}")
print(f"  初始损失: {torch_losses[0]:.6f}")
print(f"  最终损失: {torch_losses[-1]:.6f}")
print(f"  损失下降: {torch_losses[0] - torch_losses[-1]:.6f}")

print(f"\n对比:")
print(f"  MSE差异: {abs(manual_mse - torch_mse):.6f}")
print(f"  最终损失差异: {abs(losses[-1] - torch_losses[-1]):.6f}")

# 10. 调试信息
print("\n" + "=" * 60)
print("调试信息")
print("=" * 60)

# 检查输出范围
print(f"手动网络输出范围: [{manual_final.min():.4f}, {manual_final.max():.4f}]")
print(f"PyTorch网络输出范围: [{torch_final.min():.4f}, {torch_final.max():.4f}]")
print(f"真实y范围: [{y_numpy.min():.4f}, {y_numpy.max():.4f}]")

# 检查是否有nan
print(f"\n检查nan值:")
print(f"  手动网络预测有nan: {np.any(np.isnan(manual_final))}")
print(f"  PyTorch网络预测有nan: {np.any(np.isnan(torch_final))}")

# 检查激活值
with torch.no_grad():
    z1_check = torch.matmul(X[:5], W1) + b1
    a1_check = tanh_activation(z1_check)
    z2_check = torch.matmul(a1_check, W2) + b2
    a2_check = tanh_activation(z2_check)
    print(f"\n前5个样本的激活值检查:")
    print(f"  Layer1 z范围: [{z1_check.min():.4f}, {z1_check.max():.4f}]")
    print(f"  Layer1 a范围: [{a1_check.min():.4f}, {a1_check.max():.4f}]")
    print(f"  Layer2 z范围: [{z2_check.min():.4f}, {z2_check.max():.4f}]")
    print(f"  Layer2 a范围: [{a2_check.min():.4f}, {a2_check.max():.4f}]")

# 11. 额外分析：查看各层权重的变化
print("\n" + "=" * 60)
print("权重统计")
print("=" * 60)

with torch.no_grad():
    print(f"手动网络权重统计:")
    print(f"  W1: mean={W1.mean().item():.4f}, std={W1.std().item():.4f}")
    print(f"  W2: mean={W2.mean().item():.4f}, std={W2.std().item():.4f}")
    print(f"  W3: mean={W3.mean().item():.4f}, std={W3.std().item():.4f}")

    print(f"\nPyTorch网络权重统计:")
    for i, param in enumerate(model.parameters()):
        if param.dim() > 1:  # 权重矩阵
            print(f"  Layer{i // 2 + 1}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")