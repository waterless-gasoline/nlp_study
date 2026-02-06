import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch.nn as nn

# 1. 生成sin函数数据
print("生成sin函数数据...")
np.random.seed(42)  # 设置随机种子以保证结果可复现
x_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)  # 生成1000个点在[-2π, 2π]区间
y_numpy = np.sin(x_numpy) + np.random.normal(0, 0.1, size=x_numpy.shape)  # 添加噪声

# 分割训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    x_numpy, y_numpy, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
x_train_tensor = torch.from_numpy(x_train).float()
y_train_tensor = torch.from_numpy(y_train).float()
x_test_tensor = torch.from_numpy(x_test).float()
y_test_tensor = torch.from_numpy(y_test).float()

print(f"训练集大小: {len(x_train)}, 测试集大小: {len(x_test)}")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinApproximator(nn.Module):
    def __init__(self, hidden_layers=3, hidden_units=64):
        """
        多层神经网络拟合sin函数

        参数:
            hidden_layers: 隐藏层层数
            hidden_units: 每层隐藏层的神经元数量
        """
        super(SinApproximator, self).__init__()

        # 输入层 (1维输入 -> hidden_units维)
        layers = [nn.Linear(1, hidden_units), nn.ReLU()]

        # 添加隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.ReLU())

        # 输出层 (hidden_units维 -> 1维输出)
        layers.append(nn.Linear(hidden_units, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 3. 定义训练函数
def train_model(model, x_train, y_train, x_test, y_test, num_epochs=2000,
                lr=0.001, batch_size=32, model_name="模型"):
    """
    训练模型并返回训练过程中的loss记录
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    train_losses = []
    test_losses = []

    print(f"\n开始训练{model_name}...")
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0.0

        # 批量训练
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 在测试集上评估
        model.eval()
        with torch.no_grad():
            test_predictions = model(x_test)
            test_loss = criterion(test_predictions, y_test).item()
            test_losses.append(test_loss)

        # 每500个epoch打印一次进度
        if (epoch + 1) % 500 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {avg_train_loss:.6f}, "
                  f"Test Loss: {test_loss:.6f}")

    print(f"{model_name}训练完成!")
    return train_losses, test_losses


# 4. 训练不同复杂度的模型
print("=" * 50)
print("训练不同复杂度的模型拟合sin函数")
print("=" * 50)

models = {}
results = {}

# 定义不同的模型配置
model_configs = [
    {"name": "浅层网络(1层)", "hidden_layers": 1, "hidden_units": 32},
    {"name": "中等网络(2层)", "hidden_layers": 2, "hidden_units": 64},
    {"name": "深层网络(3层)", "hidden_layers": 3, "hidden_units": 128},
    {"name": "更深的网络(4层)", "hidden_layers": 4, "hidden_units": 256},
]

for config in model_configs:
    # 创建模型
    model = SinApproximator(
        hidden_layers=config["hidden_layers"],
        hidden_units=config["hidden_units"]
    )

    # 训练模型
    train_losses, test_losses = train_model(
        model, x_train_tensor, y_train_tensor, x_test_tensor, y_test_tensor,
        num_epochs=2000, lr=0.001, model_name=config["name"]
    )

    # 保存结果
    models[config["name"]] = model
    results[config["name"]] = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "final_train_loss": train_losses[-1],
        "final_test_loss": test_losses[-1]
    }

# 5. 可视化训练过程
print("\n" + "=" * 50)
print("可视化训练过程")
print("=" * 50)

plt.figure(figsize=(15, 10))

# 子图1: 训练损失对比
plt.subplot(2, 2, 1)
for name, result in results.items():
    plt.plot(result["train_losses"], label=name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Training Loss (MSE)')
plt.title('不同模型训练损失对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')  # 使用对数坐标更清晰

# 子图2: 测试损失对比
plt.subplot(2, 2, 2)
for name, result in results.items():
    plt.plot(result["test_losses"], label=name, linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Test Loss (MSE)')
plt.title('不同模型测试损失对比')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')

# 子图3: 最终拟合效果对比
plt.subplot(2, 2, 3)
# 生成用于预测的均匀分布的点
x_plot = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
x_plot_tensor = torch.from_numpy(x_plot).float()

# 绘制原始sin函数（无噪声）
plt.plot(x_plot, np.sin(x_plot), 'k-', label='True sin(x)', linewidth=3, alpha=0.7)

# 绘制各个模型的预测结果
for name, model in models.items():
    model.eval()
    with torch.no_grad():
        y_pred = model(x_plot_tensor).numpy()
    plt.plot(x_plot, y_pred, label=name, linewidth=2, alpha=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title('不同模型拟合sin函数对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 子图4: 绘制数据点和最佳模型拟合
plt.subplot(2, 2, 4)
# 找出测试损失最小的模型
best_model_name = min(results.keys(), key=lambda x: results[x]["final_test_loss"])
best_model = models[best_model_name]

plt.scatter(x_numpy, y_numpy, label='Noisy data', color='blue', alpha=0.3, s=10)

# 绘制最佳模型拟合结果
with torch.no_grad():
    y_best_pred = best_model(x_plot_tensor).numpy()

plt.plot(x_plot, np.sin(x_plot), 'k-', label='True sin(x)', linewidth=3, alpha=0.7)
plt.plot(x_plot, y_best_pred, 'r-', label=f'Best Model: {best_model_name}',
         linewidth=3, alpha=0.8)

plt.xlabel('x')
plt.ylabel('y')
plt.title(f'最佳模型拟合效果: {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sin_fitting_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 打印模型参数和性能对比
print("\n" + "=" * 60)
print("模型性能对比")
print("=" * 60)

for name, result in results.items():
    model = models[name]
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n{name}:")
    print(f"  隐藏层数: {model_configs[list(models.keys()).index(name)]['hidden_layers']}")
    print(f"  每层神经元数: {model_configs[list(models.keys()).index(name)]['hidden_units']}")
    print(f"  总参数数量: {total_params}")
    print(f"  可训练参数: {trainable_params}")
    print(f"  最终训练损失: {result['final_train_loss']:.6f}")
    print(f"  最终测试损失: {result['final_test_loss']:.6f}")

# 7. 分析最佳模型
print("\n" + "=" * 60)
print("最佳模型分析")
print("=" * 60)
print(f"最佳模型: {best_model_name}")
print(f"测试损失: {results[best_model_name]['final_test_loss']:.6f}")

# 使用最佳模型进行一些预测
print("\n最佳模型预测示例:")
test_points = torch.tensor([[0], [np.pi / 2], [np.pi], [3 * np.pi / 2], [2 * np.pi]]).float()
with torch.no_grad():
    predictions = best_model(test_points).numpy()

for i, (x_val, pred) in enumerate(zip(test_points.numpy(), predictions)):
    true_value = np.sin(x_val[0])
    error = abs(pred[0] - true_value)
    print(f"  x = {x_val[0]:.4f}: 预测值 = {pred[0]:.6f}, "
          f"真实值 = {true_value:.6f}, 误差 = {error:.6f}")

# 8. 可视化最佳模型的拟合误差
print("\n" + "=" * 60)
print("最佳模型误差分析")
print("=" * 60)

x_fine = np.linspace(-2 * np.pi, 2 * np.pi, 1000).reshape(-1, 1)
x_fine_tensor = torch.from_numpy(x_fine).float()

with torch.no_grad():
    y_pred_fine = best_model(x_fine_tensor).numpy()

true_y = np.sin(x_fine)
errors = np.abs(y_pred_fine - true_y)

plt.figure(figsize=(12, 8))

# 绘制拟合曲线和真实曲线
plt.subplot(2, 1, 1)
plt.plot(x_fine, true_y, 'k-', label='True sin(x)', linewidth=3)
plt.plot(x_fine, y_pred_fine, 'r--', label=f'Model: {best_model_name}', linewidth=2)
plt.scatter(x_train, y_train, color='blue', alpha=0.3, s=10, label='Train data')
plt.scatter(x_test, y_test, color='green', alpha=0.3, s=10, label='Test data')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'最佳模型拟合效果: {best_model_name}')
plt.legend()
plt.grid(True, alpha=0.3)

# 绘制误差曲线
plt.subplot(2, 1, 2)
plt.plot(x_fine, errors, 'b-', linewidth=2)
plt.fill_between(x_fine.flatten(), 0, errors.flatten(), alpha=0.3)
plt.xlabel('x')
plt.ylabel('Absolute Error')
plt.title('拟合误差分布')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('best_model_error_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n可视化完成！所有图表已保存为PNG文件。")
