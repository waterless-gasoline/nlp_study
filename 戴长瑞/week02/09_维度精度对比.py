import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from datetime import datetime
import warnings
import json
import os

warnings.filterwarnings('ignore')


# ==================== 1. 实验配置 ====================
class ArchitectureConfig:
    """网络结构实验配置"""
    DATA_PATH = "../Week01/data/dataset.csv"
    MAX_LEN = 40
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 20  # 增加epoch以观察loss变化
    RANDOM_SEED = 42
    SAVE_DIR = "architecture_loss_experiment"

    # 测试的架构配置 (层数, 每层节点数)
    ARCHITECTURES = [
        # (层数, 每层节点数, 描述)
        (1, 64, "单层-64节点"),
        (1, 128, "单层-128节点"),
        (1, 256, "单层-256节点"),
        (2, [128, 64], "两层-128→64"),
        (2, [256, 128], "两层-256→128"),
        (2, [512, 256], "两层-512→256"),
        (3, [256, 128, 64], "三层-256→128→64"),
        (3, [512, 256, 128], "三层-512→256→128"),
        (4, [512, 256, 128, 64], "四层-512→256→128→64"),
    ]


# ==================== 2. 灵活的网络架构 ====================
class FlexibleClassifier(nn.Module):
    """支持不同层数和节点数的灵活分类器"""

    def __init__(self, input_dim, hidden_dims, output_dim):
        super(FlexibleClassifier, self).__init__()

        self.input_dim = input_dim
        if isinstance(hidden_dims, int):
            hidden_dims = [hidden_dims]  # 单层情况
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.num_layers = len(hidden_dims)

        # 动态创建网络层
        layers = []
        prev_dim = input_dim

        # 隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # 添加少量dropout防止过拟合
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # 计算模型复杂度
        self.num_params = sum(p.numel() for p in self.parameters())
        self.trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"  创建网络: {input_dim} → {' → '.join(map(str, hidden_dims))} → {output_dim}")
        print(f"  总层数: {self.num_layers + 1} (隐藏层: {self.num_layers})")
        print(f"  参数量: {self.num_params:,} (可训练: {self.trainable_params:,})")

    def forward(self, x):
        return self.network(x)


# ==================== 3. 实验核心：记录详细的loss变化 ====================
class LossTracker:
    """详细记录loss变化的类"""

    def __init__(self, arch_name):
        self.arch_name = arch_name
        self.epoch_losses = []  # 每epoch的平均loss
        self.batch_losses = []  # 每个batch的loss
        self.val_losses = []  # 验证集loss
        self.epoch_times = []  # 每个epoch的训练时间
        self.gradient_norms = []  # 梯度范数（训练稳定性）
        self.convergence_speed = None  # 收敛速度
        self.final_loss = None  # 最终loss
        self.batch_loss_std = 0.0       # loss的标准差

    def add_epoch_result(self, epoch, train_loss, val_loss, batch_losses, epoch_time):
        """记录一个epoch的结果"""
        self.epoch_losses.append(train_loss)
        self.val_losses.append(val_loss)
        self.batch_losses.extend(batch_losses)
        self.epoch_times.append(epoch_time)

        # 计算收敛速度（loss下降到初始值10%所需的epoch数）
        if len(self.epoch_losses) >= 2:
            if self.epoch_losses[0] > 0:
                current_ratio = train_loss / self.epoch_losses[0]
                if current_ratio < 0.1 and self.convergence_speed is None:
                    self.convergence_speed = epoch + 1
        # 计算batch loss的标准差
        if len(batch_losses) > 0:
            self.batch_loss_std = np.std(batch_losses)
    def get_summary(self):
        """获取loss变化的统计摘要"""
        if not self.epoch_losses:
            return {}

        # 计算loss变化统计
        initial_loss = self.epoch_losses[0] if self.epoch_losses else 0
        final_loss = self.epoch_losses[-1] if self.epoch_losses else 0
        self.final_loss = final_loss

        # 计算loss下降的统计
        loss_reduction = initial_loss - final_loss
        loss_reduction_pct = (loss_reduction / initial_loss * 100) if initial_loss > 0 else 0

        # 计算训练稳定性（batch loss的方差）
        if len(self.batch_losses) > 10:
            recent_batch_losses = self.batch_losses[-100:]  # 最近的100个batch
            batch_loss_std = np.std(recent_batch_losses)
        else:
            batch_loss_std = 0

        return {
            'architecture': self.arch_name,
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'loss_reduction': float(loss_reduction),
            'loss_reduction_pct': float(loss_reduction_pct),
            'convergence_speed': self.convergence_speed if self.convergence_speed else ArchitectureConfig.NUM_EPOCHS,
            'avg_epoch_time': float(np.mean(self.epoch_times)) if self.epoch_times else 0,
             'batch_loss_std': float(self.batch_loss_std) if self.batch_loss_std is not None else 0.0,
            'num_epochs_tracked': len(self.epoch_losses)
        }


# ==================== 4. 训练函数（记录详细loss变化） ====================
def train_architecture(model, train_loader, val_loader, arch_name):
    """训练一个特定架构的模型，详细记录loss变化"""
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=ArchitectureConfig.LEARNING_RATE)    #手动
    optimizer = optim.Adam(model.parameters(), lr=ArchitectureConfig.LEARNING_RATE)     #自适应

    # 创建loss跟踪器
    tracker = LossTracker(arch_name)

    print(f"\n  开始训练 {arch_name}...")

    for epoch in range(ArchitectureConfig.NUM_EPOCHS):
        start_time = datetime.now()

        # 训练阶段
        model.train()
        epoch_loss = 0
        batch_losses_this_epoch = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            # 记录梯度范数（训练稳定性指标）
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            tracker.gradient_norms.append(total_norm)

            optimizer.step()

            epoch_loss += loss.item()
            batch_losses_this_epoch.append(loss.item())

            # 每25%的batch打印一次进度
            if (batch_idx + 1) % max(1, len(train_loader) // 4) == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                print(f"    Epoch {epoch + 1:2d} | 进度: {progress:5.1f}% | Batch Loss: {loss.item():.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # 验证阶段
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * correct / total
        epoch_time = (datetime.now() - start_time).total_seconds()

        # 记录结果
        tracker.add_epoch_result(epoch, avg_train_loss, avg_val_loss,
                                 batch_losses_this_epoch, epoch_time)

        # 每5个epoch打印一次详细结果
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch + 1:3d}/{ArchitectureConfig.NUM_EPOCHS} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.2f}% | "
                  f"Time: {epoch_time:.1f}s")

    return tracker, val_accuracy


# ==================== 5. 运行架构实验 ====================
def run_architecture_experiment():
    """运行不同架构的实验"""
    print("=" * 70)
    print(" 网络架构实验: 层数 vs 节点数 vs Loss变化")
    print("=" * 70)

    # 设置随机种子
    torch.manual_seed(ArchitectureConfig.RANDOM_SEED)
    np.random.seed(ArchitectureConfig.RANDOM_SEED)

    # 加载数据
    print("\n 加载数据...")
    dataset = pd.read_csv(ArchitectureConfig.DATA_PATH, sep="\t", header=None)
    texts = dataset[0].tolist()
    labels = dataset[1].tolist()

    # 预处理
    label_to_idx = {label: i for i, label in enumerate(sorted(set(labels)))}
    label_indices = [label_to_idx[label] for label in labels]

    char_to_idx = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_idx:
                char_to_idx[char] = len(char_to_idx)

    vocab_size = len(char_to_idx)
    num_classes = len(label_to_idx)

    print(f" 数据统计:")
    print(f"  • 样本数: {len(texts)}")
    print(f"  • 词汇表: {vocab_size}")
    print(f"  • 类别数: {num_classes}")
    print(f"  • 训练轮数: {ArchitectureConfig.NUM_EPOCHS}")
    print(f"  • 测试架构数: {len(ArchitectureConfig.ARCHITECTURES)}")

    # 创建数据集
    class TextDataset(Dataset):
        def __init__(self, texts, labels, char_to_idx, max_len, vocab_size):
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.features = self._create_features(texts, char_to_idx, max_len, vocab_size)

        def _create_features(self, texts, char_to_idx, max_len, vocab_size):
            features = []
            for text in texts:
                encoded = [char_to_idx.get(char, 0) for char in text[:max_len]]
                encoded += [0] * (max_len - len(encoded))
                bow = torch.zeros(vocab_size)
                for idx in encoded:
                    if idx != 0:
                        bow[idx] += 1
                features.append(bow)
            return torch.stack(features)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    dataset = TextDataset(texts, label_indices, char_to_idx,
                          ArchitectureConfig.MAX_LEN, vocab_size)

    # 划分数据集
    train_size = int(0.7 * len(dataset))  # 70%训练
    val_size = int(0.15 * len(dataset))  # 15%验证
    test_size = len(dataset) - train_size - val_size

    train_dataset, temp_dataset = random_split(dataset, [train_size, len(dataset) - train_size])
    val_dataset, test_dataset = random_split(temp_dataset, [val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=ArchitectureConfig.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=ArchitectureConfig.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=ArchitectureConfig.BATCH_SIZE, shuffle=False)

    print(f"\n 数据集划分:")
    print(f"  • 训练集: {train_size} 样本")
    print(f"  • 验证集: {val_size} 样本")
    print(f"  • 测试集: {test_size} 样本")

    # 运行所有架构实验
    print(f"\n{'=' * 70}")
    print(" 开始架构实验...")
    print(f"{'=' * 70}")

    results = []
    trackers = []

    for arch_config in ArchitectureConfig.ARCHITECTURES:
        num_layers, hidden_dims, description = arch_config

        print(f"\n 测试架构: {description}")
        print(f"{'-' * 60}")

        # 创建模型
        model = FlexibleClassifier(vocab_size, hidden_dims, num_classes)

        # 训练模型
        tracker, val_accuracy = train_architecture(model, train_loader, val_loader, description)

        # 收集结果
        summary = tracker.get_summary()
        summary.update({
            'num_layers': num_layers,
            'hidden_dims': hidden_dims if isinstance(hidden_dims, list) else [hidden_dims],
            'num_params': model.num_params,
            'final_val_accuracy': val_accuracy,
            'description': description
        })

        results.append(summary)
        trackers.append(tracker)

        print(f" 完成! 最终训练Loss: {summary['final_loss']:.4f}, "
              f"验证准确率: {val_accuracy:.2f}%")

    return results, trackers, vocab_size, num_classes


# ==================== 6. 可视化Loss变化分析 ====================
def visualize_loss_analysis(results, trackers):
    """可视化loss变化分析"""
    print(f"\n{'=' * 70}")
    print(" Loss变化可视化分析")
    print(f"{'=' * 70}")

    os.makedirs(ArchitectureConfig.SAVE_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 创建结果DataFrame
    df = pd.DataFrame(results)

    # 1. 综合对比图
    fig = plt.figure(figsize=(20, 12))

    # 图1: 各架构的Loss下降曲线对比
    ax1 = plt.subplot(2, 3, 1)
    colors = plt.cm.tab20(np.linspace(0, 1, len(trackers)))

    for idx, tracker in enumerate(trackers):
        epochs = range(1, len(tracker.epoch_losses) + 1)
        ax1.plot(epochs, tracker.epoch_losses,
                 color=colors[idx], linewidth=2.5, alpha=0.8,
                 label=tracker.arch_name)

    ax1.set_xlabel('训练轮数 (Epoch)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('训练Loss', fontsize=11, fontweight='bold')
    ax1.set_title('不同架构的训练Loss下降曲线', fontsize=13, fontweight='bold', pad=15)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')  # 对数坐标更好观察下降趋势

    # 图2: 初始Loss vs 最终Loss
    ax2 = plt.subplot(2, 3, 2)
    scatter = ax2.scatter(df['initial_loss'], df['final_loss'],
                          c=df['num_layers'], s=df['num_params'] / 1000,
                          cmap='viridis', alpha=0.7, edgecolors='black')

    # 添加连接线显示下降
    for idx, row in df.iterrows():
        ax2.plot([row['initial_loss'], row['final_loss']],
                 [row['initial_loss'], row['final_loss']],
                 'k--', alpha=0.2, linewidth=0.5)
        ax2.annotate('', xy=(row['final_loss'], row['final_loss']),
                     xytext=(row['initial_loss'], row['initial_loss']),
                     arrowprops=dict(arrowstyle='->', color='gray', alpha=0.5))

    ax2.set_xlabel('初始Loss', fontsize=11, fontweight='bold')
    ax2.set_ylabel('最终Loss', fontsize=11, fontweight='bold')
    ax2.set_title('初始Loss vs 最终Loss (大小=参数量/1000)', fontsize=13, fontweight='bold', pad=15)
    ax2.grid(True, alpha=0.3)

    # 添加颜色条
    plt.colorbar(scatter, ax=ax2, label='层数')

    # 图3: 收敛速度分析
    ax3 = plt.subplot(2, 3, 3)
    bars = ax3.bar(range(len(df)), df['loss_reduction_pct'],
                   color=plt.cm.coolwarm(df['convergence_speed'] / max(df['convergence_speed'])))

    ax3.set_xlabel('网络架构', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Loss下降百分比 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Loss下降效果与收敛速度', fontsize=13, fontweight='bold', pad=15)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels([f"L{d['num_layers']}-N{sum(d['hidden_dims']) // len(d['hidden_dims'])}"
                         for d in results], rotation=45, ha='right')

    # 在柱状图上添加收敛速度
    for i, (bar, speed) in enumerate(zip(bars, df['convergence_speed'])):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 1,
                 f'E{speed}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 图4: Batch Loss波动分析（训练稳定性）
    ax4 = plt.subplot(2, 3, 4)

    # 选择几个代表性架构展示batch loss波动
    sample_indices = [0, 3, 6, 8]  # 选择单层、两层、三层、四层各一个
    sample_colors = ['red', 'blue', 'green', 'purple']

    for idx, color in zip(sample_indices, sample_colors):
        tracker = trackers[idx]
        if len(tracker.batch_losses) > 100:
            # 取最后100个batch展示
            batch_indices = range(len(tracker.batch_losses) - 100, len(tracker.batch_losses))
            batch_losses = tracker.batch_losses[-100:]

            # 使用移动平均平滑
            window = 5
            if len(batch_losses) > window:
                smoothed = np.convolve(batch_losses, np.ones(window) / window, mode='valid')
                ax4.plot(range(len(smoothed)), smoothed,
                         color=color, linewidth=1.5, alpha=0.7,
                         # label=f"{tracker.arch_name} (std={tracker.batch_loss_std:.4f})"
                         label=f"{tracker.arch_name} (std={tracker.batch_loss_std:.4f})")

    ax4.set_xlabel('Batch序号 (最近100个)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Batch Loss', fontsize=11, fontweight='bold')
    ax4.set_title('Batch Loss波动分析 (训练稳定性)', fontsize=13, fontweight='bold', pad=15)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)

    # 图5: 层数 vs Loss下降效果
    ax5 = plt.subplot(2, 3, 5)

    # 按层数分组
    layers_grouped = df.groupby('num_layers')
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    for (layer_num, group), color in zip(layers_grouped, colors):
        ax5.scatter(group['num_params'], group['final_loss'],
                    s=150, color=color, alpha=0.7, edgecolors='black',
                    label=f'{layer_num}层网络')

    ax5.set_xlabel('模型参数量', fontsize=11, fontweight='bold')
    ax5.set_ylabel('最终训练Loss', fontsize=11, fontweight='bold')
    ax5.set_title('层数与参数量对Loss的影响', fontsize=13, fontweight='bold', pad=15)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    ax5.set_xscale('log')

    # 图6: 训练时间分析
    ax6 = plt.subplot(2, 3, 6)

    x_pos = np.arange(len(df))
    bars1 = ax6.bar(x_pos - 0.2, df['avg_epoch_time'], 0.4,
                    label='每轮时间', color='skyblue')
    bars2 = ax6.bar(x_pos + 0.2, df['convergence_speed'] * df['avg_epoch_time'], 0.4,
                    label='收敛总时间', color='lightcoral')

    ax6.set_xlabel('网络架构', fontsize=11, fontweight='bold')
    ax6.set_ylabel('时间 (秒)', fontsize=11, fontweight='bold')
    ax6.set_title('训练时间效率分析', fontsize=13, fontweight='bold', pad=15)
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels([d['description'][:15] for d in results], rotation=45, ha='right')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(f'{ArchitectureConfig.SAVE_DIR}/architecture_loss_analysis_{timestamp}.png',
                dpi=150, bbox_inches='tight')
    plt.show()

    # 2. 详细对比表格
    print(f"\n 架构性能对比表")
    print("-" * 90)
    print(f"{'架构描述':<20} {'层数':<6} {'参数量':<12} {'初始Loss':<10} {'最终Loss':<10} "
          f"{'下降%':<8} {'收敛速度':<10} {'验证准确率':<12}")
    print("-" * 90)

    df_sorted = df.sort_values('final_loss')
    for _, row in df_sorted.iterrows():
        print(f"{row['description'][:18]:<20} {row['num_layers']:<6} "
              f"{row['num_params']:<12,} {row['initial_loss']:<10.4f} "
              f"{row['final_loss']:<10.4f} {row['loss_reduction_pct']:<8.1f}% "
              f"Epoch {row['convergence_speed']:<8} {row['final_val_accuracy']:<11.2f}%")

    return df, trackers, timestamp


# ==================== 7. 生成详细分析报告 ====================
def generate_analysis_report(df, trackers, vocab_size, num_classes, timestamp):
    """生成详细的分析报告"""
    print(f"\n{'=' * 70}")
    print(" 网络架构对Loss变化影响分析报告")
    print(f"{'=' * 70}")

    # 找到最佳架构（综合考虑Loss和准确率）
    df['score'] = (100 - df['final_loss'] * 10) + (df['final_val_accuracy'] / 2)
    best_idx = df['score'].idxmax()
    best_arch = df.loc[best_idx]

    print(f"\n 实验配置:")
    print(f"  • 输入维度: {vocab_size}")
    print(f"  • 输出维度: {num_classes}")
    print(f"  • 训练轮数: {ArchitectureConfig.NUM_EPOCHS}")
    print(f"  • 测试架构数: {len(df)}")

    print(f"\n 最佳性能架构:")
    print(f"  • 架构: {best_arch['description']}")
    print(f"  • 层数: {best_arch['num_layers']}")
    print(f"  • 最终Loss: {best_arch['final_loss']:.4f}")
    print(f"  • 验证准确率: {best_arch['final_val_accuracy']:.2f}%")
    print(f"  • 收敛速度: {best_arch['convergence_speed']}个epoch")

    print(f"\n 关键发现:")

    # 分析1: 层数对Loss的影响
    layer_analysis = df.groupby('num_layers').agg({
        'final_loss': ['mean', 'min', 'max'],
        'convergence_speed': 'mean',
        'final_val_accuracy': 'mean'
    }).round(4)

    print(f"1. 层数影响分析:")
    for layers, stats in layer_analysis.iterrows():
        print(f"   {layers}层网络: 平均Loss={stats[('final_loss', 'mean')]:.4f}, "
              f"平均准确率={stats[('final_val_accuracy', 'mean')]:.2f}%, "
              f"收敛速度={stats[('convergence_speed', 'mean')]:.1f} epoch")

    # 分析2: 参数量与Loss的关系
    corr_params_loss = df['num_params'].corr(df['final_loss'])
    print(f"\n2. 参数量与Loss相关性: {corr_params_loss:.4f}")
    if corr_params_loss > 0.3:
        print("   → 参数量增加可能导致Loss上升（可能过拟合）")
    elif corr_params_loss < -0.3:
        print("   → 参数量增加有助于降低Loss")
    else:
        print("   → 参数量与Loss关系不明显")

    # 分析3: 收敛速度分析
    fastest_idx = df['convergence_speed'].idxmin()
    fastest_arch = df.loc[fastest_idx]
    print(f"\n3. 最快收敛架构: {fastest_arch['description']}")
    print(f"   仅需{fastest_arch['convergence_speed']}个epoch达到稳定")
    print(f"   最终Loss: {fastest_arch['final_loss']:.4f}")

    # 分析4: 训练稳定性分析
    stable_idx = df['batch_loss_std'].idxmin()
    stable_arch = df.loc[stable_idx]
    print(f"\n4. 最稳定训练架构: {stable_arch['description']}")
    print(f"   Batch Loss标准差: {stable_arch['batch_loss_std']:.4f}")
    print(f"   （波动越小，训练越稳定）")

    print(f"\n💡 实践建议:")

    # 基于实验结果的建议
    if best_arch['num_layers'] == 1:
        print("1. 单层网络效果最好，说明任务相对简单")
        print("2. 不需要复杂网络，可减少计算资源")
    elif best_arch['num_layers'] == 2:
        print("1. 双层网络是最佳平衡点")
        print("2. 既有足够表达能力，又不会过拟合")
    else:
        print("1. 多层网络效果最佳，但需要足够数据")
        print("2. 考虑添加更多正则化防止过拟合")

    # 效率建议
    efficient_idx = (df['final_val_accuracy'] / df['num_params'] * 1e6).idxmax()
    efficient_arch = df.loc[efficient_idx]

    print(f"\n2. 效率最佳架构: {efficient_arch['description']}")
    print(f"   每百万参数准确率: {efficient_arch['final_val_accuracy'] / efficient_arch['num_params'] * 1e6:.4f}")

    # 保存详细结果
    df.to_csv(f'{ArchitectureConfig.SAVE_DIR}/architecture_results_{timestamp}.csv',
              index=False, encoding='utf-8-sig')

    # 保存详细loss曲线数据
    loss_curves = {}
    for tracker in trackers:
        loss_curves[tracker.arch_name] = {
            'epoch_losses': tracker.epoch_losses,
            'val_losses': tracker.val_losses,
            'batch_loss_std': tracker.batch_loss_std
        }

    with open(f'{ArchitectureConfig.SAVE_DIR}/loss_curves_{timestamp}.json', 'w') as f:
        json.dump(loss_curves, f, indent=2)

    print(f"\n 结果已保存至: {ArchitectureConfig.SAVE_DIR}/")
    print(f"   图表: architecture_loss_analysis_{timestamp}.png")
    print(f"   数据: architecture_results_{timestamp}.csv")
    print(f"   Loss曲线: loss_curves_{timestamp}.json")

    return best_arch


# ==================== 8. 主函数 ====================
def main():
    """主函数"""
    print("=" * 70)
    print(" 网络架构实验: 层数 vs 节点数 vs Loss变化")
    print("=" * 70)

    try:
        # 运行实验
        results, trackers, vocab_size, num_classes = run_architecture_experiment()

        # 可视化分析
        df, trackers, timestamp = visualize_loss_analysis(results, trackers)

        # 生成报告
        best_arch = generate_analysis_report(df, trackers, vocab_size, num_classes, timestamp)

        print(f"\n{'=' * 70}")
        print(" 实验完成!")
        print(f" 最佳架构: {best_arch['description']}")
        print(f" 最终Loss: {best_arch['final_loss']:.4f}")
        print(f" 收敛速度: {best_arch['convergence_speed']}个epoch")
        print("=" * 70)

        # 生成推荐配置代码
        print(f"\n 推荐配置代码:")
        print("-" * 40)

        hidden_dims = best_arch['hidden_dims']
        if len(hidden_dims) == 1:
            layers_code = f"hidden_dim = {hidden_dims[0]}"
        else:
            layers_code = f"hidden_dims = {hidden_dims}"

        print(f"""```python
# 基于架构实验的最佳配置
vocab_size = {vocab_size}
{layers_code}
output_dim = {num_classes}

# 创建模型
model = FlexibleClassifier(vocab_size, hidden_dims, output_dim)
print(f"架构: {{model.num_layers}}层, {{'→'.join(map(str, model.hidden_dims))}}")
print(f"参数量: {{model.num_params:,}}")
print(f"预期最终Loss: {best_arch['final_loss']:.4f}")
print(f"预期验证准确率: {best_arch['final_val_accuracy']:.2f}%")
```""")

    except Exception as e:
        print(f"\n 实验出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()