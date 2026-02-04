"""
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
"""
# 分类的评估指标应该是准确率

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40

# 列出要测试的隐藏层参数和模型选择
HIDDEN_DIM = [128, 256, 512, 1024]
MODEL = ["SimpleClassifier_3", "SimpleClassifier_4"]

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier_3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_3, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class SimpleClassifier_4(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier_4, self).__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(hidden_dim//2, output_dim)

    def forward(self, x):
        x = self.relu1(self.bn1(self.fc1(x)))
        x = self.relu2(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return x


def train_test_model(model, train_dataloader, test_dataloader, criterion, optimizer):
    """
    根据模型完成训练和评估,并记录损失和准确率
    """
    epochs = 10
    train_losses = []
    test_losses = []
    test_accuracys = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # train_losses.append(running_loss / (idx + 1))
            if idx % 100 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        train_losses.append(running_loss / len(train_dataloader))

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_dataloader):.4f}")

        # 切换到评估模式,在测试集上进行测试
        model.eval()
        total_correct = 0
        with torch.no_grad():
            for inputs, labels in test_dataloader:
                losses = 0.0
                outputs = model(inputs)
                losses += criterion(outputs, labels).item()

                _, predicted = torch.max(outputs, dim=1)
                total_correct += (predicted == labels).sum().item()

        test_losses.append(losses / len(test_dataloader))
        test_accuracys.append(total_correct / len(test_dataloader.dataset))

    return train_losses, test_losses, test_accuracys


def plot_models_results_dual_y(model_results, model_name):
    """
    绘制多个模型的结果对比图
    :param model_results: 模型结果数据
    :param model_name: 模型名称
    """
    # 1. 创建画布和子图（2行3列，n为hidden_dim的个数）
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    fig.suptitle(f'{model_name} - Loss (Left) & Accuracy (Right)', fontsize=14, y=0.98)

    # 2. 定义颜色搭配（统一风格，提升可读性）
    color_loss_train = '#1f77b4'  # 训练损失：蓝色
    color_loss_test = '#ff7f0e'  # 测试损失：橙色
    color_acc = '#2ca02c'  # 测试准确率：绿色

    # 3. 遍历每个hidden_dim，绘制双y轴子图
    for idx, hidden_dim in enumerate(HIDDEN_DIM):
        # 提取当前模型的结果数据
        train_losses = model_results[hidden_dim]['train_losses']
        test_losses = model_results[hidden_dim]['test_losses']
        test_accuracys = model_results[hidden_dim]['test_accuracys']

        # 3.1 获取当前子图的主坐标轴（左侧y轴，对应Loss）
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]

        # 3.2 绘制左侧y轴：Loss曲线（主坐标轴）
        line1 = ax.plot(train_losses, label='Train Loss', color=color_loss_train, linewidth=1.5)
        line2 = ax.plot(test_losses, label='Test Loss', color=color_loss_test, linewidth=1.5)

        # 3.3 设置左侧y轴属性
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss', color=color_loss_train, fontsize=10)
        ax.tick_params(axis='y', labelcolor=color_loss_train)  # 左侧y轴刻度颜色匹配
        ax.set_title(f'Hidden Dim: {hidden_dim}', fontsize=12)
        ax.grid(alpha=0.3, linestyle='--')

        # 3.4 创建副坐标轴（右侧y轴，对应Accuracy，共享x轴）
        ax2 = ax.twinx()

        # 3.5 绘制右侧y轴：Accuracy曲线（副坐标轴）
        line3 = ax2.plot(test_accuracys, label='Test Accuracy', color=color_acc, linewidth=1.5)

        # 3.6 设置右侧y轴属性
        ax2.set_ylabel('Accuracy', color=color_acc, fontsize=10)
        ax2.tick_params(axis='y', labelcolor=color_acc)  # 右侧y轴刻度颜色匹配
        ax2.set_ylim(0, 1.0)  # 准确率固定在0~1范围，更直观

        # 3.7 合并两个坐标轴的图例（避免出现两个独立图例）
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, fontsize=8)

    # 4. 调整子图间距，避免右侧y轴标签超出画布（关键）
    plt.tight_layout()
    plt.subplots_adjust(top=0.9, wspace=0.3, hspace=0.4)  # wspace调整水平间距，适配右侧y轴

    # 5. 显示/保存图表
    plt.show()
    # fig.savefig(f'{model_name}_dual_y_results.png', dpi=300, bbox_inches='tight')

# 划分训练集和测试集
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, numerical_labels, test_size=0.2, random_state=42, stratify=numerical_labels)
print("✅训练集/测试集划分完毕!")

train_dataset = CharBoWDataset(train_texts, train_labels, char_to_index, max_len, vocab_size)
test_dataset = CharBoWDataset(test_texts, test_labels, char_to_index, max_len, vocab_size)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("✅数据集加载完毕!")


for model_name in MODEL:
    model_results = {}
    for hidden_dim in HIDDEN_DIM:
        print(f"🚀 开始训练模型：{model_name}，隐藏层维度：{hidden_dim}")
        if model_name == "SimpleClassifier_3":
            model = SimpleClassifier_3(vocab_size, hidden_dim, len(label_to_index))

        elif model_name == "SimpleClassifier_4":
            model = SimpleClassifier_4(vocab_size, hidden_dim, len(label_to_index))

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        train_losses, test_losses, test_accuracys = train_test_model(model, train_dataloader, test_dataloader, criterion, optimizer)

        model_results[hidden_dim] = {
            "hidden_dim": hidden_dim,
            "train_losses": train_losses,
            "test_losses": test_losses,
            "test_accuracys": test_accuracys
        }
        print(f"✅模型训练完毕，测试集准确率：{test_accuracys[-1]:.4f}")

    plot_models_results_dual_y(model_results, model_name)

