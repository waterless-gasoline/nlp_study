import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
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

torch.manual_seed(42)
np.random.seed(42)

class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # 将文本转换为字符索引序列
        text = self.texts[idx]
        # 1. 将每个字符转换为索引
        indices = [self.char_to_index.get(char, 0) for char in text]
        # 2. 截断或填充到固定长度
        if len(indices) > self.max_len:
            indices = indices[:self.max_len]
        else:
            indices = indices + [0] * (self.max_len - len(indices))

        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# RNN模型
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 嵌入层：将字符索引转换为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # RNN层
        self.rnn = nn.RNN(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层：将RNN输出转换为分类结果
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x形状: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]

        # RNN前向传播
        rnn_out, hidden = self.rnn(embedded)  # rnn_out: [batch_size, seq_len, hidden_dim]

        # 取最后一个时间步的输出
        last_output = rnn_out[:, -1, :]  # [batch_size, hidden_dim]

        # 通过全连接层得到分类结果
        output = self.fc(self.dropout(last_output))

        return output


# GRU模型
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # GRU层
        self.gru = nn.GRU(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        last_output = gru_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output


# LSTM模型
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # LSTM层（包含细胞状态）
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        last_output = lstm_out[:, -1, :]
        output = self.fc(self.dropout(last_output))
        return output

# 创建数据集
char_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)

# 划分训练集和验证集（80%训练，20%验证）
train_size = int(0.8 * len(char_dataset))
val_size = len(char_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(char_dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")


def train_model(model, model_name, train_loader, val_loader, num_epochs=15):
    """训练一个模型并返回训练历史"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    print(f"\n{'=' * 60}")
    print(f"开始训练 {model_name}")
    print(f"{'=' * 60}")

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_loss_avg = train_loss / len(train_loader)
        train_accuracy = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_loss_avg = val_loss / len(val_loader)
        val_accuracy = val_correct / val_total

        # 更新学习率
        scheduler.step(val_loss_avg)

        # 记录历史
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # 打印进度
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch + 1:2d}/{num_epochs}: "
              f"Train Loss: {train_loss_avg:.4f}, Train Acc: {train_accuracy:.4f}, "
              f"Val Loss: {val_loss_avg:.4f}, Val Acc: {val_accuracy:.4f}, "
              f"LR: {current_lr:.6f}")

    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'final_val_accuracy': val_accuracies[-1]
    }

# 设置模型参数
vocab_size = len(char_to_index)
embed_dim = 128
hidden_dim = 256
output_dim = len(label_to_index)
num_layers = 2

# 创建三个模型
models = {
    'RNN': RNNClassifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers),
    'GRU': GRUClassifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers),
    'LSTM': LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim, num_layers)
}

# 训练所有模型
results = {}
for name, model in models.items():
    result = train_model(model, name, train_loader, val_loader, num_epochs=15)
    results[name] = result
    print(f"\n{name} 训练完成，最终验证准确率: {result['final_val_accuracy']:.4f}")

# 打印参数数量对比
print("\n" + "="*60)
print("模型参数数量对比")
print("="*60)
for name, result in results.items():
    total_params = sum(p.numel() for p in result['model'].parameters())
    print(f"{name}: {total_params:,} 个参数")

# 设置图形
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 颜色映射
colors = {'RNN': 'blue', 'GRU': 'green', 'LSTM': 'red'}

# 1. 训练损失对比
ax1 = axes[0, 0]
for name, result in results.items():
    ax1.plot(result['train_losses'], label=name, color=colors[name], linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('训练损失对比')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 验证损失对比
ax2 = axes[0, 1]
for name, result in results.items():
    ax2.plot(result['val_losses'], label=name, color=colors[name], linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('验证损失对比')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. 验证准确率对比
ax3 = axes[1, 0]
for name, result in results.items():
    ax3.plot(result['val_accuracies'], label=name, color=colors[name], linewidth=2)
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Accuracy')
ax3.set_title('验证准确率对比')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, 1.0)


plt.tight_layout()
plt.show()

# 打印详细对比结果
print("\n" + "="*60)
print("模型对比总结")
print("="*60)
print(f"{'模型':<10} {'训练损失':<12} {'验证损失':<12} {'验证准确率':<12}")
print("-"*60)
for name, result in results.items():
    print(f"{name:<10} {result['train_losses'][-1]:<12.4f} "
          f"{result['val_losses'][-1]:<12.4f} {result['final_val_accuracy']:<12.4f}")


def test_model(model, test_texts, char_to_index, max_len, index_to_label):
    """测试模型在自定义文本上的表现"""
    model.eval()

    test_results = []
    for text in test_texts:
        # 预处理输入文本
        indices = [char_to_index.get(char, 0) for char in text]
        if len(indices) > max_len:
            indices = indices[:max_len]
        else:
            indices = indices + [0] * (max_len - len(indices))

        inputs = torch.tensor(indices, dtype=torch.long).unsqueeze(0)  # 增加批次维度

        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicted_label = index_to_label[predicted.item()]

        test_results.append((text, predicted_label))

    return test_results


# 准备测试文本
test_texts = [
    "帮我导航到北京",
    "查询明天北京的天气",
    "播放周杰伦的音乐",
    "打开客厅的空调",
    "今天有什么新闻",
    "设置明天早上7点的闹钟"
]

# 创建反向标签映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试所有模型
print("\n" + "=" * 60)
print("模型预测测试")
print("=" * 60)

for name, result in results.items():
    print(f"\n{name} 模型预测结果:")
    predictions = test_model(result['model'], test_texts, char_to_index, max_len, index_to_label)
    for text, pred_label in predictions:
        print(f"  文本: '{text:15}' → 预测: '{pred_label}'")
