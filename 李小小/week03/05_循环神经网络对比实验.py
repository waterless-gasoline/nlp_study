"""
循环神经网络(RNN/LSTM/GRU)对比实验
本文件对比了三种不同循环神经网络架构在文本分类任务上的性能表现
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
import numpy as np

# 读取数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 创建标签映射
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符到索引的映射
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# 设置最大长度
max_len = 40


class CharDataset(Dataset):
    """字符级数据集类"""
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 填充和截断
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class RNNClassifier(nn.Module):
    """RNN文本分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden = self.rnn(embedded)
        # 使用最后一个时间步的隐藏状态
        out = self.fc(hidden.squeeze(0))
        return out


class LSTMClassifier(nn.Module):
    """LSTM文本分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        # 使用最后一个时间步的隐藏状态
        out = self.fc(hidden_state.squeeze(0))
        return out


class GRUClassifier(nn.Module):
    """GRU文本分类器"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden = self.gru(embedded)
        # 使用最后一个时间步的隐藏状态
        out = self.fc(hidden.squeeze(0))
        return out


def train_model(model, train_loader, criterion, optimizer, num_epochs=4):
    """训练模型"""
    model.train()
    start_time = time.time()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    training_time = time.time() - start_time
    return training_time


def evaluate_model(model, test_loader):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy


def predict_text(text, model, char_to_index, max_len, index_to_label):
    """预测单个文本的类别"""
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()
        predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建数据集和数据加载器
dataset = CharDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 模型参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

# 定义标签索引映射
index_to_label = {i: label for label, i in label_to_index.items()}

print("=" * 60)
print("循环神经网络(RNN/LSTM/GRU)对比实验")
print("=" * 60)

# 实验结果存储
results = {}

# 1. RNN模型实验
print("\n1. 训练RNN模型...")
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.001)

rnn_training_time = train_model(rnn_model, dataloader, rnn_criterion, rnn_optimizer)
rnn_accuracy = evaluate_model(rnn_model, dataloader)

print(f"RNN - 训练时间: {rnn_training_time:.2f}s, 准确率: {rnn_accuracy:.2f}%")
results['RNN'] = {'training_time': rnn_training_time, 'accuracy': rnn_accuracy}

# 2. LSTM模型实验
print("\n2. 训练LSTM模型...")
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

lstm_training_time = train_model(lstm_model, dataloader, lstm_criterion, lstm_optimizer)
lstm_accuracy = evaluate_model(lstm_model, dataloader)

print(f"LSTM - 训练时间: {lstm_training_time:.2f}s, 准确率: {lstm_accuracy:.2f}%")
results['LSTM'] = {'training_time': lstm_training_time, 'accuracy': lstm_accuracy}

# 3. GRU模型实验
print("\n3. 训练GRU模型...")
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.001)

gru_training_time = train_model(gru_model, dataloader, gru_criterion, gru_optimizer)
gru_accuracy = evaluate_model(gru_model, dataloader)

print(f"GRU - 训练时间: {gru_training_time:.2f}s, 准确率: {gru_accuracy:.2f}%")
results['GRU'] = {'training_time': gru_training_time, 'accuracy': gru_accuracy}

# 结果汇总
print("\n" + "=" * 60)
print("实验结果汇总:")
print("=" * 60)
print(f"{'模型':<10} {'训练时间(s)':<15} {'准确率(%)':<15}")
print("-" * 40)
for model_name, metrics in results.items():
    print(f"{model_name:<10} {metrics['training_time']:<15.2f} {metrics['accuracy']:<15.2f}")

# 找出最佳模型
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
print(f"\n最佳模型: {best_model_name} (准确率: {results[best_model_name]['accuracy']:.2f}%)")

# 使用最佳模型进行预测示例
best_model = {'RNN': rnn_model, 'LSTM': lstm_model, 'GRU': gru_model}[best_model_name]
print(f"\n使用{best_model_name}模型进行预测示例:")

test_texts = ["帮我导航到北京", "查询明天北京的天气"]
for text in test_texts:
    prediction = predict_text(text, best_model, char_to_index, max_len, index_to_label)
    print(f"输入: '{text}' -> 预测类别: '{prediction}'")

# 模型架构分析
print("\n" + "=" * 60)
print("RNN、LSTM、GRU模型架构分析:")
print("=" * 60)
print("""
RNN (Recurrent Neural Network):
- 最基础的循环神经网络
- 存在梯度消失问题，难以处理长序列
- 参数量相对较少，训练速度快

LSTM (Long Short-Term Memory):
- 解决了RNN的梯度消失问题
- 通过门控机制控制信息流动
- 包含输入门、遗忘门、输出门
- 参数量较大，但能处理长序列

GRU (Gated Recurrent Unit):
- LSTM的简化版本
- 包含重置门和更新门
- 参数量介于RNN和LSTM之间
- 通常训练速度比LSTM快
""")