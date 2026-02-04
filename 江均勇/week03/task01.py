import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

# ===================== 1. 数据预处理 =====================
# 读取数据
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

# 字符编码
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40  # 文本最大长度

# ===================== 2. 自定义数据集 =====================
class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 截断+补零
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 划分训练集/验证集（8:2）
full_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# 数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ===================== 3. 三种循环模型定义 =====================
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='lstm'):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn_type = rnn_type
        
        # 根据类型选择RNN/LSTM/GRU
        if rnn_type == 'rnn':
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif rnn_type == 'gru':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'rnn', 'lstm' or 'gru'")
        
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # [batch, seq_len, embedding_dim]
        
        # RNN前向传播
        if self.rnn_type == 'lstm':
            rnn_out, (hidden_state, cell_state) = self.rnn(embedded)
        else:  # RNN/GRU只有hidden_state
            rnn_out, hidden_state = self.rnn(embedded)
        
        # 取最后一个时间步的隐藏状态（batch_first=True，hidden_state维度：[1, batch, hidden_dim]）
        out = self.fc(hidden_state.squeeze(0))
        return out

# ===================== 4. 训练与评估函数 =====================
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=4):
    """训练模型并返回训练/验证精度"""
    train_losses = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # 验证阶段（计算精度）
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = correct / total
        train_loss = running_loss / len(train_loader)
        
        train_losses.append(train_loss)
        val_accs.append(val_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"Train Loss: {train_loss:.4f} | Val Accuracy: {val_acc:.4f}")
    
    return train_losses, val_accs

# ===================== 5. 实验配置与运行 =====================
# 超参数
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
lr = 0.001
num_epochs = 4

# 存储实验结果
results = {}

# 实验1：RNN
print("\n========== Training RNN ==========")
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='rnn')
rnn_criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=lr)
rnn_losses, rnn_accs = train_model(rnn_model, train_loader, val_loader, rnn_criterion, rnn_optimizer, num_epochs)
results['rnn'] = {'losses': rnn_losses, 'accs': rnn_accs, 'final_acc': rnn_accs[-1]}

# 实验2：LSTM（原始模型）
print("\n========== Training LSTM ==========")
lstm_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='lstm')
lstm_criterion = nn.CrossEntropyLoss()
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=lr)
lstm_losses, lstm_accs = train_model(lstm_model, train_loader, val_loader, lstm_criterion, lstm_optimizer, num_epochs)
results['lstm'] = {'losses': lstm_losses, 'accs': lstm_accs, 'final_acc': lstm_accs[-1]}

# 实验3：GRU
print("\n========== Training GRU ==========")
gru_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='gru')
gru_criterion = nn.CrossEntropyLoss()
gru_optimizer = optim.Adam(gru_model.parameters(), lr=lr)
gru_losses, gru_accs = train_model(gru_model, train_loader, val_loader, gru_criterion, gru_optimizer, num_epochs)
results['gru'] = {'losses': gru_losses, 'accs': gru_accs, 'final_acc': gru_accs[-1]}

# ===================== 6. 结果对比 =====================
print("\n========== 实验结果对比 ==========")
for model_type, res in results.items():
    print(f"{model_type.upper()} | 最终验证精度: {res['final_acc']:.4f} | 最后一轮训练Loss: {res['losses'][-1]:.4f}")

# ===================== 7. 单文本预测函数（可选） =====================
def classify_text(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    return index_to_label[predicted_idx.item()]

# 示例预测
print("\n========== 示例预测 ==========")
test_texts = ["帮我导航到北京", "查询明天北京的天气"]
for text in test_texts:
    rnn_pred = classify_text(text, rnn_model, char_to_index, max_len, index_to_label)
    lstm_pred = classify_text(text, lstm_model, char_to_index, max_len, index_to_label)
    gru_pred = classify_text(text, gru_model, char_to_index, max_len, index_to_label)
    print(f"输入: {text}")
    print(f"RNN预测: {rnn_pred} | LSTM预测: {lstm_pred} | GRU预测: {gru_pred}")