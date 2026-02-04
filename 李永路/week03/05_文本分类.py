import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

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

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

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
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  # 添加padding处理
        self.rnn = nn.RNN(embedding_dim,
                          hidden_dim,
                          batch_first=True,
                          nonlinearity='tanh',  # 明确指定激活函数
                          bidirectional=True)  # 改为双向RNN

        # 双向RNN需要2倍隐藏层维度
        self.fc = nn.Sequential(
            nn.Dropout(0.2),  # 添加Dropout防止过拟合
            nn.Linear(hidden_dim * 2, hidden_dim),  # 双向输出维度调整
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hn = self.rnn(embedded)

        # 双向RNN隐藏状态处理：拼接最后时刻的前向/后向状态
        last_hn = torch.cat((hn[-2, :, :], hn[-1, :, :]), dim=1)
        out = self.fc(last_hn)
        return out


class GRUClassifier(nn.Module):  # 修改1：类名改为GRUClassifier
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 修改2：将LSTM替换为GRU，参数结构相同
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)

        # 修改3：GRU的输出结构与LSTM不同
        gru_out, hidden_state = self.gru(embedded)  # GRU只有隐藏状态，没有细胞状态

        # 使用最后一个时间步的隐藏状态进行分类
        out = self.fc(hidden_state.squeeze(0))  # hidden_state形状为(1, batch, hidden_dim)
        return out

# --- Training and Prediction ---
lstm_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

# 确定性的数据集分割
texts = dataset[0].tolist()
labels = numerical_labels

# 计算验证集大小
val_size = int(0.2 * len(texts))
train_size = len(texts) - val_size

# 创建训练集和验证集
train_texts = texts[:train_size]
train_labels = labels[:train_size]
val_texts = texts[train_size:]
val_labels = labels[train_size:]

# 创建Dataset
train_dataset = CharRNNDataset(train_texts, train_labels, char_to_index, max_len)
val_dataset = CharRNNDataset(val_texts, val_labels, char_to_index, max_len)

# 创建数据加载器
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)


embedding_dim = 128
hidden_dim = 256
output_dim = len(label_to_index)

modelRNN = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(modelRNN.parameters(),  # 使用AdamW优化器
                        lr=0.001,
                        weight_decay=1e-4)  # 添加L2正则化

# 添加学习率调度器
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=2)


# 修改后的evaluate函数实现
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total if total > 0 else 0
    return accuracy

num_epochs = 2
# --- 修改后的训练循环 ---
for epoch in range(num_epochs):
    modelRNN.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = modelRNN(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(modelRNN.parameters(), max_norm=1.0)

        optimizer.step()
        running_loss += loss.item()

        # 添加准确率计算
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        if idx % 50 == 0:
            acc = correct / total
            print(f"Batch {idx}, Loss: {loss.item():.4f}, Acc: {acc:.4f}")

    # 计算验证集准确率（需添加验证集数据）
    val_acc = evaluate(modelRNN, val_dataloader)  # 假设已实现evaluate函数
    scheduler.step(val_acc)  # 根据验证集表现调整学习率

    epoch_acc = correct / total
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

modelGRU = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelGRU.parameters(), lr=0.01)


num_epochs = 4
for epoch in range(num_epochs):
    modelGRU.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = modelGRU(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

modelLSTM = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(modelLSTM.parameters(), lr=0.01)


num_epochs = 4
for epoch in range(num_epochs):
    modelLSTM.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = modelLSTM(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
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

index_to_labelLSTM = {i: label for label, i in label_to_index.items()}

def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
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

index_to_labelGRU = {i: label for label, i in label_to_index.items()}

def classify_text_rnn(text, model, char_to_index, max_len, index_to_label):
    model.eval()
    with torch.no_grad():
        indices = [char_to_index.get(char, 0) for char in text[:max_len]]
        # 动态填充
        pad_len = max_len - len(indices)
        if pad_len > 0:
            indices += [0] * pad_len
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

        output = model(input_tensor)
        _, predicted_index = torch.max(output, 1)
        return index_to_label[predicted_index.item()]

index_to_labelRNN = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, modelLSTM, char_to_index, max_len, index_to_labelLSTM)
print(f"输入 '{new_text}' LSTM预测为: '{predicted_class}'")
predicted_class = classify_text_rnn(new_text, modelRNN, char_to_index, max_len, index_to_labelRNN)
print(f"输入 '{new_text}' RNN预测为: '{predicted_class}'")
predicted_class = classify_text_gru(new_text, modelRNN, char_to_index, max_len, index_to_labelGRU)
print(f"输入 '{new_text}' GRU预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, modelLSTM, char_to_index, max_len, index_to_labelLSTM)
print(f"输入 '{new_text_2}' LSTM预测为: '{predicted_class_2}'")
predicted_class_2 = classify_text_rnn(new_text_2, modelLSTM, char_to_index, max_len, index_to_labelRNN)
print(f"输入 '{new_text_2}' RNN预测为: '{predicted_class_2}'")
predicted_class_2 = classify_text_gru(new_text_2, modelLSTM, char_to_index, max_len, index_to_labelGRU)
print(f"输入 '{new_text_2}' GRU预测为: '{predicted_class_2}'")
