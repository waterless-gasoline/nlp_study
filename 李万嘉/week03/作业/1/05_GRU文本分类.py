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

max_len = 40

class CharLSTMDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        len0 = len(text)
        indices = []

        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return (torch.tensor(indices, dtype=torch.long),
                self.labels[idx]),


class CharGRNDataset(Dataset):
    def __init__(self, texts, labels, char2idx, max_len=40):
        self.texts = texts
        self.labels = labels
        self.char2idx = char2idx
        self.max_len = max_len
        self.pad_idx = char2idx['<pad>']
        self.unk_idx = 0

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 转换为索引
        char_indices = []
        for char in text:
            idx_char = self.char2idx.get(char, self.unk_idx)
            char_indices.append(idx_char)

        # 计算实际长度
        actual_length = len(char_indices)

        # 截断或填充
        if actual_length > self.max_len:
            char_indices = char_indices[:self.max_len]
            effective_length = self.max_len
        else:
            padding_needed = self.max_len - actual_length
            char_indices = char_indices + [self.pad_idx] * padding_needed
            effective_length = actual_length

        return {
            'input_ids': torch.tensor(char_indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long),
            'length': torch.tensor(effective_length, dtype=torch.long)  # ✅ 实际长度
        }

# --- NEW LSTM Model Class ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out

class SimpleGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_dim):
        super(SimpleGRU, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的

        self.gru = nn.GRU(embedding_dim, hidden_size, batch_first=True)
        #分类任务都这么写吧
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, input_ids, lengths):
        if lengths is None:
            # 自动计算
            lengths = (input_ids != 0).sum(dim=1)

        # batch_size = input_ids.size(0)
        max_len = input_ids.size(1)

        lengths = torch.clamp(lengths, max=max_len)

        embedded = self.embedding(input_ids)

        # 打包
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        packed_output, hidden = self.gru(packed_embedded)

        # 解包
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(
            packed_output,
            batch_first=True
        )

        # 获取最后时刻的隐藏状态
        if self.gru.bidirectional:
            hidden = hidden.view(self.gru.num_layers, 2, batch_size, self.gru.hidden_size)
            hidden_forward = hidden[-1, 0, :, :]
            hidden_backward = hidden[-1, 1, :, :]
            hidden_concat = torch.cat([hidden_forward, hidden_backward], dim=1)
        else:
            hidden_concat = hidden[-1, :, :]

        # 分类
        out = self.fc(hidden_concat)
        return out


# --- Training and Prediction ---
grn_dataset = CharGRNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(grn_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
# 应该是9
output_dim = len(label_to_index)
batch_size = 32

model = SimpleGRU(vocab_size, embedding_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = "cpu"

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, batch in enumerate(dataloader):
        # input_ids = batch['input_ids']
        # labels = batch['label']
        # lengths = batch['length']
        if isinstance(batch, dict):
            input_ids = batch.get('input_ids')
            labels = batch.get('label')
            lengths = batch.get('length')
        elif isinstance(batch, (list, tuple)):
            # 假设格式: (input_ids, labels, lengths)
            if len(batch) >= 3:
                input_ids, labels, lengths = batch[0], batch[1], batch[2]
            elif len(batch) == 2:
                input_ids, labels = batch[0], batch[1]
                lengths = None
            else:
                raise ValueError(f"无法处理的批次格式: {type(batch)}")

            # 2. 移动到设备
        input_ids = input_ids.to(device)
        labels = labels.to(device)
        if lengths is not None:
            lengths = lengths.to(device)
        optimizer.zero_grad()

        # 前向传播
        if lengths is not None:
            outputs = model(input_ids, lengths)
        else:
            # 尝试计算lengths
            lengths_calc = (input_ids != 0).sum(dim=1)
            outputs = model(input_ids, lengths_calc)

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
    len1 = min(len(text), max_len)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor, torch.tensor([len1]))

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
