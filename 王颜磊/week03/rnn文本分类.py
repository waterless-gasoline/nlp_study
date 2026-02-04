# 导入必要的库
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# 第一步：加载原始数据

dataset = pd.read_csv("E:/BaiduNetdiskDownload/八斗学院\第1周：课程介绍与大模型基础/Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()


# 第二步：标签数字化

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]


# 第三步：构建字符词汇表

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
max_len = 40


# 第四步：自定义 Dataset

class CharRNNNDataset(Dataset):
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


# 第五步：定义 RNN 分类模型

class RNNClassifier(nn.Module):
    """
    使用 Simple RNN（非 LSTM/GRU）实现文本分类
    注意：RNN 只返回 output 和 final hidden state，没有 cell state
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 使用 nn.RNN 替代 LSTM
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, emb_dim)
        # RNN 返回: (output, h_n)
        # h_n 形状: (num_layers=1, batch, hidden_dim)
        rnn_out, hidden_state = self.rnn(embedded)
        # 取最后一层的隐藏状态（这里只有一层）
        out = self.fc(hidden_state.squeeze(0))  # (batch, output_dim)
        return out


# 第六步：准备数据加载器

rnn_dataset = CharRNNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(rnn_dataset, batch_size=32, shuffle=True)


# 第七步：初始化模型、损失函数、优化器

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


# 第八步：训练模型

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 第九步：预测函数（完全通用，不依赖模型类型）

def classify_text_rnn(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_label = index_to_label[predicted_index.item()]
    return predicted_label

index_to_label = {i: label for label, i in label_to_index.items()}


# 第十步：测试预测

new_text = "帮我导航到北京"
predicted_class = classify_text_rnn(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_rnn(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
