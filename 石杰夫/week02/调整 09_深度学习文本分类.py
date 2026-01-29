import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../../第1周：课程介绍与大模型基础/Week01/dataset.csv", sep="\t", header=None)
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


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

output_dim = len(label_to_index)
num_epochs = 10

# 配置1 - 原始模型 (2层: vocab_size -> 128 -> output_dim)
print("=" * 80)
print("配置1: 原始模型 (2层, hidden_dim=128)")
print("=" * 80)

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): 
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

hidden_dim = 128  # 修改位置：隐藏层节点数
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) 
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
optimizer = optim.SGD(model.parameters(), lr=0.01)

config1_loss_history = []
# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

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

    avg_loss = running_loss / len(dataloader)
    config1_loss_history.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print(f"配置1 最终Loss: {config1_loss_history[-1]:.4f}\n")


# 配置2 - 增加隐藏节点 (2层: vocab_size -> 512 -> output_dim)
print("=" * 80)
print("配置2: 大隐藏层 (2层, hidden_dim=512)")
print("=" * 80)

class SimpleClassifier2(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): 
        # 层初始化
        super(SimpleClassifier2, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 修改位置：隐藏层节点数改为256
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

hidden_dim_2 = 512  # 修改位置：隐藏层节点数从128改为512，使变化更明显
model2 = SimpleClassifier2(vocab_size, hidden_dim_2, output_dim) 
criterion = nn.CrossEntropyLoss() 
optimizer2 = optim.SGD(model2.parameters(), lr=0.01)

config2_loss_history = []

for epoch in range(num_epochs): 
    model2.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer2.zero_grad()
        outputs = model2(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer2.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    avg_loss = running_loss / len(dataloader)
    config2_loss_history.append(avg_loss)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

print(f"配置2 最终Loss: {config2_loss_history[-1]:.4f}\n")


# 对比结果 
print("=" * 80)
print("实验结果对比")
print("=" * 80)
print(f"{'配置':<35} {'最终Loss':<15} {'Loss降幅':<15}")
print("-" * 80)
print(f"{'配置1: 原始模型 (hidden=128)':<35} {config1_loss_history[-1]:<15.4f} {config1_loss_history[0] - config1_loss_history[-1]:<15.4f}")
print(f"{'配置2: 大隐藏层 (hidden=512)':<35} {config2_loss_history[-1]:<15.4f} {config2_loss_history[0] - config2_loss_history[-1]:<15.4f}")
print("=" * 80)


# 使用配置1模型进行预测
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
