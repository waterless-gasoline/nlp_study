import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 数据加载与预处理
dataset = pd.read_csv("./data/dataset.csv", sep="\t", header=None)
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

# 数据集类
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

# 动态构建不同层数的分类器
class DynamicClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        动态分类器：支持多层隐藏层
        :param input_dim: 输入维度（词汇表大小）
        :param hidden_dims: 隐藏层维度列表，如[64]（1层）、[128, 64]（2层）、[256, 128, 64]（3层）
        :param output_dim: 输出维度（类别数）
        """
        super(DynamicClassifier, self).__init__()
        layers = []
        prev_dim = input_dim
        # 构建隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

# 训练函数
def train_model(input_dim, hidden_dims, output_dim, dataloader, num_epochs=10, lr=0.01):
    """
    训练模型并返回每轮的Loss
    :param input_dim: 输入维度
    :param hidden_dims: 隐藏层维度列表
    :param output_dim: 输出维度
    :param dataloader: 数据加载器
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :return: 每轮epoch的Loss列表
    """
    # 初始化模型、损失函数、优化器
    model = DynamicClassifier(input_dim, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    epoch_losses = []  # 记录每轮的平均Loss
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
        
        # 计算本轮平均Loss
        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"模型结构 {hidden_dims} | Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return epoch_losses

# 初始化数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义待对比的模型配置（层数+节点数）
model_configs = {
    "1层-64节点": [64],
    "1层-128节点": [128],
    "1层-256节点": [256],
    "2层-128+64节点": [128, 64],
    "2层-256+128节点": [256, 128],
    "3层-256+128+64节点": [256, 128, 64]
}

# 训练所有配置的模型，记录Loss
all_losses = {}
output_dim = len(label_to_index)
num_epochs = 10
for config_name, hidden_dims in model_configs.items():
    print(f"\n========== 开始训练 {config_name} ==========")
    all_losses[config_name] = train_model(vocab_size, hidden_dims, output_dim, dataloader, num_epochs)

# 可视化Loss变化对比
plt.figure(figsize=(12, 8))
for config_name, losses in all_losses.items():
    plt.plot(range(1, num_epochs+1), losses, label=config_name, marker='o')

plt.xlabel("Epoch（训练轮数）")
plt.ylabel("Average Loss（平均损失）")
plt.title("不同模型层数/节点数的Loss变化对比")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("model_loss_comparison.png")
plt.show()

# 测试示例（选最优模型，这里选3层-256+128+64节点）
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

# 加载最优模型测试
index_to_label = {i: label for label, i in label_to_index.items()}
best_model = DynamicClassifier(vocab_size, [256, 128, 64], output_dim)
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, best_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
