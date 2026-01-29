"""
对比不同模型结构对文本分类任务的影响
通过调整模型的层数和节点个数，观察loss变化情况
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import matplotlib
# 设置matplotlib支持中文字体显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 指定默认字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
import numpy as np
np.random.seed(42)

# ... (Data loading and preprocessing remains the same) ...
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


# 定义不同复杂度的分类器模型
class SimpleClassifier(nn.Module):
    """简单模型：1个隐藏层"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class MediumClassifier(nn.Module):
    """中等复杂度模型：2个隐藏层"""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MediumClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


class ComplexClassifier(nn.Module):
    """复杂模型：3个隐藏层"""
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(ComplexClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2])
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(hidden_dims[2], output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out


def train_model(model, dataloader, num_epochs=10, learning_rate=0.01):
    """训练模型并返回损失历史"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    model.train()
    loss_history = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    return loss_history


# 创建数据集和数据加载器
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义不同模型配置
output_dim = len(label_to_index)

# 模型1: 简单模型 - 1个隐藏层，32个节点
model1 = SimpleClassifier(vocab_size, 32, output_dim)
print("\n模型1: 简单模型 (1个隐藏层，32个节点)")
print(model1)

# 模型2: 简单模型 - 1个隐藏层，128个节点
model2 = SimpleClassifier(vocab_size, 128, output_dim)
print("\n模型2: 简单模型 (1个隐藏层，128个节点)")
print(model2)

# 模型3: 中等复杂度模型 - 2个隐藏层，[64, 32]个节点
model3 = MediumClassifier(vocab_size, [64, 32], output_dim)
print("\n模型3: 中等复杂度模型 (2个隐藏层，[64, 32]个节点)")
print(model3)

# 模型4: 复杂模型 - 3个隐藏层，[128, 64, 32]个节点
model4 = ComplexClassifier(vocab_size, [128, 64, 32], output_dim)
print("\n模型4: 复杂模型 (3个隐藏层，[128, 64, 32]个节点)")
print(model4)

# 训练所有模型并记录损失历史
models = [model1, model2, model3, model4]
model_names = ["简单模型(32)", "简单模型(128)", "中等模型(64,32)", "复杂模型(128,64,32)"]
loss_histories = []

print("\n" + "="*50)
print("开始训练模型1...")
loss1 = train_model(model1, dataloader, num_epochs=15, learning_rate=0.01)
loss_histories.append(loss1)

print("\n" + "="*50)
print("开始训练模型2...")
loss2 = train_model(model2, dataloader, num_epochs=15, learning_rate=0.01)
loss_histories.append(loss2)

print("\n" + "="*50)
print("开始训练模型3...")
loss3 = train_model(model3, dataloader, num_epochs=15, learning_rate=0.01)
loss_histories.append(loss3)

print("\n" + "="*50)
print("开始训练模型4...")
loss4 = train_model(model4, dataloader, num_epochs=15, learning_rate=0.01)
loss_histories.append(loss4)

# 绘制损失比较图
plt.figure(figsize=(14, 10))

# 损失比较图
plt.subplot(2, 1, 1)
for i, (loss_history, name) in enumerate(zip(loss_histories, model_names)):
    epochs = range(1, len(loss_history) + 1)
    plt.plot(epochs, loss_history, label=f'{name}', marker='o', markersize=4)

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同模型结构的训练损失比较')
plt.legend()
plt.grid(True, alpha=0.3)

# 最终损失柱状图
plt.subplot(2, 1, 2)
final_losses = [history[-1] for history in loss_histories]
bars = plt.bar(model_names, final_losses, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
plt.ylabel('Final Loss')
plt.title('不同模型的最终损失比较')
plt.xticks(rotation=45)

# 在柱状图上添加数值标签
for bar, loss in zip(bars, final_losses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{loss:.3f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 打印总结
print("\n" + "="*50)
print("模型性能总结:")
for i, (name, final_loss) in enumerate(zip(model_names, final_losses)):
    print(f"{name}: 最终损失 = {final_loss:.4f}")

# 根据损失值找出最佳模型
best_model_idx = final_losses.index(min(final_losses))
print(f"\n最佳模型: {model_names[best_model_idx]} (最终损失: {min(final_losses):.4f})")