import random

import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# 修改1：固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    # 如果用GPU需要使用torch.cuda.manual_seed_all(seed)


dataset = pd.read_csv("./Week02/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {"<pad>": 0}
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
            tokenized = [
                self.char_to_index.get(char, 0) for char in text[: self.max_len]
            ]
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


# class SimpleClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SimpleClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # 手动实现每层的计算
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out


# 修改模型类使其支持可变层数（hidden_dims列表）
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    # *：解包符：将壳迭代的对象展开为单个元素（列表、元组、集合、字符串等）
    # nn.Sequential接收的是多个参数（每个参数使一个神经网络）
    # 假设 layers = [nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128)]
    # nn.Sequential(*layers) 等同于 nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128))
    def forward(self, x):
        return self.network(x)


char_dataset = CharBoWDataset(
    texts, numerical_labels, char_to_index, max_len, vocab_size
)  # 读取单个样本


# dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# hidden_dim = 128
# output_dim = len(label_to_index)
# model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
# criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

# num_epochs = 10
# for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
#     model.train()
#     running_loss = 0.0
#     for idx, (inputs, labels) in enumerate(dataloader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if idx % 50 == 0:
#             print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 修改2：把“训练循环”抽象为函数
def train_one_model(model, num_epochs, dataloader, criterion, optimizer):
    epoch_losses = []
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

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}],Loss:{avg_loss:.4f}")
    print("*" * 10)
    return epoch_losses


# # 首先只做宽度（节点数）对比
results_width = {}
num_epochs_width = 10
output_dim = len(label_to_index)

hidden_dim_list = [16, 32, 48, 64]

for hidden_dim in hidden_dim_list:
    set_seed(42)

    g = torch.Generator()  # 管理伪随机数生成状态的类
    g.manual_seed(42)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True, generator=g)

    model = SimpleClassifier(vocab_size, [hidden_dim], output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = train_one_model(model, num_epochs_width, dataloader, criterion, optimizer)
    train_name = f"1layer_h{hidden_dim}"
    results_width[train_name] = (
        losses  # results存着每一个hidden_dim的loss曲线（每个epoch一个值）
    )

# 然后在做“深度对比”：固定结点数，改层数（hidden_dims）
result_depth = {}
num_epochs_depth = 30
hidden_dims_list = [[128], [128, 128], [128, 128, 128]]

for hidden_dims in hidden_dims_list:
    set_seed(42)

    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True, generator=g)

    model = SimpleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = train_one_model(model, num_epochs_depth, dataloader, criterion, optimizer)
    train_name = f"depth{len(hidden_dims)}_{hidden_dims}"
    result_depth[train_name] = (
        losses  # results存执每一个hidden_dim的loss曲线（每个epoch一个值）
    )


# 结果可视化函数
def plt_loss_curves(result_dict, num_epochs, title):
    plt.figure()
    epochs = range(num_epochs)

    for name, losses in result_dict.items():
        plt.plot(epochs, losses, marker="o", linewidth=1.8, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.show()


plt_loss_curves(
    results_width, num_epochs_width, "Loss vs Epoch(Width comparison:1 hidden layer)"
)
plt_loss_curves(
    result_depth, num_epochs_depth, "Loss vs Epoch(Depth comparision:hidden_dim = 128)"
)
