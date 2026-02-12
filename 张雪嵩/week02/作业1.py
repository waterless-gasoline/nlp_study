import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===================== 1. 数据加载与预处理（保持不变） =====================
dataset = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
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


# ===================== 2. 自定义数据集（保持不变） =====================
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


# ===================== 3. 可配置的模型类（支持自定义层数/节点数） =====================
class ConfigurableClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        可配置层数和节点数的分类器
        :param input_dim: 输入维度（词典大小）
        :param hidden_dims: 列表，每个元素为对应隐藏层的节点数（如[512, 1024]表示2层隐藏层，节点数分别为512、1024）
        :param output_dim: 输出维度（标签类别数）
        """
        super(ConfigurableClassifier, self).__init__()
        self.layers = nn.ModuleList()
        # 输入层 -> 第一层隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        # 隐藏层之间的连接
        for i in range(1, len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            self.layers.append(nn.ReLU())
        # 最后一层隐藏层 -> 输出层
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# ===================== 4. 训练函数（复用训练逻辑） =====================
def train_model(model, dataloader, num_epochs=10, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.Adam(model.parameters(), lr=lr)
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

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"模型配置 {model_config} | Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses


# ===================== 5. 实验配置与执行 =====================
# 初始化数据集（所有实验共用同一数据集）
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# 定义对比实验（key=模型配置描述，value=隐藏层节点数列表）
experiments = {
    "2层-512节点": [512],  # 输入层→512→输出层（共2层线性层）
    "2层-1024节点": [1024],  # 原代码等效配置（输入层→1024→2048→输出层是3层，这里先对齐基础对比）
    "3层-512-1024节点": [512, 1024],  # 输入层→512→1024→输出层
    "3层-1024-2048节点": [1024, 2048],  # 原代码配置（输入层→1024→2048→输出层）
    "4层-256-512-1024节点": [256, 512, 1024]  # 更多层数+递减节点
}

# 存储所有实验的Loss结果
all_losses = {}

# 执行每一组实验
for model_config, hidden_dims in experiments.items():
    print(f"\n========== 开始训练：{model_config} ==========")
    # 初始化模型
    model = ConfigurableClassifier(
        input_dim=vocab_size,
        hidden_dims=hidden_dims,
        output_dim=len(label_to_index)
    )
    # 训练并记录Loss
    epoch_losses = train_model(model, dataloader, num_epochs=10, lr=0.01)
    all_losses[model_config] = epoch_losses

# ===================== 6. Loss可视化对比 =====================
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 解决中文显示
plt.figure(figsize=(10, 6))
for config, losses in all_losses.items():
    plt.plot(range(1, 11), losses, label=config, marker='o')

plt.xlabel("训练轮数（Epoch）")
plt.ylabel("平均Loss")
plt.title("不同模型层数/节点数的Loss变化对比")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("model_loss_comparison.png")
plt.show()


# ===================== 7. 推理函数（保持不变，可选验证） =====================
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


# 可选：用最后一个训练的模型做推理示例
index_to_label = {i: label for label, i in label_to_index.items()}
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"\n输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")