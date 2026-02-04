import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("../Week01/data/dataset.csv", sep="\t", header=None)
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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim): # 层的个数 和 验证集精度
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

#防止过拟合的场景，增加dorpout 随机丢弃神经元
class BetterClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.3):
        super(BetterClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)  # 添加Dropout
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)  # 随机丢弃部分神经元
        out = self.fc2(out)
        return out

char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据
# 一、隐藏层节点数的影响
# 1.如果减少节点数：
# hidden_dim = 64  # 欠拟合风险
# 如果增加节点数：
# hidden_dim = 256  # 更好的拟合能力，但可能过拟合
# 节点数太少（如32-64）：
# loss下降缓慢
# 可能无法充分学习复杂模式
# 训练和验证loss都较高
# 2.节点数适中（如128-256）：
# loss能够较快下降
# 有足够的表达能力
# 当前128节点是较好的平衡点
# 3.节点数太多（如512+）：
# loss下降很快
# 容易过拟合（训练loss低，但泛化差）
# 计算资源消耗大

hidden_dim = 128
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
# model = BetterClassifier(vocab_size, 128, output_dim, dropout_rate=0.3)
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# optimizer = optim.SGD(model.parameters(), lr=0.01)    #随机梯度下降 需要手动调整学习率 容易陷入局部最优 收敛慢
optimizer = optim.Adam(model.parameters(), lr=0.001)    #使用Adam优化器 自适应学习率 收敛速度较SGD快！
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 10     #10->20
for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
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

    scheduler.step()  # 每个epoch后调整学习率
    # 打印当前学习率
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}, LR: {current_lr:.6f}")



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
