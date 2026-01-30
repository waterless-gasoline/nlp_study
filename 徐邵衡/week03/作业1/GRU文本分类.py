"""
这是一个基于GRU（门控循环单元）的文本分类器实现。
使用字符级别的嵌入，能够处理中文文本分类任务。
"""
import pandas as pd  # 导入pandas库，用于数据处理
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器

# 加载数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)  # 读取数据集，制表符分隔，无表头
texts = dataset[0].tolist()  # 将文本列转换为列表
string_labels = dataset[1].tolist()  # 将标签列转换为列表

# 将标签转换为数值索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}  # 创建标签到索引的映射字典
numerical_labels = [label_to_index[label] for label in string_labels]  # 将字符串标签转换为数值标签

# 构建字符词典
char_to_index = {'<pad>': 0}  # 初始化字符到索引的映射字典，添加填充标记
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)  # 为每个新字符分配唯一索引

index_to_char = {i: char for char, i in char_to_index.items()}  # 创建反向映射：索引到字符
vocab_size = len(char_to_index)  # 计算词汇表大小

max_len = 40  # 设置最大文本长度


# 自定义数据集类
class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        """
        初始化数据集
        Args:
            texts: 文本列表
            labels: 标签列表
            char_to_index: 字符到索引的映射
            max_len: 最大文本长度
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        """
        获取单个数据样本
        Args:
            idx: 索引
        Returns:
            处理后的文本张量和标签张量
        """
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]  # 将字符转换为索引
        indices += [0] * (self.max_len - len(indices))  # 填充到固定长度
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# --- NEW GRU Model Class ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.GRU = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        # 全连接层定义，将隐藏层维度映射到输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 将输入词通过嵌入层转换为词向量
        embedded = self.embedding(x)
        # 将词向量输入GRU层，得到输出和隐藏状态
        gru_out, hidden_state = self.GRU(embedded)
        # 将隐藏状态通过全连接层进行分类
        out = self.fc(hidden_state.squeeze(0))
        return out


# --- Training and Prediction ---
# 创建GRU数据集和数据加载器
GRU_dataset = CharGRUDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(GRU_dataset, batch_size=32, shuffle=True)

# 定义模型参数
embedding_dim = 64  # 嵌入维度
hidden_dim = 128  # 隐藏层维度
output_dim = len(label_to_index)  # 输出维度，等于标签数量

# 初始化模型、损失函数和优化器
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()  # 使用交叉熵损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 训练参数
num_epochs = 4
# 开始训练循环
for epoch in range(num_epochs):
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 初始化运行损失
    # 遍历数据加载器
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # 清零梯度
        outputs = model(inputs)  # 前向传播
        loss = criterion(outputs, labels)  # 计算损失
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
        running_loss += loss.item()  # 累加损失
        # 每50个批次打印一次当前损失
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 定义文本分类函数
def classify_text_GRU(text, model, char_to_index, max_len, index_to_label):
    """
    使用GRU模型对输入文本进行分类
    参数:
        text: 输入文本
        model: 训练好的GRU模型
        char_to_index: 字符到索引的映射字典
        max_len: 最大文本长度
        index_to_label: 索引到标签的映射字典
    返回:
        predicted_label: 预测的标签
    """
    # 将文本转换为索引序列，并填充到固定长度
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 设置模型为评估模式，并关闭梯度计算
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    # 获取预测结果
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建索引到标签的映射字典
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试新文本的预测
new_text = "帮我导航到北京"
predicted_class = classify_text_GRU(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_GRU(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
