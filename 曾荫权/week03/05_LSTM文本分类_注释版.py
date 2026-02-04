# =============== 1. 导入必要的库 ===============
import pandas as pd  # 用于数据处理和CSV文件读取
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化算法
from torch.utils.data import Dataset, DataLoader  # 自定义数据集和数据加载器

# =============== 2. 数据加载和预处理 ===============
# 读取数据集，使用制表符分隔，无列名
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# 提取文本内容（第0列）和标签（第1列）
texts = dataset[0].tolist()  # 将文本列转换为Python列表
string_labels = dataset[1].tolist()  # 将标签列转换为Python列表

# =============== 3. 标签映射处理 ===============
# 创建标签到索引的映射字典
# 例如: {"Radio-Listen": 0, "Weather-Query": 1, ...}
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 将字符串标签转换为数值索引
numerical_labels = [label_to_index[label] for label in string_labels]

# =============== 4. 字符映射处理 ===============
# 创建字符到索引的映射字典，<pad>作为填充符，索引为0
char_to_index = {'<pad>': 0}
# 遍历所有文本，为每个新字符分配唯一索引
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)  # 新字符的索引为当前字典大小

# 创建索引到字符的反向映射（用于调试）
index_to_char = {i: char for char, i in char_to_index.items()}
# 词汇表大小（字符种类总数）
vocab_size = len(char_to_index)
# 最大输入文本长度（用于统一输入尺寸）
max_len = 40


# =============== 5. 自定义数据集类 ===============
class CharLSTMDataset(Dataset):
    """字符级LSTM数据集类，处理文本到索引序列的转换"""

    def __init__(self, texts, labels, char_to_index, max_len):
        """
        初始化数据集
        :param texts: 原始文本列表
        :param labels: 对应标签列表
        :param char_to_index: 字符到索引的映射
        :param max_len: 最大文本长度
        """
        self.texts = texts  # 文本输入
        # 将标签转换为PyTorch长整型张量
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index  # 字符到索引的映射关系
        self.max_len = max_len  # 文本最大输入长度

    def __len__(self):
        """返回数据集样本个数"""
        return len(self.texts)

    def __getitem__(self, idx):
        """
        获取单个样本
        :param idx: 样本索引
        :return: 处理后的索引序列和对应标签
        """
        text = self.texts[idx]  # 获取指定索引的文本

        # 将文本转换为索引序列，并截断到max_len
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 填充不足max_len的部分
        indices += [0] * (self.max_len - len(indices))

        # 返回索引序列张量和对应标签
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]


# =============== 6. LSTM模型定义 ===============
class LSTMClassifier(nn.Module):
    """基于LSTM的文本分类模型"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        """
        初始化网络层
        :param vocab_size: 词汇表大小
        :param embedding_dim: 词嵌入维度
        :param hidden_dim: LSTM隐藏层维度
        :param output_dim: 输出维度（类别数量）
        """
        super(LSTMClassifier, self).__init__()
        # 词嵌入层：将字符索引转换为稠密向量
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # LSTM层：处理序列数据
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        # 全连接层：将LSTM输出映射到类别空间
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量，形状为[batch_size, seq_len]
        :return: 输出张量，形状为[batch_size, output_dim]
        """
        # 词嵌入：[batch_size, seq_len] -> [batch_size, seq_len, embedding_dim]
        embedded = self.embedding(x)

        # LSTM处理：[batch_size, seq_len, embedding_dim] -> [batch_size, seq_len, hidden_dim]
        # lstm_out: 所有时间步的输出
        # hidden_state: 最后一个时间步的隐藏状态
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # 使用最后一个时间步的隐藏状态进行分类
        # [1, batch_size, hidden_dim] -> [batch_size, hidden_dim]
        out = self.fc(hidden_state.squeeze(0))
        return out


# =============== 7. 准备数据加载器 ===============
# 创建字符级LSTM数据集实例
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
# 创建数据加载器，批量大小32，打乱数据顺序
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

# =============== 8. 模型参数配置和初始化 ===============
# 词嵌入维度
embedding_dim = 64
# LSTM隐藏层维度
hidden_dim = 128
# 输出维度（类别数量）
output_dim = len(label_to_index)
# 初始化LSTM分类模型
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

# =============== 9. 训练配置 ===============
# 损失函数：交叉熵损失（包含Softmax）
criterion = nn.CrossEntropyLoss()
# 优化器：Adam
optimizer = optim.Adam(model.parameters(), lr=0.001)
# 训练轮数
num_epochs = 4

# =============== 10. 模型训练 ===============
for epoch in range(num_epochs):
    # 设置模型为训练模式（启用dropout等）
    model.train()
    running_loss = 0.0

    # 遍历数据加载器中的每个batch
    for idx, (inputs, labels) in enumerate(dataloader):
        # 梯度清零
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        # 参数更新
        optimizer.step()

        # 累计损失
        running_loss += loss.item()

        # 每50个batch打印一次损失
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# =============== 11. 文本分类函数 ===============
def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    """
    对新文本进行LSTM分类
    :param text: 输入文本
    :param model: 训练好的模型
    :param char_to_index: 字符到索引映射
    :param max_len: 最大长度
    :param index_to_label: 索引到标签映射
    :return: 预测的标签
    """
    # 1. 文本预处理：字符索引化
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充到max_len
    indices += [0] * (max_len - len(indices))

    # 2. 转换为张量并增加batch维度
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 3. 模型预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(input_tensor)
        # 获取最大概率对应的索引
        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()
        # 将索引转换回标签
        predicted_label = index_to_label[predicted_index]

    return predicted_label


# =============== 12. 创建索引到标签的映射 ===============
# 用于将模型输出的索引转换回原始标签
index_to_label = {i: label for label, i in label_to_index.items()}

# =============== 13. 测试模型 ===============
# 测试样本1
new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# 测试样本2
new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")