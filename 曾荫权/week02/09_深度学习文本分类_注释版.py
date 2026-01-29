# =============== 1. 导入必要的库 ===============
import pandas as pd  # 用于数据处理和CSV文件读取
import torch  # PyTorch深度学习框架
import torch.nn as nn  # 神经网络模块
import torch.optim as optim  # 优化算法
from torch.utils.data import Dataset, DataLoader  # 数据集和数据加载器

# 数据加载和预处理部分保持不变，但需要正确实现
# 由于原始代码中这部分被注释掉了，我们在此补充完整实现

# =============== 2. 数据加载和预处理 ===============
# 读取数据集，使用制表符分隔，无列名
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# 提取文本内容（第0列）和标签（第1列）
texts = dataset[0].tolist()  # 将文本列转换为Python列表
string_labels = dataset[1].tolist()  # 将标签列转换为Python列表

# =============== 3. 标签映射处理 ===============
# 创建标签到索引的映射字典
# 例如: {"Radio-Listen": 0, "FilmTele-Play": 1, ...}
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

# 创建索引到字符的反向映射（用于调试和可视化）
index_to_char = {i: char for char, i in char_to_index.items()}
# 词汇表大小（字符种类总数）
vocab_size = len(char_to_index)
# 最大文本长度（用于统一输入尺寸）
max_len = 40


# =============== 5. 自定义数据集类 ===============
class CharBoWDataset(Dataset):
    """字符级别的词袋(Bag of Words)数据集类"""

    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        """
        初始化数据集
        :param texts: 原始文本列表
        :param labels: 对应标签列表
        :param char_to_index: 字符到索引的映射
        :param max_len: 最大文本长度
        :param vocab_size: 词汇表大小
        """
        self.texts = texts
        # 将标签转换为PyTorch张量
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        # 预先创建词袋向量
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        创建词袋(Bag of Words)向量
        :return: 词袋向量张量
        """
        # 1. 将文本转换为索引序列
        tokenized_texts = []
        for text in self.texts:
            # 将文本截断或填充到max_len长度
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 填充不足max_len的部分
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        # 2. 转换为词袋表示
        bow_vectors = []
        for text_indices in tokenized_texts:
            # 初始化全零向量
            bow_vector = torch.zeros(self.vocab_size)
            # 统计每个字符出现次数
            for index in text_indices:
                if index != 0:  # 忽略填充符
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)

        # 转换为PyTorch张量
        return torch.stack(bow_vectors)

    def __len__(self):
        """返回数据集大小"""
        return len(self.texts)

    def __getitem__(self, idx):
        """获取指定索引的数据样本和标签"""
        return self.bow_vectors[idx], self.labels[idx]


# =============== 6. 创建数据集实例 ===============
# 实例化字符词袋数据集
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 创建数据加载器，批量大小32，打乱数据顺序
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)


# =============== 7. 定义神经网络模型 ===============
class SimpleClassifier(nn.Module):
    """简单的全连接神经网络分类器"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化网络层
        :param input_dim: 输入维度（词汇表大小）
        :param hidden_dim: 隐藏层维度
        :param output_dim: 输出维度（类别数量）
        """
        super(SimpleClassifier, self).__init__()
        # 第一个全连接层：输入 -> 隐藏层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 第二个全连接层：隐藏层 -> 输出
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        # 第三层
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        # 第四层
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.relu4 = nn.ReLU()
        # 第五层
        self.fc5 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        前向传播
        :param x: 输入张量
        :return: 输出张量
        """
        # 通过第一层全连接
        out = self.fc1(x)
        # 应用ReLU激活
        out = self.relu(out)
        # 通过第二层全连接
        out = self.fc2(out)
        out = self.relu2(out)
        # 第三层
        out = self.fc3(out)
        out = self.relu3(out)
        # 第四层
        out = self.fc4(out)
        out = self.relu4(out)
        # 第五层
        out = self.fc5(out)

        return out


# =============== 8. 初始化模型和训练参数 ===============
# 隐藏层维度
hidden_dim = 256
# 输出维度（类别数量）
output_dim = len(label_to_index)
# 初始化模型：输入维度为词汇表大小
model = SimpleClassifier(vocab_size, hidden_dim, output_dim)

# 损失函数：交叉熵损失（包含Softmax）
criterion = nn.CrossEntropyLoss()
# 优化器：随机梯度下降
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练轮数
num_epochs = 20

# =============== 9. 模型训练 ===============
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


# =============== 10. 文本分类函数 ===============
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    对新文本进行分类
    :param text: 输入文本
    :param model: 训练好的模型
    :param char_to_index: 字符到索引映射
    :param vocab_size: 词汇表大小
    :param max_len: 最大长度
    :param index_to_label: 索引到标签映射
    :return: 预测的标签
    """
    # 1. 文本预处理：字符索引化
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 填充到max_len
    tokenized += [0] * (max_len - len(tokenized))

    # 2. 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    # 3. 调整维度以适应模型输入 [batch_size, features]
    bow_vector = bow_vector.unsqueeze(0)

    # 4. 模型预测
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        output = model(bow_vector)
        # 获取最大概率对应的索引
        _, predicted_index = torch.max(output, 1)
        predicted_index = predicted_index.item()
        # 将索引转换回标签
        predicted_label = index_to_label[predicted_index]

    return predicted_label


# =============== 11. 创建索引到标签的映射 ===============
# 用于将模型输出的索引转换回原始标签
index_to_label = {i: label for label, i in label_to_index.items()}

# =============== 12. 测试模型 ===============
# 测试样本1
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# 测试样本2
new_text_2 = "西藏在天气好的时候更容易看清路"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
