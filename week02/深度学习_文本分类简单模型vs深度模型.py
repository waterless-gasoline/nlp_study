# 导入所需的库
import pandas as pd  # 用于数据处理
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
import torch.optim as op  # PyTorch优化器模块
from torch.utils.data import Dataset, DataLoader  # PyTorch数据集和数据加载器

# 从CSV文件加载数据集
# 使用制表符分隔，没有标题行
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 从数据集中提取文本和标签数据，并转换为列表格式
texts = dataset[0].tolist()
labels = dataset[1].tolist()

# 将每个分类标签转换为对应的数字下标
# 使用字典推导式，为每个唯一标签分配一个唯一的数字索引
label_to_index = {label: i for i, label in enumerate(set(labels))}

# 将原始标签列表转换为对应的数字索引列表
numerical_labels = [label_to_index[label] for label in labels]

# 初始化字符到索引的映射字典，添加特殊标记<pad>用于填充
char_to_index = {"<pad>": 0}

# 遍历所有文本中的字符，构建字符到索引的映射
# 为每个新出现的字符分配一个唯一的数字索引
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射字典
index_to_char = {i: char for char, i in char_to_index.items()}

# 计算词汇表大小，即字典中不同字符的总数
vocab_size = len(char_to_index)

max_len = 40  # 设置文本的最大长度

# 对文本进行分词和填充处理
tokenized_texts = []
for text in texts:
    tokenized_text = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized_text += [0] * (max_len - len(tokenized_text))
    tokenized_texts.append(tokenized_text)

# 将标签转换为PyTorch张量
label_tensor = torch.tensor(numerical_labels, dtype=torch.long)

# 设置文本的最大长度为40
max_len = 40


# 这是一个用于文本分类的数据集类，将字符转换为词袋(Bag of Words)表示
class CharToBowDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vcab_size):
        """
        初始化数据集类
        参数:
            texts: 文本列表
            labels: 标签列表
            char_to_index: 字符到索引的映射字典
            max_len: 文本的最大长度
            vocab_size: 词汇表大小
        """
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为tensor
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()  # 创建词袋向量

    def _create_bow_vectors(self):
        """
        创建词袋向量表示
        返回:
            一个包含所有文本词袋表示的tensor矩阵
        """
        # 初始化一个空列表，用于存储分词后的文本
        tokenized_txts = []
        # 遍历输入的文本列表
        for txt in texts:
            # 将文本中的每个字符转换为对应的索引，使用字典的get方法处理未知字符(默认为0)
            # 同时截取前max_len个字符
            tokenized = [char_to_index.get(c, 0) for c in txt[:max_len]]
            # 如果文本长度不足max_len，用0进行填充
            tokenized += [0] * (max_len - len(tokenized))
            # 将处理后的文本添加到结果列表中
            tokenized_txts.append(tokenized)

        bow_vectors = []

        # 遍历每个文本的索引列表
        for text_dices in tokenized_texts:
            # print(text_dices)
            # A. 初始化：创建一个全 0 向量，长度等于词典总数
            bow_vector = torch.zeros(vocab_size)
            # B. 计数循环：遍历当前这句话里的每一个字符索引
            for index in text_dices:
                # C. 过滤填充：如果 index 是 0（即 <pad>），我们不统计它
                if index != 0:
                    # D. 累加：在对应字符的位置上加 1
                    bow_vector[index] += 1
                    # print(bow_vector)
            # 将处理好的单句向量存入列表
            bow_vectors.append(bow_vector)
        # E. 拼接：将一堆向量垂直堆叠，形成一个大矩阵 (样本数, 词典大小)
        return torch.stack(bow_vectors)

    def __len__(self):

        """返回文本序列的长度

        Returns:
            int: texts列表中元素的数量
        """
        return len(self.texts)  # 返回内部texts列表的长度

    def __getitem__(self, item):
        """
        通过索引获取数据集中指定位置的词袋向量和标签
        参数:
            item: 索引值，用于指定要获取的数据位置
        返回:
            tuple: 包含两个元素的元组，第一个元素是词袋向量，第二个是对应的标签
        """
        return self.bow_vectors[item], self.labels[item]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):  # 层的个数 和 验证集精度
        """
        初始化神经网络分类器
        参数:
            input_dim (int): 输入特征的维度
            hidden_dim (int): 隐藏层的维度
            output_dim (int): 输出层的维度（分类数量）
        """
        # 层初始化
        super(SimpleClassifier, self).__init__()  # 调用父类nn.Module的初始化方法
        self.fc1 = nn.Linear(input_dim, hidden_dim)  # 第一个全连接层，将输入维度转换为隐藏层维度
        self.relu = nn.ReLU()  # ReLU激活函数，引入非线性
        self.fc2 = nn.Linear(hidden_dim, output_dim)  # 第二个全连接层，将隐藏层维度转换为输出维度

    def forward(self, x):
        """
        定义前向传播过程
        参数:
            x (torch.Tensor): 输入数据张量
        返回:
            torch.Tensor: 网络的输出结果
        """
        # 手动实现每层的计算
        out = self.fc1(x)  # 第一层：线性变换
        out = self.relu(out)  # 应用ReLU激活函数
        out = self.fc2(out)  # 第二层：线性变换得到最终输出
        return out  # 返回分类结果


# 模型 B: 深层、节点多 (Deep & Wide)
class DeepClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(DeepClassifier, self).__init__()
        # 第一层：输入 -> 隐藏层1 (节点多)
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        self.relu1 = nn.ReLU()

        # 第二层：隐藏层1 -> 隐藏层2 (增加深度)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.relu2 = nn.ReLU()

        # 第三层：隐藏层2 -> 输出
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)  # 激活
        out = self.fc2(out)
        out = self.relu2(out)  # 再次激活
        out = self.fc3(out)
        return out


# 创建一个字符转换为词袋(Bag of Words)的数据集实例
# 参数包括：文本数据、数字标签、字符到索引的映射、最大长度和词汇表大小
char_dataset = CharToBowDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 创建一个数据加载器(DataLoader)
# 参数包括：之前创建的数据集实例、批次大小(batch_size)设置为32、以及是否打乱数据(shuffle)设置为True
# 数据加载器用于批量处理数据，通常用于训练神经网络
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)


# ---------------------------------------------------------
# 2. 封装训练函数 (为了复用训练逻辑)
# ---------------------------------------------------------
def train_model(model, dataloader, num_epoch=10, learning_rate=0.01, name="Model"):
    """
    训练模型的函数

    参数:
        model: 要训练的神经网络模型
        dataloader: 数据加载器，用于批量提供训练数据
        num_epoch: 训练的轮数(默认为10)
        learning_rate: 学习率(默认为0.01)
        name: 模型的名称(默认为"Model")

    返回:
        loss_history: 记录每个epoch的平均损失的列表
    """
    criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数
    optimizer = op.SGD(model.parameters(), lr=learning_rate)  # 定义随机梯度下降优化器

    loss_history = []  # 记录每个Epoch的loss

    print(f"--- 开始训练 {name} ---")
    # 开始训练循环，遍历所有epoch
    for epoch in range(num_epoch):
        # 将模型设置为训练模式
        model.train()
        # 初始化运行损失为0
        running_loss = 0.0

        # 遍历数据加载器中的每个批次
        for inputs, labels in dataloader:
            # 清空梯度
            optimizer.zero_grad()
            # 前向传播，得到模型输出
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播，计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
            # 累加当前批次的损失
            running_loss += loss.item()

        # 计算平均Loss
        avg_loss = running_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[{name}] Epoch {epoch + 1}/{num_epoch}, Loss: {avg_loss:.4f}")

    return loss_history


# 参数设置
input_dim = vocab_size
output_dim = len(label_to_index)

# === 实验 1: 浅层小模型 ===
# 隐藏层只有 64 个节点
model_small = SimpleClassifier(input_dim, 64, output_dim)
loss_small = train_model(model_small, dataloader, num_epoch=3, name="Small Model")

# === 实验 2: 深层大模型 ===
# 隐藏层1: 256节点, 隐藏层2: 128节点
model_deep = DeepClassifier(input_dim, 256, 128, output_dim)
loss_deep = train_model(model_deep, dataloader, num_epoch=20, name="Deep Model")


def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    对输入的文本进行分类预测
    参数:
        text (str): 需要分类的文本
        model: 训练好的文本分类模型
        char_to_index (dict): 字符到索引的映射字典
        vocab_size (int): 词汇表大小
        max_len (int): 文本的最大长度
        index_to_label (dict): 索引到标签的映射字典
    返回:
        str: 预测的文本标签
    """
    # 将文本转换为数字序列，截取到最大长度
    # 未知字符用0表示
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 如果序列长度不足max_len，用0填充
    tokenized += [0] * (max_len - len(tokenized))

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    # 遍历tokenized序列，统计词频
    for index in tokenized:
        if index != 0:  # 忽略填充的0
            bow_vector[index] += 1

    # 添加batch维度
    bow_vector = bow_vector.unsqueeze(0)

    # 将模型设置为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 获取模型预测结果
        output = model(bow_vector)

    # 获取预测结果中概率最大的类别索引
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    # 将索引转换为对应的标签
    predicted_label = index_to_label[predicted_index]

    return predicted_label


index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model_small, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 简单模型预测为: '{predicted_class}'")

new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model_deep, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 复杂模型预测为: '{predicted_class}'")

new_text = "播放薛之谦的歌曲"
predicted_class = classify_text(new_text, model_small, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 简单模型预测为: '{predicted_class}'")

new_text = "播放薛之谦的歌曲"
predicted_class = classify_text(new_text, model_deep, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 复杂模型预测为: '{predicted_class}'")
