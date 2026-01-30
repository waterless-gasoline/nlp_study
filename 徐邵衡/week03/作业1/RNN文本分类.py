import pandas as pd  # 导入pandas库，用于数据处理
import torch  # 导入PyTorch库，用于深度学习
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
from torch.utils.data import Dataset, DataLoader  # 导入PyTorch的数据集和数据加载器

# 从CSV文件加载数据集，使用制表符分隔，没有列名
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# 将第一列（文本数据）转换为列表
texts = dataset[0].tolist()
# 将第二列（标签数据）转换为列表
string_labels = dataset[1].tolist()

# 使用 sorted 确保每次运行顺序一致
sorted_labels = sorted(list(set(string_labels)))
label_to_index = {label: i for i, label in enumerate(sorted_labels)}
# 将字符串标签转换为对应的数字索引
numerical_labels = [label_to_index[label] for label in string_labels]

# 创建字符到索引的映射字典，包含一个特殊的填充字符'<pad>'
char_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符到索引的映射
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射
index_to_char = {i: char for char, i in char_to_index.items()}
# 获取词汇表大小
vocab_size = len(char_to_index)

# 设置最大文本长度
max_len = 40


# 自定义数据集类，继承自PyTorch的Dataset
class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts  # 存储文本数据
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为PyTorch张量
        self.char_to_index = char_to_index  # 字符到索引的映射
        self.max_len = max_len  # 最大文本长度

    def __len__(self):
        return len(self.texts)  # 返回数据集大小

    def __getitem__(self, idx):
        text = self.texts[idx]  # 获取指定索引的文本
        # 将文本转换为字符索引序列，截断到最大长度，未知字符用0填充
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        # 用填充字符补齐到最大长度
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]  # 返回索引张量和标签


# --- NEW RNN Model Class ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程， 可训练的
        self.RNN = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        # 定义全连接层，将隐藏层维度转换为输出维度
        self.fc = nn.Linear(hidden_dim, output_dim)

    # 定义前向传播过程
    def forward(self, x):
        # 将输入词嵌入为向量
        embedded = self.embedding(x)
        # 通过RNN层处理嵌入向量，得到输出和隐藏状态
        rnn_out, hidden_state = self.RNN(embedded)
        # 将隐藏状态通过全连接层，得到最终输出
        out = self.fc(hidden_state.squeeze(0))
        return out


# 创建字符级RNN数据集实例
RNN_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
# 创建数据加载器，设置批次大小为32，并打乱数据
dataloader = DataLoader(RNN_dataset, batch_size=32, shuffle=True)

# 设置模型参数
embedding_dim = 64  # 词嵌入维度
hidden_dim = 128  # 隐藏层维度
output_dim = len(label_to_index)  # 输出维度，等于标签类别数

# 初始化RNN分类器模型
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# 定义Adam优化器，学习率为0.001
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 设置训练轮数为4
num_epochs = 4
for epoch in range(num_epochs):
    # 将模型设置为训练模式
    model.train()
    # 初始化运行损失
    running_loss = 0.0
    # 遍历数据加载器
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        outputs = model(inputs)
        # 计算损失
        loss = criterion(outputs, labels)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        # 累加损失
        running_loss += loss.item()
        # 每50个批次打印一次当前损失
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 打印每个epoch的平均损失
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


def classify_text_RNN(text, model, char_to_index, max_len, index_to_label):
    """
    使用RNN模型对输入文本进行分类预测
    参数:
        text (str): 需要分类的文本
        model: 训练好的RNN模型
        char_to_index (dict): 字符到索引的映射字典
        max_len (int): 文本的最大长度
        index_to_label (dict): 索引到标签的映射字典
    返回:
        str: 预测的文本标签
    """
    # 将文本转换为字符索引序列，超出max_len的部分会被截断
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 如果文本长度不足max_len，用0填充
    indices += [0] * (max_len - len(indices))
    # 将索引列表转换为PyTorch张量，并增加批次维度
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    # 将模型设置为评估模式
    model.eval()
    # 禁用梯度计算以节省计算资源
    with torch.no_grad():
        # 使用模型进行前向传播，获取输出
        output = model(input_tensor)

    # 获取输出中最大值的索引作为预测结果
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    # 将预测的索引转换为对应的标签
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建标签到索引的反向映射字典
index_to_label = {i: label for label, i in label_to_index.items()}

# 示例1：导航请求
new_text = "帮我导航到北京"
predicted_class = classify_text_RNN(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# 示例2：天气查询请求
new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_RNN(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
