import pandas as pd          # 用于数据处理和CSV文件读取
import torch               # PyTorch深度学习框架
import torch.nn as nn      # PyTorch神经网络模块
import torch.optim as optim # PyTorch优化器模块
from torch.utils.data import Dataset, DataLoader  # PyTorch数据处理工具

# 加载数据集并进行预处理
# 从CSV文件中读取数据，以制表符分隔，无表头
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
# 提取文本数据
texts = dataset[0].tolist()
# 提取标签数据
string_labels = dataset[1].tolist()

# 创建标签到索引的映射字典
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
print("label_to_index-----------")
print(label_to_index)
# 将字符串标签转换为数值标签
numerical_labels = [label_to_index[label] for label in string_labels]
print("numerical_labels-----------")
print(numerical_labels)

# 创建字符到索引的映射字典，包含填充符号
char_to_index = {'<pad>': 0}
# 遍历所有文本，构建字符到索引的映射
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
print("char_to_index-----------")
print(char_to_index)
# 创建索引到字符的反向映射字典
index_to_char = {i: char for char, i in char_to_index.items()}
print("index_to_char-----------")
print(index_to_char)
# 计算词汇表大小
vocab_size = len(char_to_index)
print("vocab_size-----------")
print(vocab_size)
# 定义最大文本长度
max_len = 40


class CharBoWDataset(Dataset):
    # 初始化数据集类，传入文本、标签、字符索引映射等参数
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts  # 存储文本数据
        self.labels = torch.tensor(labels, dtype=torch.long)  # 将标签转换为PyTorch张量
        self.char_to_index = char_to_index  # 字符到索引的映射
        self.max_len = max_len  # 最大文本长度
        self.vocab_size = vocab_size  # 词汇表大小
        # 创建词袋向量表示
        self.bow_vectors = self._create_bow_vectors()

    # 创建词袋向量的私有方法
    def _create_bow_vectors(self):
        tokenized_texts = []  # 存储分词后的文本
        
        # 遍历每个文本，将其转换为字符索引序列
        for text in self.texts:
            # 获取文本中每个字符对应的索引（超出最大长度则截断）
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            # 如果文本长度不足max_len，则用0（即'<pad>'）填充
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)
        print("tokenized_texts-----------")
        print(tokenized_texts[:2])
        bow_vectors = []  # 存储词袋向量
        
        # 为每个文本创建词袋向量
        for text_indices in tokenized_texts:
            # 创建一个零向量，维度为词汇表大小
            bow_vector = torch.zeros(self.vocab_size)
            # 统计每个字符在文本中的出现次数
            for index in text_indices:
                if index != 0:  # 忽略填充字符
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        print("bow_vectors-----------")
        print(bow_vectors[:2])
        return torch.stack(bow_vectors)  # 将所有词袋向量堆叠成一个张量

    # 返回数据集的长度
    def __len__(self):
        return len(self.texts)

    # 获取指定索引的数据项
    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]  # 返回词袋向量和对应标签


class SimpleClassifier(nn.Module):
    # 初始化简单分类器模型
    def __init__(self, input_dim, hidden_dim,hidden_dim2, hidden_dim3,output_dim): # 层的个数 和 验证集精度
        # 调用父类构造函数
        super(SimpleClassifier, self).__init__()
        # 定义第一层全连接层：输入维度到隐藏层维度
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim2)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        # 定义ReLU激活函数
        self.relu = nn.ReLU()
        # 定义第二层全连接层：隐藏层维度到输出维度（类别数）
        self.fc4 = nn.Linear(hidden_dim3, output_dim)

    # 前向传播函数
    def forward(self, x):
        # 手动实现每层的计算
        out = self.fc1(x)   # 第一层线性变换
        out = self.relu(out) # ReLU激活函数
        out = self.fc2(out)  # 第二层线性变换
        out = self.relu(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out           # 返回最终输出


# 创建数据集实例
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# 计算并打印批次信息
print(f"数据集总样本数: {len(char_dataset)}")
print(f"批次大小: {dataloader.batch_size}")
num_batches = len(dataloader)
print(f"每个epoch的批次数: {num_batches}")

hidden_dim = 64
hidden_dim2 = 16
hidden_dim3 = 8
output_dim = len(label_to_index)
model = SimpleClassifier(vocab_size, hidden_dim,hidden_dim2,hidden_dim3,output_dim) # 维度和精度有什么关系？
criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# optimizer = optim.SGD(model.parameters(), lr=0.01)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

num_epochs = 50  # 训练轮数：数据集整体迭代的次数
print(f"训练轮数: {num_epochs}")
print(f"总的训练批次次数: {num_epochs * num_batches}")

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
        # if idx % 200 == 0:
        #     print(f"Epoch [{epoch+1}/{num_epochs}], Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")


# 定义文本分类函数
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将输入文本转换为字符索引序列
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 如果文本长度不足max_len，则用0（即'<pad>'）填充
    tokenized += [0] * (max_len - len(tokenized))

    # 创建词袋向量
    bow_vector = torch.zeros(vocab_size)
    # 统计每个字符在文本中的出现次数
    for index in tokenized:
        if index != 0:  # 忽略填充字符
            bow_vector[index] += 1

    # 增加一个维度，使其成为批量大小为1的批次
    bow_vector = bow_vector.unsqueeze(0)

    # 设置模型为评估模式
    model.eval()
    # 关闭梯度计算以节省内存和加速推理
    with torch.no_grad():
        # 进行预测
        output = model(bow_vector)

    # 获取预测概率最高的类别索引
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    # 根据索引获取对应的标签名称
    predicted_label = index_to_label[predicted_index]

    return predicted_label


# 创建索引到标签的反向映射
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试模型对新文本的分类能力
new_text = "帮我导航到北京"
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
