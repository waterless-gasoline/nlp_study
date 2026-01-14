import re

import jieba
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel

dataset = pd.read_csv("data/dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())    #group by

# 定义文本预处理函数
def preprocess_text(text):
    # 清洗文本
    text = re.sub(r'[^\w\s]', '', text)  # 去标点
    # 分词
    words = jieba.lcut(text)
    # 去除停用词
    stopwords = {"的", "了", "在", "是", "我", "你", "他", "她", "它"}
    words = [word for word in words if word not in stopwords and len(word) > 1]
    return " ".join(words)

dataset['processed_text'] = dataset[0].apply(preprocess_text)
# 准备标签编码器
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(dataset[1].values)
num_classes = len(label_encoder.classes_)
print("123::",label_encoder)
print("124::",labels_encoded)
print("125::",num_classes)
# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    dataset['processed_text'].values,
    labels_encoded,
    test_size=0.2,
    random_state=42,
    stratify=labels_encoded
)

print(f"\n训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")


# ============================================================================
# 方法1: KNN分类器（老师的例子）
# ============================================================================
input_sententce = dataset[0].apply(preprocess_text) # sklearn对中文处理
vector = CountVectorizer()              # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values)      # 统计词表
input_feature = vector.transform(input_sententce.values) # 进行转换 100 * 词表大小
model = KNeighborsClassifier()          # 创建KNN分类器
model.fit(input_feature, dataset[1].values) # 创建KNN分类器\

# 使用TF-IDF替换CountVectorizer，效果应该更好
vectorizer_knn = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.9)
X_train_tfidf = vectorizer_knn.fit_transform(X_train)
X_test_tfidf = vectorizer_knn.transform(X_test)

knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='cosine'
)
knn_model.fit(X_train_tfidf, y_train)
knn_accuracy = knn_model.score(X_test_tfidf, y_test)
print(f"KNN测试集准确率: {knn_accuracy:.4f}")

# ============================================================================
# 方法2: CNN
# ============================================================================
print("\n" + "=" * 60)
print("方法2: CNN文本分类")
print("=" * 60)


class TextDataset(Dataset):
    """文本数据集类"""

    def __init__(self, texts, labels, vocab, max_len=100):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # 将文本转换为索引序列
        words = text.split()
        indices = [self.vocab.get(word, 0) for word in words[:self.max_len]]
        # 填充或截断
        if len(indices) < self.max_len:
            indices = indices + [0] * (self.max_len - len(indices))
        else:
            indices = indices[:self.max_len]
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.long)
        }

class TextCNN(nn.Module):
    """CNN文本分类模型"""
    def __init__(self, vocab_size, embed_dim, num_classes,
                 filter_sizes=[3, 4, 5], num_filters=100, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 多个卷积层
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    def forward(self, x):
        # x: [batch_size, seq_len]
        embedded = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        embedded = embedded.permute(0, 2, 1)  # [batch_size, embed_dim, seq_len]
        # 卷积和池化
        pooled_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # [batch_size, num_filters, seq_len-filter_size+1]
            pooled = torch.max(conv_out, dim=2)[0]  # [batch_size, num_filters]
            pooled_outputs.append(pooled)
        # 拼接所有池化结果
        cat = self.dropout(torch.cat(pooled_outputs, dim=1))  # [batch_size, num_filters * len(filter_sizes)]
        # 全连接层
        logits = self.fc(cat)  # [batch_size, num_classes]
        return logits

# 创建词汇表
all_words = []
for text in X_train:
    all_words.extend(text.split())

word_counts = {}
for word in all_words:
    word_counts[word] = word_counts.get(word, 0) + 1

# 只保留出现次数较多的词
min_count = 2
vocab = {'<PAD>': 0, '<UNK>': 1}
for word, count in word_counts.items():
    if count >= min_count:
        vocab[word] = len(vocab)

vocab_size = len(vocab)
print(f"词汇表大小: {vocab_size}")
print("词汇表",vocab)

# 准备数据加载器
train_dataset = TextDataset(X_train, y_train, vocab, max_len=100)
test_dataset = TextDataset(X_test, y_test, vocab, max_len=100)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 初始化CNN模型
cnn_model = TextCNN(
    vocab_size=vocab_size,
    embed_dim=128,
    num_classes=num_classes,
    filter_sizes=[2, 3, 4],
    num_filters=100
)

# 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cnn_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn_model.parameters(), lr=0.001)

# 训练CNN模型
print("训练CNN模型...")
cnn_model.train()
for epoch in range(5):  # 简单训练5个epoch
    total_loss = 0
    correct = 0
    total = 0

    for batch in train_loader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = cnn_model(texts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Epoch [{epoch + 1}/5], Loss: {total_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%')

# 测试CNN模型
print("测试CNN模型...")
cnn_model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        texts = batch['text'].to(device)
        labels = batch['label'].to(device)

        outputs = cnn_model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

cnn_accuracy = correct / total
print(f'CNN测试集准确率: {cnn_accuracy:.4f}')
#CNN END
# ============================================================================
# 方法3: K-means聚类分类
# ============================================================================
print("\n" + "=" * 60)
print("方法3: K-means聚类分类")
print("=" * 60)

class KMeansClassifier:
    """基于K-means的文本分类器"""

    def __init__(self, n_clusters=None):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.kmeans = None
        self.n_clusters = n_clusters
        self.cluster_to_label = {}
        self.label_encoder = label_encoder

    def fit(self, texts, labels):
        # 向量化
        X = self.vectorizer.fit_transform(texts)

        # 如果未指定聚类数，使用类别数
        if self.n_clusters is None:
            self.n_clusters = len(np.unique(labels))

        # K-means聚类
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=42,
            n_init=10
        )
        cluster_labels = self.kmeans.fit_predict(X)

        # 建立聚类到真实标签的映射
        self.cluster_to_label = {}
        for cluster_id in range(self.n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_true_labels = labels[cluster_indices]
                # 找到该聚类中最常见的真实标签
                unique, counts = np.unique(cluster_true_labels, return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                self.cluster_to_label[cluster_id] = most_common_label
            else:
                self.cluster_to_label[cluster_id] = 0  # 默认标签

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        cluster_ids = self.kmeans.predict(X)
        predictions = [self.cluster_to_label.get(cid, 0) for cid in cluster_ids]
        return np.array(predictions)

# 训练K-means分类器
kmeans_classifier = KMeansClassifier()
kmeans_classifier.fit(X_train, y_train)
y_pred_kmeans = kmeans_classifier.predict(X_test)
kmeans_accuracy = np.mean(y_pred_kmeans == y_test)
print(f"K-means测试集准确率: {kmeans_accuracy:.4f}")

def text_classify_using_ml_mao(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    try:
        if not text or not text.strip():
            return "请输入文本"  # 处理空文本
        test_sentence = " ".join(jieba.lcut(text))
        # print(test_sentence)
        test_feature = vector.transform([test_sentence])    #特征提取
        # print(test_feature)
        # print(model.predict(test_feature))
        return model.predict(test_feature)[0]
    except Exception as e:
            print(f"预测失败: {e}")
            return "未知"
def text_classify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    try:
        if not text or not text.strip():
            return "请输入文本"  # 处理空文本
        processed = preprocess_text(text)
        features = vectorizer_knn.transform([processed])
        prediction = knn_model.predict(features)[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
            print(f"预测失败: {e}")
            return "未知"


def text_classify_cnn(text: str) -> str:
    """CNN文本分类"""
    try:
        if not text or not text.strip():
            return "请输入文本"
        processed = preprocess_text(text)
        words = processed.split()[:100]
        indices = [vocab.get(word, 1) for word in words]  # 1表示<UNK>
        # 填充
        if len(indices) < 100:
            indices = indices + [0] * (100 - len(indices))
        # 转换为张量
        input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
        cnn_model.eval()
        with torch.no_grad():
            output = cnn_model(input_tensor)
            prediction = torch.argmax(output, dim=1).cpu().item()
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        print(f"CNN预测失败: {e}")
        return "未知"


def text_classify_kmeans(text: str) -> str:
    """K-means文本分类"""
    try:
        if not text or not text.strip():
            return "请输入文本"

        processed = preprocess_text(text)
        prediction = kmeans_classifier.predict([processed])[0]
        return label_encoder.inverse_transform([prediction])[0]
    except Exception as e:
        print(f"K-means预测失败: {e}")
        return "未知"
if __name__ == "__main__":
    # pandas 用来进行表格的加载和分析
    # numpy 从矩阵的角度进行加载和计算
    test_texts = [
        "帮我导航到南山路",
        "帮我打开收音机",
        "最新的人工智能技术发展",
        "我喜欢听歌曲帮我播放五月天",
        "电影院的排片时间表"
    ]
    for text in test_texts:
        print(f"\n原文: {text}")
        print(f"课堂-机器学习:    {text_classify_using_ml_mao(text)}")
        print(f"机器学习:    {text_classify_using_ml(text)}")
        print(f"K-means: {text_classify_kmeans(text)}")
        print(f"CNN:     {text_classify_cnn(text)}")
        # print(f"BERT:    {text_classify_bert(text)}")
        print("-" * 40)
    print("机器学习: ", text_classify_using_ml("我喜欢听歌曲帮我播放五月天"))  #经过数据清洗准确率更好了
    print("课堂机器学习: ", text_classify_using_ml_mao("我喜欢听歌曲帮我播放五月天"))
    print("CNN: ", text_classify_cnn("我喜欢听歌曲帮我播放五月天"))      #我喜欢听歌曲帮我播放五月天  不稳定，时而 Music-Play 时而FilmTele-Play
    print("K-means: ", text_classify_kmeans("我喜欢听歌曲帮我播放五月天"))

