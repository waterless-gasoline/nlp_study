import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 数据加载与预处理
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40

# 数据集类
class CharTextDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# 定义三个模型：RNN/LSTM/GRU（结构一致，仅循环层不同）
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.rnn(x)
        out = self.fc(hidden.squeeze(0))
        return out

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden.squeeze(0))
        return out

class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        _, hidden = self.gru(x)
        out = self.fc(hidden.squeeze(0))
        return out

# 通用训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if idx % 50 == 0:
                print(f"Batch {idx}, Loss: {loss.item():.4f}")
        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Train Loss: {avg_loss:.4f}")
    return model, train_losses

# 通用评估函数（计算测试集准确率）
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f"Test Accuracy: {acc:.2f}%")
    return acc

# 预测函数
def classify_text(text, model, char_to_index, max_len, index_to_label, device):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    return index_to_label[predicted_idx.item()]

# 超参数与设备初始化
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
batch_size = 32
lr = 0.001
num_epochs = 4

# 划分训练/测试集（7:3，分层采样保证标签分布一致）
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, numerical_labels, test_size=0.3, random_state=42, stratify=numerical_labels
)

# 创建DataLoader
train_dataset = CharTextDataset(train_texts, train_labels, char_to_index, max_len)
test_dataset = CharTextDataset(test_texts, test_labels, char_to_index, max_len)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失、优化器的字典（方便批量实验）
models = {
    "RNN": RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    "LSTM": LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim),
    "GRU": GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
}
criterions = {name: nn.CrossEntropyLoss() for name in models.keys()}
optimizers = {name: optim.Adam(model.parameters(), lr=lr) for name, model in models.items()}

# 实验主流程：训练+评估每个模型，记录结果
experiment_results = {}
index_to_label = {i: label for label, i in label_to_index.items()}
for name, model in models.items():
    print(f"\n==================== Training {name} ====================")
    trained_model, train_losses = train_model(
        model, train_loader, criterions[name], optimizers[name], num_epochs, device
    )
    test_acc = evaluate_model(trained_model, test_loader, device)
    experiment_results[name] = {
        "model": trained_model,
        "train_losses": train_losses,
        "test_accuracy": test_acc
    }

# 输出三个模型的精度对比
print(f"\n==================== 模型精度对比 ====================")
for name, res in experiment_results.items():
    print(f"{name}: 测试集准确率 = {res['test_accuracy']:.2f}%")

# 用每个模型对示例文本预测
test_texts = ["帮我导航到北京", "查询明天北京的天气"]
print(f"\n==================== 示例文本预测 ====================")
for text in test_texts:
    print(f"\n输入文本: {text}")
    for name, res in experiment_results.items():
        pred_label = classify_text(
            text, res["model"], char_to_index, max_len, index_to_label, device
        )
        print(f"{name} 预测标签: {pred_label}")