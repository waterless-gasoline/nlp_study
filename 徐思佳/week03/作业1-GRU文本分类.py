import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

datasets = pd.read_csv("../week01/dataset.csv", sep="\t", header=None)
texts = datasets[0].tolist()
string_labels = datasets[1].tolist()
# print(texts[:5])
# print(string_labels[:5])

# 文本标签转下标
label_to_index = {label:i for i, label in enumerate(set(string_labels))}
# print(label_to_index)
# {'Audio-Play': 0, 'HomeAppliance-Control': 1, 'Music-Play': 2, 'Weather-Query': 3, 'Other': 4, 'Calendar-Query': 5, 'FilmTele-Play': 6, 'Video-Play': 7, 'Alarm-Update': 8, 'TVProgram-Play': 9, 'Travel-Query': 10, 'Radio-Listen': 11}
numerical_labels = [label_to_index[label] for label in string_labels]
# print(numerical_labels[:5])

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char ={i:char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

MAX_LEN = 40

class CharGRUDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts =  texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        # 文本长度统一: 截断、填充
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices = indices + [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

class GRUClassifier(nn.Module):
    def __init__(self,vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.gru(x)
        x = self.fc(x[:, -1, :])
        return x

# ====== 训练模型 ======
# 1. 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(
    texts, numerical_labels, random_state=520, test_size=0.2, stratify=numerical_labels)
train_dataset = CharGRUDataset(train_x, train_y, char_to_index, MAX_LEN)
test_dataset = CharGRUDataset(test_x, test_y, char_to_index, MAX_LEN)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 2. 创建模型
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 训练
num_epoch = 4
for epoch in range(num_epoch):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(inputs)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epoch}], Loss: {running_loss / len(train_dataloader):.4f}")

    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        correct = 0
        for inputs, labels in test_dataloader:
            output = model(inputs)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
        print(f"Test Accuracy: {correct / len(test_dataset):.4f}")

# 预测数值转类型
index_to_label = {i: label for label, i in label_to_index.items()}
def classify_text_gru(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

example_text = "请查询北京的天气"
predicted_label = classify_text_gru(example_text, model, char_to_index, MAX_LEN, index_to_label)
print(f"输入 '{example_text}' 预测为: '{predicted_label}'")

example_text = "导航到杭州"
predicted_label = classify_text_gru(example_text, model, char_to_index, MAX_LEN, index_to_label)
print(f"输入 '{example_text}' 预测为: '{predicted_label}'")

'''
Batch 个数 0, 当前Batch Loss: 2.511901378631592
Batch 个数 50, 当前Batch Loss: 2.379133462905884
Batch 个数 100, 当前Batch Loss: 2.216139316558838
Batch 个数 150, 当前Batch Loss: 1.6609269380569458
Batch 个数 200, 当前Batch Loss: 1.068881869316101
Batch 个数 250, 当前Batch Loss: 0.7887413501739502
Batch 个数 300, 当前Batch Loss: 1.013693928718567
Epoch [1/4], Loss: 1.6869
Test Accuracy: 0.7558
Batch 个数 0, 当前Batch Loss: 0.7471583485603333
Batch 个数 50, 当前Batch Loss: 0.6299922466278076
Batch 个数 100, 当前Batch Loss: 0.5115516185760498
Batch 个数 150, 当前Batch Loss: 0.6662681698799133
Batch 个数 200, 当前Batch Loss: 0.48274141550064087
Batch 个数 250, 当前Batch Loss: 0.5354156494140625
Batch 个数 300, 当前Batch Loss: 0.33853456377983093
Epoch [2/4], Loss: 0.5597
Test Accuracy: 0.8434
Batch 个数 0, 当前Batch Loss: 0.31231990456581116
Batch 个数 50, 当前Batch Loss: 0.4880974590778351
Batch 个数 100, 当前Batch Loss: 0.09065496921539307
Batch 个数 150, 当前Batch Loss: 0.1334175318479538
Batch 个数 200, 当前Batch Loss: 0.34081926941871643
Batch 个数 250, 当前Batch Loss: 0.29802411794662476
Batch 个数 300, 当前Batch Loss: 0.5854791402816772
Epoch [3/4], Loss: 0.3527
Test Accuracy: 0.8599
Batch 个数 0, 当前Batch Loss: 0.4307413101196289
Batch 个数 50, 当前Batch Loss: 0.2685500979423523
Batch 个数 100, 当前Batch Loss: 0.23050308227539062
Batch 个数 150, 当前Batch Loss: 0.19228185713291168
Batch 个数 200, 当前Batch Loss: 0.3298390805721283
Batch 个数 250, 当前Batch Loss: 0.2952723801136017
Batch 个数 300, 当前Batch Loss: 0.1881243884563446
Epoch [4/4], Loss: 0.2511
Test Accuracy: 0.8682
输入 '请查询北京的天气' 预测为: 'Weather-Query'
输入 '导航到杭州' 预测为: 'Travel-Query'
'''
