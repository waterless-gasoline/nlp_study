**要达到目标**（调层数/节点数 → 对比 loss 变化），需要对 09_深度学习文本分类.py代码进行如下修改

## 一、消除随机性

### 1、固定随机种子

```python
import random

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    # 如果你之后用GPU，再加：torch.cuda.manual_seed_all(seed)
```

### 2、保证每个模型训练使用相同shuffle顺序的数据集

现在 dataloader 是一次性创建的：

```python
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)
```

`shuffle=True` 意味着：**每次迭代都会随机打乱一次索引**（更精确地说：每次创建 iterator(迭代器) 或每个 epoch，会产生一组新的随机排列）

**问题**：后面要循环跑多个模型配置时，如果 dataloader 不重新建，随机状态会一路变化，不同模型看到的数据顺序可能不同。

**解决**：把 dataloader 的创建移入“每个实验配置的循环里”，并给 DataLoader 一个固定 seed 的 generator。

## 二、代码主逻辑编写

### 1、将训练循环抽象为函数

便于后面多个模型训练反复使用

```python
def train_one_model(model, dataloader, criterion, optimizer, num_epochs):
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")
    return epoch_losses

```

### 2、修改模型定义使其支持可变层数（hidden_dims列表）

```python
class SimpleClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dims,output_dim):
        super(SimpleClassifier,self).__init__()
        dims = [input_dim]+hidden_dims+[output_dim]
        layers = []
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i],dims[i+1]))
            if i<len(dims)-2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers) 
    def forward(self,x):
        return self.network(x)
```

### 3、分别调整节点数、层数训练模型，得到loss结果

```python
# 首先只做宽度（节点数）对比
results = {}
num_epochs = 10
output_dim = len(label_to_index)

hidden_dim_list = [64,128,256,512]

for hidden_dim in hidden_dim_list:
    set_seed(42)
    
    g = torch.Generator() #管理伪随机数生成状态的类
    g.manual_seed(42)
    dataloader = DataLoader(char_dataset,batch_size = 32,shuffle = True,generator =g)

    model = SimpleClassifier(vocab_size,[hidden_dim],output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    losses = train_one_model(model,num_epochs,dataloader,criterion,optimizer)
    train_name = f"1layer_h{hidden_dim}"
    results[train_name] = losses # results存着每一个hidden_dim的loss曲线（每个epoch一个值）
    
# 然后在做“深度对比”：固定结点数，改层数（hidden_dims）
result_depth = {}

hiddden_dims_list = [
    [128],
    [128,128],
    [128,128,128]
]

for hidden_dims in hidden_dims_list:
    set_seed(42)
    
    g = torch.Generator() 
    g.manual_seed(42)
    dataloader = DataLoader(char_dataset,batch_size = 32,shuffle = True,generator =g)

    model = SimpleClassifier(vocab_size,hidden_dims,output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.01)

    losses = train_one_model(model,num_epochs,dataloader,criterion,optimizer)
    train_name = f"depth{len(hidden_dims)}_{hidden_dims}"
    results[train_name] = losses # results存执每一个hidden_dim的loss曲线（每个epoch一个值）

```

### 4、将loss变化可视化

```python
# 结果可视化函数
def plt_loss_curves(result_dict, title):
    plt.figure()
    epochs = range(num_epochs)

    for name, losses in result_dict.items():
        plt.plot(epochs, losses, marker="o", linewidth=1.8, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.show()


plt_loss_curves(results, "Loss vs Epoch(Width comparison:1 hidden layer)")
plt_loss_curves(result_depth, "Loss vs Epoch(Depth comparision:hidden_dim = 128)")
```

## 完整代码1

```python
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import random
import matplotlib.pyplot as plt


# 修改1：固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    # 如果用GPU需要使用torch.cuda.manual_seed_all(seed)


dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
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


# class SimpleClassifier(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super(SimpleClassifier, self).__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         # 手动实现每层的计算
#         out = self.fc1(x)
#         out = self.relu(out)
#         out = self.fc2(out)
#         return out

# 修改模型类使其支持可变层数（hidden_dims列表）
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SimpleClassifier, self).__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)

    # *：解包符：将壳迭代的对象展开为单个元素（列表、元组、集合、字符串等）
    # nn.Sequential接收的是多个参数（每个参数使一个神经网络）
    # 假设 layers = [nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128)]
    # nn.Sequential(*layers) 等同于 nn.Sequential(nn.Linear(784, 256), nn.ReLU(), nn.Linear(256, 128))
    def forward(self, x):
        return self.network(x)


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)  # 读取单个样本


# dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# hidden_dim = 128
# output_dim = len(label_to_index)
# model = SimpleClassifier(vocab_size, hidden_dim, output_dim) # 维度和精度有什么关系？
# criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
# optimizer = optim.SGD(model.parameters(), lr=0.01)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次

# num_epochs = 10
# for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
#     model.train()
#     running_loss = 0.0
#     for idx, (inputs, labels) in enumerate(dataloader):
#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#         if idx % 50 == 0:
#             print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")


#     print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

# 修改2：把“训练循环”抽象为函数
def train_one_model(model, num_epochs, dataloader, criterion, optimizer):
    epoch_losses = []
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{num_epochs}],Loss:{avg_loss:.4f}")
    print("*" * 10)
    return epoch_losses


# 首先只做宽度（节点数）对比
results = {}
num_epochs = 10
output_dim = len(label_to_index)

hidden_dim_list = [64, 128, 256, 512]

for hidden_dim in hidden_dim_list:
    set_seed(42)

    g = torch.Generator()  # 管理伪随机数生成状态的类
    g.manual_seed(42)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True, generator=g)

    model = SimpleClassifier(vocab_size, [hidden_dim], output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = train_one_model(model, num_epochs, dataloader, criterion, optimizer)
    train_name = f"1layer_h{hidden_dim}"
    results[train_name] = losses  # results存着每一个hidden_dim的loss曲线（每个epoch一个值）

# 然后在做“深度对比”：固定结点数，改层数（hidden_dims）
result_depth = {}

hidden_dims_list = [
    [128],
    [128, 128],
    [128, 128, 128]
]

for hidden_dims in hidden_dims_list:
    set_seed(42)

    g = torch.Generator()
    g.manual_seed(42)
    dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True, generator=g)

    model = SimpleClassifier(vocab_size, hidden_dims, output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    losses = train_one_model(model, num_epochs, dataloader, criterion, optimizer)
    train_name = f"depth{len(hidden_dims)}_{hidden_dims}"
    result_depth[train_name] = losses  # results存执每一个hidden_dim的loss曲线（每个epoch一个值）


# 结果可视化函数
def plt_loss_curves(result_dict, title):
    plt.figure()
    epochs = range(num_epochs)

    for name, losses in result_dict.items():
        plt.plot(epochs, losses, marker='o', linewidth=1.8, label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.legend()
    plt.show()


plt_loss_curves(results, "Loss vs Epoch(Width comparison:1 hidden layer)")
plt_loss_curves(result_depth, "Loss vs Epoch(Depth comparision:hidden_dim = 128)")
```

## 最终结果1

<img width="1268" height="1069" alt="image" src="https://github.com/user-attachments/assets/0aff99a1-9795-4949-9629-17a0413ebcf6" />


可以看到“单层 MLP + 64 隐藏单元”已经能把训练 loss 压到很低；继续加宽基本是浪费参数。

<img width="1275" height="1077" alt="image" src="https://github.com/user-attachments/assets/3cbe0ff9-4197-4243-a78d-e5b88a5d0530" />


可以看到层数越多，loss下降越慢，且一层神经网络的loss最低，3层神经网络可能需要更多的轮次才能把loss下降的足够低。

---

将宽度对照组的宽度从[64,128,256,512]变为[16,32,48,64]，隐藏层很窄（16）时，模型容量不足，表达能力受限，所以训练 loss 会更高。

一旦宽度到 32/48/64，模型就“够用”了；再加宽参数更多，但在**同样 epoch 数 + 同样 SGD 设置**下，训练 loss 改善就不明显了。32层就已经很好了

<img width="1273" height="1073" alt="image" src="https://github.com/user-attachments/assets/1558261d-2445-4437-aafe-aac53c5921aa" />


将深度对照组的训练轮次从10轮变为20轮，同样层数越多下降越慢，但最终2层神经网络的loss最低，可能增加训练次数3层神经网络的最终loss值会更低

<img width="1278" height="1075" alt="image" src="https://github.com/user-attachments/assets/1d0cda10-dda2-4f4f-a71d-92fcaaed5669" />

---

增加深度对照组的训练轮次从20轮变为30轮，可以看到最终是3层神经网络的loss最低

<img width="1276" height="1071" alt="image" src="https://github.com/user-attachments/assets/2dee21cf-df5d-469f-8060-f2531398f738" />

# 实验结果总结:

## 1) 实验设置回顾（实际在对比什么）

模型是：**Char-BoW（字符词袋计数向量） + MLP 分类器**
 对比维度有两个：

1. **宽度（Width）**：固定 1 个隐藏层，改变隐藏单元数（如 16/32/48/64 或 64/128/256/512）
2. **深度（Depth）**：固定每层宽度=128，改变层数（1 层、2 层、3 层）

训练设置：CrossEntropyLoss + SGD(lr=0.01)，并且固定了随机种子和 shuffle 顺序，保证对比公平。

------

## 2) 宽度实验（1 hidden layer）总结

### 现象

- 隐藏层从 **16 → 32**：loss 下降更快、最终更低，提升明显。
- 从 **32 → 48/64（甚至到 128/256/512）**：曲线越来越接近，最终 loss 差异很小（提升非常有限）。

### 结论

- **存在“足够宽度阈值”**：当宽度达到某个水平后，模型容量已经足够拟合当前任务（在训练 loss 指标上），继续加宽收益递减。
- 在你这个 Char-BoW 特征下，任务可能本身就比较“好分”，所以小网络就能拟合得很好。

一句话：**宽度不是主要瓶颈；32 以上加宽带来的训练 loss 改善很有限。**

------

## 3) 深度实验（hidden_dim=128）总结：10 epoch vs 30 epoch 的关键变化

### （A）训练轮次较少时（比如 10 epoch 的那次）

- 1 层网络下降快、最终最低；
- 2 层、3 层下降慢，尤其 3 层看起来“没学起来”，最终 loss 明显更高。

**解释：\**深层网络在你这种“纯 MLP + ReLU + SGD”设置下，通常\**收敛更慢**，轮次少时看起来就更差——本质是“优化没跑完”。

------

### （B）训练轮次增加到 30（你最新这张图）

你观察到：**最终 3 层网络的 loss 最低**（2 层其次，1 层最高）。

从曲线形态上还能看出几个典型特征：

1. **前期（0~5 epoch）**：1 层下降最快（优化最容易）。
2. **中期（大概 5~15 epoch）**：2 层、3 层开始明显加速下降，逐步追平甚至反超。
3. **后期（15~30 epoch）**：3 层继续缓慢下降，最终达到最低。

**解释：**

- 深层网络容量更强，但“优化更难/更慢”。当你给它足够训练轮次后，它能把训练 loss 压得更低。
- 这说明之前不是“深层不如浅层”，而是“深层需要更长训练才能体现优势”。

一句话：**深度提升的收益是“用更多训练轮次换来的”；轮次足够时，3 层能达到更低的训练 loss。**

------

## 4) 总体结论（把宽度 + 深度放一起）

### 结论 1：宽度收益很快饱和

- 小宽度（如 16）会欠拟合；
- 中等宽度（32/64/128）基本就够用；
- 更大宽度（256/512）在你的训练设置下对训练 loss 改善很小。

### 结论 2：深度带来的优势依赖“训练是否充分”

- 轮次少：深层看起来更差（收敛慢、像没学会）。
- 轮次多：深层最终更低（3 层最低），体现更强的拟合能力。

### 结论 3：目前所有结论都只针对“训练集拟合能力”

- 训练 loss 更低 ≠ 泛化更好。
- 更深/更宽的网络很可能更容易过拟合（训练 loss 低但验证/测试变差）。


# 
