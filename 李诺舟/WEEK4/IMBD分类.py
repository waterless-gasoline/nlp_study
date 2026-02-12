import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import BertTokenizer # 新增导入
from datasets import load_dataset # 新增导入
from torch.utils.data import DataLoader, Dataset # 新增导入


# 2. 数据加载和预处理 (适配 custom BERT)

# 实例化BertTokenizer (使用bert-base-uncased作为vocab源)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class IMDBDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, # 添加 [CLS] 和 [SEP]
            max_length=self.max_len,
            return_token_type_ids=True, # 返回 segment_ids
            padding='max_length', # 填充到max_len
            truncation=True, # 截断到max_len
            return_attention_mask=True,
            return_tensors='pt' # 返回 PyTorch tensors
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'segment_ids': encoding['token_type_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# --- 从 08_BERT.py 复制的核心 BERT 编码器组件 ---START ---

class BertEmbedding(nn.Module):
    '''
    BertEmbedding包括三部分, 三部分相加并输出:
    1. TokenEmbedding  /  2. PositionalEmbedding  /  3. SegmentEmbedding
    '''

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        super(BertEmbedding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.token_embed = TokenEmbedding(vocab_size, embed_size)
        self.position_embed = PositionalEmbedding(embed_size)
        self.segment_embed = SegmentEmbedding(embed_size)

    def forward(self, sequence, segment_label):
        x = self.token_embed(sequence) + self.position_embed(sequence) + self.segment_embed(segment_label)
        return self.dropout(x)


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_size, max_len=512):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        pe.requires_grad = False
        position = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0., embed_size, 2) * (- math.log(10000.) / embed_size))

        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 修正 PositionalEmbedding 以正确处理 batch_size
        # self.pe 的形状是 (1, max_len, embed_size)
        # x 的形状是 (batch_size, seq_len, embed_size) - 在 BertEmbedding 中处理过 token embed 后
        # 我们需要的位置编码是 (1, seq_len, embed_size)，然后通过广播与 x 相加
        return self.pe[:, :x.size(1)]


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, head, feed_forward_hidden, dropout):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadedAttention(hidden, head, dropout=dropout)
        self.feed_forward = FeedForward(hidden, feed_forward_hidden, dropout=dropout)
        self.attn_sublayer = SubLayerConnection(hidden, dropout)
        self.ff_sublayer = SubLayerConnection(hidden, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.attn_sublayer(x, lambda x: self.attention(x, x, x, mask))
        x = self.ff_sublayer(x, self.feed_forward)
        return self.dropout(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.alpha = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        x_mean = x.mean(-1, keepdim=True)
        x_std = x.std(-1, keepdim=True)
        return self.alpha * (x - x_mean) / (x_std + self.eps) + self.beta


class SubLayerConnection(nn.Module):
    def __init__(self, hidden, dropout):
        super(SubLayerConnection, self).__init__()
        self.layer_norm = LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))


def attention(q, k, v, mask=None, dropout=None):
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)

    return torch.matmul(attention_weights, v), attention_weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, hidden, head, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        self.dk = hidden // head
        self.head = head
        self.input_linears = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(3)])
        self.output_linear = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)
        if mask is not None:
            # mask 需要与 scores 的维度匹配，scores 是 (b, head, max_len, max_len)
            # mask 初始是 (b, 1, max_len, max_len) 或者 (b, max_len, max_len)
            # 如果 mask 是 (b, max_len, max_len)，需要 unsqueeze(1) 变成 (b, 1, max_len, max_len)
            # 这样在 attention 函数中进行 masked_fill 时可以广播到 head 维度
            if mask.dim() == 3: # 假设传入的mask是 (batch_size, seq_len, seq_len)
                mask = mask.unsqueeze(1) # 变成 (batch_size, 1, seq_len, seq_len)

        q, k, v = [linear(x).view(batch_size, -1, self.head, self.dk).transpose(1, 2)
                   for linear, x in zip(self.input_linears, (q, k, v))]

        x, self.attn = attention(q, k, v, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.dk)
        return self.output_linear(x)


class FeedForward(nn.Module):
    def __init__(self, hidden, ff_dim, dropout=0.1):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(hidden, ff_dim)
        self.linear2 = nn.Linear(ff_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.activation = GLUE()  # 使用GLUE激活函数

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class GLUE(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Bert(nn.Module):
    '''
    BertEmbedding + TransformerBlock
    '''

    def __init__(self, vocab_size, hidden=768, n_layers=12, head=12, dropout=0.1):
        super(Bert, self).__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.head = head
        self.feed_forward_hidden = hidden * 4
        self.embedding = BertEmbedding(vocab_size=vocab_size, embed_size=hidden, dropout=dropout)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, head, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info, mask):
        # x: (b, max_len)
        # segment_info: (b, max_len)
        # mask: (b, max_len, max_len) - 由外部传入
        x = self.embedding(x, segment_info)

        for transformer in self.transformer_blocks:
            x = transformer(x, mask)
        return x

# --- 从 08_BERT.py 复制的核心 BERT 编码器组件 ---END ---


class CustomBertForSequenceClassification(nn.Module):
    def __init__(self, vocab_size, hidden=768, n_layers=12, head=12, dropout=0.1, num_labels=2):
        super(CustomBertForSequenceClassification, self).__init__()
        self.bert = Bert(vocab_size, hidden, n_layers, head, dropout)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels) # 分类头

    def forward(self, input_ids, segment_ids, attention_mask):
        # 得到BERT编码器的输出
        # out 形状: (batch_size, sequence_length, hidden_size)
        encoded_layers = self.bert(input_ids, segment_ids, attention_mask)
        
        # 我们通常使用 [CLS] token 的输出进行分类，[CLS] token 位于序列的第一个位置
        # cls_output 形状: (batch_size, hidden_size)
        cls_output = encoded_layers[:, 0]

        # 经过 dropout 和分类器
        logits = self.classifier(self.dropout(cls_output))
        return logits


# --- 训练和评估循环 ---START ---

from transformers import BertTokenizer # 再次导入以防万一
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW # 使用AdamW优化器
from sklearn.metrics import accuracy_score
import tqdm # 用于进度条

# 1. 加载IMDB数据集
print("正在加载IMDB数据集...")
dataset = load_dataset("imdb")

# 准备训练和测试文本和标签
def prepare_data(data_split):
    texts = [item['text'] for item in data_split]
    labels = [item['label'] for item in data_split]
    return texts, labels

train_texts, train_labels = prepare_data(dataset['train'])
test_texts, test_labels = prepare_data(dataset['test'])

# 2. 创建Dataset和DataLoader
MAX_LEN = 512 # 与PositionalEmbedding的max_len一致
train_dataset = IMDBDataset(train_texts, train_labels, tokenizer, MAX_LEN)
test_dataset = IMDBDataset(test_texts, test_labels, tokenizer, MAX_LEN)

batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 3. 实例化模型、优化器和损失函数
vocab_size = tokenizer.vocab_size # 从tokenizer获取词汇表大小
hidden_size = 768
n_layers = 12
num_heads = 12
dropout_rate = 0.1
num_labels = 2

print("正在实例化CustomBertForSequenceClassification模型...")
model = CustomBertForSequenceClassification(vocab_size=vocab_size,
                                            hidden=hidden_size,
                                            n_layers=n_layers,
                                            head=num_heads,
                                            dropout=dropout_rate,
                                            num_labels=num_labels)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5) # 学习率通常会低一些
criterion = nn.CrossEntropyLoss()

# 4. 训练循环
epochs = 3 # 训练3个epoch
print(f"开始训练，共 {epochs} 个epoch...")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Training Epoch {epoch+1}")):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        segment_ids = batch['segment_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        logits = model(input_ids, segment_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} 训练平均损失: {avg_train_loss:.4f}")

    # 5. 评估模型
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_loader, desc=f"Evaluating Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, segment_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1).cpu().numpy()

            val_preds.extend(preds)
            val_true.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(val_true, val_preds)
    print(f"Epoch {epoch+1} 验证准确率: {accuracy:.4f}")

print("模型训练完成。")

# 6. 使用新样本进行测试
print("--- 使用新样本进行测试 ---")
def predict_sentiment(text, model, tokenizer, device, max_len=512):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        return_token_type_ids=True,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    segment_ids = encoding['token_type_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, attention_mask)
        prediction = torch.argmax(logits, dim=-1).item()

    return "正面评价" if prediction == 1 else "负面评价"

# 测试样本
test_samples = [
    "This movie was absolutely fantastic! I loved every minute of it.", # 正面
    "What a terrible film. I wasted my money and time.", # 负面
    "The acting was mediocre, but the plot had some interesting twists.", # 中性，模型可能倾向于负面或正面
    "I'm not sure what to think about this movie, it was very confusing." # 负面
]

for sample in test_samples:
    sentiment = predict_sentiment(sample, model, tokenizer, device, MAX_LEN)
    print(f"评论: '{sample}'")
    print(f"预测情感: {sentiment}\n")

print("任务完成。")
