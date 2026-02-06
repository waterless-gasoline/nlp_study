import os
import torch
import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding
from tqdm.auto import tqdm


# ===================== 1. 配置参数（修改这里的路径） =====================
class Config:
    model_path = "./bert-base-uncased"
    data_path = "./"
    batch_size = 4  # 小批量避免CPU内存不足
    epochs = 2  # 训练轮数（测试用，实际可设3-5）
    lr = 5e-5  # BERT推荐学习率
    num_labels = 4  # ag_news是4分类任务
config = Config()

# ===================== 2. 环境配置 =====================
# 强制使用CPU（避免CUDA驱动问题）
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# 关闭symlink缓存警告
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
# 设置设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"使用设备: {device}")


# ===================== 3. 加载本地ag_news数据集 =====================
def load_local_ag_news(data_dir):
    """加载本地train.csv/test.csv，转换为Dataset格式"""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    # 读取CSV（names对应：标签、标题、正文）
    train_df = pd.read_csv(train_path, names=["label", "title", "description"])
    test_df = pd.read_csv(test_path, names=["label", "title", "description"])

    # 合并标题+正文为text字段（适配BERT输入）
    train_df["text"] = train_df["title"] + " " + train_df["description"]
    test_df["text"] = test_df["title"] + " " + test_df["description"]

    # 标签减1（原数据标签是1-4，转成0-3适配模型）
    train_df["label"] = train_df["label"] - 1
    test_df["label"] = test_df["label"] - 1

    # 转换为HuggingFace Dataset格式
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    return {"train": train_dataset, "test": test_dataset}


# 加载数据集
dataset = load_local_ag_news(config.data_path)
print(f"训练集数量: {len(dataset['train'])}, 测试集数量: {len(dataset['test'])}")
print(f"样本示例: {dataset['train'][0]}")

# ===================== 4. 数据预处理（分词） =====================
# 加载本地BERT分词器
tokenizer = BertTokenizer.from_pretrained(config.model_path)


def preprocess_function(examples):
    """对文本进行分词、截断（适配BERT输入）"""
    return tokenizer(
        examples["text"],  # 输入文本
        truncation=True,  # 超过max_length截断
        max_length=128,  # 文本最大长度（BERT推荐128/256）
        padding=False  # 批量padding交给DataCollator
    )


# 批量处理数据集
tokenized_datasets = dataset.map(preprocess_function, batched=True)
# 保留需要的字段（label是标签，input_ids/attention_mask是BERT输入）
tokenized_datasets = tokenized_datasets.remove_columns(["title", "description", "text"])
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")  # 适配模型输入名
tokenized_datasets.set_format("torch")  # 转换为PyTorch张量

# 构建数据加载器（自动padding）
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,  # 训练集打乱
    batch_size=config.batch_size,
    collate_fn=data_collator  # 自动补齐不同长度的文本
)
test_dataloader = DataLoader(
    tokenized_datasets["test"],
    batch_size=config.batch_size,
    collate_fn=data_collator
)

# ===================== 5. 加载BERT模型并训练 =====================
# 加载本地BERT分类模型（num_labels=4对应4分类）
model = BertForSequenceClassification.from_pretrained(
    config.model_path,
    num_labels=config.num_labels
)
model.to(device)  # 模型放到CPU/GPU

# 优化器（BERT推荐AdamW）
optimizer = AdamW(model.parameters(), lr=config.lr)

# 学习率调度器（线性衰减）
num_training_steps = config.epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,  # 预热步数（新手设0即可）
    num_training_steps=num_training_steps
)

# 训练循环
print("\n开始训练...")
progress_bar = tqdm(range(num_training_steps))
model.train()  # 模型设为训练模式
for epoch in range(config.epochs):
    epoch_loss = 0.0
    for batch in train_dataloader:
        # 数据放到设备上
        batch = {k: v.to(device) for k, v in batch.items()}
        # 前向传播
        outputs = model(**batch)
        loss = outputs.loss  # 分类损失
        epoch_loss += loss.item()

        # 反向传播+更新参数
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()  # 清空梯度

        progress_bar.update(1)
    # 打印每轮损失
    print(f"Epoch {epoch + 1}/{config.epochs}, 平均损失: {epoch_loss / len(train_dataloader):.4f}")

# ===================== 6. 模型评估（测试集准确率） =====================
print("\n开始评估...")
model.eval()  # 模型设为评估模式
correct = 0
total = 0
with torch.no_grad():  # 禁用梯度计算（节省内存）
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        # 预测结果：取logits最大值对应的类别
        preds = torch.argmax(outputs.logits, dim=1)
        # 统计正确数
        correct += (preds == batch["labels"]).sum().item()
        total += len(batch["labels"])

# 计算准确率
accuracy = correct / total
print(f"\n测试集准确率: {accuracy:.4f}")
print(f"正确预测数: {correct}, 总样本数: {total}")

# ===================== 7. 保存模型（可选） =====================
save_path = "./bert_ag_news_model"
os.makedirs(save_path, exist_ok=True)
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print(f"\n模型已保存到: {save_path}")
