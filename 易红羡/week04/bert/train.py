# train.py
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
import os

print("✅ 库导入成功")

print("\n=== 第2步：加载数据集（DBpedia，14个类别） ===")
# 这会从Hugging Face官网自动下载，无需你准备文件
dataset = load_dataset('dbpedia_14')
print("数据集结构：", dataset)
print("看看第一条训练数据是什么样子：", dataset['train'][0])

print("\n=== 第3步：加载BERT分词器和模型 ===")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=14)

print(" 分词器和模型加载成功")

print("\n=== 第4步：对数据进行分词处理（将文本变成数字） ===")
def tokenize_function(examples):
    return tokenizer(examples['content'], padding='max_length', truncation=True, max_length=128)

# 对数据集的所有部分（训练集和测试集）进行分词
tokenized_datasets = dataset.map(tokenize_function, batched=True)
# 将标签列重命名为Trainer要求的‘labels’
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 格式转换为PyTorch张量
tokenized_datasets.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
print(" 数据分词完成")

print("\n=== 第5步：定义训练参数 ===")
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    evaluation_strategy="epoch",     # 每个epoch后评估
    learning_rate=2e-5,              # 学习率
    per_device_train_batch_size=8,   # 训练批次大小
    per_device_eval_batch_size=8,    # 评估批次大小
    num_train_epochs=2,              # 训练2轮即可（演示用）
    weight_decay=0.01,               # 权重衰减
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
)

print("\n=== 第6步：创建Trainer并开始训练 ===")
small_train_dataset = tokenized_datasets["train"].select(range(3000))
small_eval_dataset = tokenized_datasets["test"].select(range(600))

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    tokenizer=tokenizer,
)

print("开始训练...这可能需要几分钟，请耐心等待")
trainer.train()

print("\n=== 第7步：保存训练好的模型 ===")
os.makedirs('./saved_bert_model', exist_ok=True)
model.save_pretrained('./saved_bert_model')
tokenizer.save_pretrained('./saved_bert_model')
print(" 模型已保存至 ‘./saved_bert_model’ 文件夹")
