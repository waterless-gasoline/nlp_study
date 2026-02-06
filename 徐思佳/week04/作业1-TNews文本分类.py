import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# TNews 数据集
# 今日头条新闻分类，15分类，覆盖民生，文化，娱乐，体育等
# 训练集约5.3w, 测试集1w, 验证集1w
# 数据集从魔搭社区下载
#      pip install modelscope
#      modelscope download --dataset C-MTEB/TNews-classification

'''
0    上课时学生手机响个不停，老师一怒之下把手机摔了，家长拿发票让老师赔，大家怎么看待这种事？      7
1  商赢环球股份有限公司关于延期回复上海证券交易所对公司2017年年度报告的事后审核问询函的公告       4
2                通过中介公司买了二手房，首付都付了，现在卖家不想卖了。怎么处理？             5
3                             2018年去俄罗斯看世界杯得花多少钱？                       10
4                           剃须刀的个性革新，雷明登天猫定制版新品首发                    8
'''

# ========== 1. 加载数据 ==========
df = pd.read_parquet(path="TNews/train.parquet", engine="pyarrow", columns=["text", "label"])

num_classes = len(df["label"].unique())
print(num_classes)
# 查看每个类别数量
print(df["label"].value_counts())

# 考虑cpu效率，减少数据: 1. 控制类别数量，2. 只保留10%数据
# 去掉多个类别，只保留5个类别
remain_class = [0, 1, 2, 3, 4]
df = df[df["label"].isin(remain_class)]
df = df.sample(frac=0.1, random_state=42)

print(df["label"].value_counts())
# 4    511
# 2    481
# 1    445
# 3    396
# 0    103

# ========== 2. 划分训练测试集 ==========
x_train, x_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.2, random_state=42)

# ========== 3. 加载下载好的模型 ==========
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese', num_labels=num_classes)

train_encodings = tokenizer(x_train.tolist(), truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test.tolist(), truncation=True, padding=True, max_length=64)

# ========== 4. 处理数据便于训练时加载 ==========
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': y_train.tolist()
})

test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': y_test.tolist()
})

# ========== 5. 评估指标 ==========
def eval_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": (predictions == labels).mean()}

# ========== 6. 模型训练 ==========
# 6.1 训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

# 6.2 训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=eval_metrics,
)

# 6.3 训练
trainer.train()
trainer.evaluate()


# {'eval_loss': 1.3008118867874146, 'eval_accuracy': 0.7319587628865979, 'eval_runtime': 21.3253, 'eval_samples_per_second': 18.194, 'eval_steps_per_second': 1.172, 'epoch': 1.0}
# {'eval_loss': 0.6935203075408936, 'eval_accuracy': 0.7835051546391752, 'eval_runtime': 12.0023, 'eval_samples_per_second': 32.327, 'eval_steps_per_second': 2.083, 'epoch': 2.0}
# {'eval_loss': 0.6983920335769653, 'eval_accuracy': 0.7860824742268041, 'eval_runtime': 22.7984, 'eval_samples_per_second': 17.019, 'eval_steps_per_second': 1.097, 'epoch': 3.0}
# {'eval_loss': 0.8093361854553223, 'eval_accuracy': 0.7654639175257731, 'eval_runtime': 9.1269, 'eval_samples_per_second': 42.512, 'eval_steps_per_second': 2.739, 'epoch': 4.0}
