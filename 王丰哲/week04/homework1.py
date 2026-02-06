import os
# 设置 Hugging Face 镜像地址 (必须在导入 transformers 之前设置)
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# ---------------------------------------------------------
# 1. 准备自定义数据集 (电商意图识别)
# ---------------------------------------------------------
# 定义 3 个类别：物流咨询、产品咨询、退换货
data = [
    # 物流咨询
    ("我的快递怎么还没到？", "物流咨询"),
    ("请问发什么快递？", "物流咨询"),
    ("查一下单号，好几天了没更新", "物流咨询"),
    ("大概几天能送到北京？", "物流咨询"),
    ("发货了吗？为什么显示待揽收", "物流咨询"),
    
    # 产品咨询
    ("这款衣服偏大还是偏小？", "产品咨询"),
    ("这个电脑内存可以升级吗？", "产品咨询"),
    ("红色款还有货吗？", "产品咨询"),
    ("保质期是多久？", "产品咨询"),
    ("这个材质是纯棉的吗？", "产品咨询"),
    
    # 退换货
    ("衣服破了个洞，我要退货", "退换货"),
    ("买了不合适，怎么申请换货", "退换货"),
    ("退款什么时候到账？", "退换货"),
    ("发错货了，我要退款", "退换货"),
    ("质量太差了，我要投诉并退货", "退换货")
]


data = data * 5 

df = pd.DataFrame(data, columns=["text", "label"])

print(f"数据集大小: {len(df)}")
print(df.head())

# ---------------------------------------------------------
# 2. 数据预处理 (参考 10_BERT文本分类.py 的逻辑)
# ---------------------------------------------------------

# 将文字标签转换为数字 (0, 1, 2)
le = LabelEncoder()
df['label_id'] = le.fit_transform(df['label'])

# 划分训练集和验证集 (80% 训练, 20% 验证)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), 
    df['label_id'].tolist(), 
    test_size=0.2, 
    random_state=42
)

# 加载 BERT 分词器
model_name = "google-bert/bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 对文本进行分词编码
# truncation=True: 截断过长文本, padding=True: 补齐长度
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=64)
val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=64)

# 转换为 Hugging Face 的 Dataset 格式，这是 Trainer API 要求的格式
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
})

# ---------------------------------------------------------
# 3. 初始化模型
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"正在使用的设备: {device}")

# 加载用于分类的 BERT 模型
# num_labels=3 对应我们的 3 个类别 (物流、产品、退换货)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.to(device)

# ---------------------------------------------------------
# 4. 设置训练参数 (Trainer API)
# ---------------------------------------------------------

# 定义评估指标函数，计算准确率
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}

# 训练参数设置
training_args = TrainingArguments(
    output_dir='./results',          # 输出目录
    num_train_epochs=5,              # 训练 5 轮
    per_device_train_batch_size=2,   # 批次大小 (因为数据少，设小一点)
    per_device_eval_batch_size=2,
    warmup_steps=10,                 # 预热步数
    weight_decay=0.01,               # 权重衰减 (防止过拟合)
    logging_dir='./logs',            # 日志目录
    logging_steps=10,
    eval_strategy="epoch",           # 每个 epoch 评估一次
    save_strategy="epoch",
    load_best_model_at_end=True,     # 训练结束加载最好的模型
    report_to="none"                 # 关闭 wandb 等第三方日志工具
)

# 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# ---------------------------------------------------------
# 5. 开始训练
# ---------------------------------------------------------
print("\n--- 开始训练 ---")
trainer.train()

# ---------------------------------------------------------
# 6. 使用新数据进行预测 (推理)
# ---------------------------------------------------------
def predict_intent(text):
    model.eval() # 切换到评估模式
    
    # 1. 分词处理
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    # 2. 放到 GPU/CPU 上
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 3. 模型预测
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 4. 获取概率最大的类别索引
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()
    
    # 5. 将数字索引转回文字标签
    predicted_label = le.inverse_transform([predicted_class_id])[0]
    return predicted_label

print("\n--- 测试模型效果 ---")
test_samples = [
    "我想把这件衣服退了，颜色不喜欢", # 预期：退换货
    "这个手机电池也是原装的吗？",     # 预期：产品咨询
    "到了三天了，快递一直不动是怎么回事" # 预期：物流咨询
]

for sample in test_samples:
    prediction = predict_intent(sample)
    print(f"输入文本: {sample}")
    print(f"预测意图: {prediction}\n")
