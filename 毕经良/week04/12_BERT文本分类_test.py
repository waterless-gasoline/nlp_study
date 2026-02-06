import pandas as pd # 导入 pandas 库，用于数据处理
import torch # 导入 PyTorch 库
from sklearn.model_selection import train_test_split # 导入数据集分割工具
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments # 导入 Hugging Face Transformers 组件
from sklearn.preprocessing import LabelEncoder # 导入标签编码工具
from datasets import Dataset # 导入 Hugging Face Datasets 库
import numpy as np # 导入 numpy 库

# 加载和预处理数据
# 读取数据集文件，假设是制表符分隔的 CSV 文件
dataset_df = pd.read_csv("./Multi-Emotion.csv", sep=",", header=None)

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 拟合数据并转换前500个标签，得到数字标签 (例如: "体育" -> 0, "财经" -> 1)
# 这里的 [:500] 是为了快速演示，实际项目中通常使用全部数据
labels = lbl.fit_transform(dataset_df[1].values[:1000])
# 提取前500个文本内容
texts = list(dataset_df[0].values[:1000])

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致 (分层采样)
)

# 从预训练模型加载分词器和模型
# 加载 BERT 中文分词器
tokenizer = BertTokenizer.from_pretrained('/Users/jlbi/ai_study/models/google-bert/bert-base-chinese')
# 加载 BERT 序列分类模型
# num_labels=17: 指定分类任务的类别数量 (根据数据集实际情况调整)
model = BertForSequenceClassification.from_pretrained('/Users/jlbi/ai_study/models/google-bert/bert-base-chinese', num_labels=17)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断 (超过 max_length)
# padding=True：对齐所有序列长度，填充到 max_length (不足补 0)
# max_length=64：最大序列长度 (BERT 最大支持 512，这里设为 64 节省资源)
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)

# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
# Dataset 对象优化了内存使用，并兼容 Trainer API
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码 (1为有效，0为padding)
    'labels': train_labels                               # 对应的数字标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 定义用于计算评估指标的函数
# Trainer 会在评估时调用此函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    # axis=-1 表示在最后一个维度 (类别维度) 上取最大值
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='./results',              # 训练输出目录，用于保存模型 checkpoints 和日志
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，前 500 步学习率线性增加
    weight_decay=0.01,                   # 权重衰减系数，L2 正则化，防止过拟合
    logging_dir='./logs',                # TensorBoard 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次训练日志
    eval_strategy="epoch",               # 评估策略：每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 保存策略：每训练完一个 epoch 保存一次模型 checkpoint
    load_best_model_at_end=True,         # 训练结束后自动加载验证集上效果最好的模型
)

# 实例化 Trainer
# Trainer 封装了训练、评估和预测的完整循环
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 深度学习训练过程，数据获取，epoch batch 循环，梯度计算 + 参数更新

# 开始训练模型
# 这会自动处理 GPU/CPU 分配、梯度下降、反向传播等细节
trainer.train()

# 在测试集上进行最终评估，输出评估指标 (如 accuracy)
trainer.evaluate()

# trainer 是比较简单，适合训练过程比较规范化的模型
# 如果我要定制化训练过程，trainer无法满足
# 需要手动编写训练循环 (类似于 PyTorch 原生方式)


# agent -》 llm

# 上下文压缩 -》 文本摘要

# torchserver -》 http 服务

# 花心思 花钱 构建领域大模型，不如等下一个qwen版本

# ==========================================
# 新增测试逻辑：测试训练好的模型效果
# ==========================================

# 定义测试文本
test_texts = [ 
    "我昨天梦到有一只超大蜘蛛在追着我跑！",
    "今天去的那家餐厅真的很脏。",
    "我终于写出了这个程序！",
    "今天老师骂了我。",
    "今年的夏天似乎比较晚来。",
    "为什么他们能背后这样批评人，真是不礼貌。",
    "他们两个竟然牵手了！",
 ]

print("\n" + "="*30)
print("正在对自定义测试文本进行预测...")
print("="*30)

# 确保模型处于评估模式
model.eval()

# 对测试文本进行编码
# return_tensors='pt' 返回 PyTorch 张量
inputs = tokenizer(test_texts, padding=True, truncation=True, max_length=64, return_tensors="pt")

# 将输入数据移动到模型所在的设备 (GPU 或 CPU)
# model.device 是模型当前所在的设备
inputs = {k: v.to(model.device) for k, v in inputs.items()}

# 禁用梯度计算，节省显存并加速
with torch.no_grad():
    # 前向传播
    outputs = model(**inputs)
    # 获取 logits (未归一化的概率)
    logits = outputs.logits
    # 获取预测类别的索引 (在最后一个维度上取最大值)
    predictions = torch.argmax(logits, dim=-1)

# 将预测的数字索引转换回原始的文本标签
# predictions 是 tensor，需要转为 numpy 数组
predicted_labels = lbl.inverse_transform(predictions.cpu().numpy())

# 打印预测结果
for text, label in zip(test_texts, predicted_labels):
    print(f"文本: {text}")
    print(f"预测类别: {label}")
    print("-" * 20)
