import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import os

#
# dataset_em = pd.read_csv('dataset_emotion.csv', sep=",", header=None)
#
# # print(dataset_em)
#
# lbl = LabelEncoder()
# labels_encoded = lbl.fit_transform(dataset_em[1].values)
#
# # print(lbl.classes_)
# counts = dataset_em[1][:888].value_counts()
# print(counts)
# x_train, x_test, train_label, test_label = train_test_split(
#     list(dataset_em[0].values[:888]),
#     labels_encoded[:888],
#     test_size=0.2,
#     random_state=42,
#     shuffle=True,
#     stratify=dataset_em[1][:888].values
# )
#
# tokenizer = BertTokenizer.from_pretrained('../models/bert-base-chinese')
#
# train_encodings = tokenizer(list(x_train), truncation=True, padding=True, max_length=64)
# test_encodings = tokenizer(list(x_test), truncation=True, padding=True, max_length=64)
#
#
# class Dataset_Emotion(Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
#
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         # 关键修改：后面加上 .long()
#         item['labels'] = torch.tensor(self.labels[idx]).long()
#         return item
#
#     def __len__(self):
#         return len(self.labels)
#
#
# train_dataset = Dataset_Emotion(train_encodings, train_label)
# test_dataset = Dataset_Emotion(test_encodings, test_label)
#
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
#
# model = BertForSequenceClassification.from_pretrained('../models/bert-base-chinese', num_labels=len(lbl.classes_))
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model.to(device)
#
# optim = torch.optim.AdamW(model.parameters(), lr=1e-5)
#
#
# def flat_accuracy(preds, labels):
#     pred_flat = np.argmax(preds, axis=1).flatten()
#     labels_flat = labels.flatten()
#     return np.sum(pred_flat == labels_flat) / len(labels_flat)
#
#
# def train(epoch):
#     model.train()
#     total_train_loss = 0
#     iter_num = 0
#     total_iter = len(train_loader)
#     for batch in train_loader:
#         optim.zero_grad()
#         input_ids = batch["input_ids"].to(device)
#         attention_mask = batch["attention_mask"].to(device)
#         labels = batch["labels"].to(device)
#
#         outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#         loss = outputs.loss  # 推荐写法
#
#         total_train_loss += loss.item()
#         loss.backward()
#
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#         optim.step()
#
#         iter_num += 1
#         # 2. 建议将打印频率改小，比如每 10 次打印一次
#         if (iter_num % 10 == 0):
#             print("epoch: %d, iter_num: %d, loss: %.4f, %.2f%%" % (
#                 epoch, iter_num, loss.item(), iter_num / total_iter * 100))
#
#     print("Epoch: %d, Average training loss: %.4f" % (epoch, total_train_loss / len(train_loader)))
#
#
# def validation():
#     model.eval()
#     total_eval_loss = 0
#     total_eval_accuracy = 0
#
#     with torch.no_grad():
#         for batch in test_loader:
#             with torch.no_grad():
#                 input_ids = batch["input_ids"].to(device)
#                 attention_mask = batch["attention_mask"].to(device)
#                 labels = batch["labels"].to(device)
#                 outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
#             loss = outputs[0]
#             logits = outputs[1]
#             total_eval_loss += loss.item()
#             logits = logits.detach().cpu().numpy()
#             label_ids = labels.to(device).numpy()
#             total_eval_accuracy += flat_accuracy(logits, label_ids)
#     # 计算平均准确率
#     avg_val_accuracy = total_eval_accuracy / len(test_loader)
#     print("Accuracy: %.4f" % (avg_val_accuracy))
#     print("Average testing loss: %.4f" % (total_eval_loss / len(test_loader)))
#     print("-------------------------------")
#
#     # -------------------------- 5. 主训练循环 --------------------------
#     # 循环训练4个epoch
#
#
# for epoch in range(4):
#     print("------------Epoch: %d ----------------" % epoch)
#     # 训练模型
#     train(epoch)
#     # 验证模型
#     validation()
#
#     # --- 改进后的保存逻辑 ---
#     # 1. 定义保存目录（使用相对路径）
#     output_dir = f"./results/bert_checkpoint_epoch_{epoch}"
#
#     # 2. 如果文件夹不存在则创建
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#         print(f"创建文件夹: {output_dir}")
#
#     # 3. 保存模型和分词器
#     model.save_pretrained(output_dir)
#     tokenizer.save_pretrained(output_dir)
#
#     print(f"✅ Epoch {epoch} 模型已成功保存至: {os.path.abspath(output_dir)}")


# 1. 设置路径（指向你保存模型的那个文件夹）
model_path = "./results/bert_checkpoint_epoch_3"  # 修改为你最后保存的文件夹名

# 2. 加载模型和分词器
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 3. 准备设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()  # 切换到预测模式


# 4. 定义预测函数
def predict_emotion(text):
    # 对输入文本进行分词和编码
    inputs = tokenizer(text, truncation=True, padding=True, max_length=64, return_tensors="pt")
    dataset = pd.read_csv("dataset_emotion.csv", sep=",", header=None)
    print(dataset)

    # 初始化并拟合标签编码器，将文本标签（如“体育”）转换为数字标签（如0, 1, 2...）
    lbl = LabelEncoder()
    lbl.fit(dataset[1].values[0:888])
    # 移动数据到设备
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 获取概率最大的类别索引
    pred_idx = torch.argmax(logits, dim=1).item()

    return lbl.classes_[pred_idx]


# 5. 测试
test_text = "你这人怎么这样子！"
result = predict_emotion(test_text)
print(f"输入文本: {test_text}")
print(f"预测情感: {result}")
