import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.preprocessing import LabelEncoder
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os

# 加载和预处理数据
dataset_df = pd.read_csv("./dataset_bert.csv", sep=",", header=None)

# 初始化 LabelEncoder
lbl = LabelEncoder()
labels = lbl.fit_transform(dataset_df[1].values[:500])
texts = list(dataset_df[0].values[:500])

# 分割数据
x_train, x_test, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, stratify=labels
)

# 加载分词器
tokenizer = BertTokenizer.from_pretrained('../../../../models/google-bert/bert-base-chinese')


# 编码数据
def encode_texts(texts, labels=None):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )

    if labels is not None:
        return TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels, dtype=torch.long)
        )
    else:
        return encodings['input_ids'], encodings['attention_mask']


# 创建数据集
train_dataset = encode_texts(x_train, train_labels)
test_dataset = encode_texts(x_test, test_labels)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 加载模型
model = BertForSequenceClassification.from_pretrained(
    '../../../../models/google-bert/bert-base-chinese',
    num_labels=17
)

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 优化器
optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)


# 训练函数
def train_model(model, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc=f'训练 Epoch {epoch + 1}')
    for batch in progress_bar:
        input_ids, attention_mask, labels = batch
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        predictions = torch.argmax(outputs.logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        # 更新进度条
        progress_bar.set_postfix({
            'loss': loss.item(),
            'acc': correct / total
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total
    return avg_loss, accuracy


# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)

            total_loss += outputs.loss.item()
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_predictions, all_labels


# 预测函数
def predict_texts(model, texts):
    """预测输入文本的分类"""
    model.eval()

    # 编码文本
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=64,
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to(device)
    attention_mask = encodings['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        # 获取概率
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        max_probs = torch.max(probabilities, dim=1).values

        # 转换为标签文本
        predicted_labels = lbl.inverse_transform(predictions.cpu().numpy())

        # 返回结果
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'predicted_label': predicted_labels[i],
                'predicted_id': predictions[i].item(),
                'confidence': max_probs[i].item()
            })

    return results


# 创建保存目录
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

# 早停机制
best_accuracy = 0
patience = 3
patience_counter = 0
best_model_state = None
best_epoch = 0

# 训练循环
num_epochs = 4
print("开始训练...")
print(f"模型将保存到目录: {save_dir}")
print("-" * 50)

for epoch in range(num_epochs):
    # 训练
    train_loss, train_acc = train_model(model, train_loader, optimizer, epoch)
    print(f"Epoch {epoch + 1}: 训练损失 = {train_loss:.4f}, 训练准确率 = {train_acc:.4f}")

    # 评估
    val_loss, val_acc, _, _ = evaluate_model(model, test_loader)
    print(f"Epoch {epoch + 1}: 验证损失 = {val_loss:.4f}, 验证准确率 = {val_acc:.4f}")

    # ============ 保存每个epoch的模型 ============
    epoch_filename = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'train_accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc,
        'label_encoder': lbl,
        'tokenizer': tokenizer
    }, epoch_filename)
    print(f"💾 模型已保存到: {epoch_filename}")
    # ============================================

    # 早停逻辑
    if val_acc > best_accuracy + 0.001:
        best_accuracy = val_acc
        patience_counter = 0
        best_model_state = model.state_dict().copy()
        best_epoch = epoch + 1

        # 保存最佳模型
        best_filename = os.path.join(save_dir, f'best_model_epoch_{best_epoch}.pth')
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_accuracy': best_accuracy,
            'label_encoder': lbl,
            'tokenizer': tokenizer
        }, best_filename)
        print(f"🏆 保存最佳模型到: {best_filename}")
    else:
        patience_counter += 1
        print(f"⏳ 早停计数器: {patience_counter}/{patience}")

    # 检查早停
    if patience_counter >= patience:
        print("🛑 触发早停机制")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print(f"恢复最佳模型 (Epoch {best_epoch})")
        break

    print("-" * 50)

# 最终评估
print("\n最终评估结果:")
final_loss, final_acc, predictions, true_labels = evaluate_model(model, test_loader)
print(f"测试集损失: {final_loss:.4f}")
print(f"测试集准确率: {final_acc:.4f}")

# 保存最终模型
final_filename = os.path.join(save_dir, 'final_model.pth')
torch.save({
    'model_state_dict': model.state_dict(),
    'final_accuracy': final_acc,
    'final_loss': final_loss,
    'label_encoder': lbl,
    'tokenizer': tokenizer,
    'total_epochs': epoch + 1,
    'best_epoch': best_epoch
}, final_filename)
print(f"✅ 最终模型已保存到: {final_filename}")

# 列出保存的所有模型文件
print("\n已保存的模型文件:")
model_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
for model_file in sorted(model_files):
    filepath = os.path.join(save_dir, model_file)
    filesize = os.path.getsize(filepath) / (1024 * 1024)  # 转换为MB
    print(f"  - {model_file} ({filesize:.2f} MB)")

# # 示例：使用模型进行预测
# print("\n=== 预测示例 ===")
# test_texts = [
#     "这是一个测试文本",
#     "另一个测试样例",
#     "你好，今天天气真好"
# ]
#
# results = predict_texts(model, test_texts)
# for result in results:
#     print(f"文本: {result['text']}")
#     print(f"  预测标签: {result['predicted_label']}")
#     print(f"  置信度: {result['confidence']:.4f}")
#     print()


# 加载指定epoch模型进行预测的示例
def load_model_from_epoch(epoch_num):
    """加载指定epoch的模型"""
    model_path = os.path.join(save_dir, f'model_epoch_{epoch_num}.pth')
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)

        # 创建新模型实例
        loaded_model = BertForSequenceClassification.from_pretrained('google-bert/bert-base-chinese',
            num_labels=17
        )
        loaded_model.to(device)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])

        print(f"✅ 成功加载 Epoch {epoch_num} 的模型")
        print(f"  验证准确率: {checkpoint['val_accuracy']:.4f}")
        print(f"  训练准确率: {checkpoint['train_accuracy']:.4f}")

        return loaded_model
    else:
        print(f"❌ 找不到 Epoch {epoch_num} 的模型文件")
        return None


# 测试加载特定epoch模型
print("\n==================================================== 加载模型测试 ===============================================================")
# # 加载第1个epoch的模型
# model_epoch1 = load_model_from_epoch(1)
# if model_epoch1:
#     results = predict_texts(model_epoch1, ["测试文本"])
#     print(f"预测结果: {results[0]['predicted_label']}")

# 交互式预测
# while True:
#     user_input = input("\n请输入要分类的文本 (输入 'quit' 退出): ")
#     if user_input.lower() == 'quit':
#         break
#
#     if user_input.strip():
#         results = predict_texts(model, [user_input])
#         result = results[0]
#         print(f"预测结果: {result['predicted_label']}")
#         print(f"置信度: {result['confidence']:.4f}")

str = "一想到明天，我就对明天充满了希望"
results = predict_texts(model, [str])
result = results[0]
print(f"预测结果: {result['predicted_label']}")
print(f"置信度: {result['confidence']:.4f}")