# test.py
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 1. 加载模型和分词器
model = BertForSequenceClassification.from_pretrained('./saved_bert_model')
tokenizer = BertTokenizer.from_pretrained('./saved_bert_model')
model.eval()  # 设置为评估模式
print("模型加载完毕")

# 2. 准备几条新的文本
test_texts = [
    "Microsoft is a large multinational technology corporation based in Redmond.",  # 应该属于‘Company’
    "The Louvre Museum is the world‘s largest art museum and a historic monument in Paris.",  # 应该属于‘Building’
    "Leo Tolstoy was a Russian writer who is regarded as one of the greatest authors of all time."  # 应该属于‘Artist’
]

# DBpedia的14个类别
label_names = [
    'Company', 'EducationalInstitution', 'Artist', 'Athlete', 'OfficeHolder',
    'MeanOfTransportation', 'Building', 'NaturalPlace', 'Village', 'Animal',
    'Plant', 'Album', 'Film', 'WrittenWork'
]

# 3. 对每条文本进行预测
print("\n--- 开始预测 ---")
for text in test_texts:
    # 用分词器处理文本
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)

    # 让模型进行预测
    with torch.no_grad():  # 关闭梯度计算，节省资源
        outputs = model(**inputs)

    # 获取预测结果（概率最大的那个类别）
    predictions = torch.argmax(outputs.logits, dim=-1)
    predicted_class_id = predictions.item()
    predicted_label = label_names[predicted_class_id]

    # 打印结果
    print(f"\n文本: {text}")
    print(f"   -> 模型预测的类别ID: {predicted_class_id}")
    print(f"   -> 对应的类别名称: 【{predicted_label}】")
