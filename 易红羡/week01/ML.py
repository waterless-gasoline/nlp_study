import jieba
import pandas as pd
from sklearn.feature_extraction.text import  CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv(filepath_or_buffer="dataset.csv", sep="\t", names=["文本内容","文本类型"], nrows=10000)
# print(dataset.head(10))
# print(dataset["文本类型"].value_counts())
# 创建数据集

text_sentence = dataset["文本内容"].apply(lambda x:" ". join(jieba.lcut(x)))
# print(text_sentence[10])
# 对数据集的文本内容词汇进行分词

vector=CountVectorizer()  # 文本特征提取
vector.fit(text_sentence.values)  # 构建学习词汇表
text_feature=vector.transform(text_sentence.values)  # 转换文本为矩阵
# print(text_feature.shape)

model = KNeighborsClassifier()
model.fit(text_feature, dataset["文本类型"].values)
# print(model)
# KNN模型训练

test_query = "导航到重庆"
test_sentence = " ".join(jieba.lcut(test_query))
# print(test_sentence)
test_feature = vector.transform([test_sentence])
print("待预测的文本：", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
