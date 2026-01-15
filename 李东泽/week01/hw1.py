
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=1000)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

# 对文本进行提取特征 默认是使用标点符号分词 文本向量化
vector = CountVectorizer()
# 将数据集中出现词频进行学习统计
vector.fit(input_sententce.values)
# 用上面学习到的词汇表将数据集转换成向量
input_feature = vector.transform(input_sententce.values)

# 模型初始化 创建KNN模型
model = KNeighborsClassifier()
# 训练模型
model.fit(input_feature, dataset[1].values)

def Fenxi(text:list):
    test_sentence = [" ".join(jieba.lcut(x)) for x in text]
    test_feature = vector.transform(test_sentence)
    return model.predict(test_feature)

if __name__ == "__main__":
    sentence = input("请输入需要预测的语句(回车代表下一句，若需要暂停，请输入end)：")
    sentences = []
    sentences.append(sentence)
    while sentence != "end":
        sentence = input("请输入下一句（输入end）：")
        sentences.append(sentence)
    print(Fenxi(sentences))


