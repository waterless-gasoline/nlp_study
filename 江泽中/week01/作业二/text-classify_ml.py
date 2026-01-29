import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

dataset = pd.read_csv('../../dataset.csv', sep="\t", header=None, nrows=10000)
# print(dataset)

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
# print(input_sentence.values)

# vectorizer = CountVectorizer()

vectorizer = TfidfVectorizer(
    tokenizer=None,
    max_features=10000000000,
    min_df=2,
    max_df=0.9,
    ngram_range=(1, 2)  # 考虑1-gram和2-gram
)
vectorizer.fit(input_sentence.values)
X = vectorizer.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(X, dataset[1].values)


def classify(text : str):
    """
    对输入内容用机器学习对文本进行分类
    """
    # 先对输入内容进行分词，空格分隔
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vectorizer.transform([text_sentence])
    return model.predict(text_feature)[0]

if __name__ == '__main__':
    print(classify("导航去新疆"))

