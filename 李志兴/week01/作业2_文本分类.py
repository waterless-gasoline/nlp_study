import jieba #做文本分词
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer #词频统计
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI #调用大语言模型

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
# print(dataset[1].unique())#输出csv数据集第二列所有类别
# 对 dataset 第 0 列中的每一条中文句子，先用 jieba 分词，再用空格拼接成字符串，生成一个新的 Series，并赋值给 input_sententce。
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

# 将不定长文本转换为维度相同的向量
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values) #统计
input_feature = vector.transform(input_sententce.values) #100*词表大小

model = KNeighborsClassifier() #KNN模型
model.fit(input_feature, dataset[1].values)

client =OpenAI(
    api_key="sk-d942676exxxx9b79e726",
    base_url="https://api.deepseek.com/v1" #deepseek模型
)

def text_calssify_using_ml(text:str)->str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def tstx_classify_using_llm(text:str)->str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role":"user",
                  "content":f"""请对文本{text}进行分类，输出类别只能从下面进行选择(除了类别之外不要有其他输出内容)：
                  ['Travel-Query' 'Music-Play' 'FilmTele-Play' 'Video-Play' 'Radio-Listen'
                   'HomeAppliance-Control' 'Weather-Query' 'Alarm-Update' 'Calendar-Query'
                   'TVProgram-Play' 'Audio-Play' 'Other']
                  """}]
    )
    return response.choices[0].message.content


if __name__ =="__main__":
    str1=text_calssify_using_ml("帮我导航到天安门")
    print(f"KNN模型预测结果: {str1}")
    print(60*'*')
    str2=tstx_classify_using_llm("帮我导航到天安门")
    print(f"大语言模型预测结果：{str2}")
