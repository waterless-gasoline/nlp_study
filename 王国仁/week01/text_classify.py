import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier # KNN
from sklearn.linear_model import LogisticRegression # 逻辑回归
from sklearn.tree import DecisionTreeClassifier # 决策树
from openai import OpenAI
from fastapi import FastAPI

# 1. 数据集读取
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None, nrows=None)
# 分词
input_sentence = dataset[0].apply(lambda x:" ".join(jieba.lcut(x)))

# 2. 词频统计 特征提取
vector = CountVectorizer() # 文本特征提取
vector.fit(input_sentence) # 统计词表
input_feature = vector.transform(input_sentence.values) # 进行转换
# print(input_feature[1])

# 3. 模型训练
# KNN模型
# model = KNeighborsClassifier()
# model.fit(input_feature, dataset[1].values)
# 逻辑回归模型
# model = LogisticRegression()
# model.fit(input_feature, dataset[1].values)
# 决策树模型
model = DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)



app = FastAPI()

@app.get("/")
def hello():
    return "Hello World"

@app.get("/text_classify/ml")
def test_classify_use_ml(text: str) -> str:
    """
    使用机器学习模型进行文本分类任务
    :param text: 输入文本
    :return: 推理结果
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

@app.get("/text_classify/llm")
def text_classify_use_llm(text: str) -> str:
    """
    使用 LLM 模型进行文本分类任务
    :param text: 输入文本
    :return: 推理结果
    """
    client = OpenAI(
        api_key="sk-f4e927fa167xxxxd9119bed",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {"role": "user", "content": f"""
            帮我进行文本分类{text}，并且输出的分类只能从下面的类别中获取
            FilmTele-Play            
            Video-Play               
            Music-Play              
            Radio-Listen           
            Alarm-Update        
            Travel-Query        
            HomeAppliance-Control  
            Weather-Query          
            Calendar-Query      
            TVProgram-Play      
            Audio-Play       
            Other  
            """}
        ]
    )
    return completion.choices[0].message.content

# if __name__ == '__main__':
#     print("机器学习: ", test_classify_use_ml("帮我导航到奥体中心"))
#     print("大模型: ", text_classify_use_llm("帮我导航到奥体中心"))
