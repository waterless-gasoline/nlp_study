import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI

MAX_DATASET_ROWS = 1000 # 控制内存占用
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10)
# print(dataset[1].value_counts())
texts =  dataset[0] #文本集

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)


client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-3a4d832ccb3d4b4cae45bc239d0c4b6c", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_calssify_using_ml(text: str) -> str:
    """
    文本分类（机器学习），输入文本完成类别划分
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[{"role": "system", "content": "你是一个善于根据文本进行分类的工程师"}, # 给大模型的命令，角色的定义
            {"role": "user", "content": f"""帮我进行文本分类：{text},
            类比只能从下面给定的选项里面选,只给出来类比就行了,其他不需要任何的解释
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
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    if __name__ == "__main__":
        print("机器学习: ", text_calssify_using_ml("帮我导航到天安门"))
        print("大语言模型: ", text_calssify_using_llm("帮我导航到天安门"))
        print("机器学习: ", text_calssify_using_ml("我要听音乐电台"))
        print("大语言模型: ", text_calssify_using_llm("我要听音乐电台"))
        for i in texts:
            print ("判断模型:机器学习: ",f"文本内容是:{i}","类型是:", text_calssify_using_ml(i))
            print("判断模型:千问: ", f"文本内容是:{i}", "类型是:", text_calssify_using_llm(i))
