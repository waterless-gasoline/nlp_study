import jieba
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

test_text = [
    "帮我导航到天安门",
    "播放周杰伦的歌曲",
    "明天北京天气怎么样",
    "设置一个明天早上8点的闹钟",
    "今天是星期几",
    "我想看最近的电影",
    "把空调温度调低2度",
    "查询从上海到北京的机票"
]

dataset = pd.read_csv("dataset.csv",    #   文件或文件路径
                      sep="\t", #   分隔符
                      header=None,  #   指定列名 '0' 第一行是列名 'None' 文件没有列名 'n' 指定行数
                      nrows=12102  #  指定数量
                      # names    #   自定义列名
                      # encoding #   文件编码
                      # usecols = [8,10]  #   指定读取位置
                      # engine   #   编译器
                      )
print(dataset[1].value_counts())

input_sententce = dataset[0].apply(lambda x:",".join(jieba.lcut(x)))
print(input_sententce)

vector = CountVectorizer()
vector.fit(input_sententce)
input_feature = vector.transform(input_sententce)
# print(input_feature)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-8b47344f618342eaa3fdbab260e9e7", # 账号绑定，用来计费的

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def test_ml(text):
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def test_llm(text):
    completion = client.chat.completions.create(
        model = "qwen-flash",

        messages = [
            {
                "role":"user",
                "content":f"""请帮我进行文本分类{text}
                输出的类别只能从如下中进行选择，除了类别之外下列的类别请给出最合适的类别。
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
                """
            }
        ]
    )

    return completion.choices[0].message.content

for i, text in enumerate(test_text, 1):
    #   分割线方便查看
    print(f"\n[{i}/{len(test_text)}] 测试文本: '{text}'")
    print("-" * 40)

    #   机械输出
    ml_result = test_ml(text)
    print("ml:"+ml_result)

    #   大模型输出
    llm_result = test_llm(text)
    print("llm:"+llm_result)
