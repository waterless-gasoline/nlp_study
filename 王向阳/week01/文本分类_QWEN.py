# 导入结巴分词库，用于中文文本的分词处理
import jieba
# 导入pandas库，用于数据读取和处理
import pandas as pd
# 导入OpenAI客户端，调用Qwen大模型
from openai import OpenAI

# 初始化Qwen大模型客户端
client = OpenAI(
    # 自己注册的api_key,禁止盗用
    api_key="sk-c99b66c8cfb441xxxxxbdd88",
    # 阿里云百炼兼容OpenAI的base_url
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取整个表格
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 定义Qwen大模型分类函数
def text_classify_using_qwen(text: str) -> str:
    """
    调用Qwen大模型完成文本分类，输出指定类别
    """
    completion = client.chat.completions.create(
        model="qwen-flash", 
        messages=[
            {"role": "user", "content": f"""帮我对文本进行分类，仅输出类别名称，不要多余内容。
待分类文本：{text}
可选类别列表：
FilmTele-Play、Video-Play、Music-Play、Radio-Listen、Alarm-Update、
Travel-Query、HomeAppliance-Control、Weather-Query、Calendar-Query、
TVProgram-Play、Audio-Play、Other
"""},
        ],
        temperature=0.0,  # 固定输出，避免随机性
    )
    # 去除结果中的多余空格/换行
    return completion.choices[0].message.content.strip()

# 遍历表格每一行，输出原始文本和Qwen预测结果
for idx in range(len(dataset)):
    # 获取表格第一列的原始文本
    original_text = dataset[0][idx]
    # 调用Qwen大模型获取分类结果
    qwen_pred_result = text_classify_using_qwen(original_text)

    # 输出结果
    print("待预测的文本", original_text)
    print("Qwen模型预测结果: ", qwen_pred_result)
    print("-" * 50)
