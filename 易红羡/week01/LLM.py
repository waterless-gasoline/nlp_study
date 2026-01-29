import os
from openai import OpenAI

client = OpenAI(
    api_key="sk-d86fcc6844f84aa6bc475ed1e13aa07a",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
completion = client.chat.completions.create(
    model="qwen-flash-2025-07-28",
messages=[
        {"role": "user", "content": """帮我进行文本分类：导航到重庆
输出类别只可以从下方文本中进行选择，并且不显示如类别等的提示词：
Video-Play
FilmTele-Play
Music-Play
Radio-Listen
Alarm-Update
Travel-Query
HomeAppliance-Control
Weather-Query
Calendar-Query 
TVProgram-Play
Audio-play
Other"""},
    ]
)
print(completion.choices[0].message.content)
