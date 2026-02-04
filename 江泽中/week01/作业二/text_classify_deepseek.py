from http.client import responses

from openai import OpenAI

client = OpenAI(
    api_key="63929c25xxxxxf0b16155dfe",
    base_url="https://api.deepseek.com")

# def text_classify_deepseek(text : str) -> str:
#     responses = client.chat.completions.create(
#         model="deepseek-chat",
#         messages=[
#         {"role": "user", "content": f"""请帮我进行文本分类: {text}"
#                                     "输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。"
#                                     "FilmTele-Play"
#                                     "Video-Play"
#                                     "Music-Play"
#                                     "Radio-Listen"
#                                     "Alarm-Update"
#                                     "Travel-Query"
#                                     "HomeAppliance-Control"
#                                     "Weather-Query"
#                                     "Calendar-Query"
#                                     "TVProgram-Play"
#                                     "Audio-Play"
#                                     "Other"""},
#     ],
#     stream=False
#     )
#     return response.choices[0].message.content

if __name__ == '__main__':
    print(text_classify_deepseek("导航去新疆"))
