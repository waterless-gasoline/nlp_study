import random
import csv
import json

# -------------------------- 1. 配置数据集参数（可自定义） --------------------------
# 分类标签（可增删、修改，比如改成数字[0,1,2,3]）
LABELS = ["积极", "消极", "工作", "生活"]
# 每类标签的样本数量
SAMPLES_PER_LABEL = 500
# 数据集保存路径
CSV_SAVE_PATH = "text_classify_dataset.csv"
JSON_SAVE_PATH = "text_classify_dataset.json"

# -------------------------- 2. 文本模板（模拟真实表述，可扩展） --------------------------
# 每类标签对应的文本模板，会随机拼接生成不同样本
TEXT_TEMPLATES = {
    "积极": [
        "今天的心情真的太{}了！", "这次{}特别顺利，太开心了", "遇到了超{}的人，感觉世界很美好",
        "终于完成了{}，成就感拉满", "{}的风景超美，治愈一切不开心", "收到了{}的惊喜，幸福感爆棚",
        "尝试了{}，结果超出预期的好", "身边的人都很{}，太温暖了", "今天的{}超棒，一整天都有活力",
        "{}的经历让我收获满满，特别满足"
    ],
    "消极": [
        "今天的心情糟透了，{}真的很让人无语", "做{}一直失败，感觉自己好没用", "遇到了{}的事，一整天都郁闷",
        "{}的天气让人心烦，什么都不想做", "被{}影响了心情，特别委屈", "{}的结果太让人失望了",
        "一整天都很{}，提不起任何精神", "{}的经历让我特别难受", "身边的{}事，让人很无奈",
        "做{}总出问题，心态崩了"
    ],
    "工作": [
        "今天要完成{}的项目报告，得加班了", "开了一上午的{}会议，讨论项目进度", "对接{}客户，确认合作细节",
        "整理{}的工作数据，做月度总结", "学习{}的办公技能，提升工作效率", "和{}同事协作，完成任务",
        "修改{}的方案，满足领导要求", "跟进{}的项目进度，确保按时交付", "处理{}的工作邮件，回复需求",
        "制定{}的工作计划，安排下周任务"
    ],
    "生活": [
        "周末和朋友去{}逛街，买了很多东西", "晚上做了{}的家常菜，味道超棒", "早上起来去{}跑步，锻炼身体",
        "看了{}的电影，剧情特别精彩", "养的{}开花了，特别好看", "和家人去{}旅行，放松心情",
        "收拾{}的房间，整理家务", "煮了{}的奶茶，解腻又好喝", "看了{}的书，收获很多",
        "遛{}的时候，遇到了可爱的小伙伴"
    ]
}

# 文本模板填充词（丰富样本多样性）
FILL_WORDS = {
    "积极": ["开心", "愉快", "幸运", "暖心", "顺利", "美好", "惊喜", "满足"],
    "消极": ["糟糕", "倒霉", "烦心", "无语", "失望", "委屈", "糟糕", "压抑"],
    "工作": ["产品", "市场", "运营", "技术", "客户", "项目", "财务", "行政"],
    "生活": ["超市", "公园", "厨房", "影院", "绿植", "郊外", "卧室", "阳台"]
}

# -------------------------- 3. 生成带标注的数据集 --------------------------
def generate_text_dataset():
    dataset = []
    for label in LABELS:
        templates = TEXT_TEMPLATES[label]
        fill_words = FILL_WORDS[label]
        for _ in range(SAMPLES_PER_LABEL):
            # 随机选模板+随机选填充词，生成唯一文本
            template = random.choice(templates)
            fill_word = random.choice(fill_words)
            text = template.format(fill_word)
            # 每条样本包含：文本、标签（可新增字段，如id）
            dataset.append({"text": text, "label": label})
    # 随机打乱数据集，避免同类样本连续
    random.shuffle(dataset)
    return dataset

# -------------------------- 4. 保存数据集（CSV/JSON） --------------------------
def save_dataset(dataset):
    # 保存为CSV（推荐，Pandas/Scikit-learn可直接读取）
    with open(CSV_SAVE_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["text", "label"])
        writer.writeheader()  # 写入表头
        writer.writerows(dataset)
    print(f"CSV格式数据集已保存：{CSV_SAVE_PATH}")

    # 保存为JSON（适合跨语言使用）
    with open(JSON_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print(f"JSON格式数据集已保存：{JSON_SAVE_PATH}")

# -------------------------- 5. 主函数执行 --------------------------
if __name__ == "__main__":
    # 生成数据集
    text_dataset = generate_text_dataset()
    # 保存数据集
    save_dataset(text_dataset)
    # 控制台打印前5条样本，预览效果
    print("\n数据集前5条样本预览：")
    for i, sample in enumerate(text_dataset[:5]):
        print(f"{i+1}. 文本：{sample['text']} | 标签：{sample['label']}")