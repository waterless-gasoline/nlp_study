import torch
import pandas as pd
import os


class Config:
    # 数据路径
    train_path = 'datasets/train.csv'
    model_path = 'D:\\AI\\bert-base-chinese'

    # 训练参数
    max_length = 128
    batch_size = 16
    learning_rate = 2e-5
    num_epochs = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 保存路径
    save_dir = 'models/'
    best_model_path = 'models/best_model.pth'

    # 类别映射
    category_mapping = {
        0: "科技",
        1: "金融",
        2: "医疗",
        3: "教育",
        4: "旅游",
        5: "其他"
    }

    # 不在这里初始化num_classes，在需要时获取
    num_classes = None

    @classmethod
    def get_num_classes(cls):
        """获取类别数量"""
        if cls.num_classes is not None:
            return cls.num_classes

        try:
            if os.path.exists(cls.train_path):
                df = pd.read_csv(cls.train_path)
                if 'label' in df.columns:
                    cls.num_classes = df['label'].nunique()
                    print(f"从数据中读取类别数: {cls.num_classes}")
                    return cls.num_classes
        except Exception as e:
            print(f"读取类别数失败: {e}")

        # 默认值
        cls.num_classes = len(cls.category_mapping)
        print(f"使用默认类别数: {cls.num_classes}")
        return cls.num_classes

    @classmethod
    def initialize(cls):
        """初始化配置"""
        cls.get_num_classes()
        print(f"配置初始化完成，类别数: {cls.num_classes}")

# 可选：在导入时自动初始化
# Config.initialize()