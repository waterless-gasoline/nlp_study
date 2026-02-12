import os
import sys

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)


def main():
    """主函数"""
    print("BERT文本分类系统")
    print("=" * 50)

    # 检查模型文件
    model_file = "text_classifier/models/best_model.pth"

    if not os.path.exists(model_file):
        print("模型不存在，开始训练...")
        print("-" * 50)

        try:
            # 直接运行训练脚本
            from text_classifier.train import train
            train()
            print("\n训练完成，启动API服务...")
        except Exception as e:
            print(f"\n训练失败: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        print("模型已存在，启动API服务...")

    print("-" * 50)

    try:
        # 直接运行API服务
        import subprocess
        subprocess.run([sys.executable, "text_classifier/api_server.py"])
    except KeyboardInterrupt:
        print("\nAPI服务已停止")
    except Exception as e:
        print(f"启动API服务失败: {e}")


if __name__ == "__main__":
    main()