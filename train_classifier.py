import os
import sys
import json
import csv
from pathlib import Path

# 确保能正确导入 Utils 里的模块
project_root = str(Path(__file__).parent.absolute())
sys.path.append(project_root)

from utils.Classifier.classifier import TextClassifier
from utils.Classifier.data_utils import DataAugmenter

# 读取配置
def load_config():
    config_path = os.path.join(project_root, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 新增：专门用于读取 CSV 数据的函数
def load_training_data(csv_path):
    questions = []
    labels = []
    # 使用 utf-8-sig 防止 Windows 下的 BOM 乱码
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 过滤掉空行防报错
            text = row.get('text', '').strip()
            label_str = row.get('label', '').strip()
            
            if not text or not label_str:
                continue
                
            questions.append(text)
            labels.append(int(label_str))
            
    return questions, labels

def main():
    CONFIG = load_config()
    model_path = CONFIG['model_settings']['classifier_path']
    
    print(f"🚀 准备启动分类器训练工程...\n🎯 模型保存路径: {model_path}")
    
    # ==================== 数据装载区 ====================
    # 自动定位项目根目录下的 CSV 文件
    csv_file_path = os.path.join(project_root, "classifier_corpus.csv")
    
    if not os.path.exists(csv_file_path):
        print(f"❌ 致命错误: 找不到训练数据文件！\n请确保 '{csv_file_path}' 存在。")
        return
        
    try:
        questions, labels = load_training_data(csv_file_path)
    except Exception as e:
        print(f"❌ 读取 CSV 训练数据失败: {e}")
        return
    # ====================================================
    
    # 初始化分类器
    classifier = TextClassifier(model_path, num_labels=2)
    augmenter = DataAugmenter()
    
    # 🔴 关键修复：必须先调用 load_model 初始化模型结构（无论是加载旧的还是新建基础的）
    print("正在初始化基础模型结构...")
    classifier.load_model()
    
    print(f"📦 成功从 CSV 文件中读取了 {len(questions)} 条语料数据！")
    print("⏳ 开始微调训练，请耐心等待...")
    
    # 执行训练
    classifier.train(questions, labels, batch_size=4, epochs=5, augmenter=augmenter)
    
    # 保存权重文件
    classifier.save_model()
    print("✅ 训练圆满完成！模型权重已保存。以后启动 simple.py 将实现秒开！")

if __name__ == '__main__':
    main()