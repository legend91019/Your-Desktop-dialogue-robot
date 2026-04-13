import sys
from pathlib import Path
import requests

project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import datetime
from utils.Classifier.classifier import TextClassifier
from utils.Classifier.data_utils import DataAugmenter
from utils.Retriever.retriever import create_rag_retriever

app = Flask(__name__)

CORS(app, resources={
    r"/api/*": {
        "origins": "*",  # 只允许前端地址
        "methods": ["GET", "POST", "OPTIONS", "DELETE"],
        "allow_headers": ["Content-Type"],
        "supports_credentials": True,  # 关键！允许携带 Cookie
    }
})

# 存储对话历史的全局变量
chat_history = []

# 确保上传文件夹存在
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def init_model():
    # 配置参数
    model_path = r'D:\Downloads\bert-base-chinese-finetuned'
    
    # 初始化分类器
    classifier = TextClassifier(model_path, num_labels=2)
    
    # 加载模型
    if not classifier.load_model():
        return
    
# 示例数据
    questions = [
    # Weather/Environment (1)
    "上海明天的气温是多少？", 
    "洛杉矶现在下雨了吗？",
    "台风路径实时查询",
    "未来一周广州的天气预报",
    "当前北极的臭氧层状况如何？",

    # Technology/Programming (0)
    "Python和Java哪个更容易学习？",
    "如何用Pandas合并两个DataFrame？",
    "解释一下区块链的工作原理",
    "神经网络中的反向传播是什么？",
    "Git rebase和merge的区别是什么？",

    # History/Facts (0)
    "明朝有多少位皇帝？",
    "第一次工业革命开始于哪一年？",
    "爱因斯坦获得诺贝尔奖的贡献是什么？",
    "金字塔是如何建造的？",
    "丝绸之路的起点和终点在哪里？",

    # Current Events (1)
    "最新一期Nature期刊的主要内容",
    "2024年奥运会奖牌榜",
    "美联储最近的利率决策是什么？",
    "特斯拉最新车型的发布时间",
    "近期国际原油价格走势",

    # Health/Medicine (mixed)
    "新冠疫苗的常见副作用有哪些？",  # 0
    "北京协和医院心内科专家号怎么挂？",  # 1
    "糖尿病患者的饮食建议",  # 0
    "最近的诺贝尔医学奖得主是谁？",  # 1
    "针灸治疗失眠有效吗？",  # 0

    # Entertainment (mixed)
    "最新一季《权力的游戏》评分如何？",  # 1
    "《红楼梦》的主要人物关系",  # 0
    "Taylor Swift演唱会上海站门票价格",  # 1
    "奥斯卡最佳影片评选标准",  # 0
    "Netflix上评分最高的科幻剧",  # 1

    # Daily Life (mixed)
    "如何去除衣服上的油渍？",  # 0
    "上海浦东机场T2航站楼餐饮店铺",  # 1
    "自制面包的简单配方",  # 0
    "北京到广州高铁时刻表",  # 1
    "室内养什么植物净化空气好？",  # 0

    # Science (0)
    "量子纠缠的基本原理",
    "黑洞是如何形成的？",
    "DNA双螺旋结构的特点",
    "相对论和量子力学的矛盾点",
    "光合作用的光反应和暗反应",

    # Economy/Finance (mixed)
    "美联储加息对A股的影响分析",  # 1
    "GDP的计算方法",  # 0
    "最新人民币兑美元汇率",  # 1
    "价值投资的基本原则",  # 0
    "2023年全球富豪榜前十名",  # 1

    # Education (mixed)
    "清华大学计算机专业录取分数线",  # 1
    "如何提高英语听力水平？",  # 0
    "雅思和托福哪个更容易？",  # 0
    "哈佛大学最新招生政策",  # 1
    "记忆宫殿法的具体操作步骤",  # 0

    # Travel (mixed)
    "日本旅游签证最新办理要求",  # 1
    "巴黎铁塔的开放时间",  # 1
    "东南亚旅游最佳季节",  # 0
    "携程上三亚五星酒店特价信息",  # 1
    "独自旅行安全注意事项",  # 0
    
    # Philosophy (0)
    "康德的先验哲学指什么？",
    "道家思想的核心观点",
    "苏格拉底的产婆术是什么？",
    "尼采的超人哲学如何理解",
    "唯物主义和唯心主义的区别",
    
    # Sports (mixed)
    "NBA最新赛季得分王是谁？",  # 1
    "羽毛球比赛规则",  # 0
    "中国男足最新世界排名",  # 1
    "马拉松训练计划",  # 0
    "F1赛车2024赛季赛程",  # 1

    # Cooking (0)
    "红烧肉的正宗做法",
    "如何判断牛排的熟度？",
    "蛋糕胚不塌陷的技巧",
    "素食者蛋白质补充方案",
    "咖啡豆的烘焙程度区别",
    
    # Law (mixed)
    "中国最新劳动法修订内容",  # 1
    "正当防卫的法律界定",  # 0
    "上海购房限购政策2024",  # 1
    "知识产权保护期限",  # 0
    "最新个人所得税起征点",  # 1
    
    # Psychology (0)
    "马斯洛需求层次理论",
    "认知失调理论的应用",
    "如何克服社交恐惧？",
    "抑郁症的早期表现",
    "正念冥想的具体方法",
    
    # Art (mixed)
    "卢浮宫镇馆三宝是什么？",  # 0
    "2024威尼斯双年展主题",  # 1
    "中国山水画的特点",  # 0
    "毕加索蓝色时期的代表作",  # 0
    "最近故宫特展的开放时间",  # 1
    
    # Miscellaneous
    "如何修复破损的PDF文件？",  # 0
    "Windows11最新更新内容",  # 1
    "手机电池保养技巧",  # 0
    "小米14Pro的摄像头参数",  # 1
    "时间管理的四象限法则"  # 
    ]
    labels = [
    1,1,1,1,1,  # Weather
    0,0,0,0,0,  # Tech
    0,0,0,0,0,  # History
    1,1,1,1,1,  # Current Events
    0,1,0,1,0,  # Health
    1,0,1,0,1,  # Entertainment
    0,1,0,1,0,  # Daily Life
    0,0,0,0,0,  # Science
    1,0,1,0,1,  # Economy
    1,0,0,1,0,  # Education
    1,1,0,1,0,  # Travel
    0,0,0,0,0,  # Philosophy
    1,0,1,0,1,  # Sports
    0,0,0,0,0,  # Cooking
    1,0,1,0,1,  # Law
    0,0,0,0,0,  # Psychology
    0,1,0,0,1,  # Art
    0,1,0,1,0   # Miscellaneous
    ]  # 1需要检索 0直接生成
    
    # 初始化数据增强器
    augmenter = DataAugmenter()
    
    # 训练模型
    classifier.train(questions, labels, batch_size=4, epochs=5, augmenter=augmenter)
    
    # 保存模型
    classifier.save_model()

    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    docx_file = os.path.join(project_root, "input.docx")
    
    retrieve_answer = create_rag_retriever(docx_file)

    return classifier, retrieve_answer
    

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def handle_chat():
    if request.method == 'OPTIONS':
        # 直接返回 200，让浏览器继续发送 POST
        return jsonify({}), 200
    try:
        try:
            data = request.json
            print("解析的JSON数据:", data)
        except Exception as e:
            print("JSON解析错误:", str(e))
            return jsonify({"error": "无效的JSON格式"}), 400

        if data is None:
            # 尝试以表单形式解析（以防前端发送的是表单数据）
            form_data = request.form
            print("尝试以表单形式解析:", form_data)
            user_message = form_data.get('message', '')
            if not user_message:
                return jsonify({"error": "消息不能为空或格式错误"}), 400
        else:
            user_message = data.get('message', '')
            if not user_message:
                return jsonify({"error": "消息不能为空"}), 400
        
        ai_response = ""
        
        # 1. 让分类器判断意图
        questions = [user_message]
        predictions = classifier.predict(questions)
        pred = predictions[0] # 获取判断结果（1或0）

        # 2. 根据判断结果走不同的处理分支
        if pred == 1:
            # 【分支A：需要检索】 -> 调用学长的本地文档检索
            ai_response += f"[知识库检索模式] 预测结果: 1\n"
            ai_response += "-"*30 + "\n"
            doc_answer = retrieve_answer(user_message)
            ai_response += f"{doc_answer}"
            
        else:
            # 【分支B：直接生成】 -> 调用本地的 Ollama Qwen2.5 模型
            ai_response += f"[大模型闲聊模式] 预测结果: 0\n"
            ai_response += "-"*30 + "\n"
            
            # 告诉 Ollama 我们要用哪个模型，以及用户问了什么
            ollama_url = "http://localhost:11434/api/generate"
            payload = {
                "model": "qwen2.5:3b",
                "prompt": user_message,
                "stream": False # 等模型全部想完再一起返回
            }
            try:
                # 向 Ollama 发送网络请求，超时时间设为 60 秒
                res = requests.post(ollama_url, json=payload, timeout=60)
                result = res.json()
                ai_response += result.get("response", "大模型好像卡壳了...")
            except Exception as e:
                ai_response += f"连接大模型失败，请检查电脑右下角 Ollama 软件是否运行。报错信息: {e}"
        
        # 记录对话历史
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({
            "type": "user",
            "content": user_message,
            "timestamp": timestamp
        })
        chat_history.append({
            "type": "ai",
            "content": ai_response,
            "timestamp": timestamp
        })
        
        return jsonify({
            "response": ai_response,
            "timestamp": timestamp
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def handle_upload():
    """处理文件上传"""
    if 'file' not in request.files:
        return jsonify({"error": "没有文件"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "未选择文件"}), 400
    
    if file:
        # 保存文件
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        
        # 记录上传历史并返回固定消息
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({
            "type": "user",
            "content": f"上传了文件：{file.filename}",
            "timestamp": timestamp
        })
        
        ai_response = f"文件「{file.filename}」已接收，这是固定的处理结果"
        chat_history.append({
            "type": "ai",
            "content": ai_response,
            "timestamp": timestamp
        })
        
        return jsonify({
            "response": ai_response,
            "filename": file.filename,
            "timestamp": timestamp
        })

@app.route('/api/history', methods=['GET'])
def get_history():
    """获取对话历史"""
    return jsonify({
        "history": chat_history
    })

@app.route('/api/history', methods=['DELETE'])
def clear_history():
    """清空对话历史"""
    global chat_history
    chat_history = []
    return jsonify({"message": "历史记录已清空"})



if __name__ == '__main__':
    classifier, retrieve_answer = init_model()

    app.run(debug=True, host='0.0.0.0', port=5000)