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

import threading
import hashlib

app = Flask(__name__)

import json

# 加载配置文件
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()

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


def extract_and_save_memory(user_msg):
    """异步记忆提取器：利用大模型分析用户输入，提取长期价值信息并存入 ChromaDB"""
    
    bot_name = CONFIG['bot_settings']['name']
    
    # 1. 组装专门用于“提取记忆”的 Prompt
    extract_prompt = f"""
    请你作为一个无感情的记忆提取机器。分析用户的这句话："{user_msg}"
    如果这句话包含用户的个人喜好、习惯、重要经历等长期价值内容，请完成以下两步：
    第一步：提取为一句第三人称客观陈述句（以“主人”为主语）。
    第二步：提取出 1 到 2 个最核心的名词实体（作为日后唤醒这条记忆的专属触发词）。
    请严格按照格式输出：陈述句 | 实体1,实体2
    
    如果没有包含这类信息，请严格只回复一个字：“无”。
    不要有任何解释，不要包含标点符号。

    例如：
    输入："我今天去吃了一家超好吃的日料，我最喜欢吃三文鱼了" -> 输出：主人最喜欢吃三文鱼 | 日料,三文鱼
    输入："今天天气真好" -> 输出：无
    """
    
    payload = {
        "model": CONFIG['model_settings']['ollama_model_id'], 
        "prompt": extract_prompt,
        "stream": False
    }
    
    try:
        # 2. 呼叫模型进行意图判断
        res = requests.post(CONFIG['model_settings']['ollama_url'], json=payload, timeout=30)
        memory_text = res.json().get("response", "").strip()
        
        # 3. 如果提取到了有效记忆（滤除“无”和乱码长句）
        if memory_text and "无" not in memory_text and len(memory_text) < 50:
            parts = memory_text.split("|")
            statement = parts[0].strip()
            keywords = [k.strip() for k in parts[1].split(",") if k.strip()]
            
            print(f"\n[🧠 触发动态学习] {bot_name} 捕捉到新记忆：{memory_text}")
            print(f"[🏷️ 自动提炼唤醒词] {keywords}")
            
            # 为了线程安全，在这个独立线程中临时连一下数据库
            import chromadb
            from sentence_transformers import SentenceTransformer
            
            # 读取配置中的路径和模型
            db_dir = os.path.join(os.path.dirname(__file__), '..', CONFIG['path_settings']['chroma_db_dir'])
            client = chromadb.PersistentClient(path=db_dir)
            collection = client.get_or_create_collection(name="qbit_memory")
            embed_model = SentenceTransformer(CONFIG['model_settings']['embedding_model'])
            
            # 将新记忆转为向量并生成唯一 ID
            emb = embed_model.encode([memory_text], normalize_embeddings=True).tolist()[0]
            mem_id = hashlib.md5(memory_text.encode('utf-8')).hexdigest()[:12]
            
            # 悄悄写入 ChromaDB 硬盘
            collection.upsert(
                ids=[mem_id],
                documents=[memory_text],
                embeddings=[emb],
                metadatas=[{
                    "type": "user_preference", # 🔴 这是一个全新的元数据标签：用户偏好
                    "source": "dynamic_memory", 
                    "title": "主人动态画像",
                    "chunk_index": 9999
                }]
            )
            print("[✅ 记忆写入完成] 该记忆已永久存入边缘设备芯片！\n")
            
            keywords_file = os.path.join(os.path.dirname(__file__), '..', 'dynamic_keywords.txt')
            with open(keywords_file, 'a', encoding='utf-8') as f:
                for kw in keywords:
                    if len(kw) > 1: # 过滤掉单字垃圾词
                        f.write(f"{kw}\n")
            
    except Exception as e:
        print(f"后台记忆写入出错，但这不影响前端聊天: {e}")
        
    


def init_model():
    # 配置参数
    model_path = CONFIG['model_settings']['classifier_path']
    
    # 初始化分类器
    classifier = TextClassifier(model_path, num_labels=2)
    
    # 纯粹的加载逻辑，没有任何训练包袱
    if not classifier.load_model():
        print("❌ 致命警告：未找到训练好的分类器模型！请先在根目录运行 python train_classifier.py")
    else:
        print("✅ 交通警察 (分类器权重) 加载成功！")

    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))
    md_file = os.path.join(project_root, "knowledge.md")
    
    retrieve_answer = create_rag_retriever(md_file)

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
        
        # ==================== 智能路由中枢 (双引擎并行) ====================
        rule_triggered = False
        
        # 1. 优先从 config.json 读取静态配置词，如果没有配，就用这套基础版兜底
        static_keywords = CONFIG.get('routing_settings', {}).get('force_rag_keywords', [
            "王勇顺", "周子铠", "杰哥", "侯立坤", "同桌", "团队", "架构师", "阿顺", "记得", "喜欢", "谁", "什么", "怎么"
        ])
        force_rag_keywords = set(static_keywords)
        
        # 2. 动态加载大模型学习到的唤醒词
        keywords_file = os.path.join(os.path.dirname(__file__), '..', 'dynamic_keywords.txt')
        if os.path.exists(keywords_file):
            with open(keywords_file, 'r', encoding='utf-8') as f:
                dynamic_words = [line.strip() for line in f if line.strip()]
                force_rag_keywords.update(dynamic_words) # 把动态词合并进拦截池
        
        # 3. 开始精准拦截判断 (防御机制：长度大于1才触发，防误触)
        if len(user_message) > 1: 
            for keyword in force_rag_keywords:
                if keyword in user_message:
                    rule_triggered = True
                    print(f"⚠️ [双引擎路由] 规则捕获到实体词「{keyword}」，强制切换为 RAG 模式！")
                    pred = 1
                    break
        
        if not rule_triggered:
            mode_str = "RAG检索" if pred == 1 else "直接生成"
            print(f"🧠 [双引擎路由] 规则未命中，分类器模型推断结果为：{mode_str} (pred={pred})")
        # ===================================================================
        #===================== 模型短期记忆，实现方法就是提示词工程，把history_text拼接到context =======================================
        history_text = ""
        
        recent_history = chat_history[-10:] if len(chat_history) > 0 else []

        for msg in recent_history:
            role = "User" if msg["type"] == "User" else "Assistant"
            history_text += f"{role}:{msg['content']}\n"
            
            
        # ==================== 提示词组装 (Skill工程 + Context) ====================
        bot_name = CONFIG['bot_settings']['name']
        
        if pred == 1:
            # 【分支A：RAG 增强模式】
            ai_response += f"[知识库增强生成模式]\n"
            
            # 第一步：去知识库里搜索相关的文本片段
            context_text = retrieve_answer(user_message)
            
            # 第二步：把搜索到的文本作为 Context，拼接到 Prompt 中
            final_prompt = f"""
                你是一个叫「{bot_name}」的聪明、贴心的桌面陪伴机器人。请严格根据下面提供的【参考资料】来回答用户的问题。
                如果你在【参考资料】中找不到答案，请诚实地说明你不知道，千万不要自己瞎编。
                回答要简洁、口语化，像人类正常对话一样。

                【参考资料】:
                {context_text}
                
                【近期对话历史】:
                {history_text}

                【用户当前提问】:
                {user_message}
            """
            
        else:
            # 【分支B：自由闲聊模式】
            ai_response += f"[自由闲聊模式]\n"
            
            # 闲聊模式不需要搜索，直接给大模型设定人设
            final_prompt = f"""
                你是一个叫「{bot_name}」的幽默、可爱的桌面陪伴机器人。用户现在正在和你闲聊。
                请用生动、带一点小情绪的语气回答，偶尔可以使用 Emoji 表情。
                回答尽量简短，不要长篇大论。
                
                【近期对话历史】:
                {history_text}

                【用户当前提问】:
                {user_message}
            """

        # ==================== 统一请求大模型生成 ====================
        
        # 把最终拼好的 prompt 发给 Ollama
        ollama_url = CONFIG["model_settings"]["ollama_url"]
        payload = {
            "model": CONFIG['model_settings']['ollama_model_id'],
            "prompt": final_prompt,
            "stream": False 
        }
        
        try:
            # 调用大模型，把 Context 消化后生成人类自然语言
            res = requests.post(ollama_url, json=payload, timeout=300)
            result = res.json()
            
            # 拿到大模型最终生成的回答！
            ai_response += result.get("response", "我脑袋有点晕，没想出来...")
            
        except Exception as e:
            ai_response = f"连接大模型失败，请检查电脑右下角 Ollama 软件是否运行。报错信息: {e}"

        # 记录对话历史 (后面的代码保持原样不要动)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({
            "type": "User",
            "content": user_message,
            "timestamp": timestamp
        })
        chat_history.append({
            "type": "Assistant",
            "content": ai_response,
            "timestamp": timestamp
        })
        
        # ==================== 触发异步记忆提取 ====================
        # 把用户刚说的话，丢给后台的记忆提取器慢慢分析
        threading.Thread(target=extract_and_save_memory, args=(user_message,)).start()
        # ==========================================================
        
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
            "type": "User",
            "content": f"上传了文件：{file.filename}",
            "timestamp": timestamp
        })
        
        ai_response = f"文件「{file.filename}」已接收，这是固定的处理结果"
        chat_history.append({
            "type": "Assistant",
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