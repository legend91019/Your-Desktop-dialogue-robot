import sys
from pathlib import Path
import requests
from flask_cors import CORS # 如果运行报错，请在终端执行 pip install flask-cors
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.append(project_root)

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import datetime

from utils.Classifier.classifier import TextClassifier
from utils.Classifier.data_utils import DataAugmenter
from utils.Retriever.retriever import create_rag_retriever
from tools.time_tool import get_current_time_str

import threading
import hashlib

app = Flask(__name__)


import json

# ==================== 队友新增：好感度持久化逻辑 ====================
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
FAVORABILITY_FILE = os.path.join(BACKEND_DIR, "favorability.json")

def get_favorability():
    """获取当前好感度分数"""
    if os.path.exists(FAVORABILITY_FILE):
        try:
            with open(FAVORABILITY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f).get('score', 50)
        except:
            return 50
    save_favorability(50)
    return 50

def save_favorability(score):
    """保存好感度分数并限制范围 [0, 100]"""
    score = max(0, min(100, score))
    with open(FAVORABILITY_FILE, 'w', encoding='utf-8') as f:
        json.dump({"score": score}, f, ensure_ascii=False, indent=2)
    return score

# 加载配置文件
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
# --- 新增：保存配置的函数 ---
def save_config(new_config):
    global CONFIG
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(new_config, f, indent=4, ensure_ascii=False)
    CONFIG = new_config

# --- 新增：接收前端弹窗数据的 API 接口 ---
@app.route('/api/settings', methods=['POST'])
def update_settings():
    data = request.json
    
    # 确保字典结构存在
    if 'user_settings' not in CONFIG: CONFIG['user_settings'] = {}
    if 'api_settings' not in CONFIG: CONFIG['api_settings'] = {}
    
    # 接收前端传来的数据
    if 'master_name' in data: CONFIG['user_settings']['master_name'] = data['master_name']
    if 'occupation' in data: CONFIG['user_settings']['occupation'] = data['occupation']
    if 'current_status' in data: CONFIG['user_settings']['current_status'] = data['current_status']
    if 'api_key' in data: CONFIG['api_settings']['deepseek_api_key'] = data['api_key']
    
    save_config(CONFIG)
    return jsonify({"message": "芯宝的初始核心设定已保存！"})

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
    
    deepseek_api_key = CONFIG.get('api_settings', {}).get('deepseek_api_key', '')
    
    if not deepseek_api_key:
        # 异步记忆提取里如果没key直接 return，闲聊路由里可以返回下面这句话
        print("哎呀，芯宝还没有接入云端神经元网络呢，请先在设置里输入 API Key 呀~")
        return 
    
    deepseek_url = "https://api.deepseek.com/chat/completions"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_api_key}"
    }

    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": extract_prompt}
        ],
        "stream": False
    }
    
    try:
        # 2. 呼叫云端模型提取记忆和实体词
        res = requests.post(deepseek_url, headers=headers, json=payload, timeout=30)
        
        if res.status_code == 200:
            memory_text = res.json()['choices'][0]['message']['content'].strip()
        else:
            memory_text = "无" # 如果 API 报错，就当做没提取到记忆，防止程序崩溃
            print(f"后台记忆提取 API 报错: {res.text}")
        
        # 3. 如果提取到了有效记忆（滤除“无”和乱码长句）
        if memory_text and "无" not in memory_text and len(memory_text) < 50:
            parts = memory_text.split("|")
            statement = parts[0].strip()
            keywords = [k.strip() for k in parts[1].split(",") if k.strip()]
            
            print(f"\n[🧠 触发动态学习] {bot_name} 捕捉到新记忆：{memory_text}")
            print(f"[🏷️ 自动提炼唤醒词] {keywords}")
            
            # 🔴 直接呼叫我们刚才在最下面加载好的全局“公共模型”
            global embed_model, collection
            
            # 将新记忆转为向量并生成唯一 ID
            emb = embed_model.encode([memory_text], normalize_embeddings=True).tolist()[0]
            mem_id = hashlib.md5(memory_text.encode('utf-8')).hexdigest()[:12]
            
            memory_time = get_current_time_str()
            
            # 悄悄写入 ChromaDB 硬盘
            collection.upsert(
                ids=[mem_id],
                documents=[memory_text],
                embeddings=[emb],
                metadatas=[{
                    "type": "user_preference", # 🔴 这是一个全新的元数据标签：用户偏好
                    "source": "dynamic_memory", 
                    "timestamp": memory_time, #长期记忆，加入时间观念
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
        
    

# 🔴 新增：在函数外面声明两个全局变量，当作“公共大厅”
embed_model = None
collection = None

def init_model():
    global embed_model, collection
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
    
    # ==================== 🔴 以下是新增的修改 ====================
    # 3. 启动时，一次性把“几百兆的向量模型”和“数据库连接”加载好
    import chromadb
    from sentence_transformers import SentenceTransformer
    print("⏳ 正在启动后台记忆处理引擎 (只加载一次，防止内存爆炸)...")
    
    db_dir = os.path.join(project_root, CONFIG['path_settings']['chroma_db_dir'])
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name="qbit_memory")
    embed_model = SentenceTransformer(CONFIG['model_settings']['embedding_model'])
    
    print("✅ 后台记忆处理引擎已稳固挂载！")
    # ==============================================================

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
        
        # ==================== 队友新增：好感度计算与拦截系统 ====================
        fav_score = get_favorability()
        favor_tip = ""

        # [拦截] 1. 查询好感度：如果用户直接问，直接回复，不消耗大模型 API
        if any(word in user_message for word in ["好感度", "好感值", "喜欢我吗", "你有多喜欢我"]):
            if fav_score > 80:
                ai_response = f"🥰 主人～当前好感度：{fav_score}！我超级超级喜欢你！要贴贴要抱抱～"
            elif fav_score < 30:
                ai_response = f"💢 哼，好感度只有 {fav_score} 而已…谁叫你老是欺负我！"
            else:
                ai_response = f"✨ 当前好感度：{fav_score}，继续触发亲密话术可以提升好感哦～"
            
            # 直接返回，记录历史
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_history.append({"type": "User", "content": user_message, "timestamp": timestamp})
            chat_history.append({"type": "Assistant", "content": ai_response, "timestamp": timestamp})
            return jsonify({"response": ai_response, "timestamp": timestamp})

        # [计算] 2. 好感度奖惩系统
        add_words = ["乖", "真棒", "厉害", "太聪明了", "好可爱", "超可爱", "爱你", "喜欢你", "贴贴", "抱抱", "摸摸头", "揉揉头"]
        sub_words = ["笨", "讨厌", "很烦", "坏", "傻", "闭嘴", "滚", "走开", "不理你", "没用", "差劲"]

        new_fav = fav_score
        hit_add = any(w in user_message for w in add_words)
        hit_sub = any(w in user_message for w in sub_words)

        if hit_add:
            new_fav += 3
            favor_tip = f"\n\n💖 好感度 +3  当前：{new_fav}"
        elif hit_sub:
            new_fav -= 5
            favor_tip = f"\n\n💔 好感度 -5  当前：{new_fav}"
        
        if hit_add or hit_sub:
            save_favorability(new_fav)

        # [设定] 3. 情绪驱动 Mood
        if new_fav >= 80:
            mood = "你超级喜欢主人！语气软萌撒娇，用～、🥰、❤️、蹭蹭、贴贴，非常粘人，说话害羞可爱。"
        elif new_fav <= 30:
            mood = "你现在很生气、傲娇、委屈，说话带💢，会哼、不理你、别碰我，但保持可爱不恶毒。"
        elif new_fav <= 50:
            mood = "你心情一般，有点小傲娇，回答简洁，偶尔吐槽，不太热情。"
        else:
            mood = "你阳光可爱，和主人关系不错，会开玩笑，会温柔回应。"
        # ==============================================================
        
        # 1. 让分类器判断意图
        questions = [user_message]
        predictions = classifier.predict(questions)
        pred = predictions[0] # 获取判断结果（1或0）
        
        # ==================== 智能路由中枢 (双引擎并行) ====================
        rule_triggered = False
        
        # 1. 优先从 config.json 读取静态配置词，如果没有配，就用这套基础版兜底
        static_keywords = CONFIG.get('routing_settings', {}).get('force_rag_keywords', [
            "芯宝", "团队", "创造者", "开发", "架构师", 
            "王勇顺", "阳泽怡", "徐启恒", "杨赛宇","徐语乐",
            "记得", "喜欢", "习惯", "谁", "什么", "怎么", "以前", "过去", "聊过"
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
        # 实时从内存中读取最新的主人设定
        user_name = CONFIG.get('user_settings', {}).get('master_name', '阿顺')
        user_occ = CONFIG.get('user_settings', {}).get('occupation', '未知')
        user_status = CONFIG.get('user_settings', {}).get('current_status', '未知')
        
        bot_name = CONFIG['bot_settings']['name']
        
        current_time_str = get_current_time_str()
        
        if pred == 1:
            # 【分支A：RAG 增强模式】
            ai_response += f"[知识库增强生成模式]\n"
            
            # 第一步：去知识库里搜索相关的文本片段
            context_text = retrieve_answer(user_message)
            
            context_text = context_text.replace("{{MASTER_NAME}}", user_name)
            context_text = context_text.replace("{{OCCUPATION}}", user_occ)
            context_text = context_text.replace("{{CURRENT_STATUS}}", user_status)
            
            
            # 🔴 2. 动态检索 ChromaDB，获取带时间戳的记忆（加入置信度过滤）

            # 【获取全局单例】从大厅拿到加载好的向量化模型和 ChromaDB 数据库句柄，防止内存爆炸。
            global embed_model, collection

            # 【问题降维】把用户刚刚说的话（比如：“我上周吃了啥？”），转换成一个几百维的浮点数数学向量（Embedding）。
            query_emb = embed_model.encode([user_message], normalize_embeddings=True).tolist()[0]

            # 【空间检索】带着这个数学向量，去 ChromaDB 里的多维空间寻找离它“最近（最相似）”的 3 条记忆。
            results = collection.query(query_embeddings=[query_emb], n_results=3)

            dynamic_context = ""

            # 【防御性非空校验】确保数据库真的返回了距离分数，防止空指针报错。
            if results['distances'] and len(results['distances'][0]) > 0:
                for i in range(len(results['distances'][0])):
        
                    # 【核心：置信度评估】获取这条记忆和当前问题的“空间距离 (Distance)”。
                    # 在向量库里，距离越小 = 语义越接近。距离越大 = 毫不相干。
                    dist = results['distances'][0][i]
        
                    # 【阈值拦截器】这是防幻觉的关键！
                    # 如果距离大于 1.2，说明数据库里根本没有相关的记忆。它只是“矮子里拔将军”硬凑了 3 条垃圾数据出来。
                    # 所以我们设定 1.2 为及格线，不及格的记忆直接抛弃，不喂给大模型！
                    if dist < 1.2:  
            
                        # 【提取内容】只有及格的优质记忆，才把具体的文字和元数据提取出来
                        doc = results['documents'][0][i]
                        meta = results['metadatas'][0][i]
            
                        # 【时序对齐】把提取时盖上的“时间戳钢印”取出来。
                        mem_time = meta.get('timestamp', '未知时间')
            
                        # 【记忆包装】把纯文本包装成大模型极易理解的时序日志格式，比如：“[2026-05-16] 主人今天吃了苹果”
                        dynamic_context += f"[{mem_time}] {doc}\n"

            # 【终极融炉】
            # static_context：提供固定的人设、团队介绍、世界观。
            # dynamic_context：提供用户刚刚被筛选出来的、极其精确的、带时间的私人动态记忆。
            # 两者合并，组成无懈可击的最强 Prompt 上下文。
            final_context_text = f"【底层设定资料】:\n{context_text}\n\n【主人动态时序记忆】:\n{dynamic_context}"
            
            
            # 第二步：把搜索到的文本作为 Context，拼接到 Prompt 中
            final_prompt = f"""
                你是一个叫「{bot_name}」的聪明、贴心的桌面陪伴机器人。
                
                【当前对主人的好感度】: {new_fav}/100 ({mood})
                
                【核心时间锚点】（极其重要）：
                现在的真实时间是：{current_time_str}。请以此为基准，理解用户说的“今天、昨天、上周”等时间概念。
                
                下面提供的【参考资料】中包含了静态设定以及带有[时间戳]的动态记忆。

                【你的回答法则】（非常重要）：
                1. 时序推理：如果用户问及历史行为，请对比当前时间和记忆的时间戳，进行正确的逻辑推导。
                2. 溯源引用：只要你的回答使用到了【主人动态时序记忆】中的内容，你必须在相关句子的末尾加上类似 ^[来源：YYYY-MM-DD] 的脚注标明出处。
                3. 私人问题兜底：如果主人问他自己的事，但在【参考资料】找不到，你可以可爱地撒娇说如“芯宝暂时还没记住这个呢QwQ”的句子，表明你不知道。
                4. 通用世界知识：如果主人问历史、文学（如《雾都孤儿》）、科学等通用常识，请无视资料限制，直接调动你自己的渊博知识库回答！
                【参考资料】:
                {final_context_text}
                
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
                你是一个叫「{bot_name}」的幽默、可爱的桌面陪伴机器人。
                
                【当前对主人的好感度】: {new_fav}/100 ({mood})
                
                【核心时间锚点】：
                现在的真实时间是：{current_time_str}。如果聊天中涉及时间，请以此为基准。
                
                用户现在正在和你闲聊。
                请用生动、带一点小情绪的语气回答，偶尔可以使用 Emoji 表情。
                回答尽量简短，不要长篇大论。
                
                【近期对话历史】:
                {history_text}

                【用户当前提问】:
                {user_message}
            """

        # ==================== 统一请求大模型生成 ====================
        
        # ==================== 统一请求大模型生成 (DeepSeek 云端版) ====================
        
        deepseek_api_key = CONFIG.get('api_settings', {}).get('deepseek_api_key', '')
        
        if not deepseek_api_key:
            # 异步记忆提取里如果没key直接 return，闲聊路由里可以返回下面这句话
            
            return jsonify({"response":"哎呀，芯宝还没有接入云端神经元网络呢，请先在设置里输入 API Key 呀~"})
        
        deepseek_url = "https://api.deepseek.com/chat/completions"
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_api_key}"
        }
        
        payload = {
            "model": "deepseek-chat",  # 指定 DeepSeek 对话模型
            "messages": [
                {"role": "user", "content": final_prompt} # 将你精心组装的 Prompt 发送过去
            ],
            "stream": True 
        }
        
        # 🔴 修改点：创建一个生成器函数，用于源源不断地向前端吐数据
        def generate_stream():
            nonlocal ai_response # 允许我们在内部修改外部的 ai_response 变量
            
            try:
                print("🚀 正在呼叫云端超级大脑 (DeepSeek 流式模式)...")
                # 🔴 修改点：requests.post 必须加上 stream=True 参数
                res = requests.post(deepseek_url, headers=headers, json=payload, stream=True, timeout=30)
                
                bot_reply = "" # 临时存放这次大模型生成的纯净回复
                
                if res.status_code == 200:
                    # 🔴 核心：逐行解析流式数据块 (Server-Sent Events 格式)
                    for line in res.iter_lines():
                        if line:
                            line = line.decode('utf-8')
                            if line.startswith('data: '):
                                data_str = line[6:]
                                if data_str.strip() == '[DONE]': # 官方结束标志
                                    break
                                
                                try:
                                    chunk_data = json.loads(data_str)
                                    if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                        # 注意：流式返回的字段叫 delta，不是 message
                                        content = chunk_data['choices'][0].get('delta', {}).get('content', '')
                                        if content:
                                            bot_reply += content
                                            # 把这一个字打包成 JSON 字符串，吐给前端
                                            yield f"data: {json.dumps({'chunk': content})}\n\n"
                                except json.JSONDecodeError:
                                    pass
                    
                    # 流输出完毕后，拼接好感度提示
                    if favor_tip:
                        bot_reply += favor_tip
                        yield f"data: {json.dumps({'chunk': favor_tip})}\n\n"
                    
                    ai_response += bot_reply
                    
                    # 发送结束标记给前端
                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    yield f"data: {json.dumps({'done': True, 'timestamp': timestamp})}\n\n"
                    
                    # 🔴 修改点：数据全部推完后，再记录历史并触发记忆提取
                    chat_history.append({"type": "User", "content": user_message, "timestamp": timestamp})
                    chat_history.append({"type": "Assistant", "content": ai_response, "timestamp": timestamp})
                    threading.Thread(target=extract_and_save_memory, args=(user_message,)).start()

                else:
                    print(f"云端 API 报错: {res.text}")
                    yield f"data: {json.dumps({'chunk': '芯宝的大脑服务器开小差了，稍后再试哦 QwQ', 'done': True})}\n\n"
            
            except Exception as e:
                yield f"data: {json.dumps({'chunk': f'连接云端大脑失败，检查一下网络哦。报错: {e}', 'done': True})}\n\n"

        # 🔴 修改点：不再返回 jsonify，而是返回一个流式 Response 对象
        return Response(stream_with_context(generate_stream()), mimetype='text/event-stream')

    # 最外层的 except (捕捉整个 handle_chat 的异常) 保持不变
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

    # 🔴 引入生产级 WSGI 服务器
    from waitress import serve
    print("🚀 芯宝后端已启动！(基于 Waitress 生产级容器运行中...)")
    print("🌐 监听地址: http://0.0.0.0:5000")
    
    # 替代原本脆弱的 app.run()，开启 4 个并发线程处理前端请求
    serve(app, host='0.0.0.0', port=5000, threads=4)