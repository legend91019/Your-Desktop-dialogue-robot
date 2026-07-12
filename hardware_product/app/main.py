import sys
from pathlib import Path

# 把项目根目录加入 Python 路径
PROJECT_ROOT = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, PROJECT_ROOT)

import asyncio
try:
    import edge_tts
except ImportError:
    edge_tts = None
import re
import glob
import time
import json
import os
import datetime
import requests
import threading
import hashlib

from flask import Flask, request, jsonify, Response, stream_with_context, send_from_directory
from flask_cors import CORS

from utils.Classifier.classifier import TextClassifier
from utils.Retriever.retriever import create_rag_retriever
from app.tools.time_tool import get_current_time_str

app = Flask(__name__)

# ==================== 好感度持久化 ====================
FAVORABILITY_FILE = os.path.join(PROJECT_ROOT, "favorability.json")


@app.route("/static/<path:filename>")
def serve_audio(filename):
    static_dir = os.path.join(PROJECT_ROOT, "static")
    return send_from_directory(static_dir, filename)


@app.route("/")
def serve_frontend():
    frontend_dir = os.path.join(PROJECT_ROOT, "frontend")
    return send_from_directory(frontend_dir, "robot.html")


def get_favorability():
    if os.path.exists(FAVORABILITY_FILE):
        try:
            with open(FAVORABILITY_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("score", 50)
        except Exception:
            return 50
    save_favorability(50)
    return 50


def save_favorability(score):
    score = max(0, min(100, score))
    with open(FAVORABILITY_FILE, "w", encoding="utf-8") as f:
        json.dump({"score": score}, f, ensure_ascii=False, indent=2)
    return score


def load_config():
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(new_config):
    global CONFIG
    config_path = os.path.join(PROJECT_ROOT, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=4, ensure_ascii=False)
    CONFIG = new_config


@app.route("/api/settings", methods=["POST"])
def update_settings():
    data = request.json
    if "user_settings" not in CONFIG:
        CONFIG["user_settings"] = {}
    if "api_settings" not in CONFIG:
        CONFIG["api_settings"] = {}

    if "master_name" in data:
        CONFIG["user_settings"]["master_name"] = data["master_name"]
    if "occupation" in data:
        CONFIG["user_settings"]["occupation"] = data["occupation"]
    if "current_status" in data:
        CONFIG["user_settings"]["current_status"] = data["current_status"]
    if "api_key" in data:
        CONFIG["api_settings"]["deepseek_api_key"] = data["api_key"]

    save_config(CONFIG)
    return jsonify({"message": "芯宝的初始核心设定已保存!"})


CONFIG = load_config()

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": "*",
            "methods": ["GET", "POST", "OPTIONS", "DELETE"],
            "allow_headers": ["Content-Type"],
            "supports_credentials": True,
        },
        r"/static/*": {"origins": "*"},
    },
)

chat_history = []
wakeup_events = []
wakeup_lock = threading.Lock()

UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


@app.route("/api/wakeup", methods=["GET", "POST"])
def handle_wakeup_event():
    global wakeup_events

    if request.method == "GET":
        with wakeup_lock:
            latest = wakeup_events[-1] if wakeup_events else None
            return jsonify({"latest": latest, "events": wakeup_events[-10:]})

    data = request.json or {}
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    event = {
        "id": int(time.time() * 1000),
        "timestamp": timestamp,
        "keyword": data.get("keyword") or data.get("result") or "xiao3 wei1 xiao3 wei1",
        "score": data.get("score"),
        "angle": data.get("angle"),
        "beam": data.get("beam"),
        "raw": data,
    }

    with wakeup_lock:
        wakeup_events.append(event)
        wakeup_events = wakeup_events[-50:]

    print(f"[M260C] Wakeup received: {event}")
    return jsonify({"ok": True, "event": event})


def extract_and_save_memory(user_msg):
    bot_name = CONFIG["bot_settings"]["name"]

    extract_prompt = f"""
    请你作为一个无感情的记忆提取机器。分析用户的这句话："{user_msg}"
    如果这句话包含用户的个人喜好、习惯、重要经历等长期价值内容，请完成以下两步：
    第一步：提取为一句第三人称客观陈述句（以"主人"为主语）。
    第二步：提取出 1 到 2 个最核心的名词实体（作为日后唤醒这条记忆的专属触发词）。
    请严格按照格式输出：陈述句 | 实体1,实体2

    如果没有包含这类信息，请严格只回复一个字："无"。
    不要有任何解释，不要包含标点符号。

    例如：
    输入："我今天去吃了一家超好吃的日料，我最喜欢吃三文鱼了" -> 输出：主人最喜欢吃三文鱼 | 日料,三文鱼
    输入："今天天气真好" -> 输出：无
    """

    deepseek_api_key = CONFIG.get("api_settings", {}).get("deepseek_api_key", "")

    if not deepseek_api_key:
        print("芯宝还没有接入云端神经元网络，请先在设置里输入 API Key")
        return

    deepseek_url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {deepseek_api_key}",
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": extract_prompt}],
        "stream": False,
    }

    try:
        res = requests.post(deepseek_url, headers=headers, json=payload, timeout=30)

        if res.status_code == 200:
            memory_text = res.json()["choices"][0]["message"]["content"].strip()
        else:
            memory_text = "无"
            print(f"后台记忆提取 API 报错: {res.text}")

        if memory_text and "无" not in memory_text and len(memory_text) < 50:
            parts = memory_text.split("|")
            statement = parts[0].strip()
            keywords = [k.strip() for k in parts[1].split(",") if k.strip()]

            print(f"\n[🧠 触发动态学习] {bot_name} 捕捉到新记忆：{memory_text}")
            print(f"[🏷️ 自动提炼唤醒词] {keywords}")

            global embed_model, collection

            emb = embed_model.encode([memory_text], normalize_embeddings=True).tolist()[0]
            mem_id = hashlib.md5(memory_text.encode("utf-8")).hexdigest()[:12]
            memory_time = get_current_time_str()

            collection.upsert(
                ids=[mem_id],
                documents=[memory_text],
                embeddings=[emb],
                metadatas=[{
                    "type": "user_preference",
                    "source": "dynamic_memory",
                    "timestamp": memory_time,
                    "title": "主人动态画像",
                    "chunk_index": 9999,
                }],
            )
            print("[✅ 记忆写入完成] 该记忆已永久存入边缘设备芯片!\n")

            keywords_file = os.path.join(PROJECT_ROOT, "dynamic_keywords.txt")
            with open(keywords_file, "a", encoding="utf-8") as f:
                for kw in keywords:
                    if len(kw) > 1:
                        f.write(f"{kw}\n")

    except Exception as e:
        print(f"后台记忆写入出错，但这不影响前端聊天: {e}")


# 全局模型实例
embed_model = None
collection = None
reranker_model = None


def init_models():
    global embed_model, collection, reranker_model

    # 1. 意图分类器
    classifier_path = os.path.join(
        PROJECT_ROOT, CONFIG["model_settings"]["classifier_path"]
    )
    classifier = TextClassifier(classifier_path, num_labels=2)
    if not classifier.load_model():
        print("❌ 致命警告：未找到训练好的分类器模型! 请先运行 python train_classifier.py")
    else:
        print("✅ 交通警察 (分类器权重) 加载成功!")

    # 2. 知识库 RAG 检索器 (共享 embedding 模型)
    md_file = os.path.join(PROJECT_ROOT, CONFIG["path_settings"]["knowledge_base"])

    # 3. 加载共享的向量模型和 ChromaDB
    import chromadb
    from sentence_transformers import SentenceTransformer, CrossEncoder

    print("⏳ 正在启动后台记忆处理引擎 (只加载一次，防止内存爆炸)...")

    db_dir = os.path.join(PROJECT_ROOT, CONFIG["path_settings"]["chroma_db_dir"])
    client = chromadb.PersistentClient(path=db_dir)
    collection = client.get_or_create_collection(name="qbit_memory")

    embedding_path = os.path.join(
        PROJECT_ROOT, CONFIG["model_settings"]["embedding_model"]
    )
    embed_model = SentenceTransformer(embedding_path)

    print("✅ 后台记忆处理引擎已稳固挂载!")

    # 4. BGE 精排模型
    print("⏳ 正在挂载交叉注意力精排引擎 (Reranker)...")
    reranker_path = os.path.join(
        PROJECT_ROOT, CONFIG["model_settings"]["reranker_model"]
    )
    reranker_model = CrossEncoder(reranker_path)
    print("✅ 后台记忆与精排引擎已稳固挂载!")

    # 5. 知识库检索器 (复用 embed_model 和 collection)
    retrieve_answer = create_rag_retriever(
        md_file,
        embed_model=embed_model,
        collection=collection,
        top_k=CONFIG["rag_settings"]["top_k"],
    )

    return classifier, retrieve_answer


@app.route("/api/chat", methods=["POST", "OPTIONS"])
def handle_chat():
    if request.method == "OPTIONS":
        return jsonify({}), 200

    try:
        try:
            data = request.json
        except Exception:
            return jsonify({"error": "无效的JSON格式"}), 400

        if data is None:
            form_data = request.form
            user_message = form_data.get("message", "")
            if not user_message:
                return jsonify({"error": "消息不能为空或格式错误"}), 400
        else:
            user_message = data.get("message", "")
            if not user_message:
                return jsonify({"error": "消息不能为空"}), 400

        ai_response = ""

        # ==================== 好感度系统 ====================
        fav_score = get_favorability()
        favor_tip = ""

        # 拦截好感度查询
        if any(
            word in user_message
            for word in ["好感度", "好感值", "喜欢我吗", "你有多喜欢我"]
        ):
            if fav_score > 80:
                ai_response = f"🥰 主人~当前好感度：{fav_score}! 我超级超级喜欢你! 要贴贴要抱抱~"
            elif fav_score < 30:
                ai_response = f"💢 哼，好感度只有 {fav_score} 而已...谁叫你老是欺负我!"
            else:
                ai_response = f"✨ 当前好感度：{fav_score}，继续触发亲密话术可以提升好感哦~"

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            chat_history.append({"type": "User", "content": user_message, "timestamp": timestamp})
            chat_history.append({"type": "Assistant", "content": ai_response, "timestamp": timestamp})
            return jsonify({"response": ai_response, "timestamp": timestamp})

        # 奖惩计算
        add_words = ["乖", "真棒", "厉害", "太聪明了", "好可爱", "超可爱", "爱你", "喜欢你", "贴贴", "抱抱", "摸摸头", "揉揉头"]
        sub_words = ["笨", "讨厌", "很烦", "坏", "傻", "闭嘴", "滚", "走开", "不理你", "没用", "差劲"]

        new_fav = fav_score
        hit_add = any(w in user_message for w in add_words)
        hit_sub = any(w in user_message for w in sub_words)

        change_type = "none"
        if hit_add:
            new_fav += 3
            change_type = "up"
            favor_tip = f"\n\n💖 好感度 +3  当前：{new_fav}"
        elif hit_sub:
            new_fav -= 5
            change_type = "down"
            favor_tip = f"\n\n💔 好感度 -5  当前：{new_fav}"

        if hit_add or hit_sub:
            save_favorability(new_fav)

        # 情绪驱动
        if new_fav >= 80:
            mood = "你超级喜欢主人! 语气软萌撒娇，用~、🥰、❤️、蹭蹭、贴贴，非常粘人，说话害羞可爱。"
        elif new_fav <= 30:
            mood = "你现在很生气、傲娇、委屈，说话带💢，会哼、不理你、别碰我，但保持可爱不恶毒。"
        elif new_fav <= 50:
            mood = "你心情一般，有点小傲娇，回答简洁，偶尔吐槽，不太热情。"
        else:
            mood = "你阳光可爱，和主人关系不错，会开玩笑，会温柔回应。"

        # ==================== 意图分类 ====================
        questions = [user_message]
        predictions = classifier.predict(questions)
        pred = predictions[0]

        # ==================== 智能路由 ====================
        rule_triggered = False

        static_keywords = CONFIG.get("routing_settings", {}).get("force_rag_keywords", [])
        force_rag_keywords = set(static_keywords)

        keywords_file = os.path.join(PROJECT_ROOT, "dynamic_keywords.txt")
        if os.path.exists(keywords_file):
            with open(keywords_file, "r", encoding="utf-8") as f:
                dynamic_words = [line.strip() for line in f if line.strip()]
                force_rag_keywords.update(dynamic_words)

        if len(user_message) > 1:
            for keyword in force_rag_keywords:
                if keyword in user_message:
                    rule_triggered = True
                    print(f"⚠️ [双引擎路由] 规则捕获到实体词「{keyword}」，强制切换为 RAG 模式!")
                    pred = 1
                    break

        if not rule_triggered:
            mode_str = "RAG检索" if pred == 1 else "直接生成"
            print(f"🧠 [双引擎路由] 规则未命中，分类器模型推断结果为：{mode_str} (pred={pred})")

        # ==================== 对话历史 ====================
        history_text = ""
        recent_history = chat_history[-10:] if len(chat_history) > 0 else []
        for msg in recent_history:
            role = "User" if msg["type"] == "User" else "Assistant"
            history_text += f"{role}:{msg['content']}\n"

        # ==================== 提示词组装 ====================
        user_name = CONFIG.get("user_settings", {}).get("master_name", "阿顺")
        user_occ = CONFIG.get("user_settings", {}).get("occupation", "未知")
        user_status = CONFIG.get("user_settings", {}).get("current_status", "未知")
        bot_name = CONFIG["bot_settings"]["name"]
        current_time_str = get_current_time_str()

        if pred == 1:
            # RAG 增强模式
            ai_response += f"[知识库增强生成模式]\n"

            context_text = retrieve_answer(user_message)
            context_text = context_text.replace("{{MASTER_NAME}}", user_name)
            context_text = context_text.replace("{{OCCUPATION}}", user_occ)
            context_text = context_text.replace("{{CURRENT_STATUS}}", user_status)

            global embed_model, collection, reranker_model

            # 阶段一：向量粗排
            query_emb = embed_model.encode([user_message], normalize_embeddings=True).tolist()[0]
            results = collection.query(query_embeddings=[query_emb], n_results=10)

            dynamic_context = ""
            candidate_docs = []

            if results["distances"] and len(results["distances"][0]) > 0:
                for i in range(len(results["distances"][0])):
                    dist = results["distances"][0][i]
                    if dist < 1.5:
                        doc = results["documents"][0][i]
                        meta = results["metadatas"][0][i]
                        candidate_docs.append((doc, meta))

            # 阶段二：精排
            if candidate_docs:
                pairs = [[user_message, doc_info[0]] for doc_info in candidate_docs]
                scores = reranker_model.predict(pairs)
                scored_docs = list(zip(scores, candidate_docs))
                scored_docs.sort(key=lambda x: x[0], reverse=True)

                print("\n🔍 [精排引擎] 候选记忆打分结果：")

                top_k = 0
                for score, (doc, meta) in scored_docs:
                    print(f"   -> 得分: {score:.4f} | 内容: {doc}")
                    if score > 0 and top_k < 3:
                        mem_time = meta.get("timestamp", "未知时间")
                        dynamic_context += f"[{mem_time}] {doc}\n"
                        top_k += 1

            final_context_text = f"【底层设定资料】:\n{context_text}\n\n【主人动态时序记忆】:\n{dynamic_context}"

            final_prompt = f"""
                你是一个叫「{bot_name}」的聪明、贴心的桌面陪伴机器人。

                【当前对主人的好感度】: {new_fav}/100 ({mood})

                【核心时间锚点】（极其重要）：
                现在的真实时间是：{current_time_str}。请以此为基准，理解用户说的"今天、昨天、上周"等时间概念。

                下面提供的【参考资料】中包含了静态设定以及带有[时间戳]的动态记忆。

                【你的回答法则】（非常重要）：
                1. 时序推理：如果用户问及历史行为，请对比当前时间和记忆的时间戳，进行正确的逻辑推导。
                2. 溯源引用：只要你的回答使用到了【主人动态时序记忆】中的内容，你必须在相关句子的末尾加上类似 ^[来源：YYYY-MM-DD] 的脚注标明出处。
                3. 私人问题兜底：如果主人问他自己的事，但在【参考资料】找不到，你可以可爱地撒娇说如"芯宝暂时还没记住这个呢QwQ"的句子，表明你不知道。
                4. 通用世界知识：如果主人问历史、文学（如《雾都孤儿》）、科学等通用常识，请无视资料限制，直接调动你自己的渊博知识库回答!
                【参考资料】:
                {final_context_text}

                【近期对话历史】:
                {history_text}

                【用户当前提问】:
                {user_message}
            """
        else:
            # 自由闲聊模式
            ai_response += f"[自由闲聊模式]\n"

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

        # ==================== 请求大模型 ====================
        deepseek_api_key = CONFIG.get("api_settings", {}).get("deepseek_api_key", "")

        if not deepseek_api_key:
            return jsonify({"response": "芯宝还没有接入云端神经元网络，请先在设置里输入 API Key"})

        deepseek_url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {deepseek_api_key}",
        }
        payload = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": final_prompt}],
            "stream": True,
        }

        def generate_stream():
            nonlocal ai_response

            # 首包推送好感度状态
            yield f"data: {json.dumps({'favorability': new_fav, 'change': change_type}, ensure_ascii=False)}\n\n"

            try:
                print("🚀 正在呼叫云端超级大脑 (DeepSeek 流式模式)...")
                res = requests.post(
                    deepseek_url, headers=headers, json=payload, stream=True, timeout=60
                )

                bot_reply = ""

                if res.status_code == 200:
                    for line in res.iter_lines():
                        if line:
                            line = line.decode("utf-8")
                            if line.startswith("data: "):
                                data_str = line[6:]
                                if data_str.strip() == "[DONE]":
                                    break
                                try:
                                    chunk_data = json.loads(data_str)
                                    if "choices" in chunk_data and len(chunk_data["choices"]) > 0:
                                        content = (
                                            chunk_data["choices"][0]
                                            .get("delta", {})
                                            .get("content", "")
                                        )
                                        if content:
                                            bot_reply += content
                                            yield f"data: {json.dumps({'chunk': content})}\n\n"
                                except json.JSONDecodeError:
                                    pass

                    if favor_tip:
                        bot_reply += favor_tip
                        yield f"data: {json.dumps({'chunk': favor_tip})}\n\n"

                    ai_response += bot_reply

                    # ==================== TTS 语音合成 ====================
                    try:
                        if edge_tts is None:
                            print("⚠️ 未安装 edge_tts，跳过语音合成，仅返回文字。")
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            yield f"data: {json.dumps({'done': True, 'timestamp': timestamp})}\n\n"
                            return

                        clean_text = re.sub(r"\[.*?\]", "", ai_response)
                        clean_text = re.sub(r"\(.*?\)|\（.*?\）", "", clean_text)
                        clean_text = re.sub(r"[*#`~]", "", clean_text).strip()

                        if clean_text:
                            static_dir = os.path.join(PROJECT_ROOT, "static")
                            now = time.time()
                            for f in glob.glob(os.path.join(static_dir, "*.mp3")):
                                if os.stat(f).st_mtime < now - 180:
                                    try:
                                        os.remove(f)
                                    except Exception:
                                        pass

                            audio_filename = f"reply_{hashlib.md5(clean_text.encode('utf-8')).hexdigest()[:8]}.mp3"
                            audio_path = os.path.join(static_dir, audio_filename)

                            async def generate_audio():
                                communicate = edge_tts.Communicate(clean_text, "zh-CN-XiaoyiNeural")
                                await communicate.save(audio_path)

                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            try:
                                loop.run_until_complete(generate_audio())
                            finally:
                                loop.close()

                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            yield f"data: {json.dumps({'done': True, 'timestamp': timestamp, 'audio_url': f'/static/{audio_filename}'})}\n\n"
                        else:
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            yield f"data: {json.dumps({'done': True, 'timestamp': timestamp})}\n\n"

                    except Exception as e:
                        print(f"⚠️ 语音生成失败: {e}")
                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        yield f"data: {json.dumps({'done': True, 'timestamp': timestamp})}\n\n"

                    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    chat_history.append({"type": "User", "content": user_message, "timestamp": timestamp})
                    chat_history.append({"type": "Assistant", "content": ai_response, "timestamp": timestamp})
                    threading.Thread(target=extract_and_save_memory, args=(user_message,)).start()

                else:
                    print(f"云端 API 报错: {res.text}")
                    yield f"data: {json.dumps({'chunk': '芯宝的大脑服务器开小差了，稍后再试哦 QwQ', 'done': True})}\n\n"

            except Exception as e:
                yield f"data: {json.dumps({'chunk': f'连接云端大脑失败，检查一下网络。报错: {e}', 'done': True})}\n\n"

        return Response(stream_with_context(generate_stream()), mimetype="text/event-stream")

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/upload", methods=["POST"])
def handle_upload():
    if "file" not in request.files:
        return jsonify({"error": "没有文件"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "未选择文件"}), 400

    if file:
        filename = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filename)

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        chat_history.append({
            "type": "User",
            "content": f"上传了文件：{file.filename}",
            "timestamp": timestamp,
        })

        ai_response = f"文件「{file.filename}」已接收，这是固定的处理结果"
        chat_history.append({
            "type": "Assistant",
            "content": ai_response,
            "timestamp": timestamp,
        })

        return jsonify({"response": ai_response, "filename": file.filename, "timestamp": timestamp})


@app.route("/api/history", methods=["GET"])
def get_history():
    return jsonify({"history": chat_history})


@app.route("/api/history", methods=["DELETE"])
def clear_history():
    global chat_history
    chat_history = []
    return jsonify({"message": "历史记录已清空"})


if __name__ == "__main__":
    classifier, retrieve_answer = init_models()

    from waitress import serve

    print("🚀 芯宝后端已启动! (基于 Waitress 生产级容器)")
    print("🌐 监听地址: http://0.0.0.0:5000")
    print("📱 前端页面: http://<香橙派IP>:5000/")
    serve(app, host="0.0.0.0", port=5000, threads=4)
