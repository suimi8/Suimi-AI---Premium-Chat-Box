from flask import Flask, render_template, request, Response, stream_with_context, jsonify
from openai import OpenAI
import json
import httpx
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
PORT = int(os.environ.get("PORT", 4000))

# --- 数据库初始化 ---
DB_PATH = 'chat.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db_connection()
    # 对话会话表
    conn.execute('''CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        title TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )''')
    # 消息记录表
    conn.execute('''CREATE TABLE IF NOT EXISTS messages (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT,
                        role TEXT,
                        content TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (session_id) REFERENCES sessions (id)
                    )''')
    conn.commit()
    conn.close()

init_db()

@app.route('/')
def index():
    return render_template('index.html')

# --- API 接口：历史记录管理 ---

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    conn = get_db_connection()
    sessions = conn.execute('SELECT * FROM sessions ORDER BY updated_at DESC').fetchall()
    conn.close()
    return jsonify([dict(s) for s in sessions])

@app.route('/api/sessions', methods=['POST'])
def create_session():
    data = request.json
    session_id = data['id']
    title = data.get('title', '新对话')
    conn = get_db_connection()
    conn.execute('INSERT INTO sessions (id, title) VALUES (?, ?)', (session_id, title))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/sessions/<session_id>', methods=['DELETE'])
def delete_session(session_id):
    conn = get_db_connection()
    conn.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
    conn.execute('DELETE FROM sessions WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()
    return jsonify({"status": "success"})

@app.route('/api/messages/<session_id>', methods=['GET'])
def get_messages(session_id):
    conn = get_db_connection()
    messages = conn.execute('SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC', (session_id,)).fetchall()
    conn.close()
    return jsonify([dict(m) for m in messages])

# --- 核心聊天接口 (带保存逻辑) ---

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    config = data.get('config', {})
    session_id = data.get('sessionId')
    
    base_url = config.get('baseUrl', 'https://api.openai.com/v1')
    api_key = config.get('apiKey', '')
    model = config.get('model', 'gpt-3.5-turbo')
    
    timeout = httpx.Timeout(600.0, connect=30.0, read=600.0)
    client = OpenAI(api_key=api_key, base_url=base_url, http_client=httpx.Client(timeout=timeout))

    def generate():
        full_assistant_content = ""
        try:
            # 1. 保存用户最后一条消息
            if messages and session_id:
                last_user_msg = messages[-1]
                if last_user_msg['role'] == 'user':
                    save_message(session_id, 'user', last_user_msg['content'])

            # 2. 调用 API
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                extra_body={"chat_template_kwargs": {"thinking": True}}
            )
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        full_assistant_content += content
                        yield f"data: {json.dumps({'content': content})}\n\n"
            
            # 3. 保存 AI 的完整回复
            if session_id and full_assistant_content:
                save_message(session_id, 'assistant', full_assistant_content)
                # 更新会话时间
                update_session_time(session_id, full_assistant_content[:50])

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

def save_message(session_id, role, content):
    conn = get_db_connection()
    # 检查会话是否存在，不存在则创建
    exists = conn.execute('SELECT id FROM sessions WHERE id = ?', (session_id,)).fetchone()
    if not exists:
        conn.execute('INSERT INTO sessions (id, title) VALUES (?, ?)', (session_id, content[:20]))
    
    conn.execute('INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)', (session_id, role, content))
    conn.commit()
    conn.close()

def update_session_time(session_id, title_hint):
    conn = get_db_connection()
    # 如果是第一条回复，把对话标题改得更有意义一点
    conn.execute('UPDATE sessions SET updated_at = CURRENT_TIMESTAMP WHERE id = ?', (session_id,))
    conn.commit()
    conn.close()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT)
