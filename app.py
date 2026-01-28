from flask import Flask, render_template, request, Response, stream_with_context
from openai import OpenAI
import json
import httpx
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    messages = data.get('messages', [])
    config = data.get('config', {})
    
    base_url = config.get('baseUrl', 'https://api.openai.com/v1')
    api_key = config.get('apiKey', '')
    model = config.get('model', 'gpt-3.5-turbo')
    
    # 终极超时配置，防止 Zeabur 或 Cloudflare 超时
    timeout = httpx.Timeout(600.0, connect=30.0, read=600.0)
    
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=timeout)
    )

    def generate():
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True,
                extra_body={"chat_template_kwargs": {"thinking": True}} # 兼容深度思考模型
            )
            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    content = chunk.choices[0].delta.content
                    if content:
                        yield f"data: {json.dumps({'content': content})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream')

if __name__ == '__main__':
    # Zeabur 部署会自动提供 PORT 环境变量
    port = int(os.environ.get("PORT", 4000))
    app.run(host='0.0.0.0', port=port)
