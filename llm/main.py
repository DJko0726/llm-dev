from flask import Flask, request, jsonify
import os
import sys
import requests
from Elastic.elastic import Elasticsearch
from OpenAI.openai import OpenAI
import tiktoken

app = Flask(__name__)
# CORS(app)
# encoding = tiktoken.get_encoding("cl100k_base")


@app.route("/")
def hello():
    return "Hello World!"

# 加入rag要觸發加入向量資料庫
@app.route("/es.add", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or 'documents' not in data or data['documents'] == "":
        return jsonify({'error': 'No documents provided'}), 400
    try:
        es = Elasticsearch(os.getenv("ES_INDEX"))
        actions = es.insert(data['documents'])
        return jsonify({"success": actions})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/prompt.generate", methods=["POST"])
def get_all_data():
    data = request.get_json()
    if not data or 'query' not in data or data['query'] == "":
        return jsonify({'error': 'No query provided'}), 400

    try:
        es = Elasticsearch(os.getenv("ES_INDEX"))
        response = es.generate_prompt(query=data['query'], top_k=data['top_k'])
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
# 透過LLM回傳title 
@app.route("/title.generate", methods=["POST"])
def generate_title():
    data = request.get_json()
    if not data or 'content' not in data or data['content'] == "":
        return jsonify({'error': 'No content provided'}), 400
    #不同語言模型實作
    try:
        content = '\n\n'.join(data['content'])
        # 檢查 content 是否為字串
        openai = OpenAI()
        response = openai.generate_title(content, max_tokens=20, temperature=0.5)
        new_title = response["response"]["choices"][0]["message"]["content"]
        return {"title": new_title, "prompt": response["prompt"]}
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/chat_history.openai.reply", methods=["POST"])
def chat_history_reply():
    data = request.get_json()
    if not data or 'question' not in data or data['question'] == "" or 'docs' not in data or data['docs'] == "" or 'history' not in data:
        return jsonify({'error': 'Missing question or docs'}), 400
    
    try:
        openai = OpenAI()
        response = openai.reply(data, max_tokens=200, temperature=0.5)
        reply = response["response"]["choices"][0]["message"]["content"]
        return {"reply": reply}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
