import sys
import os
import tiktoken
import pandas as pd
from datetime import datetime, timedelta
from Prompt.prompt import Prompt
import requests

class OpenAI:
    def __init__(self):
        self.url = os.getenv("AZURE_OPENAI_URL")
        self.api_key = os.getenv("API_KEY")
        self.prompt = Prompt()
        self.headers = {
            "Content-Type": "application/json",
            "api-key": self.api_key
        }
        
    def generate_title(self, content, max_tokens=100, temperature=0.7):
        # prompt = self.prompt.title_prompt(content)
        prompt_template = self.prompt.get_prompt(
            "generate_title"
        ) 
        prompt = str(prompt_template.format(question=content))
        post_data = {
            "messages": [
                {"role": "system", "content": "你是一位命名標題的專家。"},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        try:
            response = requests.post(self.url, headers=self.headers, json=post_data) 
            return {"response": response.json(), "prompt": prompt}
        except (KeyError, IndexError, ValueError):
            return {"error": "Unexpected response format", "details": response.text, "prompt": prompt}
        
        
    def reply(self, content, max_tokens=100, temperature=0.7):
        prompt_template = self.prompt.get_prompt(
            "reply"
        ) 
        prompt = str(prompt_template.format(question=content['question'], docs=content['docs']))
        messages = [ {"role": "system", "content": "你是一位來自台灣的知識小幫手，能夠接續對話並提供具體、精確的回應。如果查詢語句本身不是一個問句，請幫我組合成一個合適的問題，協助解決這個問題"}]
        messages.extend(content['history'])
        messages.append({"role": "user", "content": prompt})
        post_data = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        try:
            response = requests.post(self.url, headers=self.headers, json=post_data) 
            return {"response": response.json()}
        except (KeyError, IndexError, ValueError):
            return {"error": "Unexpected response format", "details": response.text}