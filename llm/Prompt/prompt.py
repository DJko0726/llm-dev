import sys
import os
from langchain.prompts import ChatPromptTemplate

class Prompt:
    def __init__(self):
        self.templates = {
            "reply": """
                    使用者的查詢語句：{question}

                    請參考來自向量資料庫的相關資料：{docs}

                    請根據以上資訊回復。
                """,
            "get_token": """
                查詢語句: {question}\n請將查詢語句擷取重要的詞彙以json格式呈現
                """,
            "generate_title": """
                查詢語句: {question}\n請將查詢語句擷取重要的詞彙，並命名一個符合查詢的標題，不可以是問句，最多10個字，不需要多餘的解釋。
                """,
        }
       
    def get_prompt(self, template_name):
        if template_name in self.templates:
            template = self.templates[template_name]
            # return ChatPromptTemplate.from_messages([("user", template)])
            return template
        else:
            raise ValueError(f"Template '{template_name}' not found.")
        