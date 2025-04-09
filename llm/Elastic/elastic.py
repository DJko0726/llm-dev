import openai
import sys
import os
import tiktoken
import pandas as pd
from datetime import datetime, timedelta
from elasticsearch import Elasticsearch as ESClient, helpers
from langchain_huggingface import HuggingFaceEmbeddings

class Elasticsearch:
    def __init__(self, index):
        self.host = os.getenv("ES_HOST", "localhost")
        self.port = int(os.getenv("ES_PORT", 9200))
        self.scheme = "http"
        self.username = os.getenv("ES_USER")
        self.password = os.getenv("ES_PASSWORD")
        self.client = ESClient(
            [{"scheme":self.scheme, "host": self.host, "port": self.port}],
            basic_auth=(self.username , self.password) if self.username and self.password else None
        )

        self.index = os.getenv("ES_INDEX")
        self.index_body = {
            "mappings": {
                "properties": {
                    "text": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768,
                        "index": True,
                        "similarity": "cosine"
                    }
                }
            }
        }
        self.chinese_embedding_name = "shibing624/text2vec-base-chinese"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.chinese_embedding_name
        )


    def create_index(self):
        if not self.client.indices.exists(index=self.index):
            self.client.indices.create(index=self.index, body=self.index_body)
            print(f"索引 {self.index} 已創建")
        else:
            print(f"索引 {self.index} 已存在")

    def delete_index(self):
        if self.client.indices.exists(index=self.index):
            self.client.indices.delete(index=self.index)
            print(f"索引 {self.index} 已刪除")
        else:
            print(f"索引 {self.index} 不存在")

    def chunk_text(self, text, chunk_size=512, model="text-embedding-ada-002"):
        """將文本 chunk"""
        if text is None:
            print("Warning: Received None text to chunk")
            return []
        if not isinstance(text, str):
            try:
                text = str(text)
                print(f"Warning: Converting non-string text to string: {type(text)}")
            except:
                print(f"Error: Cannot convert text of type {type(text)} to string")
                return []

        if not text.strip():
            print("Warning: Received empty text to chunk")
            return []
        
        try:
            tokenizer = tiktoken.encoding_for_model(model)
            tokens = tokenizer.encode(text)

            chunks = [tokens[i:i + chunk_size] for i in range(0, len(tokens), chunk_size)]
            return [tokenizer.decode(chunk) for chunk in chunks]
        except Exception as e:
            print(f"Error chunking text: {e}")
            return [text[i:i + 500] for i in range(0, len(text), 500)]

    def insert(self,  texts):
        if not self.client.indices.exists(index=self.index):
            self.create_index()
        """將 Chunking 後的資料插入 Elasticsearch"""
        actions = []
        for doc in texts:
                chunks = self.chunk_text(doc)

                for idx, chunk in enumerate(chunks):
                    query_vector = self.embeddings.embed_documents([chunk])[0]

                    action = {
                        "_index": self.index,  
                        "_source": {
                            "text": chunk,
                            "embedding": query_vector
                        }
                    }
                    actions.append(action)

        helpers.bulk(self.client, actions)
        return actions

    def search(self, query, top_k=5):
        """向量檢索"""
        query_vector = self.embeddings.embed_documents([query])[0]
        search_query = {
            "size": top_k,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": """
                            cosineSimilarity(params.query_vector, 'embedding') + 1.0
                        """,
                        "params": {
                            "query_vector": query_vector
                        }
                    }
                }
            }
        }
        try:
            response = self.client.search(index=self.index, body=search_query)
            hits = response['hits']['hits']
            results = []
            for hit in hits:
                results.append({
                    "id": hit["_id"],
                    "text": hit["_source"]["text"],
                    "embedding": hit["_source"]["embedding"],
                    "score": hit["_score"]
                })
            return results
        except Exception as e:
            return []

    def generate_prompt(self, query, top_k=5):
        """執行 RAG 生成答案"""
        retrieved_docs = self.search(query, top_k=top_k)
        return [doc['text'] for doc in retrieved_docs if 'text' in doc]

