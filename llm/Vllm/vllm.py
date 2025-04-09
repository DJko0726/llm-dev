import sys
import os
import json, requests

class Vllm:
    def __init__(self, model):
        self.model = model
        
    def llm_stream(self, sampling_params):
        bytes_arr = b""
        to_decode = ""
        sampling_params['model'] = self.model
        sampling_params['stream'] = True
        sent_msg = ""
        # 一般而言 vllm 會去監聽 8000 port, 如果有改請跟著改
        vllm_port = 8000
        for item in requests.request(
            method='post',
            url='http://localhost:'+str(vllm_port)+'/v1/completions',
            headers={'accept': 'text/event-stream', 'Content-Type': 'application/json; charset=utf-8;'},
            json=sampling_params,
            allow_redirects=False,
            stream=True):
            if not item:
                continue
            bytes_arr += item
            bytes_idx = len(bytes_arr)
            while bytes_idx>0:
                try:
                    to_decode += bytes_arr[:bytes_idx].decode('utf-8')
                    bytes_arr = bytes_arr[bytes_idx:]
                    break
                except:
                    bytes_idx -= 1
            try:
                accumulated_text = ""
                for one_line in to_decode.split("\n"):
                    if one_line.startswith("data: "):
                        try:
                            accumulated_text += json.loads(one_line[len("data: "):])["choices"][0]["text"]
                        except:
                            pass
                if len(accumulated_text) > len(sent_msg):
                    for new_char in accumulated_text[len(sent_msg):]:
                        yield new_char
                    sent_msg = accumulated_text
            except:
                pass