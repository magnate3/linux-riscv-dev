import time
import requests

url = "http://localhost:8000/v1/completions"
MODEL_NAME = "/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1"

# 缩短一点 Prompt 长度（约 1800 tokens），加快冷启动排查速度，但足以触发 LMCache
#long_context = "很久很久以前，在一个遥远的国度里..." * 100 
#
#payload = {
#    "model": MODEL_NAME,
#    "prompt": f"背景信息：{long_context}\n\n问题：总结上述故事的主旨是什么？\n答案：",
#    "temperature": 0.1,
#    "max_tokens": 15, 
#    "stream": False
#}
# 换用带有实质语义的英文句子，强迫 GPU 进行完整的 Attention 矩阵计算
long_context = "Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain. " * 150 

# 修改 test_cache.py 第 18 行：
#long_context = "Deep learning is a subset of machine learning, which is essentially a neural network with three or more layers. These neural networks attempt to simulate the behavior of the human brain. " * 600

payload = {
            "model": MODEL_NAME,
                "prompt": f"Context information: {long_context}\n\nBased on the text above, tell me what is deep learning in one sentence:\nAnswer:",
                    "temperature": 0.0, # 锁定温度，确保两次请求路径完全一致
                        "max_tokens": 15, 
                            "stream": False
                            }

# 修改后的 payload 块
#payload = {
#            "model": MODEL_NAME,
#                "prompt": f"System: You are a helpful assistant. Read the following context and answer the question short.\n\nContext: {long_context}\n\nUser: What is deep learning?\nAssistant: Deep learning is",
#            "temperature": 0.0, 
#            "max_tokens": 10,  # 只需要生成几个 token 即可，把时间留给 LMCache
#            "stream": False
#           }

headers = {"Content-Type": "application/json"}
session = requests.Session()

def send_request(label):
    print(f"正在发送 {label} ...")
    start_time = time.time()
    try:
        response = session.post(url, headers=headers, json=payload, timeout=60)
        end_time = time.time()
        
        res_json = response.json()
        if "error" in res_json:
            print(f"❌ vLLM 引擎返回错误: {res_json['error']['message']}")
            return True, 0
            
        ttft = end_time - start_time
        # 打印原始返回的文本，加上中括号防止其为空时看不见
        raw_text = res_json['choices'][0]['text']
        print(f"💬 模型实际续写内容为: [{raw_text}]")
        return True, ttft
    except Exception as e:
        print(f"❌ 请求发生异常: {e}")
        return False, 0

# ---- 第 1 次请求 ----
success1, ttft1 = send_request("第 1 次请求（冷启动）")
if success1:
    print(f"⏱️ 第 1 次请求耗时: {ttft1:.4f} 秒\n")
    
    print("等待 3 秒中...")
    time.sleep(3)
    
    # ---- 第 2 次请求 ----
    success2, ttft2 = send_request("第 2 次请求（预期命中 LMCache）")
    if success2:
        print(f"⏱️ 第 2 次请求耗时: {ttft2:.4f} 秒\n")
        print(f"🚀 === LMCache 节省了: {ttft1 - ttft2:.4f} 秒 ===")
