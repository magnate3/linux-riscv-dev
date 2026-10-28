



```
sudo docker run -d --gpus all \
           --shm-size 32g \
           -p 8001:8001 \
           -v /pytorch/models/:/models \
           --ipc=host \
           --name sglang \
          swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/lmsysorg/sglang:v0.4.6.post5-cu124  \
    python3 -m sglang.launch_server \
           --model /models/qwen2.5-0.5b \
           --tp 1 \
           --trust-remote-code \
           --host 0.0.0.0 \
           --port 8001 \
           --mem-fraction-static 0.75

```




```
sudo docker  exec -it sglang bash
root@3e00549618b6:/sgl-workspace# curl http://localhost:8001/v1/models
{"object":"list","data":[{"id":"/models/qwen2.5-0.5b","object":"model","created":1792143956,"owned_by":"sglang","root":"/models/qwen2.5-0.5b","max_model_len":32768}]}

```

```
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "你好，请问你是谁？"}]
  }'
```

```
root@3e00549618b6:/sgl-workspace# curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/qwen2.5-0.5b",
    "messages": [{"role": "user", "content": "你好，请问你是谁？"}]
  }'
{"id":"8c84b611363d442b80e4956f1cd066a0","object":"chat.completion","created":1792143989,"model":"/models/qwen2.5-0.5b","choices":[{"index":0,"message":{"role":"assistant","content":"我是来自阿里云的Qwen模型，专门针对自然语言处理和对话系统等领域而设计的。如果您有任何问题或需要帮助，请随时告诉我。","reasoning_content":null,"tool_calls":null},"logprobs":null,"finish_reason":"stop","matched_stop":151645}],"usage":{"prompt_tokens":35,"total_tokens":68,"completion_tokens":33,"prompt_tokens_details":null}}
root@3e00549618b6:/sgl-workspace# 
```