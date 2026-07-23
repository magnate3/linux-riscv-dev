import time
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-placeholder"
)

# 模拟约 20k Token 的长文本
LONG_DOCUMENT = """
CacheBlend 是一种针对大语言模型（LLM）的先进 KV Cache 传输与融合技术。
传统的 vLLM Prefix Caching 要求历史文本必须从第一个字符开始完全一致才能复用缓存。
而 CacheBlend 突破了这一限制。它允许将长文档放置在 Prompt 的任意位置，甚至是多个文档交叉组合。
通过在块级别（Chunk-level）对 KV Cache 进行动态检索、匹配和 Selective Recomputation（选择性重计算），
CacheBlend 能够将非前缀匹配情况下的首字延迟（TTFT）降低 50% 以上，极大地节约了 GPU 算力。
""" * 10

def send_request(prompt_text, stage_name):
    print(f"\n🚀 === 正在执行: {stage_name} ===")
    
    start_time = time.time()
    ttft = None
    
    # 💡 核心修改：从 chat.completions 换成 completions 接口
    #model="/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1/",
    response = client.completions.create(
        model="/models/qwen2.5-0.5b",
        prompt=prompt_text,  # 使用 prompt 代替 messages
        temperature=0.0,
        max_tokens=20,
        stream=True
    )
    
    # 流式读取
    for chunk in response:
        if chunk.choices[0].text:  # 结构微调：由 delta.content 变为 text
            if ttft is None:
                ttft = time.time() - start_time
                print(f"⏱️ 首字延迟 (TTFT): {ttft:.4f} 秒")
            print(chunk.choices[0].text, end="", flush=True)
            
    print(f"\n✅ 请求完成。总耗时: {time.time() - start_time:.4f} 秒")
    return ttft

if __name__ == "__main__":
    # 第一次请求
    prompt_1 = f"以下是参考材料：\n{LONG_DOCUMENT}\n请总结材料的核心内容："
    ttft_1 = send_request(prompt_1, "第一次请求（冷启动 / 写入 KV 缓存）")
    
    time.sleep(2)
    
    # 第二次请求（颠倒前缀，破坏前缀一致性）
    prompt_2 = f"【加急处理】请阅读随附材料：\n{LONG_DOCUMENT}\n请总结材料的核心内容："
    ttft_2 = send_request(prompt_2, "第二次请求（改变前缀 / 测试 CacheBlend 命中）")
    
    # 对比结果
    print("\n📊 ===== CacheBlend 性能对比 =====")
    print(f"第一次请求 (冷启动 TTFT): {ttft_1:.4f} 秒")
    print(f"第二次请求 (混合复用 TTFT): {ttft_2:.4f} 秒")
    if ttft_2 < ttft_1:
        speedup = (ttft_1 - ttft_2) / ttft_1 * 100
        print(f"🎉 成功！首字延迟降低了 {speedup:.2f}%！")
    else:
        print("❌ 警告：延迟未下降，请检查 vLLM 后台 LMCache 日志。")

