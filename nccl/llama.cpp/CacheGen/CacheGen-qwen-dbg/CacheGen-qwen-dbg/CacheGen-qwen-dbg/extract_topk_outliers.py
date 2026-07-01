import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

jsonl_path = "test_data/longchat.jsonl"
longchat_text = ""

# ==========================================
# 0. 定义离群通道提取函数
# ==========================================
def extract_topk_outliers(kv_tensor, outlier_ratio=0.01):
    """
    输入 kv_tensor 形状: [Layer, Head, Seq_Len, Channel] 或 [Seq_Len, Channel]
    """
    head_dim = kv_tensor.shape[-1]
    topk_count = max(1, int(head_dim * outlier_ratio)) # 计算需要保留的通道数，至少为1
    
    # 统计通道最大绝对值
    flat_tensor = kv_tensor.reshape(-1, head_dim).abs()
    max_per_channel, _ = torch.max(flat_tensor, dim=0)
    
    # 直接提取前 Top-K 个最强势的通道
    _, topk_indices = torch.topk(max_per_channel, k=topk_count)
    
    return topk_indices.sort()[0].tolist()

# ==========================================
# 1. 初始化模型与分词器
# ==========================================
MODEL_ID = "/workspace/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1/" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# ==========================================
# 2. 读取并处理文本数据
# ==========================================
try:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        first_line = f.readline()
        data = json.loads(first_line)
        
        if "context" in data:
            longchat_text = data["context"]
        elif "text" in data:
            longchat_text = data["text"]
        elif "conversation" in data:
            longchat_text = "\n".join([f"{m['role']}: {m['content']}" for m in data["conversation"]])
        else:
            longchat_text = " ".join([str(v) for v in data.values() if isinstance(v, str)])
            
    print(f"✅ 成功读取本地数据集 '{jsonl_path}'")
except FileNotFoundError:
    print(f"❌ 未找到文件 '{jsonl_path}'，触发备用长文本。")
    longchat_text = "The key performance index of this cluster is throughput. " * 300

inputs = tokenizer(longchat_text, return_tensors="pt").to(model.device)
seq_len = inputs.input_ids.shape[1]
print(f"成功加载 LongChat 任务文本，Token 总序列长度 (Seq_Len): {seq_len}")

# ==========================================
# 3. 动态提取无损 Key Cache 并转换为 4D 巨张量
# ==========================================
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values  

# layer[0] 原始形状: [batch_size=1, num_key_heads=8, seq_len, head_dim=128]
# 去掉 batch 维度，转换为: [num_key_heads, seq_len, head_dim]
layers_k_list = [layer[0].squeeze(0).cpu().float() for layer in past_key_values]

# 真正拼接成 4D 巨张量，形状为: [Layer=32, Head=8, Seq_Len, Channel=128]
final_k_tensor = torch.stack(layers_k_list, dim=0)
print(f"KV 巨张量构建成功，最终 Key 形状: {final_k_tensor.shape}")

# ==========================================
# 4. 分析并输出结果
# ==========================================
# 提取全局最强势的 1% 通道（128 * 0.01 向上取整为 1 个通道）
global_outliers = extract_topk_outliers(final_k_tensor, outlier_ratio=0.01)
print(f"🚀 全局 Key Cache 离群通道索引 (Top 1%): {global_outliers}")

# 如果你想逐层分析：
print("\n--- 逐层离群通道分析 (Top 2%): ---")
for layer_idx in range(final_k_tensor.shape[0]):
    layer_tensor = final_k_tensor[layer_idx] # [Head, Seq_Len, Channel]
    layer_outliers = extract_topk_outliers(layer_tensor, outlier_ratio=0.02)
    print(f"Layer {layer_idx:02d} Outliers Indices: {layer_outliers}")
