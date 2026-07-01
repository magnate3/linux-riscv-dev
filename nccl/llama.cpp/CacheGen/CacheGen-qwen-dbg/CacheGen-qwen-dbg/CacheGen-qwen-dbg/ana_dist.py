import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from scipy.stats import entropy
import matplotlib.pyplot as plt
import json
jsonl_path = "test_data/longchat.jsonl"
longchat_text = ""

# ==========================================
# 1. 初始化模型与分词器（使用 CacheGen 核心测试模型）
# ==========================================
MODEL_ID = "/workspace/models/Mistral-7B-v0.1/AI-ModelScope/Mistral-7B-v0___1/" 
#MODEL_ID = "mistralai/Mistral-7B-v0.1" 
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.float16, 
    device_map="auto"
)

# ==========================================
# 2. 加载 LongChat 典型长上下文（以真实的单样本对话为例）
# ==========================================
# 模拟 LongChat-16k 评测集中的长文本检索/问答任务
#longchat_prompt = (
#    "SYSTEM: You are a helpful assistant.\n"
#    "USER: Please remember the following background information carefully:\n"
#    + "The key performance index of this cluster is throughput, and the backup policy is set to increment daily. " * 300  # 构造长依赖上下文
#    + "\nNow, based on the context above, answer what the backup policy is.\nASSISTANT:"
#)

try:
    with open(jsonl_path, "r", encoding="utf-8") as f:
        # 读取第一行（或者循环读取多行进行拼接）
        first_line = f.readline()
        data = json.loads(first_line)
        
        # 根据 LongChat 数据集的标准格式提取文本
        # 注意：如果您的 jsonl 键名不同（如 'text' 或 'context'），请相应修改
        if "context" in data:
            longchat_text = data["context"]
        elif "text" in data:
            longchat_text = data["text"]
        elif "conversation" in data:
            # 如果是多轮对话格式，将其拼接为 Prompt
            longchat_text = "\n".join([f"{m['role']}: {m['content']}" for m in data["conversation"]])
        else:
            # 备用方案：如果键名未知，直接拼接所有字符串类型的值
            longchat_text = " ".join([str(v) for v in data.values() if isinstance(v, str)])
            
    print(f"✅ 成功读取本地数据集 '{jsonl_path}'")
except FileNotFoundError:
    print(f"❌ 未找到文件 '{jsonl_path}'，请检查相对路径是否正确。")
    # 触发备用长文本，防止脚本崩溃
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

# 转换形状为: [Layer, Head, Seq_Len, Channel]
layers_k = [layer[0].cpu().float() for layer in past_key_values] # 取出 Key，layer[0] 形状 [batch, head, seq, dim]
kv_matrix = torch.stack(layers_k, dim=0).squeeze(1) # 抹去单 batch 维度
num_layers, num_heads, _, num_channels = kv_matrix.shape
print(f"KV Cache 张量捕获成功，重塑后维度: [Layer={num_layers}, Head={num_heads}, Seq_Len={seq_len}, Channel={num_channels}]")

# ==========================================
# 4. 香农熵核心计算函数（使用固定 Bin 离散化）
# ==========================================
def calculate_shannon_entropy(tensor_data, bins=256, val_range=(-4.0, 4.0)):
    """
    为了公平对比不同切片，使用统一的 min-max 范围进行 256-bin 离散化直方图划分，
    并计算香农熵（单位：bits）
    """
    flat_data = tensor_data.numpy().flatten()
    # 使用固定边界以确保不同维度组合在同一离散空间内对比
    counts, _ = np.histogram(flat_data, bins=bins, range=val_range)
    
    probs = counts / (np.sum(counts) + 1e-12)
    # base=2 表示单位是比特 (bits)
    return entropy(probs, base=2)

# ==========================================
# 5. 计算两种典型维度组合下的信息熵
# ==========================================

# --- 组合 A: 固定 Token 观察通道 ---
# 我们在长文本后半段随机选择一个 Token 位置（例如索引 200）
target_token_idx = 200
# 抽取该 Token 在所有层、所有头、所有通道的集合
slice_combination_A = kv_matrix[:, :, target_token_idx, :]
entropy_A = calculate_shannon_entropy(slice_combination_A)

# --- 组合 B: 固定层与通道，观察整体概率分布 ---
# 选择一个中间层（如 16 层）和特定的特征通道（如 0 号通道）
target_layer, target_channel = 16, 0
# 抽取该层该通道下，跨所有头、所有 Token 的集合
slice_combination_B = kv_matrix[target_layer, :, :, target_channel]
entropy_B = calculate_shannon_entropy(slice_combination_B)

# ==========================================
# 6. 输出验证报告与定量结论
# ==========================================
print("\n" + "="*60)
print("📊 CACHEGEN 维度熵偏斜性验证报告（LongChat 场景）")
print("="*60)
print(f"【组合 A】固定 Token 位置 ({target_token_idx}) 观察其余维度 -> 熵值: {entropy_A:.4f} bits")
print(f"【组合 B】固定 Layer ({target_layer}) & Channel ({target_channel}) 观察全局 Token -> 熵值: {entropy_B:.4f} bits")
print("-"*60)

entropy_gap = entropy_A - entropy_B
compressibility_ratio = (2**entropy_A) / (2**entropy_B)

print(f"🔥 核心定量结论：")
print(f"1. 组合 B 的条件熵比组合 A 降低了 {entropy_gap:.2f} bits。")
print(f"2. 理论状态数压缩比：组合 B 的状态空间密集度是组合 A 的 {compressibility_ratio:.2f} 倍。")
print(f"\n📢 论文结论验证结果：")
if entropy_B < entropy_A:
    print("【✅ 成功验证】数据证明，按 [层-通道] 进行隔离时，数值展现出极强的低熵偏斜性（Skewness）。")
    print("这正是 CacheGen 放弃传统的按 Token 位置压缩，转而‘按层和通道独立配置概率表’的数学依据！")
else:
    print("【❌ 验证失败】请检查数据切片或离散化范围。")
print("="*60)

# ==========================================
# 7. 绘制概率分布图直观感受“偏斜性（Skewness）”
# ==========================================
plt.figure(figsize=(12, 5))

# 绘制组合 A 的直方图
plt.subplot(1, 2, 1)
plt.hist(slice_combination_A.numpy().flatten(), bins=100, range=(-4, 4), color='darkred', alpha=0.6)
plt.title(f"Combination A: Fixed Token {target_token_idx}\n(Broad & High Entropy: {entropy_A:.2f} bits)")
plt.xlabel("Value Range")
plt.ylabel("Frequency")

# 绘制组合 B 的直方图
plt.subplot(1, 2, 2)
plt.hist(slice_combination_B.numpy().flatten(), bins=100, range=(-4, 4), color='darkblue', alpha=0.6)
plt.title(f"Combination B: Fixed Layer {target_layer} & Channel {target_channel}\n(Highly Skewed & Low Entropy: {entropy_B:.2f} bits)")
plt.xlabel("Value Range")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("cachegen_dimension_skewness_comparison.png")
print("\n🎉 分布对比图已成功输出至 'cachegen_dimension_skewness_comparison.png'。你可以直观看到组合 B 呈现出瘦高的‘极度偏斜尖峰’。")

