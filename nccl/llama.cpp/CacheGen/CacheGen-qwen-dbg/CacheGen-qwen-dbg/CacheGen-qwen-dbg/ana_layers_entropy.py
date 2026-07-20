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
with torch.no_grad():
    outputs = model(**inputs, use_cache=True)
    past_key_values = outputs.past_key_values  

# 解包 Tuple，提取无损 4D 巨张量: [Layer, Head, Seq_Len, Channel]
layers_k = []
for layer_pair in past_key_values:
    layers_k.append(layer_pair[0].cpu().float())
kv_matrix = torch.stack(layers_k, dim=0).squeeze(1)
num_layers, num_heads, seq_len, num_channels = kv_matrix.shape

# ==========================================
# 3. CacheGen 差分核心算法（解耦连续内存版）
# ==========================================
def compute_cachegen_delta(kv_tensor, chunk_size=20):
    kv_contiguous = kv_tensor.detach().clone().contiguous()
    delta_tensor = kv_contiguous.clone()
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        if end - start <= 1:
            continue
        anchor = kv_contiguous[:, :, start:start+1, :].clone().contiguous()
        block_tokens = kv_contiguous[:, :, start+1:end, :].clone().contiguous()
        delta_tensor[:, :, start+1:end, :] = block_tokens - anchor
    return delta_tensor

delta_kv_matrix = compute_cachegen_delta(kv_matrix, chunk_size=20)

def calculate_shannon_entropy(tensor_data, bins=256):
    flat_data = tensor_data.numpy().flatten()
    dynamic_range = (flat_data.min(), flat_data.max())
    counts, _ = np.histogram(flat_data, bins=bins, range=dynamic_range)
    probs = counts / (np.sum(counts) + 1e-12)
    return entropy(probs, base=2)

# ==========================================
# 4. 自动化全层扫描（Layer Scan）
# ==========================================
print("\n🚀 开始全层动态熵扫描，正在分析分层异构特性...")

layers_raw_entropy = []
layers_delta_entropy = []
layers_entropy_gain = []
target_channel = 0  # 固定观察 0 号通道

for idx in range(num_layers):
    # 提取该层原始状态
    slice_B_raw = kv_matrix[idx, :, :, target_channel]
    ent_B = calculate_shannon_entropy(slice_B_raw)
    
    # 提取该层差分状态
    slice_C_delta = delta_kv_matrix[idx, :, :, target_channel]
    ent_C = calculate_shannon_entropy(slice_C_delta)
    
    # 计算本层的差分无损增益（正值越大代表该层越适合被压缩，特性越稳定内敛）
    gain = ent_B - ent_C
    
    layers_raw_entropy.append(ent_B)
    layers_delta_entropy.append(ent_C)
    layers_entropy_gain.append(gain)
    
    # 打印部分关键层的实时进度
    if idx % 4 == 0 or idx == num_layers - 1:
        print(f"  [Layer {idx:02d}] 原始熵: {ent_B:.3f} bits | 差分熵: {ent_C:.3f} bits | 差分收益: {gain:.3f} bits")

# ==========================================
# 5. 打印分层敏感度验证报告
# ==========================================
shallow_avg_gain = np.mean(layers_entropy_gain[:10])   # 浅层（1-10层）
deep_avg_gain = np.mean(layers_entropy_gain[-10:])     # 深层（后10层）

print("\n" + "="*70)
print("📊 CACHEGEN “分层异构敏感度” 铁律定性验证报告")
print("="*70)
print(f"【模型浅层（0-09层）】平均差分编码收益: {shallow_avg_gain:.4f} bits")
print(f"【模型深层（22-31层）】平均差分编码收益: {deep_avg_gain:.4f} bits")
print("-"*70)
print("🔥 终极科研结论：")
if deep_avg_gain > shallow_avg_gain:
    ratio = deep_avg_gain / (shallow_avg_gain + 1e-12)
    print(f"【✅ 成功验证铁律】深层 KV Cache 压缩收益是浅层的 {ratio:.2f} 倍！")
    print(" 物理真相：浅层承载了长文本的微观词法特征，数值敏感、混乱且多变，稍加压缩就会引发熵值震荡（精度剧损）。")
    print(" 深层则高度凝聚了宏观语义，特征极度收敛。CacheGen 论文据此设计了：‘浅层高保真、深层高压缩’的异构流式传输机制。")
else:
    print("【❓ 规律持平】请检查长文本语料语义是否过于单一。")
print("="*70)

# ==========================================
# 6. 绘制全层敏感度趋势曲线图（Line Plot）
# ==========================================
plt.figure(figsize=(12, 6))
x_layers = np.arange(num_layers)

plt.plot(x_layers, layers_raw_entropy, marker='o', linestyle='-', color='#00008b', label='Raw KV Entropy (Combination B)')
plt.plot(x_layers, layers_delta_entropy, marker='s', linestyle='--', color='#006400', label='Delta KV Entropy (Combination C)')

# 阴影填充：高亮压缩收益空间
plt.fill_between(x_layers, layers_raw_entropy, layers_delta_entropy, where=(np.array(layers_raw_entropy) >= np.array(layers_delta_entropy)),
                 color='#98fb98', alpha=0.4, label='Compression Gain (Bits Saved)')

plt.title("CacheGen Layer-wise Heterogeneous Sensitivity Analysis", fontsize=14, fontweight='bold')
plt.xlabel("LLM Layer Index (0 to 31)", fontsize=12)
plt.ylabel("Shannon Entropy (bits)", fontsize=12)
plt.xticks(x_layers)
plt.grid(axis='both', linestyle=':', alpha=0.6)
plt.legend(fontsize=11, loc='upper left')

# 标注浅层与深层区域
plt.axvspan(0, 9, color='red', alpha=0.07)
plt.text(3, min(layers_delta_entropy)+0.1, "Shallow Layers\n(High Sensitivity)", color='darkred', ha='center', fontweight='bold')

plt.axvspan(22, 31, color='green', alpha=0.07)
plt.text(26.5, min(layers_delta_entropy)+0.1, "Deep Layers\n(High Tolerance)", color='darkgreen', ha='center', fontweight='bold')

plt.tight_layout()
output_img = "cachegen_layer_sensitivity_trend.png"
plt.savefig(output_img, dpi=300)
print(f"\n🎉 [趋势分析完成] 全层敏感度曲线图已保存至本地：'{output_img}'")

