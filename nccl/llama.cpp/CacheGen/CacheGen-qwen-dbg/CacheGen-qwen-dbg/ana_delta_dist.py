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

# 转换形状为: [Layer, Head, Seq_Len, Channel]
#layers_k = [layer.cpu().float() for layer in past_key_values]
# ==========================================
# 3. 动态提取无损 Key Cache 并转换为 4D 巨张量（修复 Tuple 报错）
# ==========================================
layers_k = []
for layer_pair in past_key_values:
    # layer_pair 是一个元组: (key_tensor, value_tensor)
    # 我们只提取 Key 张量 (索引为 0)
    k_tensor = layer_pair[0].cpu().float()
    layers_k.append(k_tensor)

# 转换为 4D 巨张量: [Layer, Head, Seq_Len, Channel]
# 如果模型输出包含 batch 维度且为 1，使用 .squeeze(1) 抹去（注意：Llama3/Mistral 的 K 通常在 dim=1 或 dim=0）
# 这里我们直接 stack，并确保最终形状对齐
kv_matrix = torch.stack(layers_k, dim=0)

# 如果第一维度保留了 batch 导致形状变成了 [Layer, Batch, Head, Seq_Len, Channel]
if len(kv_matrix.shape) == 5:
    kv_matrix = kv_matrix.squeeze(1)  # 抹去 Batch=1 的维度

num_layers, num_heads, seq_len, num_channels = kv_matrix.shape
print(f"✅ KV Cache 张量捕获成功，重塑后维度: [Layer={num_layers}, Head={num_heads}, Seq_Len={seq_len}, Channel={num_channels}]")
#kv_matrix = torch.stack(layers_k, dim=0).squeeze(1)
#num_layers, num_heads, seq_len, num_channels = kv_matrix.shape

# ==========================================
# 3. 核心算法：实现 CacheGen 局部锚点差分编码
# ==========================================
def compute_cachegen_delta(kv_tensor, chunk_size=20):
    """
    鲁棒版 CacheGen 差分计算：
    强制深度拷贝并刷新内存连续性，确保广播相减逐元素对齐。
    """
    # 强制将整个张量在内存中平铺连续化，避免非连续切片带来的广播 Bug
    kv_contiguous = kv_tensor.detach().clone().contiguous()
    delta_tensor = kv_contiguous.clone()
    
    num_layers, num_heads, length, num_channels = kv_contiguous.shape
    
    for start in range(0, length, chunk_size):
        end = min(start + chunk_size, length)
        if end - start <= 1:
            continue
            
        # 1. 显式提取锚点 Token，并强制 clone 内存，确保其形状与母体彻底解耦
        anchor = kv_contiguous[:, :, start:start+1, :].clone().contiguous()
        
        # 2. 提取当前块需要做差分的所有其余 Token 
        block_tokens = kv_contiguous[:, :, start+1:end, :].clone().contiguous()
        
        # 3. 显式做差分（此时 block_tokens 和 anchor 内存完全干净、对齐）
        diff = block_tokens - anchor
        
        # 4. 把计算好的纯净残差安全写回
        delta_tensor[:, :, start+1:end, :] = diff
        
    return delta_tensor

def calculate_shannon_entropy(tensor_data, bins=256):
    """
    自适应量程熵计算（还原 CacheGen 真实状态）：
    不再使用固定的 (-4, 4)，而是让 256 个 Bin 100% 紧密包裹当前数据的实际极值。
    """
    flat_data = tensor_data.numpy().flatten()
    # 动态获取当前切片的极值范围，这等同于无损算术编码前的自适应缩放（Adaptive Scaling）
    dynamic_range = (flat_data.min(), flat_data.max())
    
    counts, _ = np.histogram(flat_data, bins=bins, range=dynamic_range)
    probs = counts / (np.sum(counts) + 1e-12)
    return entropy(probs, base=2)

# 执行差分，采用 CacheGen 推荐的窗口大小 20
CHUNK_SIZE = 20
delta_kv_matrix = compute_cachegen_delta(kv_matrix, chunk_size=CHUNK_SIZE)

# ==========================================
# 4. 香农熵核心计算函数（统一量程离散化）
# ==========================================
#def calculate_shannon_entropy(tensor_data, bins=256, val_range=(-4.0, 4.0)):
#    flat_data = tensor_data.numpy().flatten()
#    counts, _ = np.histogram(flat_data, bins=bins, range=val_range)
#    probs = counts / (np.sum(counts) + 1e-12)
#    return entropy(probs, base=2)

# ==========================================
# 5. 定量提取三组维度的切片并计算指标
# ==========================================
target_layer, target_channel = 16, 0
target_token_idx = 200

# 【组合 A】固定 Token 混杂状态
slice_A = kv_matrix[:, :, target_token_idx, :]
entropy_A = calculate_shannon_entropy(slice_A)

# 【组合 B】层-通道隔离状态 (原始连续值)
slice_B = kv_matrix[target_layer, :, :, target_channel]
entropy_B = calculate_shannon_entropy(slice_B)

# 🔥【组合 C】层-通道隔离 + 叠加 Delta 差分编码
slice_C = delta_kv_matrix[target_layer, :, :, target_channel]
entropy_C = calculate_shannon_entropy(slice_C)

# 计算统计学方差变化
var_B = torch.var(slice_B).item()
var_C = torch.var(slice_C).item()

# ==========================================
# 6. 打印定量验证报告
# ==========================================
print("\n" + "="*70)
print("📊 CACHEGEN + DELTA 差分编码极端熵减报告")
print("="*70)
print(f"【组合 A】固定 Token 混杂状态             -> 熵值: {entropy_A:.4f} bits")
print(f"【组合 B】层-通道隔离状态 (原始连续分布)     -> 熵值: {entropy_B:.4f} bits")
print(f"【组合 C】层-通道隔离 + 叠加 Delta 差分    -> 熵值: {entropy_C:.4f} bits")
print("-"*70)
print(f"🔥 核心定量飞跃：")
print(f"1. 纯空间隔离效益(A->B)   : 熵值降低了 {entropy_A - entropy_B:.2f} bits。")
print(f"2. 叠加 Delta 编码效益(B->C): 熵值进一步暴跌了 {entropy_B - entropy_C:.2f} bits！")
print(f"3. 📈 波动收紧：差分后数据方差从 {var_B:.5f} 缩减至 {var_C:.5f} (收拢了 {var_B/var_C:.2f} 倍)")
print("="*70)

# ==========================================
# 7. 动态全景作图分析 (Plotting)
# ==========================================
plt.figure(figsize=(18, 5))

# 📊 子图 1：组合 A 分布 (无序混杂)
plt.subplot(1, 3, 1)
plt.hist(slice_A.numpy().flatten(), bins=100, range=(-4, 4), color='#8b0000', alpha=0.7, edgecolor='none')
plt.title(f"Combination A: Fixed Token {target_token_idx}\n(Broad & High Entropy: {entropy_A:.2f} bits)")
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 📊 子图 2：组合 B 分布 (初步展现偏斜)
plt.subplot(1, 3, 2)
plt.hist(slice_B.numpy().flatten(), bins=100, range=(-4, 4), color='#00008b', alpha=0.7, edgecolor='none')
plt.title(f"Combination B: Fixed Layer {target_layer} & Channel {target_channel}\n(Skewed Distribution: {entropy_B:.2f} bits)")
plt.xlabel("Value Range")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 📊 子图 3：组合 C 分布 (叠加 Delta 后的极端低熵尖峰)
plt.subplot(1, 3, 3)
plt.hist(slice_C.numpy().flatten(), bins=100, range=(-4, 4), color='#006400', alpha=0.7, edgecolor='none')
plt.title(f"Combination C: Layer-Channel + Delta\n(Extreme Low Entropy: {entropy_C:.2f} bits)")
plt.xlabel("Delta Value (Centered near 0)")
plt.ylabel("Frequency")
plt.grid(axis='y', linestyle='--', alpha=0.5)

# 保存与输出
plt.tight_layout()
output_img = "cachegen_delta_analysis.png"
plt.savefig(output_img, dpi=300)
print(f"\n🎉 [作图分析完成] 可视化全景对比图已成功保存至本地：'{output_img}'")

