import os
import torch
import numpy as np

def generate_static_cdf_profiles(
    raw_kv_path="./kv_output/raw_kv_0.pt", 
    output_cdf_path="./LMCache/lmcache/storage_backend/serde/profiles/qwen_cdf.pt",
    bins=16, # 默认 4-bit 量化
    chunk_size=256
):
    """
    分析原始 KV 缓存的残差分布，计算并导出全局多维离线静态 CDF 查找表矩阵
    """
    if not os.path.exists(raw_kv_path):
        print(f"❌ 错误：未找到原始材料文件 {raw_kv_path}")
        return

    print("⏳ [Profile] 开始全局直方图统计与定点化 CDF 矩阵构建...")
    raw_kv = torch.load(raw_kv_path, map_location="cuda", weights_only=False)
    num_layers = len(raw_kv)
    
    # 构造 CacheGen 经典的高维离线大表格式
    # 维度: [Layers, 2(K/V), 2(0:Anchor, 1:Delta), bins + 1]
    global_cdf_matrix = torch.zeros((num_layers, 2, 2, bins + 1), dtype=torch.int32)

    for layer_idx in range(num_layers):
        for kv_idx, tensor in enumerate([raw_kv[layer_idx][0], raw_kv[layer_idx][1]]): # 0:Key, 1:Value
            # tensor 形状: [1, Num_Heads, Seq_Len, Head_Dim]
            tensor_flat = tensor.squeeze(0).float()
            num_heads, seq_len, head_dim = tensor_flat.shape

            # ----------------------------------------
            # 1. 提取并计算 Anchor (锚点) 分布
            # ----------------------------------------
            anchors = tensor_flat[:, 0::chunk_size, :] # 每隔 chunk_size 采一个锚点
            # 模拟线性量化到 [0, bins-1] 区间
            max_a = torch.max(torch.abs(anchors)) + 1e-6
            quant_anchors = torch.round((anchors / max_a) * (bins // 2 - 1)) + (bins // 2)
            quant_anchors = quant_anchors.clamp(0, bins - 1)

            # 统计 Anchor 直方图频次
            hist_anchor = torch.histc(quant_anchors, bins=bins, min=0, max=bins-1)
            global_cdf_matrix[layer_idx, kv_idx, 0, :] = counts_to_fixed_cdf(hist_anchor)

            # ----------------------------------------
            # 2. 提取并计算 Delta (残差) 分布
            # ----------------------------------------
            # 计算 Chunk 内与锚点的残差
            deltas = torch.zeros_like(tensor_flat)
            for i in range(seq_len):
                anchor_idx = (i // chunk_size) * chunk_size
                deltas[:, i, :] = tensor_flat[:, i, :] - tensor_flat[:, anchor_idx, :]
            
            # 模拟残差量化
            max_d = torch.max(torch.abs(deltas)) + 1e-6
            quant_deltas = torch.round((deltas / max_d) * (bins // 2 - 1)) + (bins // 2)
            quant_deltas = quant_deltas.clamp(0, bins - 1)

            # 统计 Delta 直方图频次 (你会发现 0 附近的频次呈尖锐的拉普拉斯分布)
            hist_delta = torch.histc(quant_deltas, bins=bins, min=0, max=bins-1)
            global_cdf_matrix[layer_idx, kv_idx, 1, :] = counts_to_fixed_cdf(hist_delta)

    # 3. 永久固化导出
    os.makedirs(os.path.dirname(output_cdf_path), exist_ok=True)
    torch.save(global_cdf_matrix, output_cdf_path)
    print(f"✅ 全局静态 CDF 矩阵计算完毕并成功固化导出至:\n🔗 {output_cdf_path}")

def counts_to_fixed_cdf(counts):
    """将直方图频次规约转化为 16-bit 定点递增的规范 CDF 数组"""
    probs = counts.float() / (counts.sum() + 1e-8)
    float_cdf = torch.zeros(len(probs) + 1, device=counts.device)
    float_cdf[1:] = torch.cumsum(probs, dim=0)
    int_cdf = torch.round(float_cdf * 65536).to(torch.int32)
    int_cdf[-1] = 65536
    # 强制单调递增防御
    for i in range(1, len(int_cdf)):
        if int_cdf[i] <= int_cdf[i - 1]:
            int_cdf[i] = int_cdf[i - 1] + 1
    int_cdf[-1] = 65536
    return int_cdf

if __name__ == "__main__":
    generate_static_cdf_profiles()

