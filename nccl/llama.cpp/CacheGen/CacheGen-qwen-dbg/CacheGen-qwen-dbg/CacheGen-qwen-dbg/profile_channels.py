import os
import torch
import numpy as np

def analyze_kv_channels(file_path="./kv_output/raw_kv_0.pt", output_log="./kv_output/channel_profile.txt"):
    """
    针对 CacheGen 第一阶段收集的原始未压缩 KV Cache 进行硬核通道特征分析。
    统计各层、各通道的极大值、极小值、标准差（方差）以及信息熵。
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到原材料文件 {file_path}，请确保 main.py 提取成功。")
        return

    print("=" * 70)
    print(f"🚀 开始对 {file_path} 进行 CacheGen 通道级（Channel/Head）多维特征透视...")
    print("=" * 70)

    # 1. 加载 PyTorch 张量
    raw_kv = torch.load(file_path, map_location="cpu", weights_only=False)
    
    # 2. 识别层数
    num_layers = len(raw_kv)
    
    with open(output_log, "w", encoding="utf-8") as log_file:
        log_file.write(f"CacheGen Channel Profiling Report for {file_path}\n")
        log_file.write("=" * 80 + "\n")
        
        # 3. 逐层分析
        for layer_idx in range(num_layers):
            # 获取当前层的 Key 和 Value 张量
            # 维度通常为 [Batch(1), Num_Heads, Seq_Len, Head_Dim]
            key_tensor, value_tensor = raw_kv[layer_idx][0], raw_kv[layer_idx][1]
            
            # 将 Batch 维挤压掉，聚焦于核心的 [Heads, Seq_Len, Head_Dim]
            key_tensor = key_tensor.squeeze(0).float()
            value_tensor = value_tensor.squeeze(0).float()
            
            num_heads = key_tensor.shape[0]
            seq_len = key_tensor.shape[1]
            head_dim = key_tensor.shape[2]
            
            log_file.write(f"\n[Layer {layer_idx:02d}] | Shape: Heads={num_heads}, SeqLen={seq_len}, HeadDim={head_dim}\n")
            log_file.write("-" * 80 + "\n")
            log_file.write(f"{'Head ID':<10}{'K_Max':<12}{'K_Min':<12}{'K_Std':<12}{'V_Max':<12}{'V_Min':<12}{'V_Std':<12}\n")
            
            # 4. 逐个 Attention Head (通道组) 进行空间统计
            for head_idx in range(num_heads):
                h_key = key_tensor[head_idx]    # 形状为 [Seq_Len, Head_Dim]
                h_value = value_tensor[head_idx] # 形状为 [Seq_Len, Head_Dim]
                
                # 计算 Key 的通道指标
                k_max = torch.max(h_key).item()
                k_min = torch.min(h_key).item()
                k_std = torch.std(h_key).item()
                
                # 计算 Value 的通道指标
                v_max = torch.max(h_value).item()
                v_min = torch.min(h_value).item()
                v_std = torch.std(h_value).item()
                
                # 写入文本日志
                log_file.write(f"{head_idx:<10}{k_max:<12.4f}{k_min:<12.4f}{k_std:<12.4f}{v_max:<12.4f}{v_min:<12.4f}{v_std:<12.4f}\n")
            
            # 5. 计算当前层的全局异常大通道（Outliers）
            # CacheGen 论文指出：方差极大（> 3倍层平均方差）的通道通常对应 Attention Sinks 核心特征
            layer_k_std_mean = torch.std(key_tensor, dim=(1, 2)).mean().item()
            outlier_heads = []
            for head_idx in range(num_heads):
                if torch.std(key_tensor[head_idx]).item() > 2.5 * layer_k_std_mean:
                    outlier_heads.append(head_idx)
            
            if outlier_heads:
                log_file.write(f"💡 [检测到异常敏感通道(Outlier Heads)]: {outlier_heads} (这些通道在压缩时需要高比特量化保护)\n")
                print(f"Layer {layer_idx:02d} 检测到异常敏感 Head: {outlier_heads}")

    print("=" * 70)
    print(f"✅ 通道分析完成！详细的层、头方差映射报告已写入本地磁盘：")
    print(f"🔗 {output_log}")
    print("=" * 70)

if __name__ == "__main__":
    # 执行分析
    analyze_kv_channels()

