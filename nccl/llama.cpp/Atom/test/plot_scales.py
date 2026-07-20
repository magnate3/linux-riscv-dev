import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch

import os
import torch
import numpy as np
from typing import Dict, Union
import torch
import numpy as np
from typing import Dict

def analyze_hessian_sensitivity(act_scales: Dict[str, torch.Tensor], protect_ratio: float = 0.2):
    """
    基于 Hessian 矩阵对角线元素对 LLaMA 模型进行二阶敏感度分析，并自动分配混合精度 Bits。
    
    参数:
    - act_scales: 包含各层统计量的字典 (必须使用 metric='hessian' 收集得到)
    - protect_ratio: 全球最敏感的层中，百分之多少需要升级为 8-bit 保护区（默认 20%）
    """
    print("=" * 100)
    print(f"🚀 开始执行基于 Hessian 矩阵的二阶敏感度分析 (保护比例: {protect_ratio*100}%)")
    print("=" * 100)
    
    # 严格锁定的 6 个核心投影层
    target_ops = ["q_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"]
    
    layer_sensitivity = {}
    total_model_hessian_trace = 0.0

    # ----------------------------------------------------
    # 1. 提取并计算各算子的 Hessian Trace (二阶扰动敏感度)
    # ----------------------------------------------------
    for name, scale_tensor in act_scales.items():
        # 确保是 Hessian 统计量且是我们关心的层输入
        if ".input" not in name or not any(op in name for op in target_ops):
            continue
            
        # 提取层 ID 和算子名 (例如: layers.0.self_attn.q_proj)
        clean_name = name.replace("model.", "").replace(".input", "")
        
        # Hessian 矩阵的迹 (Trace) 即为其对角线元素的总和
        # 迹越大，说明这一层引入量化噪声时，模型 Loss 飙升得越剧烈
        hessian_trace = float(scale_tensor.sum().cpu().item())
        
        layer_sensitivity[clean_name] = hessian_trace
        total_model_hessian_trace += hessian_trace

    # ----------------------------------------------------
    # 2. 全局敏感度排序与归一化
    # ----------------------------------------------------
    # 按照 Hessian Trace 降序排序（敏感度从高到低）
    sorted_sensitivity = sorted(layer_sensitivity.items(), key=lambda x: x[1], reverse=True)
    
    # 计算需要被高位保护（采用 8-bit）的算子数量
    num_to_protect = int(len(sorted_sensitivity) * protect_ratio)
    
    print(f"{'算子物理路径 (Operator Path)':<45} | {'Hessian 绝对迹 (Trace)':<22} | {'相对敏感度贡献':<14} | {'决策分配 (Bit)'}")
    print("-" * 100)
    
    bit_allocations = {}
    
    for rank, (op_name, trace_val) in enumerate(sorted_sensitivity):
        # 计算该算子对全模型二阶误差的贡献占比
        contribution = (trace_val / (total_model_hessian_trace + 1e-9)) * 100
        
        # 贪心决策：排名前列的敏感算子升级为 8-bit，其余下放到 4-bit
        if rank < num_to_protect:
            decision_bit = 8
            visual_tag = "🔴 [HIGH SENSITIVE] 🚀 升级保护"
        else:
            decision_bit = 4
            visual_tag = "🟢 [ROBUST]          🟢 压缩放行"
            
        bit_allocations[op_name] = decision_bit
        print(f"{op_name:<45} | {trace_val:<22.4f} | {contribution:>12.4f}% | {decision_bit}-bit ({visual_tag})")
        
    print("=" * 100)
    print(f"📊 寻优总结: 全模型共 {len(sorted_sensitivity)} 个核心线性层")
    print(f"   -> 已将 {num_to_protect} 个危险层保护为 8-bit")
    print(f"   -> 已将 {len(sorted_sensitivity) - num_to_protect} 个稳健层压缩为 4-bit")
    print("=" * 100)
    
    print(bit_allocations)
    #return bit_allocations

def analyze_llama_outliers(
    act_scales: Dict[str, Union[torch.Tensor, np.ndarray]], 
    threshold_sigma: float = 3.0, 
    top_n: int = 5
):
    """
    针对 LLaMA 核心投影层的激活值统计字典进行 Outlier 通道审计。
    
    参数:
    - act_scales: 包含各层统计量的字典 (如由 get_act_stats_llama 收集到的结果)
    - threshold_sigma: 判定为离群值的标准（偏离均值的标准差倍数，默认 3.0）
    - top_n: 详细打印前 N 个最大、最恶性的通道 ID
    """
    print("=" * 95)
    print(f"{'层与算子名称 (Layer & Operator)':<50} | {'总通道数':<8} | {'离群通道数':<10} | {'占比 (%)':<8} | {'最大值/均值':<10}")
    print("-" * 95)
    
    # 严格锁定的 6 个核心投影层（分析它们的输入 input 特征）
    target_operators = [
        "self_attn.q_proj.input", 
        "self_attn.k_proj.input", 
        "self_attn.v_proj.input",
        "mlp.gate_proj.input", 
        "mlp.up_proj.input", 
        "mlp.down_proj.input"
    ]
    
    detailed_reports = {}
    
    for name, scale_data in act_scales.items():
        # 过滤：确保只分析我们关心的那 6 个算子
        if not any(op in name for op in target_operators):
            continue
            
        # 安全转换为 numpy.float32 数组
        if isinstance(scale_data, torch.Tensor):
            scales = scale_data.float().cpu().numpy()
        else:
            scales = np.array(scale_data, dtype=np.float32)
            
        num_channels = len(scales)
        mean_val = np.mean(scales)
        std_val = np.std(scales)
        max_val = np.max(scales)
        
        # 数学定义：超过 (均值 + 3 * 标准差) 的通道判定为 Outlier
        cutoff = mean_val + threshold_sigma * std_val
        outlier_indices = np.where(scales > cutoff)[0]
        num_outliers = len(outlier_indices)
        outlier_ratio = (num_outliers / num_channels) * 100
        
        # 计算离群值峰值相比于常规平均水平被放大了多少倍
        ratio_max_mean = max_val / (mean_val + 1e-9)
        
        # 降序排序，揪出最严重的 Top-N 个“大魔王”通道
        top_indices = np.argsort(scales)[::-1][:top_n]
        top_values = scales[top_indices]
        
        # 简化打印键名
        clean_name = name.replace("model.", "")
        print(f"{clean_name:<50} | {num_channels:<8} | {num_outliers:<10} | {outlier_ratio:>7.2f}% | {ratio_max_mean:>9.1f}x")
        
        detailed_reports[name] = {
            "top_channels": top_indices.tolist(),
            "top_values": top_values.tolist(),
            "mean": float(mean_val),
            "max": float(max_val)
        }
        
    print("=" * 95)
    print(f"\n🔥 详细诊断：各算子 Top-{top_n} 极值通道分布 (格式: 通道号 #ID [激活强度])")
    print("-" * 95)
    for name, report in detailed_reports.items():
        clean_name = name.replace("model.", "")
        channels_str = ", ".join([f"#{ch} [{val:.2f}]" for ch, val in zip(report['top_channels'], report['top_values'])])
        print(f"🔹 {clean_name:<45} -> {channels_str}")
    print("=" * 95)



def plot_peak_to_rms_divergence(act_scales, save_dir="./saved_plots"):
    """
    计算并绘制全模型跨层的【峰值能量对数差 (Peak-to-RMS Divergence)】演变图
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\n[Divergence Master] 正在精确计算全模型各层的峰值能量对数差...")
    
    target_suffix = "mlp.gate_proj.input"
    layer_divergence = {}
    
    for full_name, stat_tensor in act_scales.items():
        clean_name = full_name.replace("model.", "")
        if target_suffix in clean_name:
            try:
                layer_idx = None
                for part in clean_name.split('.'):
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                
                if layer_idx is not None:
                    raw_arr = stat_tensor.detach().float().cpu().numpy()
                    
                    # 1. 计算当前层的最大值 (Peak)
                    peak_val = np.max(raw_arr)
                    # 2. 模拟标准的 RMSNorm 分母：计算全层 4096 个通道的均方根 (RMS)
                    rms_val = np.sqrt(np.mean(raw_arr ** 2))
                    
                    # 3. 核心数学公式：计算峰值能量对数差
                    # 这一步反映的是经过 RMSNorm 归一化缩放后，最强的那个离群值通道还剩下多少个数量级的“突兀度”
                    divergence = np.log10(peak_val / (rms_val + 1e-8))
                    layer_divergence[layer_idx] = divergence
            except Exception:
                continue

    sorted_layers = sorted(layer_divergence.keys())
    if not sorted_layers:
        print("[Divergence Master Warning] 未能匹配到标准层算子，跳过对数差图绘制。")
        return

    divergence_vals = [layer_divergence[l] for l in sorted_layers]

    # ========================================================
    # 绘制峰值能量对数差演变图
    # ========================================================
    plt.figure(figsize=(10, 5))
    plt.plot(sorted_layers, divergence_vals, color='#9467bd', linewidth=2.5, marker='D', markersize=6, label='Peak-to-RMS Divergence')
    
    plt.title('Cross-Layer Peak-to-RMS Divergence Evolution', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Mistral Transformer Layer Index', fontsize=12)
    plt.ylabel('Divergence Score (Log10 Scale)', fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    #plt.legend(loc='upper left')
    
    # 标注出关键转折点（突变区）
    plt.axvspan(0, 4, color='red', alpha=0.1, label='Init Alert')
    plt.axvspan(28, 31, color='red', alpha=0.1, label='Collapse Alert')
    plt.legend(loc='upper left', fontsize=10)
    # 打印终端数值报告，方便写论文复制
    print("\n--- 📊 全模型峰值能量对数差 (Peak-to-RMS Divergence) 定量报告 ---")
    print(f"{'层数 (Layer)':<12} | {'对数差得分 (Divergence Score)':<25} | {'学术学诊断 (Diagnosis)'}")
    print("-" * 65)
    for l in sorted_layers:
        score = layer_divergence[l]
        if score > 1.8:
            diag = "🚨 极致畸变 (RMSNorm 彻底卸防)"
        elif score > 1.4:
            diag = "⚠️ 畸变抬头 (防线吃紧)"
        else:
            diag = "✅ 完美平抑 (气垫稳固)"
        print(f"Layer {l:<7} | {score:<25.4f} | {diag}")
    print("-" * 65)

    plot_path = os.path.join(save_dir, "peak_to_rms_divergence.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [✓] 成功绘制峰值能量对数差演变图 -> {plot_path}")

def evaluate_residual_vs_norm(act_scales):
    """
    量化评估全模型 32 层中【残差累积】与【归一化缩放】的对抗状态
    """
    print("\n" + "="*25 + " 神经系统不均匀性与失衡评估报告 " + "="*25)
    print(f"{'层数 (Layer)':<12} | {'最大能量 (Max)':<14} | {'全层波动 (STD)':<14} | {'畸变比 (Max/Median)':<20} | {'健康诊断 (Status)'}")
    print("-" * 85)
    
    target_suffix = "mlp.gate_proj.input"
    layer_metrics = {}
    
    for full_name, stat_tensor in act_scales.items():
        clean_name = full_name.replace("model.", "")
        if target_suffix in clean_name:
            # 提取层数
            layer_idx = None
            for part in clean_name.split('.'):
                if part.isdigit():
                    layer_idx = int(part)
                    break
            if layer_idx is not None:
                raw_arr = stat_tensor.detach().float().cpu().numpy()
                layer_metrics[layer_idx] = raw_arr
                
    for l in sorted(layer_metrics.keys()):
        data = layer_metrics[l]
        max_val = np.max(data)
        std_val = np.std(data)
        median_val = np.median(data)
        ratio = max_val / (median_val + 1e-8)
        
        # 根据畸变比自动进行学术诊断分级
        if ratio > 90:
            status = "🚨 严重畸变 (天花板破裂，必须重排护航)"
        elif ratio > 60:
            status = "⚠️ 中度风险 (气垫吃紧，建议 Keeper 覆盖)"
        else:
            status = "✅ 完美抑制 (水涨船高，皮实钝感区)"
            
        print(f"Layer {l:<7} | {max_val:<14.2f} | {std_val:<14.2f} | {ratio:<20.2f} | {status}")
    print("=" * 85)

def plot_layer_variance_analysis(act_scales, save_dir="./saved_plots"):
    
    os.makedirs(save_dir, exist_ok=True)
    print(f"\n[Fluctuation Master] 正在计算全模型各层通道的方差与波动程度...")
    
    target_suffix = "mlp.gate_proj.input"
    layers_max = {}
    layers_std = {} # 用标准差（方差开根）直接反应通道物理波动的绝对幅度
    
    for full_name, stat_tensor in act_scales.items():
        clean_name = full_name.replace("model.", "")
        if target_suffix in clean_name:
            try:
                # 动态提取层数数字
                layer_idx = None
                for part in clean_name.split('.'):
                    if part.isdigit():
                        layer_idx = int(part)
                        break
                
                if layer_idx is not None:
                    raw_arr = stat_tensor.detach().float().cpu().numpy()
                    
                    # 1. 统计当前层所有通道的最大绝对值
                    layers_max[layer_idx] = np.max(raw_arr)
                    # 2. 统计当前层整层所有通道的波动程度（标准差）
                    layers_std[layer_idx] = np.std(raw_arr)
            except Exception:
                continue

    sorted_layers = sorted(layers_max.keys())
    if not sorted_layers:
        print("[Fluctuation Master Warning] 未能匹配到标准层算子，跳过方差图绘制。")
        return

    # 组织绘图数据
    max_vals = [layers_max[l] for l in sorted_layers]
    std_vals = [layers_std[l] for l in sorted_layers]

    # ========================================================
    # 开始绘制双轴联动演变图（极其硬核的消融论据图）
    # ========================================================
    fig, ax1 = plt.subplots(figsize=(11, 5.5))

    # 轴 1：绘制全模型最大激活值的雪崩曲线上涨
    color = '#d62728' # 红色代表高能爆发
    ax1.set_xlabel('Mistral Transformer Layer Index', fontsize=12)
    ax1.set_ylabel('Layer Maximum Absolute Value', color=color, fontsize=12)
    line1 = ax1.plot(sorted_layers, max_vals, color=color, linewidth=2.5, marker='o', label='Max Channel Value')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_yscale('log') # 使用对数纵轴，因为跨度极大
    ax1.grid(True, which="both", ls="--", alpha=0.3)

    # 轴 2：在右侧镜像绘制整层通道方差（标准差）波动曲线
    ax2 = ax1.twinx()  
    color = '#1f77b4' # 蓝色代表降温安全垫
    ax2.set_ylabel('Whole-Layer Channels Fluctuation (Standard Deviation)', color=color, fontsize=12)
    line2 = ax2.plot(sorted_layers, std_vals, color=color, linewidth=2.5, linestyle='--', marker='s', label='Layer Volatility (STD)')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_yscale('log')

    # 合并图例
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.title('The Interplay of Maximum Outlier and Whole-Layer Volatility across Transformer Layers', fontsize=13, fontweight='bold', pad=15)
    
    # 加上硬核文字诊断注释
    plt.text(12, min(std_vals)*3, "💡 Fluctuation Buffer:\nAlthough mid-layers have larger values,\ntheir overall variance expands proportionally,\nmaking them robust (Dull Zone) under RMSNorm.", 
             bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'), fontsize=10)

    plot_path = os.path.join(save_dir, "layer_fluctuation_vs_max.png")
    plt.savefig(plot_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [✓] 成功绘制全模型方差波动联动演变图 -> {plot_path}")

def diagnose_specific_layer(act_scales, target_layer_idx=0):
    """
    定向诊断特定层中最敏感的前 10 个特征通道
    """
    print(f"\n {target_layer_idx} layer  (mlp.gate_proj.input)  Top-10 ..")
    
    target_key = f"layers.{target_layer_idx}.mlp.gate_proj.input"
    
    found_tensor = None
    for k, v in act_scales.items():
        if target_key in k.replace("model.", ""):
            found_tensor = v
            break
            
    if found_tensor is None:
        print(f"not found {target_layer_idx} ")
        return

    raw_arr = found_tensor.detach().float().cpu().numpy()
    
    sorted_indices = np.argsort(raw_arr)[::-1]
    
    print(f"---  {target_layer_idx} layer Top-10  ---")
    print(f"{' (Rank)':<12} | {'(Channel ID)':<20} | {'Hessian/Scale score':<20}")
    print("-" * 60)
    
    for rank in range(10):
        channel_id = sorted_indices[rank]
        score = raw_arr[channel_id]
        print(f"Top {rank+1:<8} | Channel {channel_id:<12} | {score:<20.4f}")
        
    max_val = raw_arr[sorted_indices[0]]
    median_val = np.median(raw_arr)
    ratio = max_val / (median_val + 1e-8)
    print("-" * 60)
    print(f"outlier max_val/median_val 【{ratio:.1f}】 ！")

def plot_hessian_analysis(act_scales, save_dir="./saved_plots"):
    """
    对 get_act_stats_llama 计算出的 Hessian 敏感度或 Scales 进行多维度图表分析
    """
    #os.makedirs(save_dir, exist_ok=True)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(f"\n start Hessian sensitivity: {save_dir}）...")
    
    target_suffix = "mlp.gate_proj.input"
    layer_stats = {}
    
    for full_name, stat_tensor in act_scales.items():
        clean_name = full_name.replace("model.", "")
        if target_suffix in clean_name:
            try:
                layer_idx = int(clean_name.split('.')[1])
                layer_stats[layer_idx] = stat_tensor.detach().float().cpu().numpy()
            except Exception:
                continue

    if not layer_stats:
        print("[Warning] in act_scales not math mlp.gate_proj.input")
        for i, (k, v) in enumerate(list(act_scales.items())[:10]):
            layer_stats[i] = v.detach().float().cpu().numpy()

    sorted_layers = sorted(layer_stats.keys())
    print(sorted_layers) 
    # ========================================================
    # ========================================================
    plt.figure(figsize=(10, 5))
    example_layer = sorted_layers[len(sorted_layers) // 3] # 取靠前的一个中间层
    sample_data = layer_stats[example_layer]
    
    sorted_data = np.sort(sample_data)[::-1]
    
    plt.plot(sorted_data, color='#1f77b4', linewidth=2.5, label=f'Layer {example_layer}')
    plt.yscale('log') # 因为大数和小数差好几个数量级，必须用对数纵轴
    plt.title(f"Hessian Sensitivity Distribution (Layer {example_layer} - Sorted)", fontsize=14, fontweight='bold')
    plt.xlabel("Sorted Channel Index", fontsize=12)
    plt.ylabel("Sensitivity Value (Log Scale)", fontsize=12)
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend()
    
    plt.annotate('Outliers (Top 1%)', xy=(5, sorted_data[0]), xytext=(len(sorted_data)//4, sorted_data[0]/10),
                 arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6))
    
    plot_path1 = os.path.join(save_dir, "channel_long_tail_distribution.png")
    plt.savefig(plot_path1, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [1/2] 成功绘制通道长尾断崖分布图 -> {plot_path1}")

    # ========================================================
    # 图 2：大模型跨层敏感度热力图（Cross-layer Heatmap）
    # ========================================================
    # 构建热力图矩阵：[层数, 通道数]，由于通道可能很多（如 11008），我们对其按分位数进行滑窗聚合或降采样
    heatmap_matrix = []
    for l in sorted_layers:
        # 对每层的数据降序后，均匀抽取100个采样点看宏观趋势
        sorted_arr = np.sort(layer_stats[l])[::-1]
        sampled_arr = sorted_arr[np.linspace(0, len(sorted_arr)-1, 128, dtype=int)]
        heatmap_matrix.append(sampled_arr)
    
    heatmap_matrix = np.log10(np.array(heatmap_matrix) + 1e-6) # 取对数避免色彩被极大值吃掉
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(heatmap_matrix, cmap="rocket_r", xticklabels=20, yticklabels=sorted_layers)
    plt.title("Cross-Layer Activation Hessian Sensitivity Map (Log10 Scale)", fontsize=14, fontweight='bold')
    plt.xlabel("Top Channels (Sorted by Importance ->)", fontsize=12)
    plt.ylabel("Transformer Layer Index", fontsize=12)
    
    plot_path2 = os.path.join(save_dir, "cross_layer_sensitivity_heatmap.png")
    plt.savefig(plot_path2, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" [2/2] 成功绘制跨层全局敏感度热力图 -> {plot_path2}")

if __name__ == '__main__':
    save_dir="saved"
    model_name=""
    dataset="wikitext2"
    act_sort_metric="hessian"
    scales_save_path = f'{save_dir}/{model_name}_act_scales_{dataset}_{act_sort_metric}.pt'
  
    if os.path.exists(scales_save_path):
        print(f" [✓] 检测到本地已存在激活值特征缓存，正在直接从磁盘秒加载: {scales_save_path}")
        act_scales = torch.load(scales_save_path, map_location='cpu') # 先加载到cpu，防止瞬间挤爆显存
        
        analyze_llama_outliers(act_scales, threshold_sigma=3.0, top_n=5)
        analyze_hessian_sensitivity(act_scales, 0.2)
        #plot_hessian_analysis(act_scales, save_dir="./saved_plots")
        #plot_layer_variance_analysis(act_scales, save_dir="./saved_plots")
        #plot_peak_to_rms_divergence(act_scales, save_dir="./saved_plots")
        #evaluate_residual_vs_norm(act_scales)
        #diagnose_specific_layer(act_scales, target_layer_idx=0) 
        #diagnose_specific_layer(act_scales, target_layer_idx=15)
        #diagnose_specific_layer(act_scales, target_layer_idx=31)
    #print("Getting reording index...")
    #index_filename = f'{args.save_dir}/{model_name}_reorder_index_{args.dataset}.pt'
    #if os.path.exists(index_filename):
    #    print("Loading cached reording index from disk...")
    #    reorder_index = torch.load(index_filename)
