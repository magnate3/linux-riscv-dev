
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
import json
from src.utils import *
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--bins", type=int)
p.add_argument("--results_dir", type=str, default = None)
p.add_argument("--results_str", type=str, default = "gt")
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", type=int)
args = p.parse_args()
import torch

import torch

def mixed_bit_channels_quant(raw_kv_data, base_bit=4, protected_bit=8, outlier_multiplier=2.5):
    """
    【学术对赌修复版】：完美对齐层轴的跨通道多比特混合量化器
    """
    # 判断传入的数据是列表/元组格式，还是已经被堆叠成的一个大张量
    is_list_or_tuple = isinstance(raw_kv_data, (list, tuple))
    num_layers = len(raw_kv_data)
    
    processed_layers = []
    
    # 💡 核心修正：必须一层一层地处理，才能让 25,079,808 降解为单层的 1,045,004
    for layer_idx in range(num_layers):
        layer_data = raw_kv_data[layer_idx]
        
        # 兼容处理：确保 layer_data 内部处理时转换为 float 并克隆
        orig_dtype = layer_data.dtype if hasattr(layer_data, 'dtype') else torch.float32
        
        # 动态反向提取单层真正的核心维度 [Heads, Seq, Dim]
        num_heads = layer_data.shape[-3]
        seq_len   = layer_data.shape[-2]
        head_dim  = layer_data.shape[-1]
        
        # 将这一层的数据拉平，并一分为二（平分出这层的 Key 和 Value）
        flat_layer = layer_data.clone().float().flatten()
        half_size = flat_layer.numel() // 2
        
        k_layer = flat_layer[:half_size].view(num_heads, seq_len, head_dim)
        v_layer = flat_layer[half_size:].view(num_heads, seq_len, head_dim)
        
        # 自动探测物理设备（多卡安全对齐）
        target_device = layer_data.device if hasattr(layer_data, 'device') else 'cuda'
        k_layer = k_layer.to(target_device)
        v_layer = v_layer.to(target_device)
        
        k_processed = torch.zeros_like(k_layer)
        v_processed = torch.zeros_like(v_layer)
        
        # 🔍 动态计算这一层每个 Head 的标准差，筛选出当前层的 Outlier 通道
        head_stds = torch.tensor([torch.std(k_layer[h]).item() for h in range(num_heads)], device=target_device)
        layer_std_mean = head_stds.mean().item()
        layer_outliers = [h for h in range(num_heads) if head_stds[h].item() > outlier_multiplier * layer_std_mean]
        
        # ✂️ 跨通道多比特分配
        for h in range(num_heads):
            if h in layer_outliers:
                # 异常通道守护：拓宽格子到 2^protected_bit
                outlier_bins = 1 << protected_bit
                k_processed[h] = simulate_uniform_quant_core(k_layer[h].clone(), bins=outlier_bins)
                v_processed[h] = simulate_uniform_quant_core(v_layer[h].clone(), bins=outlier_bins)
            else:
                # 普通通道降级：限制在 2^base_bit
                normal_bins = 1 << base_bit
                k_processed[h] = simulate_uniform_quant_core(k_layer[h].clone(), bins=normal_bins)
                v_processed[h] = simulate_uniform_quant_core(v_layer[h].clone(), bins=normal_bins)
                
        # 将这层切开加工完的 K 和 V 重新合并复原为单层的原始形态
        layer_final_flat = torch.cat([k_processed.flatten(), v_processed.flatten()])
        layer_restored = layer_final_flat.view(layer_data.shape).to(orig_dtype).contiguous()
        
        processed_layers.append(layer_restored)
        
    # 根据原本的全局数据类型，将其重新封装返回
    if is_list_or_tuple:
        return tuple(processed_layers) if isinstance(raw_kv_data, tuple) else processed_layers
    else:
        # 如果原本是一个被 stack 起来的大张量，重新在 dim=0 上拼接
        return torch.stack(processed_layers).contiguous()

def simulate_uniform_quant_core(sub_tensor, bins):
    """ 底层标准对称量化与反量化闭环 """
    max_val = torch.max(torch.abs(sub_tensor)).item()
    if max_val == 0:
        return sub_tensor
    MAX = bins // 2 - 1
    scale = max_val / MAX
    quantized = torch.round(sub_tensor / scale).clamp(-MAX, MAX)
    return quantized * scale


if __name__ == "__main__":
    # Check if save_dir exists
    
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    print("Model and tokenizer loaded")
    data =  load_testcases(DATASET_TO_PATH[args.dataset_name])
    layer_to_device_id = {}
    kv = pickle.load(open(f"{args.save_dir}/raw_kv_{args.start}.pkl", "rb"))
    for i in range(len(kv)):
        layer_to_device_id[i] = kv[i][0].device.index
    average_acc = []
    average_size = []
    for doc_id in range(args.start, args.end):
        raw_kv = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        # ⬇️ 【强行插入以下拦截代码：将 raw_kv 在原地重新执行跨通道多比特处理】 ⬇️
        # 我们对普通通道分配 4-bit (base_bit=4)，对高方差通道强制进行 8-bit 的高阶精度留存
        print(f"📡 [混合比特基线] 正在执行多比特通道分配：普通通道 4-bit, 敏感通道 8-bit 保护...")
        raw_kv = mixed_bit_channels_quant(
            raw_kv, 
            base_bit=4,           # 👈 改变这个数值调整基础比特
            protected_bit=8,      # 👈 改变这个数值调整保护通道比特
            outlier_multiplier=2.5
        )
        # ⬆️ 【拦截完毕】 ⬆️
        kv, max_tensors = default_quantization(raw_kv, args.bins, layer_to_device_id)
        torch.save(kv, f"{args.save_dir}/quant_kv_{doc_id}.pt")
        torch.save(max_tensors, f"{args.save_dir}/max_tensors_{doc_id}.pt")
        #read the file and compute the size of the kv
        file_size = os.path.getsize(f"{args.save_dir}/quant_kv_{doc_id}.pt") + \
                    os.path.getsize(f"{args.save_dir}/max_tensors_{doc_id}.pt")
        average_size += [file_size/1e6]
        
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        # Load and dequantize 
        kv = torch.load(f"{args.save_dir}/quant_kv_{doc_id}.pt")
        max_tensors = torch.load(f"{args.save_dir}/max_tensors_{doc_id}.pt")
        kv = dequantize_kv(kv, max_tensors, args, layer_to_device_id)
        
        generated = model.generate(input_ids, past_key_values=kv, max_new_tokens = 20)
        prediction = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"doc id: {doc_id}", tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))
        if args.calculate_metric == 1:
            if args.dataset_name == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                average_acc += [metric]
            elif args.dataset_name == "nqa" or args.dataset_name == "tqa":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                average_acc += [metric]
    if args.dataset_name == "longchat":
        metric_name = "accuracy"
    else:
        metric_name = "F1 score"
    if args.calculate_metric == 1:
        print(f"Average quantization {metric_name} is: ", np.mean(average_acc))
    print("Average size is: ", np.mean(average_size))
