
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
    
    # 判定阈值（方差是层平均方差的 2.5 倍以上判定为 Outlier）
    outlier_multiplier = 2.5 

    for doc_id in range(args.start, args.end):
        raw_kv = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        
        # ====================================================================
        # 🛡️ 【守护者机制第一阶段：完美兼容 3 维或 4 维的张量克隆原件】
        # ====================================================================
        outliers_backup = {} 
        
        for layer_idx in range(len(raw_kv)):
            k_tensor, v_tensor = raw_kv[layer_idx], raw_kv[layer_idx]
            
            # 💡 动态判断维度：如果已经是3维 [Heads, Seq, Dim]，我们就将其包回4维好做统一处理
            if len(k_tensor.shape) == 3:
                k_tensor_4d = k_tensor.unsqueeze(0)
                v_tensor_4d = v_tensor.unsqueeze(0)
            else:
                k_tensor_4d = k_tensor
                v_tensor_4d = v_tensor
                
            num_heads = k_tensor_4d.shape[1]
            head_stds = torch.zeros(num_heads, device=k_tensor.device)
            
            for h in range(num_heads):
                # 💡 使用 4 维安全的张量计算标准差
                head_stds[h] = torch.std(k_tensor_4d[0, h, :, :].float())
                
            layer_std_mean = head_stds.mean().item()
            
            layer_outliers = []
            for h in range(num_heads):
                if head_stds[h].item() > outlier_multiplier * layer_std_mean:
                    layer_outliers.append(h)
                    
            if layer_outliers:
                # 备份高精度原件（保持和原张量一模一样的维度）
                outliers_backup[layer_idx] = {
                    "heads": layer_outliers,
                    "is_3d": len(k_tensor.shape) == 3, # 记录原始维度状态
                    "k_raw": [k_tensor_4d[:, h:h+1, :, :].clone() for h in layer_outliers],
                    "v_raw": [v_tensor_4d[:, h:h+1, :, :].clone() for h in layer_outliers]
                }
        # ====================================================================

        # 运行原有的粗暴 4-bit 量化
        kv, max_tensors = default_quantization(raw_kv, args.bins, layer_to_device_id)
        torch.save(kv, f"{args.save_dir}/quant_kv_{doc_id}.pt")
        torch.save(max_tensors, f"{args.save_dir}/max_tensors_{doc_id}.pt")
        
        file_size = os.path.getsize(f"{args.save_dir}/quant_kv_{doc_id}.pt") + \
                    os.path.getsize(f"{args.save_dir}/max_tensors_{doc_id}.pt")
        average_size += [file_size/1e6]

        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        
        # 读取并反量化
        kv = torch.load(f"{args.save_dir}/quant_kv_{doc_id}.pt")
        max_tensors = torch.load(f"{args.save_dir}/max_tensors_{doc_id}.pt")
        kv = dequantize_kv(kv, max_tensors, args, layer_to_device_id)

        # ====================================================================
        # 🛡️ 【守护者机制第二阶段：多维自适应缝合写回】
        # ====================================================================
        if outliers_backup:
            print(f"💡 [CacheGen 语义保护激活] 正在为 doc_{doc_id} 恢复敏感通道的高精度数据...")
            new_kv = []
            for layer_idx in range(len(kv)):
                k_assigned, v_assigned = kv[layer_idx], kv[layer_idx]
                
                if layer_idx in outliers_backup:
                    is_3d = outliers_backup[layer_idx]["is_3d"]
                    
                    for idx, h in enumerate(outliers_backup[layer_idx]["heads"]):
                        raw_k_head_4d = outliers_backup[layer_idx]["k_raw"][idx]
                        raw_v_head_4d = outliers_backup[layer_idx]["v_raw"][idx]
                        
                        if is_3d:
                            # 如果反量化后是 3 维结构，强行降维覆盖对应通道
                            k_assigned[h:h+1, :, :] = raw_k_head_4d.squeeze(0).squeeze(0).to(k_assigned.dtype)
                            v_assigned[h:h+1, :, :] = raw_v_head_4d.squeeze(0).squeeze(0).to(v_assigned.dtype)
                        else:
                            # 如果是 4 维结构，正常按通道覆盖
                            k_assigned[:, h:h+1, :, :] = raw_k_head_4d.to(k_assigned.dtype)
                            v_assigned[:, h:h+1, :, :] = raw_v_head_4d.to(v_assigned.dtype)
                        
                new_kv.append((k_assigned, v_assigned))
            
            kv = tuple(new_kv)
        # ====================================================================

        # 喂给大模型生成
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

