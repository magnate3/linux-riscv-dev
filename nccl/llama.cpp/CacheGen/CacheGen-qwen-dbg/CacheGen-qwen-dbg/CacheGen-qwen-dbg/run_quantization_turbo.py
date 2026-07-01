
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
import numpy as np

# 检查当前 NumPy 是否缺失 trapz，如果缺失则使用 scipy 或替代方案进行动态修复
if not hasattr(np, "trapz"):
    try:
        import scipy.integrate as integrate
        np.trapz = integrate.trapezoid
    except ImportError:
        # 如果没有安装 scipy，使用替代写法（也可以直接实现一个简单的梯形积分）
        def _trapz(y, x=None, dx=1.0, axis=-1):
            if x is not None:
                return np.sum((y[..., 1:] + y[..., :-1]) * np.diff(x) / 2.0, axis=axis)
            return np.sum((y[..., 1:] + y[..., :-1]) * dx / 2.0, axis=axis)
        np.trapz = _trapz

# 之后再正常引入其他库
import torch
import turboquant
from transformers import DynamicCache

import openai

# =========== 针对 OpenAI >= 1.0.0 的热修复兼容层 ===========
if hasattr(openai, "OpenAI"):
    # 实例化新版客户端
    _client = openai.OpenAI(
        api_key=openai.api_key or "YOUR_API_KEY", # 确保读取了环境变量或在此处填写
        base_url=getattr(openai, "api_base", None) # 如果使用了中转代理
    )
    
    # 猴子补丁：用新版 API 模拟旧版的 ChatCompletion.create
    class CompatibilityChatCompletion:
        @staticmethod
        def create(*args, **kwargs):
            # 将旧版参数关键字转换为新版
            if "model" not in kwargs:
                kwargs["model"] = "gpt-3.5-turbo"
            # 移除旧版可能存在的过时参数
            kwargs.pop("api_base", None)
            return _client.chat.completions.create(*args, **kwargs)

    # 动态挂载回旧版路径，让 calculate_acc 内部的调用无缝通过
    openai.ChatCompletion = CompatibilityChatCompletion
# =========================================================




p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--bins", type=int)
p.add_argument("--bits", type=int)
p.add_argument("--results_dir", type=str, default = None)
p.add_argument("--results_str", type=str, default = "gt")
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", type=int)
p.add_argument("--method", type=str, default = "turboquant")
args = p.parse_args()
def turboquant_quantization(raw_kv, bits, layer_to_device_id):
    """
    使用 TurboQuant 算法量化 KV Cache
    """
    quant_kv = []
    meta_tensors = []  # 用来存 Lloyd-Max 的码表、缩放因子或者旋转矩阵参数
    
    # 遍历每一层 (每一层包含 key 和 value)
    for layer_idx, (key, value) in enumerate(raw_kv):
        device_id = layer_to_device_id[layer_idx]
        key = key.to(f"cuda:{device_id}")
        value = value.to(f"cuda:{device_id}")
        
        # 1. 实例化或调用 TurboQuant 的量化器 (根据你的实际 turboquant API 调整)
        # 这里的 dim 通常是 head_dim
        dim = key.shape[-1] 
        quantizer = turboquant.TurboQuantMSE(dim=dim, bits=bits, device=f"cuda:{device_id}")
        
        # 2. 对 Key 和 Value 进行量化
        # TurboQuant 核心：先进行 Walsh-Hadamard 变换旋转，再执行 Lloyd-Max 量化
        k_indices, k_codes = quantizer.quantize(key)
        v_indices, v_codes = quantizer.quantize(value)
        
        # 3. 保存量化结果与元数据 (保持与原脚本结构一致，转回 CPU 准备持久化)
        quant_kv.append((k_indices.cpu(), v_indices.cpu()))
        meta_tensors.append((k_codes.cpu(), v_codes.cpu()))
        
    return quant_kv, meta_tensors

def turboquant_dequantize_kv(quant_kv, meta_tensors, layer_to_device_id, bits):
    """
    将 TurboQuant 量化后的数据解量化，恢复为模型可读的 FP16/BF16 格式
    """
    dequant_kv = []
    
    for layer_idx, (k_indices, v_indices) in enumerate(quant_kv):
        device_id = layer_to_device_id[layer_idx]
        k_indices = k_indices.to(f"cuda:{device_id}")
        v_indices = v_indices.to(f"cuda:{device_id}")
        k_codes, v_codes = meta_tensors[layer_idx]
        k_codes = k_codes.to(f"cuda:{device_id}")
        v_codes = v_codes.to(f"cuda:{device_id}")
        
        # 重建量化器
        dim = k_indices.shape[-1] # 或者从原始配置获取 head_dim
        quantizer = turboquant.TurboQuantMSE(dim=dim, bits=bits, device=f"cuda:{device_id}")
        
        # 反量化（逆旋转 + 逆 Lloyd-Max）
        dequant_k = quantizer.dequantize(k_indices, k_codes)
        dequant_v = quantizer.dequantize(v_indices, v_codes)
        
        dequant_kv.append((dequant_k, dequant_v))
        
    return tuple(dequant_kv)

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
        
        # ======= 修改点 1：根据方法选择不同的量化函数 =======
        if getattr(args, "method", "default") == "turboquant":
            # TurboQuant 通常使用 3-bit 或 4-bit，这里可以用 args.bits 传入
            target_bits = getattr(args, "bits", 3) 
            kv, max_tensors = turboquant_quantization(raw_kv, target_bits, layer_to_device_id)
        else:
            # CacheGen 原版的均匀量化
            kv, max_tensors = default_quantization(raw_kv, args.bins, layer_to_device_id)
        # ====================================================

        torch.save(kv, f"{args.save_dir}/quant_kv_{doc_id}.pt")
        torch.save(max_tensors, f"{args.save_dir}/max_tensors_{doc_id}.pt")
        
        # 计算文件大小（由于 TurboQuant 存的是低比特 index，文件体积会显著缩小）
        file_size = os.path.getsize(f"{args.save_dir}/quant_kv_{doc_id}.pt") + \
                    os.path.getsize(f"{args.save_dir}/max_tensors_{doc_id}.pt")
        average_size += [file_size/1e6]

        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        
        # 从磁盘加载
        kv = torch.load(f"{args.save_dir}/quant_kv_{doc_id}.pt")
        max_tensors = torch.load(f"{args.save_dir}/max_tensors_{doc_id}.pt")
        
        # ======= 修改点 2：根据方法选择不同的反量化函数 =======
        #if getattr(args, "method", "default") == "turboquant":
        #    target_bits = getattr(args, "bits", 3)
        #    kv = turboquant_dequantize_kv(kv, max_tensors, layer_to_device_id, target_bits)
        #else:
        #    kv = dequantize_kv(kv, max_tensors, args, layer_to_device_id)
        #    # ==================== 新增以下修复代码 ====================
        #    # 将反量化后的 KV cache 转换为模型当前参数的实际数据类型（如 bfloat16）
        #    kv = tuple(tuple(t.to(dtype=model.dtype) for t in layer) for layer in kv)
        
        
        # 1. 执行反量化
        if getattr(args, "method", "default") == "turboquant":
            target_bits = getattr(args, "bits", 3)
            kv = turboquant_dequantize_kv(kv, max_tensors, layer_to_device_id, target_bits)
        else:
            kv = dequantize_kv(kv, max_tensors, args, layer_to_device_id)

        # ==================== 强制数据类型与维度对齐 ====================
        from transformers import DynamicCache
        
        kv_cache_obj = DynamicCache()
        
        for layer_idx, (k_tensor, v_tensor) in enumerate(kv):
            # 💡 核心修复点 1：强制将反量化出来的 float 转换为模型需要的 bfloat16
            k_tensor = k_tensor.to(torch.bfloat16)
            v_tensor = v_tensor.to(torch.bfloat16)
            
            # 💡 核心修复点 2：强制确保张量处于 4 维状态 [batch, num_heads, seq_len, head_dim]
            if k_tensor.ndim == 3:
                k_tensor = k_tensor.unsqueeze(0)
                v_tensor = v_tensor.unsqueeze(0)
                
            kv_cache_obj.update(k_tensor, v_tensor, layer_idx=layer_idx)
        # =========================================================================

        # 3. 将类型和维度完美对齐的缓存对象传给模型
        generated = model.generate(
            input_ids, 
            past_key_values=kv_cache_obj,  
            max_new_tokens=20
        )


        prediction = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(f"doc id: {doc_id}", tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True))
        # ==================== 智能防误杀清洗逻辑 ====================
        ## 1. Token 级别切片
        #prompt_len = input_ids.shape[1]
        #new_tokens = generated[0][prompt_len:]
        #prediction = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        #
        ## 2. 智能文本切分：如果是包含长指引的复读，通常答案在 "USER:" 或 "Question:" 之前
        ## 或者是类似 "The second topic is [答案]" 的结构
        #for stop_word in ["USER:", "ASSISTANT:", "Question:", "Only give me"]:
        #    if stop_word in prediction:
        #        # 答案通常在这些新出现的模板引导词之前
        #        prediction = prediction.split(stop_word)[0].strip()
        #
        ## 3. 针对 LongChat 任务的特异性清洗：
        ## 如果模型输出了类似 "What is the second topic we discussed? Only give me the topic name." 这样的 Prompt 残留
        ## 我们寻找它后面可能跟着的真正答句，例如 "The second topic is..." 
        #lower_pred = prediction.lower()
        #for pattern in ["the second topic was the topic of", "the second topic is"]:
        #    if pattern in lower_pred:
        #        idx = lower_pred.find(pattern)
        #        prediction = prediction[idx + len(pattern):].strip()
        #        
        ## 4. 去除可能残留在句尾的标点符号
        #prediction = prediction.rstrip(".?，。？ ")
        #
        #print(f"doc id: {doc_id} | Final Cleaned Predict: {prediction}")
        ## =========================================================================

        
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
