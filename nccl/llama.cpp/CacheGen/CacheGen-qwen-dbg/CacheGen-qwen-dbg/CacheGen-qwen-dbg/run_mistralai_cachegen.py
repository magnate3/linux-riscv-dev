
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import numpy as np
import os
import time
import pickle
import torch
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward
from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
from lmcache.storage_backend.serde.cachegen_decoder import CacheGenDeserializer
import json
from src.utils import *

p = argparse.ArgumentParser()

p.add_argument("--model_id", type = str, default = "lmsys/longchat-7b-16k")
p.add_argument("--save_dir", type=str, default = None)
p.add_argument("--num_gpus", type=int, default = 1)
p.add_argument("--max_gpu_memory", type=int, default=48, help="Default max GPU memory in GiB on A40")
p.add_argument("--path_to_context", type=str, help="The directory where the contexts are stored. ")
p.add_argument("--start", type=int, default = 0)
p.add_argument("--end", type=int, default = 1)
p.add_argument("--encoded_dir", type=str, default = None)
p.add_argument("--results_dir", type=str, default = None)
p.add_argument("--results_str", type=str, default = "results")
p.add_argument("--dataset_name", type=str)
p.add_argument("--calculate_metric", type=int)

args = p.parse_args()

if __name__ == "__main__":
    # Check if encoded_dir is exists
    if not os.path.exists(args.encoded_dir):
        os.makedirs(args.encoded_dir, exist_ok=True)
    # Check if results_dir is exists
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir, exist_ok=True)
    # Read data from jsonl
    data =  load_testcases(DATASET_TO_PATH[args.dataset_name])
    os.environ["QUANT_LEVEL"] = "2"
    kv_tokens = []
    # Start encoding
    layer_to_device_id = {}
    kv = pickle.load(open(f"{args.save_dir}/raw_kv_{args.start}.pkl", "rb"))
    for i in range(len(kv)):
        layer_to_device_id[i] = kv[i][0].device.index
    avg_size = []
    for doc_id in range(args.start, args.end):
        key_value = torch.load(f"{args.save_dir}/raw_kv_{doc_id}.pt")
        # ==================== 针对 LMCache 新版核心修复 ====================
        # 1. 从加载的 raw_kv 中动态获取各个维度的形状
        num_layers = len(key_value)
        # 拿到第一层的 key tensor 形状，通常为 [batch_size, num_kv_heads, seq_len, head_dim]
        # 注：如果前向传播时 batch 维度被压掉了（ndim=3），则补为 1
        first_key_tensor = key_value[0][0]
        if first_key_tensor.ndim == 3:
            batch_size = 1
            num_kv_heads, seq_len, head_dim = first_key_tensor.shape
        else:
            batch_size, num_kv_heads, seq_len, head_dim = first_key_tensor.shape
        
        # 2. 组装成 LMCache 期待的 kv_shape 元组
        # 根据 LMCache 官方设计，标准的 kv_shape 内部格式通常包含：(batch_size, num_kv_heads, seq_len, head_dim) 
        # 或者某些版本要求包含层数，这里我们传入标准的 4 维格式或 5 维格式
        calculated_kv_shape = (batch_size, num_kv_heads, seq_len, head_dim)
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=key_value.shape[-2])
        meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0, kv_shape=calculated_kv_shape)
        cachegen_serializer = CacheGenSerializer(lmcache_config, meta_data)
        bytes = cachegen_serializer.to_bytes(key_value)
        pickle.dump(bytes, open(f"{args.encoded_dir}/{doc_id}.pkl", "wb"))
        kv_tokens += [key_value.shape[-2]]
        # Averaging the size of KV cache 
        avg_size += [len(bytes)/1e6]
    # Start inferencing 
    decoded_kvs = []
    average_acc = []
    for doc_id in range(args.start, args.end):
        os.environ['DOC_ID'] = str(doc_id)
        print("Running inference for doc_id: ", doc_id)
        lmcache_config = LMCacheEngineConfig.from_defaults(chunk_size=kv_tokens[doc_id])
        meta_data = LMCacheEngineMetadata(model_name=args.model_id, fmt="huggingface", world_size=1, worker_id=0,kv_shape=calculated_kv_shape)
        deserializer = CacheGenDeserializer(lmcache_config, meta_data)
        bytes = pickle.load(open(f"{args.encoded_dir}/{doc_id}.pkl", "rb"))
        decoded_kv = deserializer.from_bytes(bytes)
        decoded_kvs += [decoded_kv.cpu()]
        
        
    model, tokenizer = define_model_and_tokenizer(args.model_id, num_gpus=args.num_gpus, max_gpu_memory=args.max_gpu_memory)
    for doc_id in range(args.start, args.end):
        decoded_kv = decoded_kvs[doc_id].cuda()
        decoded_kv = tensor_to_tuple(decoded_kv, layer_to_device_id)
        text = data[doc_id]['prompt']
        input_ids = tokenizer(text, return_tensors="pt").input_ids.cuda()
        #output = model.generate(input_ids, past_key_values=decoded_kv, max_new_tokens=20)
        # ==================== 针对 run_cachegen.py 的 BFloat16 彻底锁死修复 ====================
        from transformers import DynamicCache
        #import torch

        # 假设 run_cachegen.py 中解压恢复出来的 KV 变量名叫 decompressed_kv 或 kv
        # 如果你的脚本里变量名不同，请将下方的 'kv' 替换为你脚本里的实际变量名
        
        kv_cache_obj = DynamicCache()
        
        for layer_idx, (k_tensor, v_tensor) in enumerate(kv):
            # 1. 强制将张量移到 GPU 并锁死在 bfloat16 存储格式
            k_tensor = k_tensor.to(device=input_ids.device, dtype=torch.bfloat16)
            v_tensor = v_tensor.to(device=input_ids.device, dtype=torch.bfloat16)
            
            # 2. 强制确保张量处于 4 维状态 [batch, num_heads, seq_len, head_dim]
            if k_tensor.ndim == 3:
                k_tensor = k_tensor.unsqueeze(0)
                v_tensor = v_tensor.unsqueeze(0)
                
            # 3. 填充到标准容器，这会重置计数指针，彻底欺骗过 accelerate 的钩子
            kv_cache_obj.update(k_tensor, v_tensor, layer_idx=layer_idx)
        # ===================================================================================

        # 4. 找到原本的 model.generate，确保传入的是我们刚刚全新组装的 kv_cache_obj
        output = model.generate(
            input_ids, 
            past_key_values=kv_cache_obj,  # <--- 必须修改为这个变量
            max_new_tokens=20              # 评测时可保持原样或放大
        )

        prediction = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        if args.calculate_metric == 1:
            if args.dataset_name == "longchat":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id]['label'])
                average_acc += [metric]
            elif args.dataset_name == "nqa" or args.dataset_name == "tqa":
                metric = calculate_acc(args.dataset_name, prediction, data[doc_id])
                average_acc += [metric]
        if args.dataset_name == "longchat":
            print(prediction, data[doc_id]['label'][0])
    if args.dataset_name == "longchat":
        metric_name = "accuracy"
    else:
        metric_name = "F1 score"
    if args.calculate_metric == 1:
        print(f"Average cachegen {metric_name} is: ", np.mean(average_acc))
    print(f"Average size of KV cache: {np.mean(avg_size)}MB")
