
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import time
import pickle
import torch
import numpy as np

from lmcache.config import LMCacheEngineConfig, LMCacheEngineMetadata
from lmcache.storage_backend.serde.cachegen_encoder import CacheGenSerializer
import json
from src.utils import *
from src.attention_monkey_patch import replace_llama_forward_with_reuse_forward
from turboquant import TurboQuantCache
import torch
from transformers import DynamicCache

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
import numpy as np

# 🛠️ 核心修复：兼容新版 NumPy，动态将 trapz 重定向为 trapezoid 或 scipy 别名
if not hasattr(np, 'trapz'):
    # 如果是 NumPy 2.x 移除了 trapz
    if hasattr(np, 'trapezoid'):
        np.trapz = np.trapezoid
    else:
        # 极端保底：使用 scipy 或者是手动实现梯形积分
        try:
            from scipy.integrate import trapz
            np.trapz = trapz
        except ImportError:
            # 极简梯形积分实现
            def _trapz(y, x):
                return np.sum((y[1:] + y[:-1]) * np.diff(x)) / 2.0
            np.trapz = _trapz

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
        #kv = dequantize_kv(kv, max_tensors, args, layer_to_device_id)
        
        # ==========================================
        # 1. 干净地创建 TurboQuant 顶级缓存实例
        # ==========================================
        kv_cache = TurboQuantCache()
        
        import turboquant.cache as tq_cache
        layer_class = None
        for name in ["TurboQuantLayer", "QuantizedLayer", "TurboLayer", "DynamicLayer"]:
            if hasattr(tq_cache, name):
                layer_class = getattr(tq_cache, name)
                break
        
        # ==========================================
        # 2. 🚨 核心注入：伪装层数 (解决锁死 q_len=1 的技术关键)
        # ==========================================
        # 我们定义一个特殊的 layers 属性：
        # - 在第一步（Prefill）时，让它返回空列表 []，这样 Hugging Face 就会认为“缓存里啥都没有”，从而允许大 Prompt (q_len=8165) 正常进入。
        # - 在第一步之后（Decoding），我们再把真正的 24 个层对象塞进去。
        
        real_layers = []
        num_layers = model.config.num_hidden_layers
        
        for i in range(num_layers):
            layer_obj = layer_class()
            layer_obj.is_initialized = False  # 必须是 False，让它第一步自己去初始化 _key_indices
            layer_obj.seen_tokens = 0
            
            # 显式初始化 TurboQuant 缺少的内部变量，防止 None 报错
            layer_obj._residual_keys = None
            layer_obj._residual_values = None
            layer_obj._key_indices = torch.empty(0, dtype=torch.int32, device="cuda") # 💡 彻底修复 num_errors 报错
            layer_obj._value_indices = torch.empty(0, dtype=torch.int32, device="cuda")
            
            layer_obj.bits = 4
            layer_obj.key_quant_bits = 4
            layer_obj.value_quant_bits = 4
            real_layers.append(layer_obj)
        
        # 挂载动态劫持的 layers 属性
        # 使用 property 拦截器
        class LayerProxy:
            def __init__(self):
                self.is_prefill = True
            def __len__(self):
                return 0 if self.is_prefill else len(real_layers)
            def __getitem__(self, idx):
                return real_layers[idx]
        
        proxy = LayerProxy()
        kv_cache.layers = proxy  # 将伪装代理挂载上去
        
        # 补齐高版本 Hugging Face 必需的长度和容量方法
        kv_cache.get_seq_length = lambda layer_idx=0: 0 if proxy.is_prefill else (real_layers[layer_idx]._key_indices.shape[-2] if real_layers[layer_idx]._key_indices.numel() > 0 else real_layers[layer_idx].seen_tokens)
        kv_cache.get_max_length = lambda: 131072
        
        # ==========================================
        # 3. 拦截模型的 forward，在第一步结束的刹那切换代理状态
        # ==========================================
        original_model_forward = model.forward
        
        def patched_model_forward(*args, **kwargs):
            # 正常执行前向传播
            res = original_model_forward(*args, **kwargs)
            # 💡 只要经历过一次 forward（即大 Prompt 推理完成），立即将代理切换为真实层，开启自回归模式
            proxy.is_prefill = False
            return res
        
        model.forward = patched_model_forward
        
        # ==========================================
        # 4. 完美运行标准的 model.generate
        # ==========================================
        print("🚀 启动整合型 TurboQuant 长文本推理...")
        outputs = model.generate(
            input_ids=input_ids,
            past_key_values=kv_cache,
            use_cache=True,
            max_new_tokens=32,  # 允许自回归走 32 步
        )
        
        print("🎉 推理成功完成！")

        
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
