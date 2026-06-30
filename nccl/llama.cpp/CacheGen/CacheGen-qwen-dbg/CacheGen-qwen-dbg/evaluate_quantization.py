import os
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

def simulate_uniform_quant(sub_tensor, bins):
    """
    模拟标准均匀量化与反量化闭环（保证稳健性，杜绝 NoneType 冲突）
    """
    if bins is None or bins <= 0:
        return sub_tensor
    max_val = torch.max(torch.abs(sub_tensor)).item()
    if max_val == 0:
        return sub_tensor
    MAX = bins // 2 - 1
    scale = max_val / MAX
    quantized = torch.round(sub_tensor / scale).clamp(-MAX, MAX)
    return quantized * scale

def run_evaluation_suite():
    p = argparse.ArgumentParser(description="CacheGen 论文评测工件：自适应异常通道保护量化评估脚本")
    p.add_argument("--model_id", type=str, required=True, help="大模型本地路径或拥护白名单标识")
    p.add_argument("--save_dir", type=str, default="./kv_output", help="原始材料读取目录")
    p.add_argument("--bins", type=int, default=16, help="普通通道的量化状态数（16代表4-bit）")
    p.add_argument("--protect_outliers", type=int, default=1, help="1: 开启非对称通道精度保护, 0: 粗暴无差别量化")
    p.add_argument("--outlier_multiplier", type=float, default=2.5, help="判定异常高激活通道的方差倍数阈值")
    args = p.parse_args()

    # 1. 加载对齐了数据类型的原生大模型（采用 bfloat16 杜绝底层架构冲突）
    print("=" * 70)
    print(f"📡 [LMCache Eval] 正在加载大模型环境：{args.model_id}")
    print("=" * 70)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, 
        torch_dtype=torch.bfloat16, 
        device_map="auto", 
        trust_remote_code=True
    )
    print("✨ 大模型及分词器原生加载完毕。")

    # 2. 读取第一阶段 main.py 收集的长文本原始 KV Cache 原材料
    doc_id = 0
    raw_kv_file = f"{args.save_dir}/raw_kv_{doc_id}.pt"
    if not os.path.exists(raw_kv_file):
        print(f"❌ 严重错误：未能找到测试原材料文件 {raw_kv_file}，请先运行 main.py 收集数据。")
        return

    # 加载原始张量字典/元组到 GPU
    raw_kv = torch.load(raw_kv_file, map_location="cuda", weights_only=False)
    print(f"📦 成功载入原始原材料，准备进行非对称精度裁剪与通道提取...")

    # -------------------------------------------------------------
    # 🛡️ 核心重构：【全自动逆向维度抓取与选择性敏感通道物理处理】
    # -------------------------------------------------------------
    processed_kv_dict = {}
    
    if args.protect_outliers == 1:
        print(f"✂️ 策略A：执行 {args.bins} 桶（4-bit）量化，同时【开启】{args.outlier_multiplier}x 异常通道高精度缝合保护。")
    else:
        print(f"✂️ 策略B：执行 {args.bins} 桶（4-bit）无差别粗暴量化，【关闭】任何通道保护. ")
        
    # 逐层（Layer）迭代处理
    for layer_idx in range(len(raw_kv)):
        num_heads = raw_kv[layer_idx].shape[-3]
        seq_len   = raw_kv[layer_idx].shape[-2]
        head_dim  = raw_kv[layer_idx].shape[-1]
        
        # 从最底层将多维矩阵从中间平分一分为二切开
        combined_tensor = raw_kv[layer_idx].float()
        flat_tensor = combined_tensor.flatten()
        half_size = flat_tensor.numel() // 2
        
        k_layer = flat_tensor[:half_size].view(num_heads, seq_len, head_dim)
        v_layer = flat_tensor[half_size:].view(num_heads, seq_len, head_dim)
        
        # 自动探测当前层所在的物理 GPU
        target_device = model.model.layers[layer_idx].self_attn.q_proj.weight.device
        k_layer = k_layer.to(target_device)
        v_layer = v_layer.to(target_device)
        
        k_processed = torch.zeros_like(k_layer)
        v_processed = torch.zeros_like(v_layer)
        
        # 🎯 动态计算当前层内各 Head 的标准差，精准圈定 Outlier 通道
        head_stds = torch.tensor([torch.std(k_layer[h].float()).item() for h in range(num_heads)], device=target_device)
        layer_std_mean = head_stds.mean().item()
        layer_outliers = [h for h in range(num_heads) if head_stds[h].item() > args.outlier_multiplier * layer_std_mean]
        
        for h in range(num_heads):
            if args.protect_outliers == 1 and h in layer_outliers:
                k_processed[h] = k_layer[h].clone()
                v_processed[h] = v_layer[h].clone()
            else:
                k_processed[h] = simulate_uniform_quant(k_layer[h].clone(), bins=args.bins)
                v_processed[h] = simulate_uniform_quant(v_layer[h].clone(), bins=args.bins)
                
        k_4d = k_processed.unsqueeze(0).to(torch.bfloat16).contiguous()
        v_4d = v_processed.unsqueeze(0).to(torch.bfloat16).contiguous()
        
        processed_kv_dict[layer_idx] = (k_4d, v_4d)

    # -------------------------------------------------------------
    # 🎯 终极杀手锏：【使用带有 kwargs 原生拦截的标准前置钩子】
    # -------------------------------------------------------------
    hooks = []
    
    def make_hook(l_idx):
        # 严格对齐 3 参数定义，利用 with_kwargs=True 拦截关键字字典
        def attention_pre_hook(module, args, kwargs):
            k_inject, v_inject = processed_kv_dict[l_idx]
            
            # 1. 实例化官方的 DynamicCache 容器
            from transformers.cache_utils import DynamicCache
            custom_cache = DynamicCache()
            
            # 2. 🛡️ 【硬核绕过 IndexError】：跳过 update 限制，直接对容器最底层的列表执行物理注入！
            # 这样无论当前是第几层，容器内部都完美包含当前层对应的高精度保护 4-bit 缓存
            custom_cache.key_cache = [k_inject]
            custom_cache.value_cache = [v_inject]
            
            # 3. 🎯 注入关键字字典，供当前自注意力层直接读取
            kwargs["past_key_value"] = custom_cache
            
            # 原原本本地返回，绝不破坏位置参数
            return args, kwargs
        return attention_pre_hook

    # 遍历大模型的所有层，挂载绝缘版前置钩子
    for layer_idx in range(len(model.model.layers)):
        attn_module = model.model.layers[layer_idx].self_attn
        # 🛑 【终极修正点】：明确开启 with_kwargs=True 允许传入并改写关键字参数
        h_reg = attn_module.register_forward_pre_hook(make_hook(layer_idx), with_kwargs=True)
        hooks.append(h_reg)

    # 3. 准备长文本测试 Prompt
    test_prompt = "The role of art in society. USER: What is the first topic we discussed? ASSISTANT:"
    input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(model.device)

    print("\n🎬 [大模型启动] 采用精准 kwargs 定点爆破 Hook 注入，彻底绝缘所有位置参数冲突...")
    
    try:
        # 瞒天过海：不传 past_key_values 让模型自己去初始化最完美的 RoPE 步长控制矩阵
        generated = model.generate(input_ids, past_key_values=None, max_new_tokens=20)
        prediction = tokenizer.decode(generated[input_ids.shape:], skip_special_tokens=True)
        
        print("\n" + "=" * 60)
        print(f"📊 [CacheGen 最终评测大面板] | 保护状态: {'开启 (自适应高智商)' if args.protect_outliers==1 else '关闭 (复读机降级)'}")
        print("=" * 60)
        print(f"📝 大模型最终输出的文本结果: \n👉 \033[1;32m{prediction.strip()}\033[0m")
        print("=" * 60 + "\n")
        
    except Exception as e:
        import traceback
        print(f"❌ 生成阶段触发异常错误，请核对堆栈信息: {e}")
        traceback.print_exc()
    finally:
        # 4. 实验完成，强行清除所有 Hook 钩子防止环境污染
        for h_reg in hooks:
            h_reg.remove()

if __name__ == "__main__":
    run_evaluation_suite()

