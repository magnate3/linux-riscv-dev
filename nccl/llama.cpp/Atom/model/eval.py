import torch
import torch.nn as nn
from tqdm import tqdm
import fnmatch

def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print(model)
    print(model.model)
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'cache_position': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['cache_position'] = kwargs.get('cache_position', None)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    cache_position = cache['cache_position']
    if cache_position is None:
        cache_position = torch.arange(model.seqlen, device=dev)

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            position_embeddings = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)
            seq_len = inps[j].unsqueeze(0).shape[1]
            current_attention_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=dev))
            current_attention_mask = (1.0 - current_attention_mask) * torch.finfo(dtype).min
            outs[j] = layer(
                hidden_states=inps[j].unsqueeze(0), 
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                past_key_values=None, 
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
        
        layers[i] = layer.cpu()
        del layer
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        print(f'Sample {i+1}/{nsamples}, NLL: {neg_log_likelihood.item()}')
        print(shift_labels.shape,shift_logits.shape, shift_logits[0][-1].max(), shift_logits[0][-1].min(), shift_logits[0][-1][shift_labels[0][-1]])
        print(f'shift_logits: {shift_logits}')
        print(f'shift_labels: {shift_labels}')
        print(f'loss: {loss.item()}')
        print(f'neg_log_likelihood: {neg_log_likelihood.item()}')
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))

    return ppl.item()
@torch.no_grad()
def opt_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev) 
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev) 
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']

    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.decoder.final_layer_norm is not None:
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
    if model.model.decoder.project_out is not None:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.decoder.final_layer_norm is not None:
            hidden_states = model.model.decoder.final_layer_norm(hidden_states)
        if model.model.decoder.project_out is not None:
            hidden_states = model.model.decoder.project_out(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache
    return ppl.item()

@torch.no_grad()
def qwen_eval(model, testenc, dev):
    """适用于Qwen3模型的评估函数"""
    print(f"Evaluating Qwen3 model on device: {dev}")
    
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    layers = model.model.layers

    # 获取Qwen3特有的配置
    has_sliding_layers = getattr(model.model, 'has_sliding_layers', False)
    print(f"Model has sliding layers: {has_sliding_layers}")
    
    # 将必要的组件移到设备上
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None, 'cache_position': None}

    # 使用与GPTQ量化相同的Catcher类
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self._module = module  # 保存原始模块引用
            
        def __getattr__(self, name):
            # 动态代理所有属性访问到原始模块
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self._module, name)
            
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            cache['cache_position'] = kwargs.get('cache_position', None)
            raise ValueError
    
    # 捕获输入
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    
    print(f"Captured {cache['i']} samples")
    
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    cache_position = cache['cache_position']
    
    if cache_position is None:
        cache_position = torch.arange(model.seqlen, device=dev)

    # 为Qwen3准备掩码映射
    causal_mask_mapping = {}
    
    # 导入Qwen3的掩码函数（如果可用）
    try:
        from transformers.models.qwen3.modeling_qwen3 import create_causal_mask, create_sliding_window_causal_mask
        qwen_mask_functions_available = True
        print("Using Qwen3 mask functions")
    except ImportError:
        qwen_mask_functions_available = False
        print("Warning: Could not import Qwen3 mask functions, using fallback masks")
        
        # 简单的替代实现
        def create_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids):
            seq_len = input_embeds.shape[1]
            mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=input_embeds.device))
            mask = mask.masked_fill(mask == 0, float('-inf'))
            return mask
        
        def create_sliding_window_causal_mask(config, input_embeds, attention_mask, cache_position, past_key_values, position_ids):
            seq_len = input_embeds.shape[1]
            sliding_window = getattr(config, 'sliding_window', 4096)  # 默认值
            mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=input_embeds.device))
            # 创建滑动窗口掩码（简化的实现）
            if seq_len > sliding_window:
                for i in range(seq_len):
                    mask[:, :, i, max(0, i-sliding_window):i+1] = 1
                mask = mask.masked_fill(mask == 0, float('-inf'))
            return mask

    # 为每一层创建掩码映射
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        attention_type = getattr(layer, 'attention_type', 'full_attention')
        
        if attention_type not in causal_mask_mapping:
            # 创建该类型对应的掩码
            mask_kwargs = {
                "config": model.config,
                "input_embeds": inps[0].unsqueeze(0) if nsamples > 0 else torch.randn(1, model.seqlen, model.config.hidden_size, device=dev),
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            
            if attention_type == "sliding_attention" and qwen_mask_functions_available:
                causal_mask_mapping[attention_type] = create_sliding_window_causal_mask(**mask_kwargs)
            else:
                causal_mask_mapping[attention_type] = create_causal_mask(**mask_kwargs)
        
        layer = layer.cpu()
        del layer
        torch.cuda.empty_cache()

    print(f"Created masks for attention types: {list(causal_mask_mapping.keys())}")

    # 逐层处理
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        
        # 获取当前层的注意力类型
        attention_type = getattr(layer, 'attention_type', 'full_attention')
        
        for j in range(nsamples):
            # 生成位置嵌入
            position_embeddings = model.model.rotary_emb(inps[j].unsqueeze(0), position_ids)
            
            # 选择正确的掩码
            # if attention_type in causal_mask_mapping:
            #     current_attention_mask = causal_mask_mapping[attention_type]
            # else:
                # 默认使用全注意力掩码
            seq_len = inps[j].unsqueeze(0).shape[1]
            current_attention_mask = torch.tril(torch.ones(1, 1, seq_len, seq_len, device=dev))
            current_attention_mask = (1.0 - current_attention_mask) * torch.finfo(dtype).min
            # print(current_attention_mask)
            # 前向传播
            outs[j] = layer(
                hidden_states=inps[j].unsqueeze(0), 
                attention_mask=current_attention_mask,
                position_ids=position_ids,
                past_key_values=None, 
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings
            )
        
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        
        # 交换输入输出
        inps, outs = outs, inps

    # 处理最后的归一化层和lm_head
    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        
        # 计算损失
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        
        # 打印调试信息
        print(f'Sample {i+1}/{nsamples}, NLL: {neg_log_likelihood.item()}')
        print(shift_labels.shape,shift_logits.shape, shift_logits[0][-1].max(), shift_logits[0][-1].min(), shift_logits[0][-1][shift_labels[0][-1]])
        print(f'shift_logits: {shift_logits}')
        print(f'shift_labels: {shift_labels}')
        print(f'loss: {loss.item()}')
        print(f'neg_log_likelihood: {neg_log_likelihood.item()}')
        # 检查logits分布
        if shift_logits.numel() > 0:
            logits_flat = shift_logits.view(-1)
            print(f'  Logits stats - Min: {logits_flat.min().item():.4f}, Max: {logits_flat.max().item():.4f}, Mean: {logits_flat.mean().item():.4f}')
            print(f'  Loss: {loss.item():.6f}')
        
        nlls.append(neg_log_likelihood)
    
    # 计算最终PPL
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"\nFinal PPL: {ppl.item():.4f}")

    return ppl.item()